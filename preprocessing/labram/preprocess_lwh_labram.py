"""Preprocess BDF EEG into the official LaBraM pretraining HDF5 layout.

References: https://github.com/935963004/LaBraM
  dataset_maker/shock/utils/eegUtils.py: 0.1-75 Hz, 50 Hz notch, 200 Hz, uV.
  data_processor/dataset.py: /<recording>/eeg and its chOrder attribute.
  engine_for_pretraining.py: divide uV by 100, reshape to (B,C,A,200).

Nonstandard physical units require an explicit --source-unit declaration.
No class labels, reference changes, ICA, z-scoring, or dataset splits are inferred.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import h5py
import mne
import numpy as np

DEFAULT_INPUT = Path.home() / (
    "rustfs/oi-data/EMOVID_RAW/20261022161317_20261022_lwh/"
    "20261022161317_20261022_lwh/lwh/lwh.bdf"
)
UPSTREAM = "https://github.com/935963004/LaBraM"
# Position vocabulary from upstream utils.py, preserving its exact indices/case.
LABRAM_CHANNELS = (
    "FP1 FPZ FP2 AF9 AF7 AF5 AF3 AF1 AFZ AF2 AF4 AF6 AF8 AF10 "
    "F9 F7 F5 F3 F1 FZ F2 F4 F6 F8 F10 FT9 FT7 FC5 FC3 FC1 FCZ "
    "FC2 FC4 FC6 FT8 FT10 T9 T7 C5 C3 C1 CZ C2 C4 C6 T8 T10 "
    "TP9 TP7 CP5 CP3 CP1 CPZ CP2 CP4 CP6 TP8 TP10 "
    "P9 P7 P5 P3 P1 PZ P2 P4 P6 P8 P10 PO9 PO7 PO5 PO3 PO1 POZ "
    "PO2 PO4 PO6 PO8 PO10 O1 OZ O2 O9 CB1 CB2 IZ O10 T3 T5 T4 T6 "
    "M1 M2 A1 A2 CFC1 CFC2 CFC3 CFC4 CFC5 CFC6 CFC7 CFC8 "
    "CCP1 CCP2 CCP3 CCP4 CCP5 CCP6 CCP7 CCP8 "
    "T1 T2 FTT9h TTP7h TPP9h FTT10h TPP8h TPP10h "
    "FP1-F7 F7-T7 T7-P7 P7-O1 FP2-F8 F8-T8 T8-P8 P8-O2 "
    "FP1-F3 F3-C3 C3-P3 P3-O1 FP2-F4 F4-C4 C4-P4 P4-O2"
).split()
POSITION = {name.upper(): i for i, name in enumerate(LABRAM_CHANNELS)}
REFERENCE_CHANNELS = {"M1", "M2", "A1", "A2", "REF", "GND"}
UNIT_TO_VOLTS = {"V": 1.0, "mV": 1e-3, "uV": 1e-6}


def log(message):
    print(message, flush=True)


def read_bdf_header(path):
    """Read only the header; avoid a full remote read for --inspect."""
    fields = (
        ("label", 16), ("transducer", 80), ("unit", 8),
        ("physical_min", 8), ("physical_max", 8), ("digital_min", 8),
        ("digital_max", 8), ("prefilter", 80), ("samples_per_record", 8),
        ("reserved", 32),
    )
    with Path(path).open("rb") as stream:
        fixed = stream.read(256)
        if len(fixed) != 256 or fixed[:8] != b"\xffBIOSEMI":
            raise ValueError("Input is not a supported 24-bit BDF file.")
        text = lambda start, end: fixed[start:end].decode("latin1").strip()
        count = int(text(252, 256))
        header_bytes = int(text(184, 192))
        if count < 1 or header_bytes != 256 * (count + 1):
            raise ValueError("Invalid BDF header length/channel count.")
        channels = [{} for _ in range(count)]
        for key, width in fields:
            block = stream.read(count * width)
            if len(block) != count * width:
                raise ValueError("Truncated BDF channel header.")
            for i, channel in enumerate(channels):
                channel[key] = block[i * width:(i + 1) * width].decode("latin1").strip()
    records = int(text(236, 244))
    record_seconds = float(text(244, 252))
    if records < 1 or not np.isfinite(record_seconds) or record_seconds <= 0:
        raise ValueError("A positive finalized record count/duration is required.")
    if text(192, 236).startswith("BDF+D"):
        raise ValueError("Discontinuous BDF+D needs a gap-aware preprocessing pipeline.")
    expected_size = header_bytes + records * sum(
        int(channel["samples_per_record"]) * 3 for channel in channels
    )
    if Path(path).stat().st_size != expected_size:
        raise ValueError("BDF file size disagrees with its finalized header.")
    return {
        "format": text(192, 236), "header_bytes": header_bytes,
        "records": records, "record_seconds": record_seconds,
        "duration_seconds": records * record_seconds,
        "start_as_stored": text(168, 176) + " " + text(176, 184),
        "channels": channels,
    }


def select_eeg(header):
    selected = []
    for channel in header["channels"]:
        name = channel["label"].upper()
        transducer = channel["transducer"].upper()
        if (name in POSITION and name not in REFERENCE_CHANNELS
                and not any(kind in transducer for kind in ("EOG", "ECG", "HECG", "EMG"))):
            selected.append(channel)
    if not selected:
        raise ValueError("No EEG electrodes matching the LaBraM position vocabulary.")
    if len({c["label"].upper() for c in selected}) != len(selected):
        raise ValueError("Duplicate canonical EEG channel names.")
    rates = {int(c["samples_per_record"]) / header["record_seconds"] for c in selected}
    if len(rates) != 1 or next(iter(rates)) <= 150:
        raise ValueError("EEG channels must have one sample rate above 150 Hz.")
    return sorted(selected, key=lambda c: POSITION[c["label"].upper()])


def mne_unit_factor(unit):
    # Match MNE's BDF physical-unit conversion before applying an override.
    if unit in ("uV", "µV", "μV", "\x83\xcaV"):
        return 1e-6
    return 1e-3 if unit == "mV" else 1.0


def calibration_factors(channels, source_unit):
    factors = {}
    for channel in channels:
        unit = channel["unit"]
        if source_unit == "auto":
            if unit not in ("V", "mV", "uV", "µV", "μV"):
                raise ValueError(
                    f"Nonstandard unit {unit!r} on {channel['label']}. "
                    "After confirming acquisition calibration, specify --source-unit uV, mV, or V."
                )
            factors[channel["label"]] = 1.0
        else:
            factors[channel["label"]] = UNIT_TO_VOLTS[source_unit] / mne_unit_factor(unit)
    return factors


def preprocess(input_path, output_path, *, source_unit="auto", start_seconds=0.0,
               duration_seconds=None, trim_tail_seconds=0.0, group_name="lwh"):
    """Return metadata after writing /group_name/eeg in microvolts."""
    input_path, output_path = Path(input_path), Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Output exists; choose a new filename: {output_path}")
    if not group_name or "/" in group_name or group_name in (".", ".."):
        raise ValueError("group_name must be a single nonempty HDF5 group name.")
    values = [start_seconds, trim_tail_seconds]
    if duration_seconds is not None:
        values.append(duration_seconds)
    if not all(np.isfinite(v) for v in values) or min(values) < 0:
        raise ValueError("Crop and trim parameters must be finite and nonnegative.")
    if duration_seconds == 0:
        raise ValueError("duration_seconds must be positive.")
    header = read_bdf_header(input_path)
    channels = select_eeg(header)
    factors = calibration_factors(channels, source_unit)
    selected_names = [c["label"] for c in channels]
    excluded = [c["label"] for c in header["channels"] if c["label"] not in selected_names]
    log(f"Reading {input_path.name}: {len(channels)} EEG channels; dropping {excluded}")
    log("MNE extracts annotations at open time, which can scan the remote BDF even with preload=False.")
    raw = mne.io.read_raw_bdf(
        str(input_path), preload=False, exclude=excluded, stim_channel=None, verbose="warning"
    )
    try:
        original_rate = float(raw.info["sfreq"])
        original_first_time = raw.first_time
        original_onsets = raw.annotations.onset.copy() - original_first_time
        durations = raw.annotations.duration.copy()
        descriptions = raw.annotations.description.copy()
        start = int(round(start_seconds * original_rate))
        stop = raw.n_times
        if duration_seconds is not None:
            stop = min(stop, start + int(round(duration_seconds * original_rate)))
        stop -= int(round(trim_tail_seconds * original_rate))
        if start < 0 or stop > raw.n_times or stop - start < 34 * original_rate:
            raise ValueError("Selected interval must contain at least 34 s for the 0.1 Hz FIR filter.")
        raw.crop(tmin=start / original_rate, tmax=(stop - 1) / original_rate)
        raw.load_data(verbose="warning")
        raw.reorder_channels(selected_names)
        # In MNE memory, EEG is represented in volts. An override must account
        # for scaling already applied by the reader to avoid a 1e6 error.
        for factor in sorted(set(factors.values())):
            picks = [name for name in selected_names if factors[name] == factor]
            if factor != 1.0:
                raw.apply_function(lambda data, factor=factor: data * factor, picks=picks)
        rename = {name: LABRAM_CHANNELS[POSITION[name.upper()]] for name in selected_names}
        raw.rename_channels(rename)
        log("Filtering continuous EEG: 0.1–75 Hz FIR, 50 Hz notch; resampling to 200 Hz.")
        raw.filter(0.1, 75.0, picks="eeg", n_jobs=1, verbose="warning")
        raw.notch_filter(50.0, picks="eeg", n_jobs=1, verbose="warning")
        raw.resample(200.0, n_jobs=1, verbose="warning")
        channel_names = raw.ch_names
        input_chans = [0] + [POSITION[name.upper()] + 1 for name in channel_names]
        crop_start = start / original_rate
        relative_onsets = original_onsets - crop_start
        target_samples = np.rint(relative_onsets * 200).astype(np.int64)
        in_output = ((relative_onsets >= 0) & (relative_onsets < (stop - start) / original_rate)
                     & (target_samples >= 0) & (target_samples < raw.n_times))
        metadata = {
            "source_path": str(input_path.resolve()), "source_size_bytes": input_path.stat().st_size,
            "source_start_as_stored": header["start_as_stored"],
            "source_timezone": "unspecified in BDF header",
            "source_rate_hz": original_rate, "source_duration_seconds": header["duration_seconds"],
            "source_unit_declaration": source_unit,
            "source_units": {c["label"]: c["unit"] for c in channels},
            "reader_to_volts_corrections": factors,
            "dropped_channels": excluded, "channel_names": channel_names,
            "crop_start_seconds": crop_start, "crop_stop_seconds": stop / original_rate,
            "trim_tail_seconds": trim_tail_seconds,
            "output_shape": [len(channel_names), int(raw.n_times)], "output_unit": "uV",
            "output_rate_hz": 200, "bandpass_hz": [0.1, 75.0], "notch_hz": [50.0],
            "reference": "acquisition reference unchanged", "ica_applied": False,
            "zscore_applied": False, "amplitude_divisor_at_model_input": 100.0,
            "patch_samples": 200, "suggested_window_seconds": 8,
            "input_chans_including_cls": input_chans,
            "event_count_original": len(original_onsets), "event_count_in_output": int(in_output.sum()),
            "event_code_counts": dict(Counter(descriptions.tolist())),
            "event_semantics": "original descriptions; no class-label interpretation",
            "upstream_repository": UPSTREAM,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "versions": {"mne": mne.__version__, "numpy": np.__version__, "h5py": h5py.__version__},
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        strings = h5py.string_dtype("utf-8")
        # Exclusive creation: original BDF and existing outputs are never overwritten.
        with h5py.File(output_path, "x") as output:
            output.attrs["processing_status"] = "incomplete"
            output.attrs["metadata_json"] = json.dumps(metadata, ensure_ascii=False)
            group = output.create_group(group_name)
            eeg = group.create_dataset(
                "eeg", (len(channel_names), raw.n_times), dtype="float32",
                chunks=(len(channel_names), 200), compression="lzf",
            )
            eeg.attrs.update({"lFreq": 0.1, "hFreq": 75.0, "rsFreq": 200,
                              "unit": "uV", "model_input_divisor": 100.0})
            eeg.attrs["chOrder"] = np.asarray(channel_names, dtype=strings)
            group.create_dataset("input_chans", data=np.asarray(input_chans, dtype=np.int64))
            for first in range(0, raw.n_times, 20000):
                end = min(first + 20000, raw.n_times)
                block = raw.get_data(start=first, stop=end, units="uV")
                if not np.isfinite(block).all():
                    raise ValueError("Nonfinite samples in processed EEG; output remains marked incomplete.")
                eeg[:, first:end] = block.astype(np.float32)
            events = group.create_group("events")
            events.create_dataset("onset_original_seconds", data=original_onsets)
            events.create_dataset("onset_output_seconds", data=relative_onsets)
            events.create_dataset("duration_seconds", data=durations)
            events.create_dataset("description", data=np.asarray(descriptions.tolist(), dtype=strings))
            events.create_dataset("sample_original", data=np.rint(original_onsets * original_rate).astype(np.int64))
            events.create_dataset("sample_output", data=target_samples)
            events.create_dataset("in_output", data=in_output)
            events.attrs["sample_index_base"] = 0
            events.attrs["rounding"] = "nearest sample (numpy.rint); original seconds retained"
            output.attrs["processing_status"] = "complete"
        log(f"Saved {output_path}: {metadata['output_shape']} float32 uV, {in_output.sum()} in-range events.")
        return metadata
    finally:
        raw.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=Path("processed/lwh_labram.hdf5"))
    parser.add_argument("--source-unit", choices=["auto", "uV", "mV", "V"], default="auto",
                        help="Actual unit of BDF physical values; uV/mm requires an explicit confirmed override")
    parser.add_argument("--start-seconds", type=float, default=0)
    parser.add_argument("--duration-seconds", type=float, help="Optional short interval for validation; default is the full recording")
    parser.add_argument("--trim-tail-seconds", type=float, default=0,
                        help="Optional tail removal; official CNT example uses 10 s, default here preserves all data")
    parser.add_argument("--group", default="lwh")
    parser.add_argument("--inspect", action="store_true", help="Read only the header; do not filter or create output")
    args = parser.parse_args()
    try:
        if args.inspect:
            header = read_bdf_header(args.input)
            selected = select_eeg(header)
            print(json.dumps({
                "input": str(args.input), "duration_seconds": header["duration_seconds"],
                "original_channel_count": len(header["channels"]),
                "selected_eeg_count": len(selected), "source_units": sorted({c["unit"] for c in selected}),
                "channel_order": [LABRAM_CHANNELS[POSITION[c["label"].upper()]] for c in selected],
            }, indent=2, ensure_ascii=False))
            return 0
        preprocess(args.input, args.output, source_unit=args.source_unit,
                   start_seconds=args.start_seconds, duration_seconds=args.duration_seconds,
                   trim_tail_seconds=args.trim_tail_seconds, group_name=args.group)
    except (ValueError, OSError) as exc:
        print(f"Preprocessing failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
