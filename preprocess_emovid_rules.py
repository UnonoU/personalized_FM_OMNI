"""Preprocess mapped EMOVID EEG using the dataset-rules example and base profile.

The specification supplies example parameters, not an executable algorithm.
This implementation chooses MNE zero-phase FIR filters and scipy resample_poly,
keeps supervised emotion labels, and exports one Lance table plus TOML metadata.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import mne
import numpy as np
import scipy
from scipy.signal import resample_poly

from batch_emovid_segments import atomic_json
from emovid_segments import LABEL_IDS, Store, split_s3

RULES_COMMIT = "502c9d1dea27d9930813aa11076ef086e4ba2de3"
RULES_URL = f"https://github.com/Omni-Intel/dataset-rules/blob/{RULES_COMMIT}"
CHANNELS = tuple((
    "Fpz Fp1 Fp2 AF3 AF4 AF7 AF8 Fz F1 F2 F3 F4 F5 F6 F7 F8 "
    "FCz FC1 FC2 FC3 FC4 FC5 FC6 FT7 FT8 Cz C1 C2 C3 C4 C5 C6 T7 T8 "
    "CP1 CP2 CP3 CP4 CP5 CP6 TP7 TP8 Pz P3 P4 P5 P6 P7 P8 "
    "POz PO3 PO4 PO5 PO6 PO7 PO8 Oz O1 O2"
).upper().split())
UNIT_TO_UV = {"V": 1e6, "mV": 1e3, "uV": 1.0, "µV": 1.0, "μV": 1.0}
PRIMARY_LABEL_IDS = {**LABEL_IDS, "rest": 3}
WINDOW_SECONDS = 3.0
WINDOW_SAMPLES = 600
DATASET_VERSION = "0.2.0"


def canonical_channel(name):
    name = name.strip().upper()
    return name[4:].strip() if name.startswith("EEG ") else name


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class SignalReader:
    """Calibrated record-range reads of continuous EDF/BDF without annotation scans.

    Decode digital samples into the physical values in the header first. Unit
    conversion is explicit and independent of MNE's treatment of unknown units.
    """

    def __init__(self, store, uri, *, source_unit="auto", channels=CHANNELS):
        self.store, self.uri = store, uri
        fixed = store.read(uri, (0, 255))
        if len(fixed) != 256 or fixed[:8] not in (b"\xffBIOSEMI", b"0       "):
            raise ValueError("Expected EDF or BDF")
        self.sample_bytes = 3 if fixed[:8] == b"\xffBIOSEMI" else 2
        self.n_channels = int(fixed[252:256])
        self.header_bytes = int(fixed[184:192])
        self.n_records = int(fixed[236:244])
        self.record_seconds = float(fixed[244:252])
        if (self.n_channels <= 0 or self.header_bytes != 256*(self.n_channels+1)
                or self.n_records <= 0 or not math.isfinite(self.record_seconds) or self.record_seconds <= 0):
            raise ValueError("Invalid finalized EDF/BDF header")
        if b"+D" in fixed[192:236]:
            raise ValueError("Discontinuous EDF/BDF requires a gap-aware implementation")
        payload = store.read(uri, (256, self.header_bytes-1))
        if len(payload) != self.header_bytes-256:
            raise ValueError("Truncated channel header")
        fields, offset = {}, 0
        for key, width in (("label", 16), ("transducer", 80), ("unit", 8),
                           ("physical_min", 8), ("physical_max", 8), ("digital_min", 8),
                           ("digital_max", 8), ("prefilter", 80), ("samples_per_record", 8), ("reserved", 32)):
            fields[key] = [payload[offset+i*width:offset+(i+1)*width].decode("latin1").strip()
                           for i in range(self.n_channels)]
            offset += width*self.n_channels
        names = [canonical_channel(name) for name in fields["label"]]
        self.channel_names = tuple(channels)
        if any(names.count(name) != 1 for name in channels):
            raise ValueError("Missing or duplicate required scalp channels")
        self.picks = [names.index(name) for name in channels]
        self.samples_per_record = [int(value) for value in fields["samples_per_record"]]
        rates = {self.samples_per_record[i]/self.record_seconds for i in self.picks}
        if len(rates) != 1 or min(rates) <= 150:
            raise ValueError("EEG channels must share a sampling rate above 150 Hz")
        self.sfreq = rates.pop()
        self.duration = self.n_records*self.record_seconds
        self.n_times = self.n_records*self.samples_per_record[self.picks[0]]
        self.record_bytes = sum(self.samples_per_record)*self.sample_bytes
        self.expected_size = self.header_bytes + self.n_records*self.record_bytes
        self.units = [fields["unit"][i] for i in self.picks]
        self.dropped_channels = [fields["label"][i] for i in range(self.n_channels) if i not in self.picks]
        self.parameters = []
        for i in self.picks:
            pmin, pmax = float(fields["physical_min"][i]), float(fields["physical_max"][i])
            dmin, dmax = int(fields["digital_min"][i]), int(fields["digital_max"][i])
            if not np.isfinite([pmin, pmax]).all() or pmax <= pmin or dmax <= dmin:
                raise ValueError("Invalid channel calibration")
            unit = fields["unit"][i] if source_unit == "auto" else source_unit
            if unit not in UNIT_TO_UV:
                raise ValueError(f"Nonstandard unit {unit!r}; explicitly declare --source-unit after checking calibration")
            scale = (pmax-pmin)/(dmax-dmin)
            self.parameters.append((scale, pmin-dmin*scale, UNIT_TO_UV[unit]))
        self.channel_offsets = np.cumsum([0] + self.samples_per_record[:-1])*self.sample_bytes

    def read_samples(self, start, stop):
        if not 0 <= start < stop <= self.n_times:
            raise ValueError("Sample read outside source recording")
        count = self.samples_per_record[self.picks[0]]
        first_record, last_record = start//count, (stop+count-1)//count
        data = self.store.read(self.uri, (self.header_bytes+first_record*self.record_bytes,
                                          self.header_bytes+last_record*self.record_bytes-1))
        n_records = last_record-first_record
        if len(data) != n_records*self.record_bytes:
            raise ValueError("Truncated EDF/BDF record range")
        selected = np.empty((len(self.picks), n_records*count), dtype=np.float64)
        for row, (pick, (scale, offset, factor)) in enumerate(zip(self.picks, self.parameters)):
            if self.sample_bytes == 3:
                octets = np.ndarray((n_records, count, 3), dtype=np.uint8, buffer=data,
                                   offset=int(self.channel_offsets[pick]), strides=(self.record_bytes, 3, 1))
                values = (octets[:, :, 0].astype(np.int32) | (octets[:, :, 1].astype(np.int32) << 8)
                          | (octets[:, :, 2].astype(np.int32) << 16))
                values = (values ^ 0x800000) - 0x800000
            else:
                values = np.ndarray((n_records, count), dtype="<i2", buffer=data,
                                    offset=int(self.channel_offsets[pick]), strides=(self.record_bytes, 2))
            selected[row] = (values.reshape(-1)*scale + offset)*factor
        return selected[:, start-first_record*count:stop-first_record*count].copy()


def filter_and_resample(data_uv, source_rate, target_rate):
    """Reference unchanged, zero-phase FIR bandpass/notch, polyphase resampling."""
    if not np.isfinite(data_uv).all():
        raise ValueError("Nonfinite source samples")
    data = mne.filter.filter_data(data_uv, source_rate, 0.3, 75.0, method="fir", phase="zero",
                                  fir_design="firwin", pad="reflect_limited", n_jobs=1, verbose=False)
    data = mne.filter.notch_filter(data, source_rate, [50.0], method="fir", phase="zero",
                                   notch_widths=0.25, trans_bandwidth=1.0, fir_design="firwin",
                                   pad="reflect_limited", n_jobs=1, verbose=False)
    fraction = Fraction(target_rate/source_rate).limit_denominator(10000)
    return resample_poly(data, fraction.numerator, fraction.denominator, axis=-1)


def window_plan(segments, *, target_rate=200, window_seconds=WINDOW_SECONDS):
    length = round(target_rate*window_seconds)
    windows, discarded_tail_samples = [], 0
    for segment_index, segment in enumerate(segments):
        if not segment["usable"]:
            continue
        first, last = round(segment["start_seconds"]*target_rate), round(segment["stop_seconds"]*target_rate)
        discarded_tail_samples += (last-first) % length
        for start in range(first, last-length+1, length):
            stop = start+length
            windows.append({"segment_index": segment_index, "start": start, "stop": stop,
                            "window_index": (start-first)//length})
    return windows, discarded_tail_samples


def encode_window(data_uv, *, window_samples=WINDOW_SAMPLES, max_clip_fraction=0.01, flat_std_uv=1e-4):
    data_uv = np.asarray(data_uv, dtype=np.float64)
    if data_uv.ndim != 2 or data_uv.shape[1] != window_samples or not np.isfinite(data_uv).all():
        raise ValueError(f"Window must contain exactly {window_samples} finite samples per channel; no padding")
    clip_fraction = float(np.mean(np.abs(data_uv) > 200.0))
    channel_std = data_uv.std(axis=1)
    bad = channel_std <= flat_std_uv
    reasons = []
    if clip_fraction > max_clip_fraction:
        reasons.append("clip_fraction_above_threshold")
    if bad.any():
        reasons.append("flat_channel")
    values = np.clip(data_uv/200.0, -1.0, 1.0).astype(np.float32)
    return values, {"qc_pass": not reasons, "qc_reasons": reasons, "clip_fraction": clip_fraction,
                    "bad_channel_mask": bad.tolist(), "maximum_ptp_uv": float(np.ptp(data_uv, axis=1).max()),
                    "minimum_channel_std_uv": float(channel_std.min())}


def preprocessing_contract(args):
    return {"algorithm_version": "emovid-dataset-rules-v2-3s-no-padding", "rules_commit": RULES_COMMIT,
            "profile": "base", "channel_names": list(CHANNELS), "reference": "acquisition_reference_unchanged",
            "bandpass_hz": [0.3, 75.0], "notch_hz": 50.0, "notch_width_hz": 0.25,
            "notch_transition_hz": 1.0, "filter": "MNE_zero_phase_FIR_firwin_hamming_reflect_limited",
            "resampling": "scipy_resample_poly", "target_sfreq": 200.0, "window_seconds": WINDOW_SECONDS,
            "minimum_window_seconds": WINDOW_SECONDS, "tail_policy": "discard_incomplete_window",
            "normalization": "clip(x_uV / 200, -1, 1)", "source_unit_declaration": args.source_unit,
            "source_unit_status": args.unit_status, "unit_before_normalization": "uV", "output_unit": "normalized",
            "max_clip_fraction": args.max_clip_fraction, "flat_std_uv": 1e-4,
            "block_seconds": args.block_seconds, "context_seconds": args.context_seconds,
            "car_applied": False, "baseline_correction_applied": False, "ica_applied": False,
            "mvnn_applied": False, "split": "unspecified", "mne_version": mne.__version__,
            "numpy_version": np.__version__, "scipy_version": scipy.__version__}


def preprocess_recording(report, reader, config, output_folder):
    target_rate, window_samples = 200.0, WINDOW_SAMPLES
    windows, discarded = window_plan(report["segments"])
    if not windows:
        return None
    output_folder.mkdir(parents=True, exist_ok=True)
    destination = output_folder / "data.npy"
    values = np.lib.format.open_memmap(destination, mode="w+", dtype=np.float32,
                                      shape=(len(windows), len(CHANNELS), window_samples))
    grouped = {}
    for index, window in enumerate(windows):
        key = int(window["start"]//round(config["block_seconds"]*target_rate))
        grouped.setdefault(key, []).append((index, window))
    rows = [None]*len(windows)
    bandpass = mne.filter.create_filter(None, reader.sfreq, 0.3, 75.0, verbose=False)
    notch = mne.filter.create_filter(None, reader.sfreq, [50.625], [49.375],
                                      l_trans_bandwidth=0.5, h_trans_bandwidth=0.5, verbose=False)
    fraction = Fraction(target_rate/reader.sfreq).limit_denominator(10000)
    required_context = ((len(bandpass)-1)+(len(notch)-1))/(2*reader.sfreq) + 10/target_rate
    if config["context_seconds"] < required_context:
        raise ValueError(f"Insufficient filtering context: need at least {required_context:.3f}s")
    for items in grouped.values():
        first = min(w["start"] for _, w in items)/target_rate
        last = max(w["stop"] for _, w in items)/target_rate
        # Align source start to the resampler's denominator, preserving the
        # global 200 Hz sample grid across independently processed blocks.
        start = max(0, math.floor((first-config["context_seconds"])*reader.sfreq))
        start = start//fraction.denominator*fraction.denominator
        stop = min(reader.n_times, math.ceil((last+config["context_seconds"])*reader.sfreq))
        raw = reader.read_samples(start, stop)
        filtered = filter_and_resample(raw, reader.sfreq, target_rate)
        offset = round(start*target_rate/reader.sfreq)
        for index, window in items:
            segment = report["segments"][window["segment_index"]]
            a, b = window["start"]-offset, window["stop"]-offset
            if not 0 <= a < b <= filtered.shape[1]:
                raise ValueError("Filtered window outside loaded context")
            encoded, qc = encode_window(filtered[:, a:b], max_clip_fraction=config["max_clip_fraction"])
            values[index] = encoded
            sample_id = "emovid#" + canonical_hash([report["raw_uri"], segment["events_uri"],
                                                   segment["start_event_index"], window["start"], window["stop"]])[:24]
            rows[index] = {"sample_id": sample_id, "subject_id": str(segment["subject_id"]),
                           "session_id": f"session_{int(segment['session_id']):02d}",
                           "run_id": str(segment["raw_run_index"]), "modality": "eeg",
                           "shape": [len(CHANNELS), window_samples], "original_shape": [len(CHANNELS), b-a],
                           "valid_length": len(CHANNELS)*(b-a),
                           "label": segment["emotion_label_id"] if segment["segment_type"] == "emotion" else 3,
                           "label_name": segment["emotion_label"] if segment["segment_type"] == "emotion" else "rest",
                           "emotion_label": segment["emotion_label_id"], "emotion_label_name": segment["emotion_label"],
                           "segment_type": segment["segment_type"],
                           "rest_type": segment["rest_type"], "video_id": segment["video_id"],
                           "split": "unspecified", "source_path": report["raw_uri"], "events_uri": segment["events_uri"],
                           "segment_start_seconds": segment["start_seconds"], "segment_stop_seconds": segment["stop_seconds"],
                           "start_time": window["start"]/target_rate, "duration": (b-a)/target_rate,
                           "start_sample": round(window["start"]*reader.sfreq/target_rate),
                           "stop_sample": round(window["stop"]*reader.sfreq/target_rate),
                           "output_start_sample": window["start"], "output_stop_sample": window["stop"],
                           "source_sfreq": reader.sfreq, "trial_id": str(segment.get("trial_id") or ""),
                           "attempt_id": str(segment.get("attempt_id") or ""),
                           "channel_names": list(CHANNELS), "channel_mask": [True]*len(CHANNELS),
                           "boundary_time_basis": segment["start_time_basis"] + ";" + segment["stop_time_basis"], **qc}
        del raw, filtered
    values.flush()
    del values
    result = {"raw_uri": report["raw_uri"], "rows": rows, "n_windows": len(rows),
              "qc_pass_count": sum(r["qc_pass"] for r in rows),
              "qc_reasons": dict(Counter(reason for r in rows for reason in r["qc_reasons"])),
              "discarded_short_tail_seconds": discarded/target_rate, "source_header_units": reader.units,
              "source_sfreq": reader.sfreq, "dropped_channels": reader.dropped_channels,
              "bandpass_filter_taps": len(bandpass), "notch_filter_taps": len(notch),
              "required_context_seconds": required_context, "data_sha256": digest(destination)}
    return result


def arrow_schema():
    import pyarrow as pa
    definitions = [
        ("sample_id", pa.string(), False), ("subject_id", pa.string(), False),
        ("session_id", pa.string(), False), ("run_id", pa.string(), False), ("modality", pa.string(), False),
        ("data", pa.list_(pa.float32(), len(CHANNELS)*WINDOW_SAMPLES), False),
        ("shape", pa.list_(pa.int32()), False), ("original_shape", pa.list_(pa.int32()), False),
        ("valid_length", pa.int64(), False), ("qc_pass", pa.bool_(), False),
        ("label", pa.int32(), False), ("label_name", pa.string(), False),
        ("emotion_label", pa.int32(), True), ("emotion_label_name", pa.string(), True),
        ("segment_type", pa.string(), False), ("rest_type", pa.string(), True), ("video_id", pa.string(), True),
        ("split", pa.string(), False), ("source_path", pa.string(), False), ("events_uri", pa.string(), False),
        ("segment_start_seconds", pa.float64(), False), ("segment_stop_seconds", pa.float64(), False),
        ("start_time", pa.float64(), False), ("duration", pa.float64(), False),
        ("start_sample", pa.int64(), False), ("stop_sample", pa.int64(), False),
        ("output_start_sample", pa.int64(), False), ("output_stop_sample", pa.int64(), False),
        ("source_sfreq", pa.float64(), False), ("trial_id", pa.string(), False), ("attempt_id", pa.string(), False),
        ("channel_names", pa.list_(pa.string()), False), ("channel_mask", pa.list_(pa.bool_(), len(CHANNELS)), False),
        ("bad_channel_mask", pa.list_(pa.bool_(), len(CHANNELS)), False),
        ("qc_reasons", pa.list_(pa.string()), False), ("clip_fraction", pa.float64(), False),
        ("maximum_ptp_uv", pa.float64(), False), ("minimum_channel_std_uv", pa.float64(), False),
        ("boundary_time_basis", pa.string(), False),
    ]
    return pa.schema([pa.field(name, dtype, nullable=nullable) for name, dtype, nullable in definitions])


def arrow_batches(shards, schema):
    import pyarrow as pa
    for shard in shards:
        meta = json.loads((shard/"report.json").read_text())
        data = np.load(shard/"data.npy", mmap_mode="r", allow_pickle=False)
        for start in range(0, len(data), 16):
            stop = min(start+16, len(data))
            fields = []
            for field in schema:
                if field.name == "data":
                    array = pa.FixedSizeListArray.from_arrays(pa.array(data[start:stop].reshape(-1)), len(CHANNELS)*WINDOW_SAMPLES)
                else:
                    array = pa.array([r[field.name] for r in meta["rows"][start:stop]], type=field.type)
                fields.append(array)
            yield pa.RecordBatch.from_arrays(fields, schema=schema)


def write_metadata(root, schema, summary, config):
    """Write TOML with every Arrow column declared; no optional TOML writer needed."""
    def toml_value(value):
        return json.dumps(value, ensure_ascii=False)
    table_name = summary.get("table_name", "dataset.lance")
    dataset_id = summary.get("dataset_id", "emovid-emotion-rest")
    lines = ['spec_version = "0.1.0"', 'profile = "base"', '', '[dataset]']
    sections = {
        "dataset": {"dataset_id": dataset_id, "name": "EMOVID emotion and rest preliminary EEG",
                    "modality": "eeg", "version": DATASET_VERSION, "created_at": summary["created_at_utc"],
                    "updated_at": summary["created_at_utc"], "sample_unit": "window", "task_type": "classification",
                    "n_samples": summary["n_windows"], "n_subjects": summary["n_subjects"]},
        "storage": {"table_name": table_name, "lance_path": table_name, "path_type": "relative",
                    "backend": "s3" if summary.get("storage_uri") else "local"},
        "schema": {"primary_key": "sample_id", "data_column": "data", "label_column": "label"},
    }
    for section, values in sections.items():
        if section != "dataset":
            lines += ["", f"[{section}]"]
        lines += [f"{key} = {toml_value(value)}" for key, value in values.items()]
    descriptions = {"data": "Normalized row-major EEG [59,600], exactly 3 seconds; no padding.",
                    "original_shape": "[59,600], identical to shape; all time samples are real EEG.",
                    "label": "Primary state: 0 negative, 1 neutral, 2 positive, 3 rest (a recording state, not an emotion).",
                    "emotion_label": "Three-class emotion target 0/1/2; null on rest. Select segment_type=emotion for classification.",
                    "qc_pass": "Finite window, clipping fraction within threshold and no detected flat channels.",
                    "split": "Unspecified; no automatic train/validation/test division has been made."}
    for field in schema:
        lines += ["", "[[schema.columns]]", f"name = {toml_value(field.name)}", f"type = {toml_value(str(field.type))}",
                  "required = true", f"nullable = {toml_value(field.nullable)}",
                  f"description = {toml_value(descriptions.get(field.name, field.name.replace('_', ' ')))}"]
    sections = {
        "data": {"array_column": "data", "dtype": "float32", "axis_order": ["channel", "time"],
                 "padding": False, "padding_value": 0.0, "shape_column": "shape", "original_shape_column": "original_shape",
                 "valid_length_column": "valid_length", "time_padding": False},
        "eeg": {"n_channels": len(CHANNELS), "sampling_rate": 200.0, "sampling_rate_unit": "Hz", "unit": "normalized",
                "reference": "acquisition_reference_unchanged_original_unknown", "montage": "10-10 scalp channel labels; coordinates not stored",
                "channel_axis": 0, "time_axis": 1, "channel_layout": "dataset_level", "channel_names_column": "channel_names",
                "channel_mask_column": "channel_mask", "bad_channel_mask_column": "bad_channel_mask"},
        "label": {"label_column": "label", "label_name_column": "label_name", "task_type": "classification", "num_classes": 4,
                  "rest_label_policy": "primary label 3 = rest state; emotion_label is null on rest",
                  "emotion_target_column": "emotion_label", "emotion_target_num_classes": 3},
        "quality_control": {"qc_column": "qc_pass", "qc_pass_values": [True, False],
                            "qc_method": "finiteness, pre-clip amplitude fraction and per-channel flatness",
                            "max_clip_fraction": config["max_clip_fraction"], "flat_std_uv": config["flat_std_uv"],
                            "failed_rows_retained": True},
        "preprocessing": dict(config, segment_samples=WINDOW_SAMPLES, target_sfreq=200.0, bandpass_low=0.3, bandpass_high=75.0,
                              normalization_detail="x_uV / 200, then clip to [-1,1]; discard tails shorter than 3 seconds, no padding",
                              unit_caveat="Nonstandard header units require explicit declaration; status records confirmation or assumption."),
        "split": {"split_column": "split", "allowed_values": ["unspecified"],
                  "policy": "Assign by subject/session/video before training; avoid leakage across windows of the same video."},
    }
    for section, values in sections.items():
        lines += ["", f"[{section}]"]
        for key, value in values.items():
            lines.append(f"{key} = {toml_value(value)}")
    for index, name in enumerate(CHANNELS):
        lines += ["", "[[eeg.channels]]", f"index = {index}", f"name = {toml_value(name)}", 'type = "eeg"', 'unit = "normalized"']
    for name, index in sorted(PRIMARY_LABEL_IDS.items(), key=lambda item: item[1]):
        description = "Rest recording state; not an emotion label" if name == "rest" else name+" emotion stimulus label"
        lines += ["", "[[label.classes]]", f"id = {index}", f"name = {toml_value(name)}", f"description = {toml_value(description)}"]
    if summary.get("unit_confirmation"):
        lines += ["", "[provenance.unit_confirmation]"]
        lines += [f"{key} = {toml_value(value)}" for key, value in summary["unit_confirmation"].items()]
    (root/"metadata.toml").write_text("\n".join(lines)+"\n", encoding="utf-8")
    previous_versions = []
    if (root/"versions.toml").exists():
        import tomli
        previous_versions = tomli.loads((root/"versions.toml").read_text()).get("versions", [])
    version_metadata = ['spec_version = "0.1.0"', f'dataset_id = {toml_value(dataset_id)}', f'table = {toml_value(table_name)}',
                        '', '[current]', f'lance_version = {summary["lance_version"]}', f'dataset_version = "{DATASET_VERSION}"',
                        'description = "EMOVID: fixed 3-second windows, no zero padding."']
    entry = {"lance_version": summary["lance_version"], "dataset_version": DATASET_VERSION,
             "created_at": summary["created_at_utc"], "author": "preprocess_emovid_rules.py",
             "change_type": "preprocessing_update" if previous_versions else "initial_import",
             "compatible_with_previous": False,
             "description": "Fixed 3-second windows within each labeled segment; discard incomplete tails, no padding.",
             "affected_rows": summary["n_windows"], "affected_columns": schema.names}
    for entry in previous_versions + [entry]:
        version_metadata += ["", "[[versions]]"] + [f"{key} = {toml_value(value)}" for key, value in entry.items()]
    (root/"versions.toml").write_text("\n".join(version_metadata)+"\n", encoding="utf-8")


def run(args):
    import lance
    import pyarrow as pa
    manifest = json.loads(args.manifest.read_text())
    inventory = json.loads(args.manifest.with_name("inventory.json").read_text())
    objects = {obj["uri"]: obj for obj in inventory["raw_objects"]}
    config = preprocessing_contract(args)
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    previous_summary = json.loads((root/"summary.json").read_text()) if (root/"summary.json").exists() else {}
    confirmation = previous_summary.get("unit_confirmation") if args.unit_status == "confirmed" else None
    marker = root/"preprocessing_config.json"
    if not marker.exists() and any(root.iterdir()):
        raise ValueError("Output directory contains unrelated files; choose a new output directory")
    atomic_json(marker, config)
    store, shards, results = Store(args.profile), [], []
    selected = [r for r in manifest["recordings"] if r.get("usable_segment_count", 0) > 0]
    if args.limit_recordings is not None:
        selected = selected[:args.limit_recordings]
    for number, item in enumerate(selected, 1):
        uri = item["raw_uri"]
        name = canonical_hash(uri)[:16]
        folder = root/"derivatives"/"checkpoints"/name
        try:
            source = objects[uri]
            bucket, key = split_s3(uri)
            actual = store.s3.head_object(Bucket=bucket, Key=key)
            if actual["ContentLength"] != source["size"] or actual.get("ETag") != source["etag"]:
                raise ValueError("Source recording changed since event mapping; rerun mapping first")
            report_path = Path(item["report_path"])
            if not report_path.is_absolute():
                report_path = Path.cwd()/report_path
            report = json.loads(report_path.read_text())
            signature = canonical_hash({"source": source, "mapping_sha256": digest(report_path),
                                        "config": config, "code_sha256": digest(__file__)})
            checkpoint = folder/"report.json"
            if args.resume and checkpoint.exists():
                old = json.loads(checkpoint.read_text())
                if old.get("fingerprint") == signature and (folder/"data.npy").exists() and digest(folder/"data.npy") == old["data_sha256"]:
                    shards.append(folder)
                    results.append({k: v for k, v in old.items() if k != "rows"})
                    print(f"[{number}/{len(selected)}] cached windows={old['n_windows']} {uri}", flush=True)
                    continue
            reader = SignalReader(store, uri, source_unit=args.source_unit)
            if reader.expected_size != source["size"]:
                raise ValueError("Source object size disagrees with finalized EDF/BDF header")
            folder.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=name+"-", dir=folder.parent) as temporary:
                temp = Path(temporary)
                result = preprocess_recording(report, reader, config, temp)
                if result is None:
                    results.append({"raw_uri": uri, "status": "no_eligible_windows"})
                    continue
                result.update(status="success", fingerprint=signature)
                if confirmation:
                    result["unit_confirmation"] = confirmation
                atomic_json(temp/"report.json", result)
                if folder.exists():
                    shutil.rmtree(folder)
                folder.mkdir()
                for path in temp.iterdir():
                    os.replace(path, folder/path.name)
            shards.append(folder)
            results.append({k: v for k, v in result.items() if k != "rows"})
            print(f"[{number}/{len(selected)}] windows={result['n_windows']} qc_pass={result['qc_pass_count']} {uri}", flush=True)
        except Exception as exc:
            results.append({"raw_uri": uri, "status": "error", "error_type": type(exc).__name__, "reason": str(exc)})
            print(f"[{number}/{len(selected)}] ERROR {type(exc).__name__}: {exc} {uri}", flush=True)
        atomic_json(root/"progress.json", {"config": config, "recordings": results})
    if not shards:
        raise ValueError("No recording could be preprocessed; inspect progress.json")
    schema = arrow_schema()
    data_reader = pa.RecordBatchReader.from_batches(schema, arrow_batches(shards, schema))
    dataset = lance.write_dataset(data_reader, root/"dataset.lance", mode="overwrite" if (root/"dataset.lance").exists() else "create",
                                  max_rows_per_file=256, max_rows_per_group=16)
    all_rows = [row for folder in shards for row in json.loads((folder/"report.json").read_text())["rows"]]
    summary = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "config": config,
               "mapping_manifest_sha256": digest(args.manifest), "rules_reference": RULES_URL+"/examples/eeg-pretrain/metadata.toml",
               "processed_recording_count": len(shards), "selected_recording_count": len(selected),
               "n_windows": dataset.count_rows(), "qc_pass_count": sum(row["qc_pass"] for row in all_rows),
               "n_subjects": len({row["subject_id"] for row in all_rows}),
               "window_types": dict(Counter(row["segment_type"] for row in all_rows)),
               "emotion_counts": dict(Counter(row["label_name"] for row in all_rows if row["segment_type"] == "emotion")),
               "qc_pass_emotion_counts": dict(Counter(row["label_name"] for row in all_rows if row["segment_type"] == "emotion" and row["qc_pass"])),
               "padded_window_count": sum(row["original_shape"][1] < WINDOW_SAMPLES for row in all_rows),
               "discarded_short_tail_seconds": sum(r.get("discarded_short_tail_seconds", 0) for r in results),
               "lance_version": dataset.version, "lance_library_version": lance.__version__,
               "pyarrow_version": pa.__version__, "recordings": results,
               "mapping_skipped_collections": manifest.get("skipped_collections", [])}
    if confirmation:
        summary["unit_confirmation"] = confirmation
    write_metadata(root, schema, summary, config)
    atomic_json(root/"summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("processed_recording_count", "n_windows", "qc_pass_count", "window_types", "emotion_counts", "padded_window_count")}, ensure_ascii=False))
    return summary


def main():
    # Keep the old executable name, with one storage-safe public interface.
    from emovid_pipeline import main as unified_main
    return unified_main()


if __name__ == "__main__":
    main()
