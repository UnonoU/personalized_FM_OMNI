"""Validate the local EMOVID Lance table and its numerical/metadata invariants."""

import argparse
from collections import Counter, defaultdict
import json
import hashlib
from pathlib import Path

import lance
import numpy as np
import tomli

from batch_emovid_segments import atomic_json
from preprocess_emovid_rules import CHANNELS, PRIMARY_LABEL_IDS, WINDOW_SECONDS, WINDOW_SAMPLES, arrow_schema


def validate(root, *, table_uri=None, storage_options=None, expected_signal_sha256=None):
    metadata = tomli.loads((root/"metadata.toml").read_text())
    versions = tomli.loads((root/"versions.toml").read_text())
    summary = json.loads((root/"summary.json").read_text())
    dataset = lance.dataset(table_uri or root/metadata["storage"]["lance_path"], storage_options=storage_options)
    schema = arrow_schema()
    checks = {}

    def check(name, condition):
        checks[name] = bool(condition)
        if not condition:
            raise ValueError(f"Validation failed: {name}")

    check("nonempty_table", dataset.count_rows() > 0)
    check("row_count_matches_metadata", dataset.count_rows() == metadata["dataset"]["n_samples"] == summary["n_windows"])
    check("base_profile_with_label_definition", metadata["profile"] == "base" and metadata["label"]["num_classes"] == 4)
    check("schema_matches_implementation", dataset.schema == schema)
    columns = {column["name"]: column for column in metadata["schema"]["columns"]}
    check("all_columns_declared_with_matching_types", set(columns) == set(schema.names)
          and all(columns[field.name]["type"] == str(field.type) for field in schema))
    check("eeg_channel_order", [ch["name"] for ch in metadata["eeg"]["channels"]] == list(CHANNELS))
    check("eeg_channel_indexes", [ch["index"] for ch in metadata["eeg"]["channels"]] == list(range(59)))
    check("sampling_rate_and_output_unit", metadata["eeg"]["sampling_rate"] == 200 and metadata["eeg"]["unit"] == "normalized")
    check("three_seconds_without_padding", metadata["preprocessing"]["window_seconds"] == WINDOW_SECONDS
          and metadata["preprocessing"]["segment_samples"] == WINDOW_SAMPLES
          and metadata["data"]["padding"] is False and metadata["data"]["time_padding"] is False)
    check("current_lance_version", versions["current"]["lance_version"] == dataset.version == summary["lance_version"])
    required_version = {"lance_version", "dataset_version", "created_at", "author", "change_type",
                        "compatible_with_previous", "description", "affected_rows", "affected_columns"}
    check("version_entry_complete", required_version.issubset(versions["versions"][-1]))
    check("no_recording_processing_errors", summary["processed_recording_count"] == summary["selected_recording_count"]
          and all(r["status"] == "success" for r in summary["recordings"]))
    sample_ids, qc_count, type_counts, label_counts, padded = set(), 0, Counter(), Counter(), 0
    segment_windows = defaultdict(list)
    inverse_labels = {value: name for name, value in PRIMARY_LABEL_IDS.items()}
    signal_hash = hashlib.sha256()
    for batch in dataset.scanner(batch_size=8, batch_readahead=1, fragment_readahead=1, scan_in_order=True).to_batches():
        values = batch.column("data").flatten().to_numpy(zero_copy_only=False).reshape(len(batch), 59, WINDOW_SAMPLES)
        if not np.isfinite(values).all() or np.abs(values).max() > 1:
            raise ValueError("Nonfinite or out-of-range normalized EEG")
        signal_hash.update(values.tobytes(order="C"))
        records = batch.drop_columns(["data"]).to_pylist()
        for row, data in zip(records, values):
            if any(row[field.name] is None for field in schema if not field.nullable and field.name != "data"):
                raise ValueError("Null value in a required non-nullable column")
            if row["sample_id"] in sample_ids or not row["sample_id"] or not row["subject_id"]:
                raise ValueError("Duplicate/empty identity")
            sample_ids.add(row["sample_id"])
            count = row["original_shape"][1]
            if not (row["shape"] == row["original_shape"] == [59, WINDOW_SAMPLES]
                    and count == WINDOW_SAMPLES and row["valid_length"] == 59*WINDOW_SAMPLES):
                raise ValueError("Invalid shape, padding or valid length")
            if row["channel_names"] != list(CHANNELS) or row["channel_mask"] != [True]*59 or len(row["bad_channel_mask"]) != 59:
                raise ValueError("Invalid channel metadata or masks")
            if inverse_labels.get(row["label"]) != row["label_name"]:
                raise ValueError("Label id/name disagree")
            if row["segment_type"] == "emotion":
                if row["label"] not in (0, 1, 2) or row["emotion_label"] != row["label"] or row["emotion_label_name"] != row["label_name"]:
                    raise ValueError("Invalid three-class emotion target")
                label_counts[row["emotion_label_name"]] += 1
            elif row["segment_type"] == "rest":
                if row["label"] != 3 or row["emotion_label"] is not None or row["emotion_label_name"] is not None:
                    raise ValueError("Rest incorrectly assigned an emotion")
            else:
                raise ValueError("Unknown segment type")
            if row["split"] not in metadata["split"]["allowed_values"]:
                raise ValueError("Undeclared split")
            start, stop = row["output_start_sample"]/200, row["output_stop_sample"]/200
            if not (row["segment_start_seconds"]-.002501 <= start < stop <= row["segment_stop_seconds"]+.002501):
                raise ValueError("Window crosses segment boundary")
            if row["output_stop_sample"]-row["output_start_sample"] != count or abs(row["duration"]-count/200) > 1e-10:
                raise ValueError("Duration/sample grid mismatch")
            segment_key = (row["source_path"], row["events_uri"], row["segment_start_seconds"], row["segment_stop_seconds"])
            segment_windows[segment_key].append(row["output_start_sample"])
            if not (row["start_sample"] == round(start*row["source_sfreq"]) and row["stop_sample"] == round(stop*row["source_sfreq"])):
                raise ValueError("Source sample indices disagree with output grid")
            expected_qc = not (row["clip_fraction"] > metadata["quality_control"]["max_clip_fraction"] or any(row["bad_channel_mask"]))
            if row["qc_pass"] != expected_qc or row["qc_pass"] != (not row["qc_reasons"]):
                raise ValueError("QC flag and measured criteria disagree")
            qc_count += row["qc_pass"]
            type_counts[row["segment_type"]] += 1
    check("all_window_invariants", len(sample_ids) == dataset.count_rows())
    check("complete_nonoverlapping_windows_with_only_short_tails_discarded", all(
        sorted(starts) == list(range(round(key[2]*200), round(key[3]*200)-WINDOW_SAMPLES+1, WINDOW_SAMPLES))
        for key, starts in segment_windows.items()))
    check("qc_counts_match_summary", qc_count == summary["qc_pass_count"])
    check("padding_counts_match_summary", padded == summary["padded_window_count"])
    check("label_counts_match_summary", dict(label_counts) == summary["emotion_counts"])
    if expected_signal_sha256 is not None:
        check("signals_match_preprocessed_source_sha256", signal_hash.hexdigest() == expected_signal_sha256)
    result = {"status": "pass", "scope": "local structure and numerical invariants; physical calibration is not verified",
              "dataset_id": metadata["dataset"]["dataset_id"], "dataset_version": metadata["dataset"]["version"],
              "n_windows": len(sample_ids), "qc_pass_count": qc_count, "qc_fail_count": len(sample_ids)-qc_count,
              "window_types": dict(type_counts), "emotion_counts": dict(label_counts),
              "padded_windows": padded, "source_unit_status": metadata["preprocessing"]["source_unit_status"],
              "signal_sha256": signal_hash.hexdigest(), "checks": checks}
    atomic_json(root/"validation.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?", default=Path("processed/emovid_rules_v1"))
    args = parser.parse_args()
    print(json.dumps(validate(args.root), ensure_ascii=False, indent=2))
