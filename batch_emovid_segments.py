"""Scan all EMOVID_RAW recordings, map every acquisition run, and export usable segments."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from zoneinfo import ZoneInfo

from emovid_segments import (
    ClockAlignment, LABEL_IDS, PROTOCOL_COMMIT, PROTOCOL_URL, Store, event_time, load_annotations,
    normalize_events, path_timestamp, positive, recording_header, s3_uri,
    segment_events, split_s3,
)
from emovid_storage import recording_path

SCHEMA_VERSION = 1


def list_objects(store, root):
    bucket, prefix = split_s3(root)
    return [{"uri": s3_uri(bucket, obj["Key"]), "size": obj["Size"], "etag": obj.get("ETag")}
            for page in store.s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/")
            for obj in page.get("Contents", [])]


def discover_recordings(objects):
    """Prefer combined exports over data.bdf and deduplicate identical export pairs."""
    by_directory = defaultdict(list)
    for obj in objects:
        by_directory[obj["uri"].rsplit("/", 1)[0]].append(obj)
    recordings, ignored = [], []
    for directory, files in sorted(by_directory.items()):
        annotations = [f for f in files if Path(f["uri"]).name.lower() in ("evt.bdf", "evt.edf")]
        signals = [f for f in files if Path(f["uri"]).suffix.lower() in (".bdf", ".edf") and f not in annotations]
        combined = [f for f in signals if Path(f["uri"]).stem.lower() != "data"]
        if combined:
            ignored.extend({"uri": f["uri"], "reason": "signal_component_of_combined_export"}
                           for f in signals if f not in combined)
            signals = combined
        for signal in signals:
            recordings.append({"raw_uri": signal["uri"], "signal_object": signal,
                               "annotation_object": annotations[0] if annotations else None, "aliases": []})
    # Prefer the original export directory to the '(1)' copies. ETags are used
    # only together with size and the separate annotation object's identity.
    recordings.sort(key=lambda r: (bool(re.search(r"\(\d+\)", r["raw_uri"])), r["raw_uri"]))
    unique, seen = [], {}
    for recording in recordings:
        a, b = recording["signal_object"], recording["annotation_object"]
        signature = (a["etag"], a["size"], b["etag"], b["size"]) if a.get("etag") and b and b.get("etag") else None
        if signature is not None and signature in seen:
            original = seen[signature]
            original["aliases"].append(recording["raw_uri"])
            ignored.append({"uri": recording["raw_uri"], "reason": "duplicate_export_etag_and_size",
                            "canonical_uri": original["raw_uri"]})
        else:
            unique.append(recording)
            if signature is not None:
                seen[signature] = recording
    return sorted(unique, key=lambda r: r["raw_uri"]), ignored


def raw_run_intervals(raw_events, duration):
    """Keep acquisition runs independent even inside a continuous device recording."""
    runs, active = [], None
    for t, code in raw_events:
        if code == 101:
            if active is not None:
                runs.append((active, t))
            active = t
        elif code == 102 and active is not None:
            runs.append((active, t))
            active = None
    if active is not None:
        runs.append((active, duration))
    return runs or [(0.0, duration)]


def header_datetime(header):
    # EDF uses 1985..2084 for its two-digit year convention.
    value = header["start_as_stored"]
    day, month, year = map(int, value[:8].split("."))
    hour, minute, second = map(int, value[8:].split("."))
    year += 1900 if year >= 85 else 2000
    return datetime(year, month, day, hour, minute, second)


def log_wall_origin(events, uri, timezone_name):
    """Estimate the log's relative_time_sec origin from its recorded wall clock."""
    for e in events:
        unix = e.get("payload", {}).get("timestamp_unix_sec")
        if isinstance(unix, (int, float)):
            wall = datetime.fromtimestamp(unix, ZoneInfo(timezone_name)).replace(tzinfo=None)
            return wall - timedelta(seconds=event_time(e)), "payload_timestamp_unix_sec"
    return path_timestamp(uri), "directory_timestamp_fallback"


def apply_duration_filter(segments, minimum_seconds):
    for segment in segments:
        if segment["duration_seconds"] < minimum_seconds:
            segment["usable"] = False
            if "below_minimum_duration" not in segment["exclusion_reasons"]:
                segment["exclusion_reasons"].append("below_minimum_duration")
        segment["minimum_duration_seconds"] = minimum_seconds
    return segments


def reject_overlapping_runs(matches, ambiguity_seconds=2.0):
    """Do not choose different label logs for the same raw acquisition interval."""
    accepted, ambiguous = [], []
    grouped = defaultdict(list)
    for match in matches:
        grouped[match["raw_run_index"]].append(match)
    for _, group in sorted(grouped.items()):
        group.sort(key=lambda x: x["wall_clock_difference_seconds"])
        if len(group) > 1 and group[1]["wall_clock_difference_seconds"] - group[0]["wall_clock_difference_seconds"] <= ambiguity_seconds:
            ambiguous.append({"raw_run_index": group[0]["raw_run_index"],
                              "events_uris": [m["events_uri"] for m in group], "reason": "ambiguous_validated_logs"})
        else:
            accepted.append(group[0])
    return accepted, ambiguous


def process_recording(recording, log_objects, store, config):
    raw_uri = recording["raw_uri"]
    report = {"schema_version": SCHEMA_VERSION, "raw_uri": raw_uri, "aliases": recording["aliases"],
              "status": "unmatched", "segments": [], "matches": [], "rejected_candidates": []}
    header = recording_header(store, raw_uri)
    report["recording"] = header
    start = header_datetime(header)
    end = start + timedelta(seconds=header["duration_seconds"])
    candidates = []
    for obj in log_objects:
        stamp = path_timestamp(obj["uri"])
        if start - timedelta(seconds=config["search_padding_seconds"]) <= stamp <= end:
            candidates.append(obj)
    report["candidate_count"] = len(candidates)
    if not candidates:
        report["reason"] = "no_logs_in_recording_time_window"
        return report
    annotation = recording["annotation_object"]
    if annotation is None:
        report.update(status="unsupported", reason="no_separate_annotation_export")
        return report
    raw_events, annotations_uri = load_annotations(store, raw_uri, annotation["uri"])
    runs = raw_run_intervals(raw_events, header["duration_seconds"])
    report.update(annotations_uri=annotations_uri, raw_run_count=len(runs), raw_annotation_count=len(raw_events))
    accepted = []
    for obj in candidates:
        uri = obj["uri"]
        try:
            events = normalize_events(store.json(uri))
            origin, wall_basis = log_wall_origin(events, uri, config["timezone"])
            expected_start = (origin - start).total_seconds() + event_time(events[0])
            possible_runs = [(i, bounds) for i, bounds in enumerate(runs)
                             if abs(bounds[0] - expected_start) <= config["wall_tolerance_seconds"]]
            if not possible_runs:
                raise ValueError("no_device_session_start_near_log_start")
            errors = []
            for run_index, (first, last) in possible_runs:
                try:
                    run_events = [(t, c) for t, c in raw_events if first <= t <= last]
                    alignment = ClockAlignment(events, run_events, max_residual_seconds=config["max_residual_seconds"])
                    # Validate real clock proximity as well as repeated trigger order.
                    errors_wall = [abs(alignment.exact[i] - ((origin-start).total_seconds() + event_time(events[i])))
                                   for i in alignment.exact]
                    from statistics import median
                    difference = median(errors_wall)
                    if difference > config["wall_tolerance_seconds"]:
                        raise ValueError(f"wall_clock_difference={difference:.3f}s")
                    segments, issues = segment_events(events, alignment, sfreq=header["sfreq"],
                                                      duration_seconds=header["duration_seconds"],
                                                      output_sfreq=config["output_sfreq"])
                    apply_duration_filter(segments, config["min_segment_seconds"])
                    for segment in segments:
                        segment.update(raw_uri=raw_uri, events_uri=uri, raw_run_index=run_index)
                    acquisition_errors = [{"event_index": i, "payload": e.get("payload", {})}
                                          for i, e in enumerate(events) if e.get("name") == "eeg_acquisition_error"]
                    accepted.append({"events_uri": uri, "raw_run_index": run_index,
                                     "wall_clock_difference_seconds": difference, "wall_clock_basis": wall_basis,
                                     "alignment": alignment.report, "segments": segments, "pairing_issues": issues,
                                     "acquisition_errors": acquisition_errors})
                except ValueError as exc:
                    errors.append(str(exc))
            if errors and not any(m["events_uri"] == uri for m in accepted):
                raise ValueError("; ".join(errors))
        except (ValueError, KeyError, TypeError) as exc:
            report["rejected_candidates"].append({"events_uri": uri, "reason": str(exc)})
    matches, ambiguous = reject_overlapping_runs(accepted)
    report["ambiguous_matches"] = ambiguous
    report["matches"] = [{k: v for k, v in match.items() if k != "segments"} for match in matches]
    report["segments"] = sorted([segment for match in matches for segment in match["segments"]], key=lambda s: s["start_seconds"])
    matched_runs = {match["raw_run_index"] for match in matches}
    report["unmatched_runs"] = [{"raw_run_index": i, "start_seconds": first, "stop_seconds": last}
                                for i, (first, last) in enumerate(runs) if i not in matched_runs]
    if matches:
        report["status"] = "partial" if (report["unmatched_runs"] or ambiguous or any(
            m["pairing_issues"] or m["acquisition_errors"] for m in matches)) else "success"
    elif ambiguous:
        report["status"] = "ambiguous"
    report["usable_segment_count"] = sum(s["usable"] for s in report["segments"])
    report["duration_filtered_count"] = sum("below_minimum_duration" in s["exclusion_reasons"] for s in report["segments"])
    return report


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def fingerprint(recording, logs, config):
    source_files = [Path(__file__), Path(__file__).with_name("emovid_segments.py")]
    value = {"recording": recording, "logs": logs, "config": config,
             "code_hashes": [hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files]}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_csv(path, rows, fields):
    # Summaries are regenerated from completed per-recording JSON files.
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def batch_run(args):
    store = Store(args.profile)
    raw_objects, label_objects = list_objects(store, args.raw_root), list_objects(store, args.label_root)
    logs = [o for o in label_objects if re.search(r"/events(?:_part_\d+)?\.json$", o["uri"])]
    recordings, ignored = discover_recordings(raw_objects)
    owners = {}
    for recording in recordings:
        name = str(recording_path(recording["raw_uri"], args.raw_root))
        if name in owners and owners[name] != recording["raw_uri"]:
            raise ValueError(f"Acquisition-name collision: {name}; refusing to overwrite mapping reports")
        owners[name] = recording["raw_uri"]
    config = {"min_segment_seconds": args.min_segment_seconds, "output_sfreq": args.output_sfreq,
              "search_padding_seconds": args.search_padding_seconds, "timezone": args.timezone,
              "wall_tolerance_seconds": args.wall_tolerance_seconds, "max_residual_seconds": args.max_residual_seconds}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    marker = args.output_dir / "batch_config.json"
    if not marker.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Output directory contains unrelated files; use a new directory")
    atomic_json(marker, {"schema_version": SCHEMA_VERSION, "raw_root": args.raw_root,
                         "label_root": args.label_root, "config": config})
    atomic_json(args.output_dir / "inventory.json", {"raw_objects": raw_objects, "event_logs": logs, "ignored_exports": ignored})
    print(f"Discovered {len(raw_objects)} objects, {len(recordings)} unique recordings, {len(logs)} event logs", flush=True)

    def worker(recording):
        uri = recording["raw_uri"]
        name = str(recording_path(uri, args.raw_root))
        path = args.output_dir / "recordings" / f"{name}.json"
        signature = fingerprint(recording, logs, config)
        if args.resume and path.exists():
            previous = json.loads(path.read_text(encoding="utf-8"))
            if previous.get("source_fingerprint") == signature and previous.get("status") in ("success", "partial"):
                return previous, True
        try:
            result = process_recording(recording, logs, store, config)
        except Exception as exc:
            result = {"raw_uri": uri, "aliases": recording["aliases"], "status": "error", "segments": [],
                      "error_type": type(exc).__name__, "reason": str(exc)}
        result.update(source_fingerprint=signature, report_path=str(path))
        atomic_json(path, result)
        return result, False

    results = []
    # Initialize the boto client once before threads; boto clients can be shared.
    store.s3
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(worker, recording) for recording in recordings]
        for future in as_completed(futures):
            result, resumed = future.result()
            results.append(result)
            print(f"[{len(results)}/{len(recordings)}] {result['status']} "
                  f"usable={result.get('usable_segment_count', 0)} "
                  f"{'(cached) ' if resumed else ''}{result['raw_uri']}", flush=True)
    results.sort(key=lambda r: r["raw_uri"])
    bucket, root_prefix = split_s3(args.raw_root)
    def collection(uri):
        return split_s3(uri)[1][len(root_prefix.rstrip("/") + "/"):].split("/")[0]
    all_collections = {collection(o["uri"]) for o in raw_objects}
    exported_collections = {collection(r["raw_uri"]) for r in results}
    skipped = [{"collection": name, "reason": "no_edf_bdf_export"} for name in sorted(all_collections - exported_collections)]
    segments = [s for r in results for s in r.get("segments", [])]
    usable = [s for s in segments if s["usable"]]
    summary = {"schema_version": SCHEMA_VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "raw_root": args.raw_root, "label_root": args.label_root, "config": config,
               "raw_object_count": len(raw_objects), "collection_count": len(all_collections),
               "unique_recording_count": len(recordings), "event_log_count": len(logs),
               "status_counts": dict(Counter(r["status"] for r in results)),
               "skipped_collections": skipped, "ignored_exports": ignored,
               "segment_count": len(segments), "usable_segment_count": len(usable),
               "duration_filtered_count": sum("below_minimum_duration" in s["exclusion_reasons"] for s in segments),
               "usable_segment_counts": dict(Counter(s["rest_type"] or "emotion" for s in usable)),
               "usable_emotion_counts": dict(Counter(s["emotion_label"] for s in usable if s["segment_type"] == "emotion")),
               "label_ids": LABEL_IDS, "protocol_commit": PROTOCOL_COMMIT,
               "protocol_source": PROTOCOL_URL + "/src/video_eeg/utils/markers.py",
               "recordings": [{k: v for k, v in r.items() if k != "segments"} for r in results]}
    atomic_json(args.output_dir / "manifest.json", summary)
    segment_fields = ["raw_uri", "events_uri", "raw_run_index", "subject_id", "session_id", "eeg_part", "video_id", "trial_id", "attempt_id",
                      "segment_type", "rest_type", "emotion_label", "emotion_label_id", "start_seconds", "stop_seconds", "duration_seconds",
                      "sfreq", "start_sample", "stop_sample", "output_sfreq", "output_start_sample", "output_stop_sample", "usable", "exclusion_reasons",
                      "start_time_basis", "stop_time_basis"]
    write_csv(args.output_dir / "all_segments.csv", segments, segment_fields)
    write_csv(args.output_dir / "usable_segments.csv", usable, segment_fields)
    write_csv(args.output_dir / "recordings.csv", summary["recordings"],
              ["raw_uri", "status", "usable_segment_count", "duration_filtered_count", "raw_run_count", "unmatched_runs", "reason", "report_path"])
    print(json.dumps({k: summary[k] for k in ("status_counts", "usable_segment_counts", "usable_emotion_counts", "duration_filtered_count")}, ensure_ascii=False))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default="s3://oi-data/EMOVID_RAW/")
    parser.add_argument("--label-root", default="s3://oi-data/EMOVID-261029/")
    parser.add_argument("--profile", default="rustfs")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/emovid_batch"))
    parser.add_argument("--min-segment-seconds", type=float, default=3.0)
    parser.add_argument("--output-sfreq", type=float, default=200.0)
    parser.add_argument("--search-padding-seconds", type=float, default=600.0)
    parser.add_argument("--wall-tolerance-seconds", type=float, default=30.0)
    parser.add_argument("--max-residual-seconds", type=float, default=0.5)
    parser.add_argument("--timezone", default="Asia/Shanghai", help="Timezone of acquisition EDF/BDF headers")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true", help="Reuse successful reports when sources, code and configuration match")
    args = parser.parse_args()
    for name in ("min_segment_seconds", "output_sfreq", "search_padding_seconds", "wall_tolerance_seconds", "max_residual_seconds"):
        positive(getattr(args, name), name)
    if args.workers < 1:
        parser.error("--workers must be positive")
    batch_run(args)


if __name__ == "__main__":
    main()
