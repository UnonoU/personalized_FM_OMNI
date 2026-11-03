"""Find EMOVID event logs and map rest/emotion intervals onto EDF/BDF samples.

Run ``python emovid_segments.py --help``. S3 access reuses read_rustfs.client;
matching and segmentation functions do not require network access or MNE.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
from difflib import SequenceMatcher
import json
import math
from pathlib import Path
import re
import statistics
import tempfile
from urllib.parse import urlsplit


PROTOCOL_COMMIT = "637058b3b2a431132a4643e56444d03a3bb5567d"
PROTOCOL_URL = f"https://github.com/18yiba/visual-video-task/blob/{PROTOCOL_COMMIT}"
LABEL_IDS = {"negative": 0, "neutral": 1, "positive": 2}
EVENT_CODES = {
    "session_start": 101, "session_end": 102,
    "baseline_start": 110, "baseline_end": 111,
    "eyes_open_baseline_start": 112, "eyes_open_baseline_end": 113,
    "eyes_closed_baseline_start": 114, "eyes_closed_baseline_end": 115,
    "fixation_on": 130, "fixation_off": 131, "video_on": 132, "video_off": 133,
    "break_start": 136, "break_end": 137, "trial_start": 140, "trial_end": 141,
    "EMOTION_VIDEO_ONSET": 150, "EMOTION_VIDEO_OFFSET": 151,
    "VALENCE_RATING_ONSET": 152, "VALENCE_RATING_RESPONSE": 153,
    "AROUSAL_RATING_ONSET": 154, "AROUSAL_RATING_RESPONSE": 155,
    "SHORT_REST_ONSET": 156, "SHORT_REST_OFFSET": 157,
    "LIKING_ONSET": 158, "LIKING_RESPONSE": 159,
    "ALARM_ONSET": 160, "ALARM_RESPONSE": 161,
    "FATIGUE_ONSET": 162, "FATIGUE_RESPONSE": 163,
}
PAIRS = {
    "baseline_start": ("baseline_end", "baseline"),
    "eyes_open_baseline_start": ("eyes_open_baseline_end", "eyes_open_baseline"),
    "eyes_closed_baseline_start": ("eyes_closed_baseline_end", "eyes_closed_baseline"),
    "break_start": ("break_end", "short_rest"),
    "SHORT_REST_ONSET": ("SHORT_REST_OFFSET", "short_rest"),
    "rest_start": ("rest_end", "long_rest"),
    "EMOTION_VIDEO_ONSET": ("EMOTION_VIDEO_OFFSET", "emotion"),
}
ENDS = {v[0]: k for k, v in PAIRS.items()}


def positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def split_s3(uri):
    """Accept s3://bucket/key and the UI notation s3://bucket@region/key."""
    parsed = urlsplit(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError(f"Invalid S3 URI: {uri}")
    bucket = parsed.netloc.split("@", 1)[0]
    return bucket, parsed.path.lstrip("/")


def s3_uri(bucket, key):
    return f"s3://{bucket}/{key}"


def path_timestamp(path):
    stamps = re.findall(r"(?<!\d)(20\d{6}_\d{6}|20\d{12})(?!\d)", str(path))
    if not stamps:
        raise ValueError(f"No YYYYMMDDHHMMSS or YYYYMMDD_HHMMSS timestamp in {path}")
    # Raw paths repeat the acquisition stamp in nested directory names.
    unique = {datetime.strptime(s.replace("_", ""), "%Y%m%d%H%M%S") for s in stamps}
    if len(unique) != 1:
        raise ValueError(f"Conflicting directory timestamps in {path}")
    return unique.pop()


@dataclass(frozen=True)
class Candidate:
    events_uri: str
    delta_seconds: float
    subject_id: str
    session: str


def rank_candidates(raw_uri, event_uris, *, max_delta_seconds=600, subject_id=None):
    """Rank same-day logs by absolute directory-time distance; IDs are optional filters."""
    limit = positive(max_delta_seconds, "max_delta_seconds")
    raw_stamp = path_timestamp(raw_uri)
    result = []
    for uri in event_uris:
        match = re.search(r"/([^/]+)/(session_\d+)/(20\d{6}_\d{6})/events(?:_part_\d+)?\.json$", uri)
        if not match:
            continue
        subject, session, stamp = match.groups()
        if subject_id is not None and subject != str(subject_id):
            continue
        stamp = datetime.strptime(stamp, "%Y%m%d_%H%M%S")
        delta = (stamp - raw_stamp).total_seconds()
        if stamp.date() == raw_stamp.date() and abs(delta) <= limit:
            result.append(Candidate(uri, delta, subject, session))
    return sorted(result, key=lambda c: (abs(c.delta_seconds), c.events_uri))


def select_candidate(candidates, ambiguity_seconds=30):
    if not math.isfinite(ambiguity_seconds) or ambiguity_seconds < 0:
        raise ValueError("ambiguity_seconds must be finite and nonnegative")
    if not candidates:
        raise ValueError("No event log in the time window; adjust --max-delta-seconds or use --events")
    if len(candidates) > 1 and abs(candidates[1].delta_seconds) - abs(candidates[0].delta_seconds) <= ambiguity_seconds:
        raise ValueError("Ambiguous time match; use --subject-id or --events. Candidates: "
                         + json.dumps([asdict(c) for c in candidates[:5]], ensure_ascii=False))
    return candidates[0]


def event_code(event):
    payload = event.get("payload", {})
    if event.get("name") in ("rest_start", "rest_end"):
        return None  # Local-only; payload inherits the previous trial context.
    code = payload.get("marker_code")
    return int(code) if code is not None else EVENT_CODES.get(event.get("name"))


def event_time(event):
    value = float(event["relative_time_sec"])
    if not math.isfinite(value) or value < 0:
        raise ValueError("Event relative_time_sec must be finite and nonnegative")
    return value


def normalize_events(document):
    events = document if isinstance(document, list) else document.get("events")
    if not isinstance(events, list) or not events:
        raise ValueError("events.json must be a nonempty array or an object containing events[]")
    last = -1.0
    for event in events:
        if not isinstance(event, dict) or not isinstance(event.get("payload", {}), dict):
            raise ValueError("Malformed event record")
        now = event_time(event)
        if now < last:
            raise ValueError("Event clock resets: process separate EEG parts independently")
        last = now
    parts = {e.get("payload", {}).get("eeg_part") for e in events} - {None}
    if len(parts) > 1:
        raise ValueError("Multiple EEG parts: process each events_part_NNN.json independently")
    return events


class ClockAlignment:
    """Exact common trigger times plus interpolation for local-only log events."""

    def __init__(self, events, raw_events, *, min_coverage=0.8, max_residual_seconds=0.5):
        if not 0 < min_coverage <= 1:
            raise ValueError("min_coverage must be in (0, 1]")
        positive(max_residual_seconds, "max_residual_seconds")
        # Recorder relative_time_sec can be rounded to milliseconds, assigning
        # two different triggers the same time. The flip callback clock keeps
        # full precision and also exists on local-only rest events.
        timing_events = [e for e in events if event_code(e) is not None or e.get("name") in PAIRS or e.get("name") in ENDS]
        self.time_basis = ("flip_callback_time_monotonic_sec" if all(
            isinstance(e.get("payload", {}).get("flip_callback_time_monotonic_sec"), (int, float))
            and math.isfinite(e["payload"]["flip_callback_time_monotonic_sec"])
            for e in timing_events) else "relative_time_sec")
        log = [(i, event_code(e), self.source_time(e)) for i, e in enumerate(events)
               if event_code(e) is not None
               and e.get("payload", {}).get("external_marker_sent") is not False
               and e.get("payload", {}).get("marker_send_success") is not False]
        codes = {code for _, code, _ in log}
        raw = [(float(t), int(c)) for t, c in raw_events if int(c) in codes]
        if any(not math.isfinite(t) or t < 0 for t, _ in raw):
            raise ValueError("Invalid annotation onset")
        if any(a[0] > b[0] for a, b in zip(raw, raw[1:])):
            raise ValueError("Raw annotations must be ordered")
        matcher = SequenceMatcher(None, [c for _, c, _ in log], [c for _, c in raw], autojunk=False)
        anchors = [(log[b.a + j][0], log[b.a + j][2], raw[b.b + j][0])
                   for b in matcher.get_matching_blocks() for j in range(b.size)]
        coverage = len(anchors) / max(1, len(log))
        if len(anchors) < 3 or coverage < min_coverage:
            raise ValueError(f"Insufficient common triggers: {len(anchors)}/{len(log)}; wrong log or missing markers")
        xs, ys = [a[1] for a in anchors], [a[2] for a in anchors]
        if any(b <= a for a, b in zip(xs, xs[1:])) or any(b <= a for a, b in zip(ys, ys[1:])):
            raise ValueError("Common triggers must have strictly increasing timestamps")
        xbar, ybar = statistics.mean(xs), statistics.mean(ys)
        denom = sum((x - xbar)**2 for x in xs)
        slope = sum((x - xbar)*(y - ybar) for x, y in zip(xs, ys)) / denom
        intercept = ybar - slope*xbar
        residuals = [abs(y - (intercept + slope*x)) for x, y in zip(xs, ys)]
        if not 0.98 <= slope <= 1.02 or max(residuals) > max_residual_seconds:
            raise ValueError("Trigger order matches but clocks disagree; mapping rejected. "
                             f"slope={slope:.8f}, maximum residual={max(residuals):.6f}s")
        self.exact = {i: y for i, x, y in anchors}
        self.xs, self.ys = xs, ys
        self.report = {"method": "common_trigger_sequence_and_piecewise_linear_interpolation",
                       "source_clock": self.time_basis,
                       "matched_triggers": len(anchors), "log_triggers": len(log),
                       "raw_triggers_with_log_codes": len(raw), "log_coverage": coverage,
                       "clock_slope": slope, "clock_intercept_seconds": intercept,
                       "maximum_fit_residual_seconds": max(residuals),
                       "median_fit_residual_seconds": statistics.median(residuals)}

    def source_time(self, event):
        if self.time_basis == "relative_time_sec":
            return event_time(event)
        return float(event["payload"][self.time_basis])

    def map_event(self, index, event):
        if index in self.exact:
            return self.exact[index], "raw_trigger"
        x = self.source_time(event)
        at = bisect_left(self.xs, x)
        if at == 0 or at == len(self.xs):
            raise ValueError("No common triggers bracketing event; extrapolation is disabled")
        ratio = (x - self.xs[at-1]) / (self.xs[at] - self.xs[at-1])
        return self.ys[at-1] + ratio*(self.ys[at] - self.ys[at-1]), "interpolated_log_clock"


def pair_identity(event, kind):
    payload = event.get("payload", {})
    fields = ["subject_id", "session_id", "eeg_part"]
    if kind in ("emotion", "short_rest"):
        fields += ["trial_id", "attempt_id", "video_id"]
    elif kind == "long_rest":
        fields += ["rest_index"]
    return tuple(str(payload.get(key, "")) for key in fields)


def segment_events(events, alignment, *, sfreq, duration_seconds, output_sfreq=None):
    """Return half-open intervals and pairing issues; never infer rest from gaps."""
    sfreq = positive(sfreq, "sfreq")
    duration_seconds = positive(duration_seconds, "duration_seconds")
    target = positive(output_sfreq, "output_sfreq") if output_sfreq is not None else None
    active, segments, issues = {}, [], []

    def issue(index, reason, **extra):
        issues.append({"event_index": index, "event_name": events[index]["name"],
                       "issue": reason, **extra})

    for index, event in enumerate(events):
        name = event["name"]
        if name in ("session_start", "session_end"):
            for _, (start, _, _) in active.items():
                issue(start, "unclosed_at_session_boundary", boundary_event_index=index)
            active.clear()
        if name in PAIRS:
            end_name, kind = PAIRS[name]
            key = (name, pair_identity(event, kind))
            # A new start cannot complete an earlier unfinished attempt of this phase.
            for old in [key_ for key_ in active if key_[0] == name]:
                old_index, _, _ = active.pop(old)
                issue(old_index, "unclosed_before_next_start", boundary_event_index=index)
            active[key] = (index, event, kind)
        elif name in ENDS:
            start_name = ENDS[name]
            kind = PAIRS[start_name][1]
            key = (start_name, pair_identity(event, kind))
            if key not in active:
                issue(index, "end_without_matching_start")
                continue
            start_index, onset_event, kind = active.pop(key)
            p, q = onset_event.get("payload", {}), event.get("payload", {})
            try:
                start, start_basis = alignment.map_event(start_index, onset_event)
                stop, stop_basis = alignment.map_event(index, event)
            except ValueError as exc:
                issue(start_index, "unmapped_boundary", detail=str(exc))
                continue
            if not 0 <= start < stop <= duration_seconds:
                issue(start_index, "invalid_or_out_of_recording_interval", start_seconds=start, stop_seconds=stop)
                continue
            reasons = []
            label = p.get("three_class_label") if kind == "emotion" else None
            if kind == "emotion":
                if label not in LABEL_IDS:
                    reasons.append("missing_or_unknown_emotion_label")
                if q.get("three_class_label", label) != label:
                    reasons.append("conflicting_emotion_labels")
                if (q.get("completed_naturally") is False or q.get("skipped") is True
                        or q.get("status") in ("aborted", "skipped", "interrupted", "failed")
                        or q.get("abort_reason")):
                    reasons.append("incomplete_video")
            if kind != "long_rest" and (start_basis != "raw_trigger" or stop_basis != "raw_trigger"):
                reasons.append("missing_raw_boundary_trigger")
            first, last = round(start*sfreq), round(stop*sfreq)
            if last <= first:
                issue(start_index, "empty_sample_interval")
                continue
            row = {"segment_type": "emotion" if kind == "emotion" else "rest", "rest_type": None if kind == "emotion" else kind,
                   "emotion_label": label, "emotion_label_id": LABEL_IDS.get(label),
                   "start_seconds": start, "stop_seconds": stop, "duration_seconds": stop-start,
                   "start_sample": first, "stop_sample": last, "sfreq": sfreq,
                   "start_event_index": start_index, "stop_event_index": index,
                   "start_time_basis": start_basis, "stop_time_basis": stop_basis,
                   "usable": not reasons, "exclusion_reasons": reasons}
            for field in ("subject_id", "session_id", "eeg_part", "trial_id", "attempt_id", "video_id", "rest_index"):
                # Long-rest payload carries the previous video; retain only its rest identity.
                row[field] = p.get(field) if kind != "long_rest" or field not in ("trial_id", "attempt_id", "video_id") else None
            row["original_label"] = p.get("original_label") if kind == "emotion" else None
            if target is not None:
                row.update(output_sfreq=target, output_start_sample=round(start*target), output_stop_sample=round(stop*target))
            segments.append(row)
    for index, _, _ in active.values():
        issue(index, "unclosed_at_end_of_log")
    return sorted(segments, key=lambda s: s["start_seconds"]), issues


class Store:
    def __init__(self, profile="rustfs"):
        self.profile, self._s3 = profile, None

    @property
    def s3(self):
        if self._s3 is None:
            from read_rustfs import client
            self._s3 = client(self.profile)
        return self._s3

    def read(self, uri, byte_range=None):
        if uri.startswith("s3://"):
            bucket, key = split_s3(uri)
            options = {"Range": f"bytes={byte_range[0]}-{byte_range[1]}"} if byte_range else {}
            response = self.s3.get_object(Bucket=bucket, Key=key, **options)
            try:
                return response["Body"].read()
            finally:
                response["Body"].close()
        with Path(uri).open("rb") as stream:
            if byte_range:
                stream.seek(byte_range[0])
                return stream.read(byte_range[1] - byte_range[0] + 1)
            return stream.read()

    def exists(self, uri):
        if not uri.startswith("s3://"):
            return Path(uri).is_file()
        from botocore.exceptions import ClientError
        bucket, key = split_s3(uri)
        try:
            self.s3.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError as exc:
            if exc.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def list_event_uris(self, root):
        if root.startswith("s3://"):
            bucket, prefix = split_s3(root)
            for page in self.s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/"):
                for item in page.get("Contents", []):
                    if re.search(r"/events(?:_part_\d+)?\.json$", item["Key"]):
                        yield s3_uri(bucket, item["Key"])
        else:
            for path in Path(root).rglob("events*.json"):
                if re.fullmatch(r"events(?:_part_\d+)?\.json", path.name):
                    yield str(path)

    def json(self, uri):
        return json.loads(self.read(uri).decode("utf-8-sig"))


def resolve_raw(store, uri):
    if store.exists(uri):
        return uri
    suffix = Path(uri).suffix.lower()
    if suffix in (".edf", ".bdf"):
        alternate = uri[:-4] + (".bdf" if suffix == ".edf" else ".edf")
        if store.exists(alternate):
            return alternate
    raise FileNotFoundError(f"Raw recording does not exist: {uri}")


def recording_header(store, uri):
    fixed = store.read(uri, (0, 255))
    if len(fixed) != 256 or fixed[:8] not in (b"\xffBIOSEMI", b"0       "):
        raise ValueError("Expected an EDF/BDF header")
    count, length = int(fixed[252:256]), int(fixed[184:192])
    if count < 1 or length != 256*(count+1):
        raise ValueError("Invalid EDF/BDF channel header")
    if b"+D" in fixed[192:236]:
        raise ValueError("Discontinuous EDF/BDF requires a gap-aware timeline")
    channels = store.read(uri, (256, length-1))
    if len(channels) != length-256:
        raise ValueError("Truncated EDF/BDF channel header")
    labels = [channels[i*16:(i+1)*16].decode("latin1").strip() for i in range(count)]
    records, seconds = int(fixed[236:244]), positive(float(fixed[244:252]), "record_seconds")
    if records <= 0:
        raise ValueError("Finalized positive EDF/BDF record count required")
    # samples_per_record follows 216 bytes per channel of preceding fields.
    samples = [int(channels[216*count+i*8:216*count+(i+1)*8]) for i in range(count)]
    rates = {n/seconds for n, label in zip(samples, labels) if "Annotations" not in label}
    if len(rates) != 1:
        raise ValueError("Mixed-rate raw signals: an explicit EEG channel selection is required")
    return {"sfreq": positive(rates.pop(), "sfreq"), "duration_seconds": records*seconds,
            "start_as_stored": fixed[168:184].decode("latin1"), "channel_names": labels}


def load_annotations(store, raw_uri, annotations_uri=None):
    import mne

    if annotations_uri is None:
        directory = raw_uri.rsplit("/", 1)[0]
        sibling = directory + "/evt.bdf"
        if store.exists(sibling):
            annotations_uri = sibling
        elif not raw_uri.startswith("s3://"):
            annotations_uri = raw_uri
        else:
            raise ValueError("No adjacent evt.bdf; provide --annotations (local/S3 BDF, EDF or event CSV)")
    if annotations_uri.lower().endswith(".csv"):
        rows = csv.DictReader(store.read(annotations_uri).decode("utf-8-sig").splitlines())
        return [(float(row["onset_seconds"]), int(row["code"])) for row in rows], annotations_uri
    with tempfile.TemporaryDirectory(prefix="emovid-annotations-") as temp:
        path = annotations_uri
        if annotations_uri.startswith("s3://"):
            path = str(Path(temp) / Path(annotations_uri).name)
            Path(path).write_bytes(store.read(annotations_uri))
        ann = mne.read_annotations(path)
        # Adjacent evt.bdf is normally exported with the signal file's start.
        # Explicitly reject a different origin instead of silently shifting it.
        fixed = store.read(annotations_uri, (0, 255))
        raw_fixed = store.read(raw_uri, (0, 255))
        if fixed[168:184] != raw_fixed[168:184]:
            raise ValueError("Annotation file and raw recording have different header start times")
        raw_events = []
        for onset, description in zip(ann.onset, ann.description):
            if re.fullmatch(r"\d+(?:\.0+)?", description.strip()):
                raw_events.append((float(onset), int(float(description))))
        return raw_events, annotations_uri


def run(args):
    store = Store(args.profile)
    raw_uri = resolve_raw(store, args.raw)
    candidates = []
    events_uri = args.events
    if events_uri is None:
        candidates = rank_candidates(raw_uri, store.list_event_uris(args.label_root),
                                     max_delta_seconds=args.max_delta_seconds, subject_id=args.subject_id)
        events_uri = select_candidate(candidates, args.ambiguity_seconds).events_uri
    report = {"requested_raw_uri": args.raw, "raw_uri": raw_uri, "events_uri": events_uri,
              "candidate_matches": [asdict(c) for c in candidates],
              "protocol_commit": PROTOCOL_COMMIT,
              "protocol_sources": [PROTOCOL_URL + "/src/video_eeg/utils/markers.py",
                                   PROTOCOL_URL + "/src/video_eeg/experiment/emotion_runner.py"],
              "label_ids": LABEL_IDS, "interval_convention": "[start, stop)",
              "sample_rounding": "nearest integer; seconds relative to uncropped raw recording"}
    if args.search_only:
        return report
    header = recording_header(store, raw_uri)
    events = normalize_events(store.json(events_uri))
    raw_events, annotations_uri = load_annotations(store, raw_uri, args.annotations)
    alignment = ClockAlignment(events, raw_events, max_residual_seconds=args.max_residual_seconds)
    segments, issues = segment_events(events, alignment, sfreq=header["sfreq"],
                                      duration_seconds=header["duration_seconds"], output_sfreq=args.output_sfreq)
    report.update(recording=header, annotations_uri=annotations_uri, alignment=alignment.report,
                  segments=segments, pairing_issues=issues,
                  segment_counts=dict(Counter(s["rest_type"] or "emotion" for s in segments)),
                  emotion_counts=dict(Counter(s["emotion_label"] for s in segments if s["segment_type"] == "emotion")),
                  usable_segment_count=sum(s["usable"] for s in segments))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, help="Raw EDF/BDF local path or S3 URI")
    parser.add_argument("--label-root", default="s3://oi-data/EMOVID-261029")
    parser.add_argument("--events", help="Explicit events.json/part file, bypassing directory-time search")
    parser.add_argument("--subject-id", help="Optional numeric platform participant ID")
    parser.add_argument("--profile", default="rustfs")
    parser.add_argument("--max-delta-seconds", type=float, default=600)
    parser.add_argument("--ambiguity-seconds", type=float, default=30)
    parser.add_argument("--max-residual-seconds", type=float, default=0.5)
    parser.add_argument("--annotations", help="Device annotations: EDF/BDF or CSV with onset_seconds,code")
    parser.add_argument("--output-sfreq", type=float, help="Also emit indices at e.g. 200 Hz (before cropping)")
    parser.add_argument("--search-only", action="store_true")
    parser.add_argument("--output", required=True, type=Path, help="New JSON output path")
    args = parser.parse_args()
    if args.output.exists():
        parser.exit(1, f"Output already exists: {args.output}\n")
    try:
        report = run(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps({"raw_uri": report["raw_uri"], "events_uri": report["events_uri"],
                          "segment_counts": report.get("segment_counts"), "output": str(args.output)}, ensure_ascii=False))
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"Mapping failed: {exc}\n")


if __name__ == "__main__":
    main()
