"""Unified EMOVID interface: discover, map, preprocess, validate and save to RustFS.

One acquisition directory becomes one equally named .lance table. Waveforms go
directly to S3; scratch arrays live in /tmp and are removed after each recording.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from dataclasses import dataclass
import hashlib
import inspect
import json
from pathlib import Path
import tempfile

import lance
import numpy as np
import pyarrow as pa

from batch_emovid_segments import discover_recordings, list_objects, process_recording
from emovid_segments import Store, split_s3
from emovid_storage import DEFAULT_MOUNT, DEFAULT_OUTPUT_URI, OutputStore, mount_processed, recording_path
import preprocess_emovid_rules as prep
from validate_emovid_rules import validate

LAYOUT = "emovid-per-acquisition-v1"


@dataclass
class PreprocessingOptions:
    raw_root: str = "s3://oi-data/EMOVID_RAW"
    label_root: str = "s3://oi-data/EMOVID-261029"
    raw: str | None = None
    output_root: Path = DEFAULT_MOUNT
    output_uri: str = DEFAULT_OUTPUT_URI
    profile: str = "rustfs"
    source_unit: str = "uV"
    unit_status: str = "confirmed"
    max_clip_fraction: float = .01
    block_seconds: float = 120
    context_seconds: float = 15
    work_dir: Path = Path("/tmp/emovid-preprocess")
    from_manifest: Path | None = None
    import_existing: Path | None = None
    resume: bool = True
    mount: bool = True
    mount_only: bool = False

    def __post_init__(self):
        for name in ("output_root", "work_dir", "from_manifest", "import_existing"):
            if getattr(self, name) is not None:
                setattr(self, name, Path(getattr(self, name)))
        if not 0 <= self.max_clip_fraction <= 1 or self.block_seconds <= 0 or self.context_seconds <= 0:
            raise ValueError("Invalid QC, block or context parameter")
        if self.import_existing and not self.from_manifest:
            raise ValueError("import_existing requires from_manifest")


def processing_hash():
    """A storage/CLI refactor does not change the waveform recipe."""
    return prep.canonical_hash([inspect.getsource(component) for component in
                               (prep.SignalReader, prep.filter_and_resample, prep.window_plan,
                                prep.encode_window, prep.preprocess_recording)])


def resolve_report(path, manifest):
    path = Path(path)
    for candidate in (path, manifest.parent/path, Path(__file__).parent/path):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Mapping report not found: {path}")


def mapping_jobs(args, store):
    if args.from_manifest:
        manifest = json.loads(args.from_manifest.read_text())
        inventory = json.loads(args.from_manifest.with_name("inventory.json").read_text())
        objects = {obj["uri"]: obj for obj in inventory["raw_objects"]}
        jobs = [{"raw_uri": item["raw_uri"], "signal_object": objects[item["raw_uri"]],
                 "aliases": item.get("aliases", []),
                 "report": json.loads(resolve_report(item["report_path"], args.from_manifest).read_text())}
                for item in manifest["recordings"]]
        skipped = manifest.get("skipped_collections", [])
    else:
        raw, labels = list_objects(store, args.raw_root), list_objects(store, args.label_root)
        jobs, _ = discover_recordings(raw)
        for job in jobs:
            job["logs"] = [obj for obj in labels if Path(obj["uri"]).name.startswith("events")
                           and Path(obj["uri"]).suffix == ".json"]
        _, prefix = split_s3(args.raw_root)
        prefix = prefix.rstrip("/")+"/"
        collections = {split_s3(obj["uri"])[1][len(prefix):].split("/")[0] for obj in raw}
        exported = {str(recording_path(job["raw_uri"], args.raw_root)) for job in jobs}
        skipped = [{"collection": name, "reason": "no_edf_bdf_export"} for name in sorted(collections-exported)]
    if args.raw:
        requested = split_s3(args.raw)
        jobs = [job for job in jobs if any(split_s3(uri) == requested
                for uri in [job["raw_uri"]] + job.get("aliases", []))]
        if not jobs:
            raise ValueError("Requested BDF was not found in the source inventory")
    # Validate the entire set before writing: a name collision must never
    # silently replace a different source recording.
    owners = {}
    for job in jobs:
        name = str(recording_path(job["raw_uri"], args.raw_root))
        if name in owners and owners[name] != job["raw_uri"]:
            raise ValueError(f"Acquisition-name collision: {name}: {owners[name]} and {job['raw_uri']}")
        owners[name] = job["raw_uri"]
        job["name"] = name
    return jobs, skipped


def mapping_signature(report):
    return prep.canonical_hash({key: report.get(key) for key in ("raw_uri", "recording", "segments", "matches")})


def import_checkpoint(args, job, report, config):
    if not args.import_existing:
        return None
    root = args.import_existing
    summary = json.loads((root/"summary.json").read_text())
    if summary["config"] != config:
        raise ValueError("Existing preprocessed data uses a different numeric configuration")
    if args.from_manifest is None or summary["mapping_manifest_sha256"] != prep.digest(args.from_manifest):
        raise ValueError("Existing data does not match the supplied mapping manifest")
    folder = root/"derivatives/checkpoints"/prep.canonical_hash(job["raw_uri"])[:16]
    old = json.loads((folder/"report.json").read_text())
    if old["raw_uri"] != job["raw_uri"] or prep.digest(folder/"data.npy") != old["data_sha256"]:
        raise ValueError("Existing checkpoint identity or waveform checksum mismatch")
    if old["n_windows"] != len(prep.window_plan(report["segments"])[0]):
        raise ValueError("Existing checkpoint window plan disagrees with current mapping")
    return folder, old


def summarize_record(name, raw_uri, rows, report, config, dataset, output, result):
    summary = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "layout": LAYOUT,
               "raw_uri": raw_uri, "recording_name": name, "dataset_id": "emovid-"+name,
               "table_name": name+".lance", "storage_uri": output.uri(name+".lance"),
               "config": config, "n_windows": len(rows), "qc_pass_count": sum(row["qc_pass"] for row in rows),
               "n_subjects": len({row["subject_id"] for row in rows}), "selected_recording_count": 1,
               "processed_recording_count": 1, "recordings": [{"status": "success", "raw_uri": raw_uri}],
               "window_types": dict(Counter(row["segment_type"] for row in rows)),
               "emotion_counts": dict(Counter(row["label_name"] for row in rows if row["segment_type"] == "emotion")),
               "qc_pass_emotion_counts": dict(Counter(row["label_name"] for row in rows
                                                       if row["segment_type"] == "emotion" and row["qc_pass"])),
               "padded_window_count": 0, "lance_version": dataset.version,
               "discarded_short_tail_seconds": result["discarded_short_tail_seconds"],
               "mapping_signature": mapping_signature(report)}
    if result.get("unit_confirmation"):
        summary["unit_confirmation"] = result["unit_confirmation"]
    return summary


def export_recording(args, job, report, config, output, store):
    name, raw_uri = job["name"], job["raw_uri"]
    signature = prep.canonical_hash({"source": job["signal_object"], "mapping": mapping_signature(report),
                                    "config": config, "processing_hash": processing_hash()})
    summary_key, validation_key = name+".summary.json", name+".validation.json"
    reservation_key = name+".processing.json"
    table_exists = False
    if output.exists(summary_key):
        previous = output.read_json(summary_key)
        if previous.get("layout") != LAYOUT or previous.get("raw_uri") != raw_uri:
            raise ValueError(f"Output belongs to another source: {summary_key}")
        if args.resume and previous.get("signature") == signature and output.exists(validation_key):
            validation = output.read_json(validation_key)
            dataset = lance.dataset(output.uri(name+".lance"), storage_options=output.options)
            if (validation.get("status") == "pass" and dataset.version == previous["lance_version"]
                    and dataset.count_rows() == previous["n_windows"]):
                return previous, True
    else:
        # Do not overwrite an unowned table even if its sidecars are absent.
        bucket, key = split_s3(output.uri(name+".lance"))
        table_exists = bool(output.client.list_objects_v2(Bucket=bucket, Prefix=key+"/", MaxKeys=1).get("KeyCount", 0))
        owned_partial = output.exists(reservation_key) and output.read_json(reservation_key).get("raw_uri") == raw_uri
        if table_exists and not owned_partial:
            raise ValueError(f"Unowned output table already exists: {name}.lance")
    output.put_json(reservation_key, {"layout": LAYOUT, "raw_uri": raw_uri, "signature": signature})
    args.work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=name+"-", dir=args.work_dir) as temporary:
        stage = Path(temporary)
        imported = import_checkpoint(args, job, report, config)
        if imported:
            folder, result = imported
        else:
            folder = stage/"signal"
            reader = prep.SignalReader(store, raw_uri, source_unit=args.source_unit)
            if reader.expected_size != job["signal_object"]["size"]:
                raise ValueError("BDF size disagrees with source inventory")
            result = prep.preprocess_recording(report, reader, config, folder)
            if result is None:
                raise ValueError("No complete eligible 3-second windows")
            result["status"] = "success"
            (folder/"report.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        data = np.load(folder/"data.npy", mmap_mode="r", allow_pickle=False)
        if data.shape != (result["n_windows"], len(prep.CHANNELS), prep.WINDOW_SAMPLES):
            raise ValueError("Unexpected checkpoint signal shape")
        expected_hash = hashlib.sha256()
        for start in range(0, len(data), 16):
            expected_hash.update(data[start:start+16].tobytes(order="C"))
        del data
        schema = prep.arrow_schema()
        batches = pa.RecordBatchReader.from_batches(schema, prep.arrow_batches([folder], schema))
        dataset = lance.write_dataset(batches, output.uri(name+".lance"), storage_options=output.options,
                                      mode="overwrite" if table_exists or output.exists(summary_key) else "create",
                                      max_rows_per_file=256, max_rows_per_group=16)
        summary = summarize_record(name, raw_uri, result["rows"], report, config, dataset, output, result)
        summary.update(signature=signature, status="success", numeric_data_origin="existing_checkpoint" if imported else "raw_bdf")
        if output.exists(name+".versions.toml"):
            response = output.client.get_object(Bucket=output.bucket, Key=output.key(name+".versions.toml"))
            try:
                (stage/"versions.toml").write_bytes(response["Body"].read())
            finally:
                response["Body"].close()
        prep.write_metadata(stage, schema, summary, config)
        (stage/"summary.json").write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
        validation = validate(stage, table_uri=output.uri(name+".lance"), storage_options=output.options,
                              expected_signal_sha256=expected_hash.hexdigest())
        for suffix in ("metadata.toml", "versions.toml"):
            output.put(name+"."+suffix, (stage/suffix).read_bytes(), "application/toml")
        output.put_json(name+".segments.json", dict(report, report_path=output.uri(name+".segments.json")))
        output.put_json(validation_key, validation)
        output.put_json(summary_key, summary)  # Write the readiness marker last.
        return summary, False


def preprocess_dataset(args):
    store = Store(args.profile)
    output = OutputStore(store.s3, args.output_uri)
    if args.mount or args.mount_only:
        mount_processed(store.s3, args.output_root, args.output_uri)
    if args.mount_only:
        return {"output_uri": output.root, "mountpoint": str(args.output_root)}
    jobs, skipped = mapping_jobs(args, store)
    config = prep.preprocessing_contract(args)
    previous = output.read_json("manifest.json") if output.exists("manifest.json") else None
    if previous is not None and previous.get("layout") != LAYOUT:
        raise ValueError("Output manifest belongs to another dataset/layout")
    results = {r["raw_uri"]: r for r in previous.get("recordings", [])} if previous and args.raw else {}
    for number, job in enumerate(jobs, 1):
        uri, name = job["raw_uri"], job["name"]
        try:
            bucket, key = split_s3(uri)
            head = store.s3.head_object(Bucket=bucket, Key=key)
            if head["ContentLength"] != job["signal_object"]["size"] or head.get("ETag") != job["signal_object"]["etag"]:
                raise ValueError("Raw source changed since inventory/mapping")
            report = job.get("report") or process_recording(job, job["logs"], store, {
                "min_segment_seconds": 3.0, "output_sfreq": 200.0, "search_padding_seconds": 600.0,
                "timezone": "Asia/Shanghai", "wall_tolerance_seconds": 30.0, "max_residual_seconds": 0.5})
            if not report.get("usable_segment_count", 0):
                output.put_json(name+".segments.json", report)
                results[uri] = {"raw_uri": uri, "recording_name": name, "status": "skipped",
                                "reason": report.get("reason", "no_eligible_segments")}
                print(f"[{number}/{len(jobs)}] skipped {name}", flush=True)
                continue
            summary, resumed = export_recording(args, job, report, config, output, store)
            result = {key: value for key, value in summary.items() if key != "recordings"}
            result.update(lance_uri=output.uri(name+".lance"),
                          metadata_uri=output.uri(name+".metadata.toml"), validation_uri=output.uri(name+".validation.json"),
                          segments_uri=output.uri(name+".segments.json"),
                          local_lance_path=str(args.output_root/(name+".lance")) if args.mount else None)
            results[uri] = result
            print(f"[{number}/{len(jobs)}] {'cached' if resumed else 'saved'} {name}.lance "
                  f"windows={summary['n_windows']} qc_pass={summary['qc_pass_count']}", flush=True)
        except Exception as exc:
            results[uri] = {"raw_uri": uri, "recording_name": name, "status": "error",
                            "error_type": type(exc).__name__, "reason": str(exc)}
            print(f"[{number}/{len(jobs)}] ERROR {name}: {type(exc).__name__}: {exc}", flush=True)
        output.put_json("progress.json", {"layout": LAYOUT, "recordings": list(results.values())})
    ordered = sorted(results.values(), key=lambda r: r["raw_uri"])
    successful = [r for r in ordered if r["status"] == "success"]
    aggregate = {"layout": LAYOUT, "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                 "output_uri": output.root, "mountpoint": str(args.output_root), "config": config,
                 "recordings": ordered, "skipped_collections": skipped,
                 "processed_recording_count": len(successful),
                 "status_counts": dict(Counter(r["status"] for r in ordered)),
                 "n_windows": sum(r["n_windows"] for r in successful),
                 "qc_pass_count": sum(r["qc_pass_count"] for r in successful), "padded_window_count": 0}
    for key in ("window_types", "emotion_counts", "qc_pass_emotion_counts"):
        counts = Counter()
        for result in successful:
            counts.update(result[key])
        aggregate[key] = dict(counts)
    if successful and successful[0].get("unit_confirmation"):
        aggregate["unit_confirmation"] = successful[0]["unit_confirmation"]
    output.put_json("manifest.json", aggregate)
    output.put_json("summary.json", aggregate)
    output.put_json("preprocessing_config.json", config)
    output.put_json("validation.json", {"status": "pass" if not any(r["status"] == "error" for r in ordered) and successful else "fail",
                                        "layout": LAYOUT, "n_windows": aggregate["n_windows"],
                                        "qc_pass_count": aggregate["qc_pass_count"], "validated_recording_count": len(successful)})
    print(json.dumps({key: aggregate[key] for key in ("processed_recording_count", "n_windows", "qc_pass_count", "status_counts")}, ensure_ascii=False))
    return aggregate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default="s3://oi-data/EMOVID_RAW")
    parser.add_argument("--raw", help="Optional single BDF URI; otherwise process all eligible recordings")
    parser.add_argument("--label-root", default="s3://oi-data/EMOVID-261029")
    parser.add_argument("--output-root", "--output-dir", type=Path, default=DEFAULT_MOUNT)
    parser.add_argument("--output-uri", default=DEFAULT_OUTPUT_URI)
    parser.add_argument("--from-manifest", "--manifest", type=Path, help="Reuse an existing mapping manifest")
    parser.add_argument("--import-existing", type=Path, help="Migrate verified existing checkpoints without refiltering raw data")
    parser.add_argument("--profile", default="rustfs")
    parser.add_argument("--source-unit", choices=("auto", "uV", "mV", "V"), default="uV")
    parser.add_argument("--unit-status", choices=("confirmed", "assumed"), default="confirmed")
    parser.add_argument("--max-clip-fraction", type=float, default=.01)
    parser.add_argument("--block-seconds", type=float, default=120)
    parser.add_argument("--context-seconds", type=float, default=15)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/emovid-preprocess"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--mount", action="store_true", default=True, help="Mount the processed prefix for local access (default)")
    parser.add_argument("--no-mount", dest="mount", action="store_false", help="Use S3 directly without a local FUSE view")
    parser.add_argument("--mount-only", action="store_true")
    args = parser.parse_args(argv)
    if not 0 <= args.max_clip_fraction <= 1 or args.block_seconds <= 0 or args.context_seconds <= 0:
        parser.error("Invalid QC, block or context parameter")
    if args.import_existing and not args.from_manifest:
        parser.error("--import-existing requires --from-manifest for source provenance checks")
    result = preprocess_dataset(args)
    if any(r["status"] == "error" for r in result.get("recordings", [])):
        parser.exit(1, "Some recordings failed; inspect the remote manifest and retry with --resume.\n")
    return result


if __name__ == "__main__":
    main()
