#!/usr/bin/env python3
"""Incremental two-subject EEG preprocessing with session-level canonical shards."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import platform
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy.signal import butter, resample_poly, sosfiltfilt

ALLOWED_ROOT = Path("/mnt/cfs/jianyu/Demo")
COS_MOUNT_ROOT = Path("/mnt/cos")
SUPPORTED_SUBJECTS = ("sub-13467638", "sub-38861884")
SUBJECT_INDEX = {"sub-13467638": 0, "sub-38861884": 1}
POSTERIOR17 = ["Pz", "P3", "P4", "P5", "P6", "P7", "P8", "POz", "PO3",
               "PO4", "PO5", "PO6", "PO7", "PO8", "Oz", "O1", "O2"]
FULL59 = [
    "Fpz", "Fp1", "Fp2", "AF3", "AF4", "AF7", "AF8", "Fz", "F1", "F2",
    "F3", "F4", "F5", "F6", "F7", "F8", "FCz", "FC1", "FC2", "FC3",
    "FC4", "FC5", "FC6", "FT7", "FT8", "Cz", "C1", "C2", "C3", "C4",
    "C5", "C6", "T7", "T8", "CP1", "CP2", "CP3", "CP4", "CP5", "CP6",
    "TP7", "TP8", "Pz", "P3", "P4", "P5", "P6", "P7", "P8", "POz",
    "PO3", "PO4", "PO5", "PO6", "PO7", "PO8", "Oz", "O1", "O2",
]
BRANCHES = ("full_preprocessed", "full_mvnn", "visual17_mvnn_full")
EXPECTED_MVNN17_SHA = "bfb0bb3ba56e1397598811063d9d8e9a99f7bcbd465db4a73944da65dd7d5f8b"
MVNN59_SOURCE = Path("/mnt/cfs/jianyu/Demo/THINGS_not_RSVP/brainvit_20260928/window_ablation_20260928/prepare.py")
EXPECTED_MVNN59_SOURCE_SHA = "fc1467d1c04b315aa4bf6d4d0f400bcad082971de40c3299a82544632fb08ce7"
HISTORICAL_3508_SPLIT = Path("/mnt/cfs/jianyu/Demo/THINGS_not_RSVP/brainvit_20260928/cache/split.json")
EXPECTED_HISTORICAL_3508_SPLIT_SHA = "c613bcc755b0e84250fbbbbf03e6e8c6d82fca5125ecdd7203f7f2628d49fb30"
SESSION_RE = re.compile(r"^ses-[A-Za-z0-9._-]+$")

PREPROCESSING_CONTRACT = {
    "schema": "shared-preprocessing-contract-v1",
    "input_channels": FULL59,
    "channel_selection": "first_59_scalp_eeg_channels",
    "car": "continuous_mean_across_59_channels",
    "filter": {"kind": "butterworth_bandpass", "order": 4, "low_hz": 0.5,
               "high_hz": 40.0, "representation": "sos", "application": "sosfiltfilt"},
    "resample": {"method": "scipy.signal.resample_poly", "from_hz": 1000,
                 "to_hz": 250, "up": 1, "down": 4},
    "epoch_seconds": [-0.2, 1.2],
    "baseline_seconds": [-0.2, 0.0],
    "crop_seconds": [0.0, 1.2],
    "output": {"shape_tail": [59, 300], "dtype": "float32", "sampling_rate_hz": 250},
    "qc": {"sync_quality": "hardware_verified", "image_offset_required": True,
           "epoch_boundary_required": True, "finite_required": True,
           "max_channel_ptp_uv": 500.0, "global_std_min_exclusive": 1e-4,
           "channel_layout_required": FULL59},
}


def versions() -> dict:
    try:
        import sklearn
        sklearn_version = sklearn.__version__
    except ModuleNotFoundError:
        sklearn_version = "UNAVAILABLE_IN_THIS_RUNTIME"
    data = {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "scikit_learn": sklearn_version}
    print("[RUNTIME VERSIONS]")
    for key, value in data.items():
        print(f"{key}={value}")
    return data


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


PREPROCESSING_CONTRACT_HASH = canonical_hash(PREPROCESSING_CONTRACT)


def truth(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def require_input_in_demo(path: Path) -> Path:
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(ALLOWED_ROOT.resolve(strict=True))
    except ValueError as exc:
        raise RuntimeError(f"forbidden_source_path:{resolved}") from exc
    return resolved


def require_raw_input(path: Path) -> Path:
    """Allow raw reads from Demo or the read-only COS mount; never for writes."""
    resolved = path.resolve(strict=True)
    for allowed in (ALLOWED_ROOT.resolve(strict=True), COS_MOUNT_ROOT.resolve(strict=False)):
        try:
            resolved.relative_to(allowed)
            return resolved
        except ValueError:
            continue
    raise RuntimeError(f"forbidden_raw_source_path:{resolved}")


def require_cos_cache_root(path: Path) -> Path:
    """Validate the configured read root without granting output permissions."""
    resolved = path.resolve(strict=False)
    allowed_roots = (ALLOWED_ROOT.resolve(strict=True), COS_MOUNT_ROOT.resolve(strict=False))
    for allowed in allowed_roots:
        try:
            resolved.relative_to(allowed)
            return resolved
        except ValueError:
            continue
    raise RuntimeError(f"forbidden_cos_read_root:{resolved}")


def require_output_in_demo(path: Path) -> Path:
    absolute = path.absolute()
    resolved = path.resolve(strict=False)
    root = ALLOWED_ROOT.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"REFUSING UNSAFE OVERWRITE path={resolved}") from exc
    forbidden = {root, (root / "THINGS_not_RSVP").resolve(),
                 (root / "experiment").resolve(), (root / "BHA").resolve()}
    if resolved in forbidden or absolute != resolved:
        raise RuntimeError(f"REFUSING UNSAFE OVERWRITE path={resolved}")
    return resolved


def validate_identity(subject: str, session: str) -> None:
    if subject not in SUPPORTED_SUBJECTS:
        reason = "unsupported_channel_layout" if subject == "sub-Jiashuo234" else "unsupported_subject"
        raise RuntimeError(f"{reason}:{subject}")
    if not SESSION_RE.fullmatch(session):
        raise RuntimeError(f"invalid_session_id:{session}")


def session_paths(output_root: Path, branch: str, subject: str, session: str) -> tuple[Path, Path]:
    validate_identity(subject, session)
    if branch not in BRANCHES:
        raise RuntimeError(f"invalid_branch:{branch}")
    root = require_output_in_demo(output_root)
    folder = root / branch / subject
    expected_parent = folder.resolve(strict=False)
    try:
        expected_parent.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"REFUSING UNSAFE OVERWRITE path={expected_parent}") from exc
    npz = folder / f"{session}.npz"
    meta = folder / f"{session}.json"
    for path in (npz, meta):
        if path.resolve(strict=False).parent != expected_parent:
            raise RuntimeError(f"REFUSING UNSAFE OVERWRITE path={path}")
    return npz, meta


def safe_unlink_session_files(output_root: Path, branches: tuple[str, ...], subject: str, session: str) -> None:
    for branch in branches:
        for path in session_paths(output_root, branch, subject, session):
            if path.exists():
                resolved = path.resolve(strict=True)
                if resolved != path.absolute():
                    raise RuntimeError(f"REFUSING UNSAFE OVERWRITE path={resolved}")
                path.unlink()
                print(f"OVERWRITE_SESSION_REMOVED path={path}")


def atomic_json(path: Path, value: dict) -> None:
    require_output_in_demo(path.parent)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def atomic_npz(path: Path, payload: dict) -> None:
    require_output_in_demo(path.parent)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    np.savez(tmp, **payload)
    os.replace(tmp, path)


def selected_manifest_rows(path: Path, subjects: list[str]) -> list[dict]:
    rows = []
    for row in read_csv(path):
        if row.get("subject_id") not in subjects:
            continue
        if "selected_as_canonical" in row and not truth(row["selected_as_canonical"]):
            continue
        rows.append(row)
    keys = [(r["subject_id"], r["session_id"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate_canonical_session")
    for subject, session in keys:
        validate_identity(subject, session)
    return sorted(rows, key=lambda r: (r["subject_id"], r["session_id"]))


def load_image_index(path: Path) -> tuple[dict[str, int], str]:
    require_input_in_demo(path)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        image_ids = data["image_ids"] if isinstance(data, dict) else data
        pairs = [(str(image_id), index) for index, image_id in enumerate(image_ids)]
    else:
        rows = read_csv(path)
        if not rows or "image_id" not in rows[0]:
            raise RuntimeError("image_index_missing_image_id")
        index_key = next((k for k in ("canonical_image_index", "index", "selected_feature_row", "feature_row")
                          if k in rows[0]), None)
        pairs = [(r["image_id"], int(r[index_key]) if index_key else i) for i, r in enumerate(rows)]
    result = {}
    for image_id, index in pairs:
        if not image_id or image_id in result or index < 0:
            raise RuntimeError(f"invalid_or_duplicate_image_index:{image_id}:{index}")
        result[image_id] = index
    if len(set(result.values())) != len(result):
        raise RuntimeError("duplicate_numeric_image_index")
    return result, sha256(path)


def load_split(path: Path) -> tuple[dict, str]:
    require_input_in_demo(path)
    split = json.loads(path.read_text(encoding="utf-8"))
    seen = set()
    for name in ("train", "val", "test"):
        values = [int(x) for x in split[name]]
        if seen.intersection(values):
            raise RuntimeError("split_overlap")
        seen.update(values)
        split[name] = values
    return split, sha256(path)


def derived_context(args) -> dict:
    if args.image_index is None or args.split is None or args.mvnn17 is None:
        raise RuntimeError("derived_requires_image_index_split_mvnn17")
    image_index, image_index_hash = load_image_index(args.image_index)
    split, split_hash = load_split(args.split)
    if sha256(args.mvnn17) != EXPECTED_MVNN17_SHA:
        raise RuntimeError(f"mvnn17_sha_mismatch:{args.mvnn17}")
    if sha256(MVNN59_SOURCE) != EXPECTED_MVNN59_SOURCE_SHA:
        raise RuntimeError("historical_mvnn59_source_sha_mismatch")
    values = set(image_index.values())
    if any(index not in values for part in ("train", "val", "test") for index in split[part]):
        raise RuntimeError("split_index_missing_from_image_index")
    contract = {
        "schema": "derived-protocol-contract-v1", "image_protocol": args.image_protocol,
        "image_index_hash": image_index_hash, "split_hash": split_hash,
        "mvnn17_code_hash": sha256(args.mvnn17), "mvnn59_code_hash": sha256(MVNN59_SOURCE),
        "posterior17_channel_order": POSTERIOR17,
    }
    return {"image_index": image_index, "split": split, **contract,
            "derived_protocol_hash": canonical_hash(contract)}


def load_exact_mvnn17(path: Path):
    spec = importlib.util.spec_from_file_location("historical_mvnn17_exact", path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("mvnn17_import_loader_missing")
    spec.loader.exec_module(module)
    if not hasattr(module, "fit_transform"):
        raise RuntimeError("mvnn17_fit_transform_missing")
    return module


def fit_transform_mvnn59_exact(x: np.ndarray, labels: np.ndarray, train_mask: np.ndarray):
    """Exact historical 59-channel implementation from the pinned source."""
    from sklearn.discriminant_analysis import _cov
    if sha256(MVNN59_SOURCE) != EXPECTED_MVNN59_SOURCE_SHA:
        raise RuntimeError("historical_mvnn59_source_sha_mismatch")
    rows = np.flatnonzero(train_mask)
    if not len(rows) or x.ndim != 3 or x.shape[1:] != (59, 300):
        raise RuntimeError("mvnn59_input_contract_failed")
    sums, counts = {}, {}
    for idx in rows:
        label = int(labels[idx])
        covariance = _cov(x[idx].astype("float64").T, shrinkage="auto")
        sums[label] = sums.get(label, 0) + covariance
        counts[label] = counts.get(label, 0) + 1
    covariance = np.mean([sums[i] / counts[i] for i in sorted(sums)], axis=0)
    vals, vectors = np.linalg.eigh(covariance)
    floor = max(float(vals.max()) * 1e-6, 1e-12)
    whitener = (vectors * (np.maximum(vals, floor) ** -0.5)) @ vectors.T
    whitened = np.einsum("cd,ndt->nct", whitener, x, optimize=True).astype("float32")
    if not np.isfinite(whitened).all():
        raise RuntimeError("mvnn59_nonfinite")
    state = {"whitener": whitener, "covariance": covariance,
             "active": np.ones(59, dtype=bool), "fit_rows": rows}
    audit = {"fit_trials": len(rows), "fit_images": len(sums), "active_channels": 59,
             "eigenvalue_min": float(vals.min()), "eigenvalue_max": float(vals.max()),
             "eigen_floor": floor, "clipped_eigenvalues": int((vals < floor).sum()),
             "validation_and_test_used_for_fit": False,
             "source": "exact extraction of historical prepare.py lines 44-56"}
    return whitened, state, audit


def cos_key(uri: str) -> str:
    prefix = "cos://jianyu-1442740494/"
    if not uri.startswith(prefix):
        raise RuntimeError(f"unexpected_cos_uri:{uri}")
    return uri[len(prefix):]


def resolve_runtime_sources(row: dict, cos_cache_root: Path) -> tuple[Path, Path, Path]:
    if row["source_type"] == "cfs_local":
        raw, side, events = Path(row["eeg_file"]), Path(row["sidecar_file"]), Path(row["events_file"])
    elif row["source_type"] == "cos":
        raw = cos_cache_root / cos_key(row["eeg_file"])
        side, events = Path(row["metadata_mirror_sidecar"]), Path(row["metadata_mirror_events"])
    else:
        raise RuntimeError(f"unsupported_source_type:{row['source_type']}")
    return require_raw_input(raw), require_input_in_demo(side), require_input_in_demo(events)


def preprocess_session(raw_path: Path, sidecar_path: Path, session_rows: list[dict]):
    meta = json.loads(sidecar_path.read_text(encoding="utf-8"))
    shape = tuple(meta["shape"])
    if meta["dtype"] != "<f4" or meta["layout"] != "sample_major":
        raise RuntimeError("raw_contract_mismatch")
    if int(meta["sampling_rate_hz"]) != 1000 or meta["channel_names"][:59] != FULL59:
        raise RuntimeError("channel_or_sampling_mismatch")
    expected_bytes = int(np.prod(shape)) * 4
    if raw_path.stat().st_size != expected_bytes:
        raise RuntimeError(f"raw_size_mismatch:{raw_path}")
    raw = np.memmap(raw_path, mode="r", dtype="<f4", shape=shape)
    eeg = np.asarray(raw[:, :59].T, dtype=np.float64)
    eeg -= eeg.mean(axis=0, keepdims=True)
    sos = butter(4, [0.5, 40], btype="bandpass", fs=1000, output="sos")
    filtered = np.stack([resample_poly(sosfiltfilt(sos, channel), 1, 4).astype("float32")
                         for channel in eeg])
    del eeg
    kept, accepted, rejected = [], [], []
    for row in sorted(session_rows, key=lambda r: int(r["global_source_row"])):
        reason = ""
        if row["sync_quality"] != "hardware_verified":
            reason = "sync_quality_not_hardware_verified"
        elif not truth(row["has_offset"]):
            reason = "image_offset_missing"
        elif not truth(row["epoch_boundary_valid"]):
            reason = "epoch_boundary_invalid"
        onset_zero = int(row["eeg_sample_index"]) - 1
        if not reason and (onset_zero < 0 or onset_zero >= shape[0]):
            reason = "onset_out_of_raw_bounds"
        if not reason and int(round(float(raw[onset_zero, 64]))) != 101:
            reason = "trigger_channel_not_101"
        center = int(np.rint(onset_zero / 4.0))
        lo, hi = center - 50, center + 300
        if not reason and (lo < 0 or hi > filtered.shape[1]):
            reason = "epoch_out_of_bounds"
        if not reason:
            ep = filtered[:, lo:hi].copy()
            ep -= ep[:, :50].mean(axis=1, keepdims=True)
            ep = ep[:, 50:350]
            if ep.shape != (59, 300):
                reason = "unexpected_epoch_shape"
            elif not np.isfinite(ep).all():
                reason = "nonfinite"
            elif float(np.ptp(ep, axis=1).max()) > 500.0:
                reason = "max_channel_ptp_gt_500uv"
            elif float(ep.std()) <= 1e-4:
                reason = "global_std_too_small"
        if reason:
            rejected.append({"subject_id": row["subject_id"], "session_id": row["session_id"],
                             "event_seq": row["event_seq"], "image_id": row["image_id"],
                             "mapping_status": row.get("mapping_status", "unknown"), "reason": reason})
        else:
            kept.append(ep.astype("float32"))
            accepted.append(row)
    if not kept:
        return np.empty((0, 59, 300), dtype=np.float32), accepted, rejected, meta
    return np.stack(kept), accepted, rejected, meta


def raw_identity(row: dict) -> tuple[str, int]:
    raw_source = row["eeg_file"]
    raw_size = int(row.get("expected_eeg_bytes") or row.get("actual_eeg_bytes") or 0)
    return raw_source, raw_size


def npz_contract(path: Path, channels: list[str], require_mapping: bool = False) -> tuple[bool, str, int]:
    if not path.exists():
        return False, "npz_missing", 0
    try:
        with np.load(path, allow_pickle=False) as z:
            required = {"eeg", "labels", "session_rows", "source_rows", "channel_names"}
            if not required.issubset(z.files):
                return False, "npz_keys_missing", 0
            x = z["eeg"]
            n = x.shape[0] if x.ndim == 3 else 0
            if x.dtype != np.float32 or x.shape[1:] != (len(channels), 300):
                return False, "eeg_shape_or_dtype", n
            if not np.isfinite(x).all():
                return False, "eeg_nonfinite", n
            if list(z["channel_names"].astype(str)) != channels:
                return False, "channel_order", n
            for key in ("labels", "session_rows", "source_rows"):
                if len(z[key]) != n:
                    return False, f"length_{key}", n
            if require_mapping and not {"image_ids", "event_seq", "mapping_status"}.issubset(z.files):
                return False, "shared_mapping_fields_missing", n
            return True, "compatible", n
    except Exception as exc:
        return False, f"npz_unreadable:{type(exc).__name__}", 0


def session_mapping_rows(all_rows: list[dict], subject: str, session: str) -> list[dict]:
    return [r for r in all_rows if r["subject_id"] == subject and r["session_id"] == session]


def shared_compatibility(output_root: Path, manifest_row: dict, mapping_rows: list[dict]) -> tuple[bool, str]:
    subject, session = manifest_row["subject_id"], manifest_row["session_id"]
    npz_path, json_path = session_paths(output_root, "full_preprocessed", subject, session)
    ok, reason, n = npz_contract(npz_path, FULL59)
    if not json_path.exists():
        return False, reason if not ok else "json_missing"
    try:
        meta = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"json_unreadable:{type(exc).__name__}"
    if meta.get("subject_id") != subject or meta.get("session_id") != session:
        return False, "identity_mismatch"
    raw_source, raw_size = raw_identity(manifest_row)
    if "preprocessing_contract_hash" in meta:
        identity_checks = (meta.get("raw_source") == raw_source,
                           int(meta.get("raw_size", -1)) == raw_size,
                           meta.get("preprocessing_contract_hash") == PREPROCESSING_CONTRACT_HASH)
        if (not ok and reason == "npz_missing" and all(identity_checks) and
                meta.get("full_preprocessed_ready") is False and
                meta.get("shared_blocker") == "NO_RETAINED_TRIALS"):
            return False, "compatible_blocked_no_retained_trials"
        if not ok:
            return False, reason
        checks = (*identity_checks, meta.get("full_preprocessed_ready") is True)
        return all(checks), "compatible" if all(checks) else "metadata_mismatch"
    if not ok:
        return False, reason
    if any(r.get("mapping_status") != "mapped" for r in mapping_rows):
        return False, "legacy_shared_excluded_unmapped_rows"
    legacy_source_ok = meta.get("source_path_or_uri") == manifest_row.get("source_path_or_uri")
    legacy_count_ok = int(meta.get("retained_trials", -1)) == n
    legacy_channels_ok = meta.get("channels_full") == FULL59
    ok = legacy_source_ok and legacy_count_ok and legacy_channels_ok
    return ok, "legacy_compatible" if ok else "legacy_metadata_mismatch"


def legacy_ready_compatible(output_root: Path, branch: str, context: dict) -> bool:
    ready_path = output_root / "READY.json"
    if not ready_path.exists() or context["image_protocol"] != "historical_3508":
        return False
    try:
        ready = json.loads(ready_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    expected_mvnn = context["mvnn59_code_hash"] if branch == "full_mvnn" else context["mvnn17_code_hash"]
    actual_mvnn = ready.get("mvnn59_source_sha256") if branch == "full_mvnn" else ready.get("mvnn17_sha256")
    return ready.get("split_sha256") == context["split_hash"] and actual_mvnn == expected_mvnn


def derived_compatibility(output_root: Path, branch: str, manifest_row: dict,
                          context: dict) -> tuple[bool, str]:
    subject, session = manifest_row["subject_id"], manifest_row["session_id"]
    npz_path, json_path = session_paths(output_root, branch, subject, session)
    channels = FULL59 if branch == "full_mvnn" else POSTERIOR17
    ok, reason, _ = npz_contract(npz_path, channels)
    if not json_path.exists():
        return False, reason if not ok else "json_missing"
    try:
        meta = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"json_unreadable:{type(exc).__name__}"
    if meta.get("subject_id") != subject or meta.get("session_id") != session:
        return False, "identity_mismatch"
    fields = ("image_protocol", "image_index_hash", "split_hash", "derived_protocol_hash")
    protocol_matches = all(meta.get(key) == context[key] for key in fields)
    expected_mvnn = context["mvnn59_code_hash"] if branch == "full_mvnn" else context["mvnn17_code_hash"]
    if (not ok and reason == "npz_missing" and protocol_matches and
            meta.get("mvnn_code_hash") == expected_mvnn and
            meta.get("derived_blocker") == "NO_TRAIN_ROWS_FOR_CURRENT_PROTOCOL"):
        return False, "compatible_blocked_no_train_rows"
    if not ok:
        return False, reason
    if "derived_protocol_hash" not in meta:
        ok = branch == "full_mvnn" and legacy_ready_compatible(output_root, branch, context)
        return ok, "legacy_compatible" if ok else "legacy_protocol_metadata_missing"
    ok = protocol_matches
    ok = ok and meta.get("mvnn_code_hash") == expected_mvnn and meta.get(f"{branch}_ready") is True
    return ok, "compatible" if ok else "protocol_mismatch"


def shared_payload(full: np.ndarray, accepted: list[dict], image_index: dict[str, int] | None) -> dict:
    labels = np.asarray([image_index.get(r["image_id"], -1) if image_index else
                         int(r["canonical_image_index"]) if r.get("canonical_image_index") else -1
                         for r in accepted], dtype=np.int64)
    status = np.asarray(["mapped" if label >= 0 else "unmapped_current_protocol" for label in labels])
    return {"eeg": full, "labels": labels,
            "session_rows": np.arange(len(accepted), dtype=np.int64),
            "source_rows": np.asarray([int(r["global_source_row"]) for r in accepted], dtype=np.int64),
            "channel_names": np.asarray(FULL59),
            "image_ids": np.asarray([r["image_id"] for r in accepted]),
            "event_seq": np.asarray([int(r["event_seq"]) for r in accepted], dtype=np.int64),
            "mapping_status": status}


def shared_metadata(args, manifest_row: dict, accepted: list[dict], rejected: list[dict],
                    payload: dict, context: dict | None) -> dict:
    raw_source, raw_size = raw_identity(manifest_row)
    status_counts = Counter(payload["mapping_status"].astype(str))
    return {"schema": "incremental-session-shard-v1", "branch": "full_preprocessed",
            "subject_id": manifest_row["subject_id"], "session_id": manifest_row["session_id"],
            "source_type": manifest_row["source_type"], "raw_source": raw_source,
            "raw_size": raw_size, "raw_sha256": None,
            "accepted_trials": len(accepted), "rejected_trials": len(rejected),
            "rejection_reasons": dict(Counter(r["reason"] for r in rejected)),
            "mapping_status_counts": dict(status_counts),
            "preprocessing_code_sha": sha256(Path(__file__)),
            "preprocessing_contract_hash": PREPROCESSING_CONTRACT_HASH,
            "sampling_rate": 250, "channel_order": FULL59, "samples": 300,
            "image_protocol": args.image_protocol,
            "image_index_hash": context["image_index_hash"] if context else None,
            "split_hash": context["split_hash"] if context else None,
            "derived_protocol_hash": context["derived_protocol_hash"] if context else None,
            "mvnn_code_hash": ({"full_mvnn": context["mvnn59_code_hash"],
                                "visual17_mvnn_full": context["mvnn17_code_hash"]}
                               if context else None),
            "full_preprocessed_ready": True, "full_mvnn_ready": False,
            "visual17_mvnn_full_ready": False, "shared_blocker": None,
            "derived_blocker": None,
            "created_utc": datetime.now(timezone.utc).isoformat()}


def load_shared_for_derived(npz_path: Path, mapping_rows: list[dict], context: dict) -> dict:
    with np.load(npz_path, allow_pickle=False) as z:
        data = {key: z[key] for key in z.files}
    if "image_ids" not in data:
        # Legacy shards stored labels in the frozen historical_3508 row space.
        # Their global source_rows are not stable when later manifest snapshots
        # add sessions, so recover stable image identity from the pinned split.
        historical_path = require_input_in_demo(HISTORICAL_3508_SPLIT)
        if sha256(historical_path) != EXPECTED_HISTORICAL_3508_SPLIT_SHA:
            raise RuntimeError("historical_3508_split_sha_mismatch")
        historical = json.loads(historical_path.read_text(encoding="utf-8"))
        historical_ids = historical.get("image_ids", [])
        labels = np.asarray(data.get("labels", []), dtype=np.int64)
        if labels.ndim != 1 or len(labels) != len(data.get("eeg", [])):
            raise RuntimeError("legacy_shared_labels_invalid")
        if len(historical_ids) != 3508 or np.any(labels < 0) or np.any(labels >= len(historical_ids)):
            raise RuntimeError("legacy_shared_label_missing_from_historical_3508")
        image_ids = np.asarray([historical_ids[int(label)] for label in labels])
        missing = sorted({str(image_id) for image_id in image_ids
                          if str(image_id) not in context["image_index"]})
        if missing:
            raise RuntimeError(f"legacy_shared_image_missing_from_current_protocol:{missing[:5]}")
        data["image_ids"] = image_ids
    if "event_seq" not in data:
        # event_seq is provenance-only here; do not guess it from unstable rows.
        data["event_seq"] = np.full(len(data["image_ids"]), -1, dtype=np.int64)
    return data


def derived_payload(x: np.ndarray, labels: np.ndarray, shared: dict, mapped: np.ndarray,
                    channels: list[str], state: dict) -> dict:
    payload = {"eeg": x, "labels": labels,
               "session_rows": np.asarray(shared["session_rows"])[mapped],
               "source_rows": np.asarray(shared["source_rows"])[mapped],
               "channel_names": np.asarray(channels),
               "image_ids": np.asarray(shared["image_ids"])[mapped],
               "event_seq": np.asarray(shared["event_seq"])[mapped],
               "mapping_status": np.asarray(["mapped"] * len(labels))}
    payload.update({"whitener": state["whitener"], "covariance": state["covariance"],
                    "active": state["active"], "fit_rows": state["fit_rows"]})
    return payload


def derived_metadata(branch: str, manifest_row: dict, context: dict, labels: np.ndarray,
                     train_mask: np.ndarray, audit: dict | None, ready: bool,
                     blocker: str | None) -> dict:
    mvnn_hash = context["mvnn59_code_hash"] if branch == "full_mvnn" else context["mvnn17_code_hash"]
    return {"schema": "incremental-session-shard-v1", "branch": branch,
            "subject_id": manifest_row["subject_id"], "session_id": manifest_row["session_id"],
            "raw_source": manifest_row["eeg_file"], "raw_size": raw_identity(manifest_row)[1],
            "preprocessing_code_sha": sha256(Path(__file__)),
            "preprocessing_contract_hash": PREPROCESSING_CONTRACT_HASH,
            "image_protocol": context["image_protocol"],
            "image_index_hash": context["image_index_hash"], "split_hash": context["split_hash"],
            "mvnn_code_hash": mvnn_hash, "derived_protocol_hash": context["derived_protocol_hash"],
            "posterior17_channel_order": POSTERIOR17,
            "sampling_rate": 250,
            "channel_order": FULL59 if branch == "full_mvnn" else POSTERIOR17,
            "samples": 300,
            "mapped_valid_trials": int(len(labels)), "train_rows": int(train_mask.sum()),
            "train_unique_images": int(len(set(labels[train_mask].tolist()))) if train_mask.any() else 0,
            "full_preprocessed_ready": True, "full_mvnn_ready": ready if branch == "full_mvnn" else None,
            "visual17_mvnn_full_ready": ready if branch == "visual17_mvnn_full" else None,
            "derived_blocker": blocker, "mvnn_audit": audit,
            "created_utc": datetime.now(timezone.utc).isoformat()}


def update_shared_derived_status(output_root: Path, row: dict, full_ready: bool,
                                 visual_ready: bool, blocker: str | None,
                                 context: dict) -> None:
    _, path = session_paths(output_root, "full_preprocessed", row["subject_id"], row["session_id"])
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta["full_preprocessed_ready"] = True
    meta["full_mvnn_ready"] = full_ready
    meta["visual17_mvnn_full_ready"] = visual_ready
    meta["derived_blocker"] = blocker
    meta["image_protocol"] = context["image_protocol"]
    meta["image_index_hash"] = context["image_index_hash"]
    meta["split_hash"] = context["split_hash"]
    meta["derived_protocol_hash"] = context["derived_protocol_hash"]
    meta["mvnn_code_hash"] = {"full_mvnn": context["mvnn59_code_hash"],
                              "visual17_mvnn_full": context["mvnn17_code_hash"]}
    atomic_json(path, meta)


def process_shared(args, row: dict, mapping_rows: list[dict], context: dict | None) -> bool:
    raw, side, _ = resolve_runtime_sources(row, args.cos_cache_root)
    full, accepted, rejected, _ = preprocess_session(raw, side, mapping_rows)
    payload = shared_payload(full, accepted, context["image_index"] if context else None)
    npz_path, json_path = session_paths(args.output_root, "full_preprocessed",
                                        row["subject_id"], row["session_id"])
    if not accepted:
        safe_unlink_session_files(args.output_root, BRANCHES, row["subject_id"], row["session_id"])
        meta = shared_metadata(args, row, accepted, rejected, payload, context)
        meta.update(full_preprocessed_ready=False, shared_blocker="NO_RETAINED_TRIALS",
                    derived_blocker="SHARED_NOT_READY", full_mvnn_ready=False,
                    visual17_mvnn_full_ready=False)
        atomic_json(json_path, meta)
        reasons = dict(Counter(item["reason"] for item in rejected))
        print(f"SHARED_BLOCKED subject={row['subject_id']} session={row['session_id']} "
              f"blocker=NO_RETAINED_TRIALS rejection_reasons={json.dumps(reasons, sort_keys=True)}")
        return False
    atomic_npz(npz_path, payload)
    atomic_json(json_path, shared_metadata(args, row, accepted, rejected, payload, context))
    safe_unlink_session_files(args.output_root, ("full_mvnn", "visual17_mvnn_full"),
                              row["subject_id"], row["session_id"])
    print(f"PROCESS_SHARED subject={row['subject_id']} session={row['session_id']} "
          f"accepted={len(accepted)} rejected={len(rejected)}")
    return True


def process_derived(args, row: dict, mapping_rows: list[dict], context: dict,
                    need_full: bool, need_visual: bool) -> None:
    shared_path, _ = session_paths(args.output_root, "full_preprocessed",
                                   row["subject_id"], row["session_id"])
    shared = load_shared_for_derived(shared_path, mapping_rows, context)
    all_labels = np.asarray([context["image_index"].get(str(image_id), -1)
                             for image_id in shared["image_ids"]], dtype=np.int64)
    mapped = all_labels >= 0
    labels = all_labels[mapped]
    train_set = np.asarray(context["split"]["train"], dtype=np.int64)
    train_mask = np.isin(labels, train_set)
    if not train_mask.any():
        blocker = "NO_TRAIN_ROWS_FOR_CURRENT_PROTOCOL"
        for branch in ("full_mvnn", "visual17_mvnn_full"):
            if (branch == "full_mvnn" and need_full) or (branch == "visual17_mvnn_full" and need_visual):
                safe_unlink_session_files(args.output_root, (branch,), row["subject_id"], row["session_id"])
                _, meta_path = session_paths(args.output_root, branch, row["subject_id"], row["session_id"])
                atomic_json(meta_path, derived_metadata(branch, row, context, labels, train_mask,
                                                        None, False, blocker))
        update_shared_derived_status(args.output_root, row, False, False, blocker, context)
        print(f"DERIVED_BLOCKED subject={row['subject_id']} session={row['session_id']} blocker={blocker}")
        return
    x = np.asarray(shared["eeg"])[mapped]
    full_ready = not need_full
    visual_ready = not need_visual
    if need_full:
        transformed, state, audit = fit_transform_mvnn59_exact(x, labels, train_mask)
        payload = derived_payload(transformed, labels, shared, mapped, FULL59, state)
        npz_path, json_path = session_paths(args.output_root, "full_mvnn", row["subject_id"], row["session_id"])
        atomic_npz(npz_path, payload)
        atomic_json(json_path, derived_metadata("full_mvnn", row, context, labels, train_mask,
                                                audit, True, None))
        full_ready = True
        print(f"PROCESS_FULL_MVNN subject={row['subject_id']} session={row['session_id']}")
    if need_visual:
        p17 = [FULL59.index(channel) for channel in POSTERIOR17]
        mvnn17 = load_exact_mvnn17(args.mvnn17)
        transformed, state, audit = mvnn17.fit_transform(x[:, p17, :], labels, train_mask)
        payload = derived_payload(transformed, labels, shared, mapped, POSTERIOR17, state)
        npz_path, json_path = session_paths(args.output_root, "visual17_mvnn_full",
                                            row["subject_id"], row["session_id"])
        atomic_npz(npz_path, payload)
        atomic_json(json_path, derived_metadata("visual17_mvnn_full", row, context, labels,
                                                train_mask, audit, True, None))
        visual_ready = True
        print(f"PROCESS_VISUAL17_MVNN_FULL subject={row['subject_id']} session={row['session_id']}")
    update_shared_derived_status(args.output_root, row, full_ready, visual_ready, None, context)


def overwrite_tokens(values: list[str]) -> set[tuple[str, str]]:
    result = set()
    for value in values:
        parts = value.split("/")
        if len(parts) != 2:
            raise RuntimeError(f"overwrite_session_must_be_subject_slash_session:{value}")
        validate_identity(parts[0], parts[1])
        result.add((parts[0], parts[1]))
    return result


def run(args) -> None:
    bad = [subject for subject in args.subjects if subject not in SUPPORTED_SUBJECTS]
    if bad:
        reason = "unsupported_channel_layout" if "sub-Jiashuo234" in bad else "unsupported_subject"
        raise RuntimeError(f"{reason}:{bad}")
    if len(set(args.subjects)) != len(args.subjects):
        raise RuntimeError("duplicate_subject_argument")
    args.output_root = require_output_in_demo(args.output_root)
    if not args.dry_run:
        args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = selected_manifest_rows(args.session_manifest, args.subjects)
    mapping_all = read_csv(args.mapping_audit)
    context = derived_context(args) if args.stage in {"derived", "all"} else None
    forced = overwrite_tokens(args.overwrite_session)
    manifest_keys = {(row["subject_id"], row["session_id"]) for row in manifest}
    unknown = forced - manifest_keys
    if unknown:
        raise RuntimeError(f"overwrite_session_not_in_manifest:{sorted(unknown)}")
    counters = Counter()
    for row in manifest:
        subject, session = row["subject_id"], row["session_id"]
        key = (subject, session)
        rr = session_mapping_rows(mapping_all, subject, session)
        if not rr:
            raise RuntimeError(f"session_missing_mapping_rows:{subject}:{session}")
        shared_ok, shared_reason = shared_compatibility(args.output_root, row, rr)
        full_ok = visual_ok = False
        full_reason = visual_reason = "not_checked"
        if context:
            full_ok, full_reason = derived_compatibility(args.output_root, "full_mvnn", row, context)
            visual_ok, visual_reason = derived_compatibility(args.output_root, "visual17_mvnn_full", row, context)
        if key in forced:
            branches = BRANCHES if args.stage in {"shared", "all"} else ("full_mvnn", "visual17_mvnn_full")
            if not args.dry_run:
                safe_unlink_session_files(args.output_root, branches, subject, session)
            shared_ok = False if args.stage in {"shared", "all"} else shared_ok
            shared_reason = "forced_overwrite" if args.stage in {"shared", "all"} else shared_reason
            full_ok = visual_ok = False
            full_reason = visual_reason = "forced_overwrite"
        shared_blocked = shared_reason == "compatible_blocked_no_retained_trials"
        full_done = full_ok or full_reason == "compatible_blocked_no_train_rows"
        visual_done = visual_ok or visual_reason == "compatible_blocked_no_train_rows"
        if args.stage == "shared":
            if shared_ok:
                action = "REUSE_SHARED"
            elif shared_blocked:
                action = "SKIP_SHARED_BLOCKED"
            else:
                action = "PROCESS_SHARED"
        elif args.stage == "derived":
            if shared_blocked:
                action = "SKIP_SHARED_BLOCKED"
            elif not shared_ok:
                action = "BLOCKED_SHARED_MISSING_OR_INCOMPATIBLE"
            elif full_done and visual_done:
                action = "SKIP_ALL"
            else:
                action = "REBUILD_DERIVED" if (full_reason == "protocol_mismatch" or visual_reason == "protocol_mismatch") else "PROCESS_DERIVED"
        else:
            if shared_blocked:
                action = "SKIP_SHARED_BLOCKED"
            elif not shared_ok:
                action = "PROCESS_SHARED_AND_DERIVED"
            elif full_done and visual_done:
                action = "SKIP_ALL"
            else:
                action = "REBUILD_DERIVED" if (full_reason == "protocol_mismatch" or visual_reason == "protocol_mismatch") else "PROCESS_DERIVED"
        print(f"SESSION_PLAN subject={subject} session={session} source={row['source_type']} "
              f"full_preprocessed_exists={session_paths(args.output_root, 'full_preprocessed', subject, session)[0].exists()} "
              f"full_preprocessed_compatible={shared_ok} full_preprocessed_reason={shared_reason} "
              f"full_mvnn_exists={session_paths(args.output_root, 'full_mvnn', subject, session)[0].exists()} "
              f"full_mvnn_compatible={full_ok} full_mvnn_reason={full_reason} "
              f"visual17_mvnn_full_exists={session_paths(args.output_root, 'visual17_mvnn_full', subject, session)[0].exists()} "
              f"visual17_mvnn_full_compatible={visual_ok} visual17_reason={visual_reason} planned_action={action}")
        counters[action] += 1
        if args.dry_run:
            continue
        if args.stage == "shared":
            if shared_ok:
                print(f"REUSE_FULL_PREPROCESSED subject={subject} session={session}")
            elif shared_blocked:
                print(f"SKIP_SHARED_BLOCKED_NO_RETAINED_TRIALS subject={subject} session={session}")
            else:
                process_shared(args, row, rr, None)
            continue
        if args.stage == "derived":
            if shared_blocked:
                print(f"SKIP_SHARED_BLOCKED_NO_RETAINED_TRIALS subject={subject} session={session}")
                continue
            if not shared_ok:
                raise RuntimeError(f"derived_requires_compatible_shared:{subject}:{session}:{shared_reason}")
        elif shared_blocked:
            print(f"SKIP_SHARED_BLOCKED_NO_RETAINED_TRIALS subject={subject} session={session}")
            continue
        elif not shared_ok:
            shared_ready = process_shared(args, row, rr, context)
            if not shared_ready:
                continue
            full_ok = visual_ok = False
            full_done = visual_done = False
        else:
            print(f"REUSE_FULL_PREPROCESSED subject={subject} session={session}")
        if full_ok:
            print(f"SKIP_FULL_MVNN subject={subject} session={session}")
        elif full_reason == "compatible_blocked_no_train_rows":
            print(f"SKIP_FULL_MVNN_BLOCKED_NO_TRAIN_ROWS subject={subject} session={session}")
        if visual_ok:
            print(f"SKIP_VISUAL17_MVNN_FULL subject={subject} session={session}")
        elif visual_reason == "compatible_blocked_no_train_rows":
            print(f"SKIP_VISUAL17_MVNN_FULL_BLOCKED_NO_TRAIN_ROWS subject={subject} session={session}")
        if not (full_done and visual_done):
            process_derived(args, row, rr, context, not full_done, not visual_done)
    print("[INCREMENTAL SUMMARY]")
    for key in sorted(counters):
        print(f"{key}={counters[key]}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-manifest", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--split", type=Path)
    parser.add_argument("--image-index", type=Path)
    parser.add_argument("--image-protocol", default="historical_3508")
    parser.add_argument("--vision-bank", type=Path,
                        help="Accepted for command compatibility; preprocessing does not read vision features.")
    parser.add_argument("--mvnn17", type=Path)
    parser.add_argument("--mapping-audit", type=Path)
    parser.add_argument("--cos-cache-root", type=Path,
                        default=COS_MOUNT_ROOT)
    parser.add_argument("--output-root", "--output", dest="output_root", type=Path, required=True)
    parser.add_argument("--stage", choices=("shared", "derived", "all"), default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite-session", action="append", default=[], metavar="SUBJECT/SESSION")
    parser.add_argument("--save-full-preprocessed", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--save-full-mvnn", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--save-posterior17-mvnn", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.mapping_audit = args.mapping_audit or args.session_manifest.parent / "image_mapping_full_audit.csv"
    return args


def main():
    versions()
    args = parse_args()
    for path in (args.session_manifest, args.mapping_audit):
        require_input_in_demo(path)
    if args.stage in {"derived", "all"}:
        for path in (args.split, args.image_index, args.mvnn17, MVNN59_SOURCE):
            if path is None:
                raise RuntimeError("derived_requires_image_index_split_mvnn17")
            require_input_in_demo(path)
    if args.vision_bank is not None:
        require_input_in_demo(args.vision_bank)
    require_cos_cache_root(args.cos_cache_root)
    run(args)


if __name__ == "__main__":
    main()
