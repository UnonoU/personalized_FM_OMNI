"""Canonical deployment API for the formal per-session 59-channel MVNN."""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import sklearn
from sklearn.discriminant_analysis import _cov

from .contracts import (
    FULL59,
    MVNN_ALGORITHM_VERSION,
    MVNN_SCHEMA_VERSION,
    PREPROCESSING_CONTRACT_HASH,
    canonical_json_hash,
    require_channel_order,
    require_mvnn_eeg,
)

SklearnCompatibility = Literal["strict", "warn", "ignore"]


@dataclass
class MVNNState:
    whitener: np.ndarray
    covariance: np.ndarray
    fit_rows: np.ndarray
    channel_names: tuple[str, ...] = FULL59
    sfreq: float = 250.0
    eeg_unit: str | None = None
    fit_trial_count: int = 0
    fit_unique_image_count: int = 0
    subject_id: str | None = None
    session_id: str | None = None
    algorithm_version: str = MVNN_ALGORITHM_VERSION
    sklearn_version: str = sklearn.__version__
    preprocessing_contract_hash: str = PREPROCESSING_CONTRACT_HASH
    created_from: dict[str, Any] | None = None
    state_sha256: str | None = None


def select_calibration_rows(
    *,
    event_qc: Sequence[bool],
    left_splits: Sequence[str],
    right_splits: Sequence[str],
) -> np.ndarray:
    """Exact dual-image evaluator selection: QC and both candidates train."""
    qc = np.asarray(event_qc, dtype=bool)
    left = np.asarray(left_splits).astype(str)
    right = np.asarray(right_splits).astype(str)
    if qc.ndim != 1 or left.shape != qc.shape or right.shape != qc.shape:
        raise ValueError("calibration selector arrays must be equally sized 1-D arrays")
    return qc & (left == "train") & (right == "train")


def _label(value: Any) -> int | str:
    value = value.item() if isinstance(value, np.generic) else value
    if isinstance(value, (int, np.integer)):
        return int(value)
    text = str(value)
    if text.lstrip("-").isdigit():
        return int(text)
    return text


def fit_mvnn(
    eeg: np.ndarray,
    target_image_ids: Sequence[Any],
    *,
    fit_mask: Sequence[bool] | None = None,
    channel_names: Sequence[str] = FULL59,
    sfreq: float = 250.0,
    eeg_unit: str | None = None,
    subject_id: str | None = None,
    session_id: str | None = None,
    created_from: dict[str, Any] | None = None,
    warn_below_trials: int | None = None,
    warn_below_images: int | None = None,
) -> MVNNState:
    """Fit the exact image-balanced covariance whitener used by the evaluator."""
    x, _ = require_mvnn_eeg(np.asarray(eeg))
    require_channel_order(tuple(channel_names))
    if float(sfreq) != 250.0:
        raise ValueError(f"formal MVNN expects preprocessed 250 Hz EEG, got {sfreq}")
    labels = np.asarray(target_image_ids)
    if labels.ndim != 1 or len(labels) != len(x):
        raise ValueError(f"target_image_ids must be [N] for N={len(x)}, got {labels.shape}")
    selected = np.ones(len(x), dtype=bool) if fit_mask is None else np.asarray(fit_mask, dtype=bool)
    if selected.ndim != 1 or selected.shape != (len(x),):
        raise ValueError(f"fit_mask must have shape {(len(x),)}, got {selected.shape}")
    rows = np.flatnonzero(selected)
    if not len(rows):
        raise ValueError("MVNN requires at least one selected calibration trial")
    sums: dict[int | str, np.ndarray] = {}
    counts: dict[int | str, int] = {}
    for index in rows:
        label = _label(labels[index])
        covariance = _cov(x[index].astype("float64").T, shrinkage="auto")
        sums[label] = sums.get(label, 0) + covariance
        counts[label] = counts.get(label, 0) + 1
    ordered_labels = sorted(sums, key=lambda value: (str(type(value)), value))
    covariance = np.mean([sums[label] / counts[label] for label in ordered_labels], axis=0)
    values, vectors = np.linalg.eigh(covariance)
    floor = max(float(values.max()) * 1e-6, 1e-12)
    whitener = (vectors * (np.maximum(values, floor) ** -0.5)) @ vectors.T
    if whitener.shape != (59, 59) or not np.isfinite(whitener).all():
        raise RuntimeError("MVNN produced an invalid whitener")
    if warn_below_trials is not None and len(rows) < int(warn_below_trials):
        warnings.warn(
            f"MVNN calibration has {len(rows)} trials, below configured warning threshold "
            f"{warn_below_trials}; this is not a formal equivalence claim",
            RuntimeWarning,
            stacklevel=2,
        )
    if warn_below_images is not None and len(sums) < int(warn_below_images):
        warnings.warn(
            f"MVNN calibration has {len(sums)} unique images, below configured warning "
            f"threshold {warn_below_images}; this is not a formal equivalence claim",
            RuntimeWarning,
            stacklevel=2,
        )
    state = MVNNState(
        whitener=np.asarray(whitener, dtype=np.float64),
        covariance=np.asarray(covariance, dtype=np.float64),
        fit_rows=rows.astype(np.int64),
        channel_names=tuple(str(value) for value in channel_names),
        sfreq=float(sfreq),
        eeg_unit=None if eeg_unit is None else str(eeg_unit),
        fit_trial_count=int(len(rows)),
        fit_unique_image_count=int(len(sums)),
        subject_id=None if subject_id is None else str(subject_id),
        session_id=None if session_id is None else str(session_id),
        created_from=created_from,
    )
    return replace(state, state_sha256=compute_state_sha256(state))


def apply_mvnn(eeg: np.ndarray, state: MVNNState) -> np.ndarray:
    x, single = require_mvnn_eeg(np.asarray(eeg))
    validate_state(state, verify_hash=True, sklearn_compatibility="ignore")
    output = np.einsum("cd,ndt->nct", state.whitener, x, optimize=True).astype("float32")
    if not np.isfinite(output).all():
        raise RuntimeError("MVNN output contains non-finite values")
    return output[0] if single else output


def _metadata(state: MVNNState, *, include_hash: bool) -> dict[str, Any]:
    result = {
        "schema_version": MVNN_SCHEMA_VERSION,
        "subject_id": state.subject_id,
        "session_id": state.session_id,
        "channel_names": list(state.channel_names),
        "sfreq": state.sfreq,
        "eeg_unit": state.eeg_unit,
        "fit_trial_count": state.fit_trial_count,
        "fit_unique_image_count": state.fit_unique_image_count,
        "sklearn_version": state.sklearn_version,
        "algorithm_version": state.algorithm_version,
        "preprocessing_contract_hash": state.preprocessing_contract_hash,
        "created_from": state.created_from,
    }
    if include_hash:
        result["state_sha256"] = state.state_sha256
    return result


def compute_state_sha256(state: MVNNState) -> str:
    result = hashlib.sha256()
    for name, value, dtype in (
        ("whitener", state.whitener, "<f8"),
        ("covariance", state.covariance, "<f8"),
        ("fit_rows", state.fit_rows, "<i8"),
    ):
        array = np.asarray(value, dtype=dtype)
        result.update(name.encode("ascii"))
        result.update(str(tuple(array.shape)).encode("ascii"))
        result.update(array.tobytes(order="C"))
    result.update(canonical_json_hash(_metadata(state, include_hash=False)).encode("ascii"))
    return result.hexdigest()


def _version_pair(value: str) -> tuple[int, int] | None:
    try:
        fields = value.split(".")
        return int(fields[0]), int(fields[1])
    except (ValueError, IndexError):
        return None


def validate_state(
    state: MVNNState,
    *,
    verify_hash: bool = True,
    sklearn_compatibility: SklearnCompatibility = "strict",
) -> None:
    if state.whitener.shape != (59, 59) or state.covariance.shape != (59, 59):
        raise ValueError("MVNN whitener and covariance must both be [59,59]")
    if not np.isfinite(state.whitener).all() or not np.isfinite(state.covariance).all():
        raise ValueError("MVNN state contains non-finite values")
    require_channel_order(state.channel_names)
    if state.algorithm_version != MVNN_ALGORITHM_VERSION:
        raise ValueError(f"unsupported MVNN algorithm version: {state.algorithm_version}")
    if state.preprocessing_contract_hash != PREPROCESSING_CONTRACT_HASH:
        raise ValueError("MVNN preprocessing contract hash mismatch")
    if state.fit_trial_count != len(state.fit_rows) or state.fit_trial_count < 1:
        raise ValueError("MVNN fit trial provenance is inconsistent")
    if state.fit_unique_image_count < 1:
        raise ValueError("MVNN must contain at least one calibration image")
    if float(state.sfreq) != 250.0:
        raise ValueError(f"MVNN state must describe 250 Hz EEG, got {state.sfreq}")
    actual_hash = compute_state_sha256(state)
    if verify_hash and state.state_sha256 != actual_hash:
        raise ValueError(
            f"MVNN state hash mismatch expected={state.state_sha256} actual={actual_hash}"
        )
    saved_pair = _version_pair(state.sklearn_version)
    runtime_pair = _version_pair(sklearn.__version__)
    if sklearn_compatibility != "ignore" and saved_pair != runtime_pair:
        message = (
            f"MVNN was fitted with scikit-learn {state.sklearn_version}, runtime has "
            f"{sklearn.__version__}; private _cov compatibility is not guaranteed"
        )
        if sklearn_compatibility == "strict":
            raise RuntimeError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)


def save_mvnn_state(state: MVNNState, npz_path: str | Path, json_path: str | Path) -> MVNNState:
    state = replace(state, state_sha256=compute_state_sha256(state))
    validate_state(state, verify_hash=True, sklearn_compatibility="ignore")
    npz_path = Path(npz_path)
    json_path = Path(json_path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    npz_tmp = npz_path.with_name(f".{npz_path.stem}.{os.getpid()}.tmp.npz")
    json_tmp = json_path.with_name(f".{json_path.name}.{os.getpid()}.tmp")
    np.savez(
        npz_tmp,
        whitener=np.asarray(state.whitener, dtype=np.float64),
        covariance=np.asarray(state.covariance, dtype=np.float64),
        fit_rows=np.asarray(state.fit_rows, dtype=np.int64),
        channel_names=np.asarray(state.channel_names),
    )
    json_tmp.write_text(
        json.dumps(_metadata(state, include_hash=True), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(npz_tmp, npz_path)
    os.replace(json_tmp, json_path)
    return state


def load_mvnn_state(
    npz_path: str | Path,
    json_path: str | Path,
    *,
    sklearn_compatibility: SklearnCompatibility = "strict",
) -> MVNNState:
    npz_path = Path(npz_path)
    json_path = Path(json_path)
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != MVNN_SCHEMA_VERSION:
        raise ValueError(f"unsupported MVNN schema: {metadata.get('schema_version')!r}")
    with np.load(npz_path, allow_pickle=False) as arrays:
        required = {"whitener", "covariance", "fit_rows", "channel_names"}
        if set(arrays.files) != required:
            raise ValueError(f"MVNN numeric asset keys mismatch: {sorted(arrays.files)}")
        numeric_channels = tuple(str(value) for value in arrays["channel_names"].tolist())
        json_channels = tuple(str(value) for value in metadata["channel_names"])
        if numeric_channels != json_channels:
            raise ValueError("MVNN channel metadata differs between npz and json")
        state = MVNNState(
            whitener=np.asarray(arrays["whitener"], dtype=np.float64),
            covariance=np.asarray(arrays["covariance"], dtype=np.float64),
            fit_rows=np.asarray(arrays["fit_rows"], dtype=np.int64),
            channel_names=json_channels,
            sfreq=float(metadata["sfreq"]),
            eeg_unit=metadata.get("eeg_unit"),
            fit_trial_count=int(metadata["fit_trial_count"]),
            fit_unique_image_count=int(metadata["fit_unique_image_count"]),
            subject_id=metadata.get("subject_id"),
            session_id=metadata.get("session_id"),
            algorithm_version=str(metadata["algorithm_version"]),
            sklearn_version=str(metadata["sklearn_version"]),
            preprocessing_contract_hash=str(metadata["preprocessing_contract_hash"]),
            created_from=metadata.get("created_from"),
            state_sha256=metadata.get("state_sha256"),
        )
    validate_state(
        state, verify_hash=True, sklearn_compatibility=sklearn_compatibility
    )
    return state
