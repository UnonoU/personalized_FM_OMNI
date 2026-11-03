"""Derive unclipped LaBraM windows from an existing S3 EMOVID dataset.

Reuse EMOVID's exact mapping, 0.3-75 Hz filters, 200 Hz grid, 3 s windows and QC.
Re-read raw S3 EEG to retain physical microvolts before EMOVID's lossy clipping;
require every re-encoded window to match the pinned source Lance table exactly.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import inspect
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace

import h5py
import lance
import mne
import numpy as np

from preprocessing.common import preprocess_emovid_rules as prep
from preprocessing.common.emovid_pipeline import LAYOUT as SOURCE_LAYOUT, mapping_signature, processing_hash
from preprocessing.common.emovid_segments import Store, split_s3
from preprocessing.common.emovid_storage import DEFAULT_OUTPUT_URI, OutputStore
from preprocessing.labram.labram_contract import (LAYOUT, DEFAULT_LABRAM_URI, PATCH_SAMPLES, AMPLITUDE_DIVISOR,
                                           ORDER, CHANNEL_NAMES, INPUT_CHANS, validate_hdf5)
from preprocessing.common.s3_hdf5 import open_s3_hdf5

def validate_source_config(config):
    """Refuse silent recipe/version changes when reconstructing physical EEG."""
    options = SimpleNamespace(source_unit=config['source_unit_declaration'],
                              unit_status=config['source_unit_status'],
                              max_clip_fraction=config['max_clip_fraction'],
                              block_seconds=config['block_seconds'], context_seconds=config['context_seconds'])
    expected = prep.preprocessing_contract(options)
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f'Unsupported EMOVID configuration/environment: {key}')
    if config['source_unit_status'] != 'confirmed':
        raise ValueError('Physical source units must be confirmed before LaBraM conversion')


class PinnedRawStore:
    """Keep every raw BDF/EDF Range read on the same S3 snapshot."""

    def __init__(self, client, uri, head):
        self.client, self.uri, self.head = client, uri, head
        self.bucket, self.key = split_s3(uri)

    def read(self, uri, byte_range=None):
        if uri != self.uri or byte_range is None:
            raise ValueError('Only bounded reads of the pinned raw recording are supported')
        start, stop = byte_range
        kwargs = dict(Bucket=self.bucket, Key=self.key, Range=f'bytes={start}-{stop}', IfMatch=self.head['ETag'])
        if self.head.get('VersionId'):
            kwargs['VersionId'] = self.head['VersionId']
        response = self.client.get_object(**kwargs)
        try:
            data = response['Body'].read()
        finally:
            response['Body'].close()
        if len(data) != stop-start+1:
            raise OSError('Truncated raw recording Range response')
        return data


def iter_uv_blocks(report, reader, config):
    """Reproduce EMOVID block boundaries and filter halos before normalization."""
    windows, _ = prep.window_plan(report['segments'])
    grouped = {}
    for index, window in enumerate(windows):
        group = int(window['start']//round(config['block_seconds']*200))
        grouped.setdefault(group, []).append((index, window))
    bandpass = mne.filter.create_filter(None, reader.sfreq, .3, 75., verbose=False)
    notch = mne.filter.create_filter(None, reader.sfreq, [50.625], [49.375],
                                   l_trans_bandwidth=.5, h_trans_bandwidth=.5, verbose=False)
    required = ((len(bandpass)-1)+(len(notch)-1))/(2*reader.sfreq)+10/200
    if config['context_seconds'] < required:
        raise ValueError(f'Insufficient filtering context: need at least {required:.3f}s')
    fraction = Fraction(200/reader.sfreq).limit_denominator(10000)
    for items in grouped.values():
        first = min(w['start'] for _, w in items)/200
        last = max(w['stop'] for _, w in items)/200
        start = max(0, math.floor((first-config['context_seconds'])*reader.sfreq))
        start = start//fraction.denominator*fraction.denominator
        stop = min(reader.n_times, math.ceil((last+config['context_seconds'])*reader.sfreq))
        raw = reader.read_samples(start, stop)
        filtered = prep.filter_and_resample(raw, reader.sfreq, 200)
        del raw
        offset = round(start*200/reader.sfreq)
        values = []
        for _, window in items:
            a, b = window['start']-offset, window['stop']-offset
            if not 0 <= a < b <= filtered.shape[1]:
                raise ValueError('Filtered window outside loaded context')
            values.append(filtered[:, a:b])
        yield items, np.stack(values)


def ordered_rows(dataset, report):
    rows = dataset.to_table(columns=[name for name in dataset.schema.names if name != 'data']).to_pylist()
    indexed = {}
    for index, row in enumerate(rows):
        key = (row['output_start_sample'], row['output_stop_sample'])
        if key in indexed:
            raise ValueError('Duplicate EMOVID window boundaries')
        indexed[key] = (index, row)
    windows, _ = prep.window_plan(report['segments'])
    if len(windows) != len(rows) or len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('EMOVID window count or sample identities disagree with mapping')
    ordered, indices = [], []
    for window in windows:
        index, row = indexed[(window['start'], window['stop'])]
        segment = report['segments'][window['segment_index']]
        expected_id = 'emovid#'+prep.canonical_hash([report['raw_uri'], segment['events_uri'],
                        segment['start_event_index'], window['start'], window['stop']])[:24]
        expected_label = segment['emotion_label_id'] if segment['segment_type'] == 'emotion' else 3
        if (row['sample_id'] != expected_id or row['label'] != expected_label
                or row['source_path'] != report['raw_uri'] or row['channel_names'] != list(prep.CHANNELS)
                or row['segment_type'] != segment['segment_type']
                or row['subject_id'] != str(segment['subject_id'])):
            raise ValueError('EMOVID metadata/labels disagree with source mapping')
        ordered.append(row)
        indices.append(index)
    return ordered, indices


def write_hdf5(path, dataset, report, config, reader, provenance):
    """Keep all windows and QC; store uV, scale exactly once in the reader."""
    rows, source_indices = ordered_rows(dataset, report)
    if not rows:
        raise ValueError('No complete EMOVID windows')
    strings = h5py.string_dtype('utf-8')
    with h5py.File(path, 'x') as output:
        output.attrs.update(layout=LAYOUT, processing_status='writing', unit='uV', sampling_rate_hz=200.,
                            window_seconds=3., patch_samples=PATCH_SAMPLES,
                            amplitude_divisor_at_model_input=AMPLITUDE_DIVISOR,
                            metadata_json=json.dumps(provenance, ensure_ascii=False, allow_nan=False))
        output.create_dataset('channel_names', data=np.array(CHANNEL_NAMES, dtype=object), dtype=strings)
        output.create_dataset('input_chans', data=np.array(INPUT_CHANS, dtype=np.int64))
        signal = output.create_dataset('data', (len(rows), len(ORDER), prep.WINDOW_SAMPLES),
                                       dtype='float32', chunks=(1, len(ORDER), prep.WINDOW_SAMPLES),
                                       compression='gzip', compression_opts=1, shuffle=True, fletcher32=True)
        output.create_dataset('labels', data=np.array([r['label'] for r in rows], dtype=np.int64))
        output.create_dataset('emotion_labels', data=np.array([
            -1 if r['emotion_label'] is None else r['emotion_label'] for r in rows], dtype=np.int64))
        output.create_dataset('qc_pass', data=np.array([r['qc_pass'] for r in rows], dtype=bool))
        for name, field in [('sample_ids', 'sample_id'), ('subject_ids', 'subject_id'),
                            ('session_ids', 'session_id'), ('segment_type', 'segment_type')]:
            output.create_dataset(name, data=np.array([r[field] for r in rows], dtype=object), dtype=strings)
        output.create_dataset('rows_json', data=np.array([
            json.dumps(r, ensure_ascii=False, allow_nan=False) for r in rows], dtype=object), dtype=strings)
        verified = 0
        for items, values in iter_uv_blocks(report, reader, config):
            indices = [index for index, _ in items]
            source = dataset.take([source_indices[i] for i in indices], columns=['data'])
            normalized = source.column('data').combine_chunks().values.to_numpy().reshape(
                len(indices), len(prep.CHANNELS), prep.WINDOW_SAMPLES)
            for offset, index in enumerate(indices):
                encoded, qc = prep.encode_window(values[offset], max_clip_fraction=config['max_clip_fraction'])
                if not np.array_equal(encoded, normalized[offset]):
                    raise ValueError(f'Raw reconstruction differs from source Lance: {rows[index]["sample_id"]}')
                for key, value in qc.items():
                    if rows[index][key] != value:
                        raise ValueError(f'Reconstructed QC differs from source Lance: {key}')
            signal[indices] = values[:, ORDER, :].astype(np.float32)
            verified += len(indices)
        if verified != len(rows):
            raise ValueError('Incomplete LaBraM waveform reconstruction')
        output.attrs['processing_status'] = 'complete'
    return dict(n_windows=len(rows), qc_pass_count=sum(r['qc_pass'] for r in rows),
                window_types=dict(Counter(r['segment_type'] for r in rows)),
                emotion_counts=dict(Counter(r['label_name'] for r in rows if r['segment_type'] == 'emotion')),
                qc_pass_emotion_count=sum(r['qc_pass'] and r['segment_type'] == 'emotion' for r in rows),
                verified_source_windows=verified)


def export_recording(source, output, entry, *, resume=True, work_dir=None):
    name = entry['recording_name']
    if Path(name).name != name or name in ('', '.', '..'):
        raise ValueError('Unsafe recording name')
    summary = source.read_json(name+'.summary.json')
    report = source.read_json(name+'.segments.json')
    validation = source.read_json(name+'.validation.json')
    if (summary.get('layout') != SOURCE_LAYOUT or summary.get('status') != 'success'
            or validation.get('status') != 'pass' or summary['raw_uri'] != entry['raw_uri']
            or entry.get('signature') != summary.get('signature')
            or entry.get('lance_version') != summary['lance_version']
            or mapping_signature(report) != summary['mapping_signature']):
        raise ValueError('Source EMOVID manifest, summary, mapping or validation disagree')
    config = summary['config']
    validate_source_config(config)
    raw_uri = summary['raw_uri']
    bucket, key = split_s3(raw_uri)
    head = source.client.head_object(Bucket=bucket, Key=key)
    raw_snapshot = {key: head.get(key) for key in ('ETag', 'ContentLength', 'VersionId')}
    signature = prep.canonical_hash(dict(layout=LAYOUT, source_signature=summary['signature'],
        source_uri=source.root, lance_version=summary['lance_version'], mapping=summary['mapping_signature'],
        raw_snapshot=raw_snapshot, source_processing_hash=processing_hash(),
        conversion_code=[inspect.getsource(fn) for fn in (validate_source_config, iter_uv_blocks, ordered_rows, write_hdf5)],
        channel_names=CHANNEL_NAMES, input_chans=INPUT_CHANS, amplitude_divisor=AMPLITUDE_DIVISOR))
    summary_key = name+'.summary.json'
    if output.exists(summary_key):
        previous = output.read_json(summary_key)
        if previous.get('layout') != LAYOUT or previous.get('source_uri') != source.root or previous.get('raw_uri') != raw_uri:
            raise ValueError('LaBraM destination belongs to another source')
        if resume and previous.get('signature') == signature:
            file_head = output.client.head_object(Bucket=output.bucket, Key=output.key(previous['hdf5_key']))
            if (file_head.get('Metadata', {}).get('sha256') == previous['sha256']
                    and file_head['ContentLength'] == previous['size_bytes'] and file_head['ETag'] == previous['etag']):
                with open_s3_hdf5(output.client, previous['hdf5_uri'], etag=previous['etag']) as remote:
                    validate_hdf5(remote, n_windows=previous['n_windows'], signature=signature)
                return previous, True
            raise ValueError('Previously validated LaBraM object changed or is incomplete')
    dataset = lance.dataset(source.uri(summary['table_name']), version=summary['lance_version'], storage_options=source.options)
    if dataset.count_rows() != summary['n_windows']:
        raise ValueError('Pinned source Lance row count mismatch')
    raw_store = PinnedRawStore(source.client, raw_uri, head)
    reader = prep.SignalReader(raw_store, raw_uri, source_unit=config['source_unit_declaration'])
    if (reader.expected_size != head['ContentLength'] or reader.sfreq != report['recording']['sfreq']
            or reader.duration != report['recording']['duration_seconds']):
        raise ValueError('Raw recording snapshot/header mismatch')
    provenance = dict(layout=LAYOUT, signature=signature, created_at_utc=datetime.now(timezone.utc).isoformat(),
        source_uri=source.root, raw_uri=raw_uri, raw_snapshot=raw_snapshot,
        source_lance_uri=source.uri(summary['table_name']), source_lance_version=summary['lance_version'],
        source_signature=summary['signature'], source_mapping_signature=summary['mapping_signature'],
        emovid_config=config, source_channel_names=list(prep.CHANNELS), channel_permutation=ORDER,
        normalization='stored_uV; model_input=x_uV/100; no_clipping; no_zscore',
        model_input_shape=[len(CHANNEL_NAMES), 3, PATCH_SAMPLES],
        label_ids=prep.PRIMARY_LABEL_IDS, split='unspecified', numeric_data_origin='raw_s3_reconstruction')
    if work_dir is not None:
        Path(work_dir).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='emovid-labram-', dir=work_dir) as temporary:
        path = Path(temporary)/'windows.hdf5'
        result = write_hdf5(path, dataset, report, config, reader, provenance)
        checksum = prep.digest(path)
        hdf5_key = name+'.'+checksum[:16]+'.labram.hdf5'
        output.client.upload_file(str(path), output.bucket, output.key(hdf5_key), ExtraArgs={
            'ContentType': 'application/x-hdf5', 'Metadata': {'sha256': checksum, 'signature': signature}})
        remote_head = output.client.head_object(Bucket=output.bucket, Key=output.key(hdf5_key))
        response = output.client.get_object(Bucket=output.bucket, Key=output.key(hdf5_key), IfMatch=remote_head['ETag'])
        digest = hashlib.sha256()
        try:
            for chunk in response['Body'].iter_chunks(1024**2):
                digest.update(chunk)
        finally:
            response['Body'].close()
        if digest.hexdigest() != checksum or remote_head['ContentLength'] != path.stat().st_size:
            raise ValueError('S3 LaBraM upload checksum/size mismatch')
        with open_s3_hdf5(output.client, output.uri(hdf5_key), etag=remote_head['ETag']) as remote:
            validate_hdf5(remote, n_windows=result['n_windows'], signature=signature)
        result.update(provenance, status='success', recording_name=name, hdf5_key=hdf5_key,
                      hdf5_uri=output.uri(hdf5_key), sha256=checksum, size_bytes=remote_head['ContentLength'],
                      etag=remote_head['ETag'], validation_status='pass')
        output.put_json(name+'.validation.json', dict(status='pass', signature=signature, sha256=checksum,
                        verified_source_windows=result['verified_source_windows'], s3_sha256_verified=True))
        output.put_json(summary_key, result)
        return result, False


def convert_dataset(*, source_uri=DEFAULT_OUTPUT_URI, output_uri=DEFAULT_LABRAM_URI,
                    profile='rustfs', recording=None, resume=True, work_dir=None):
    source = OutputStore(Store(profile).s3, source_uri)
    output = OutputStore(source.client, output_uri)
    if source.root == output.root:
        raise ValueError('LaBraM output must use a separate prefix from EMOVID')
    manifest = source.read_json('manifest.json')
    if manifest.get('layout') != SOURCE_LAYOUT:
        raise ValueError('Expected the existing per-acquisition EMOVID S3 manifest')
    entries = [r for r in manifest['recordings'] if r['status'] == 'success']
    if recording:
        entries = [r for r in entries if r['recording_name'] == recording]
    if not entries:
        raise ValueError('No successful EMOVID recordings selected')
    previous = output.read_json('manifest.json') if output.exists('manifest.json') else None
    if previous and (previous.get('layout') != LAYOUT or previous.get('source_uri') != source.root):
        raise ValueError('Output manifest belongs to another dataset')
    results = {r['recording_name']: r for r in previous['recordings']} if previous and recording else {}
    for number, entry in enumerate(entries, 1):
        name = entry['recording_name']
        try:
            result, cached = export_recording(source, output, entry, resume=resume, work_dir=work_dir)
            results[name] = result
            print(f'[{number}/{len(entries)}] {"cached" if cached else "saved"} {name}: {result["n_windows"]} windows', flush=True)
        except Exception as exc:
            results[name] = dict(recording_name=name, raw_uri=entry['raw_uri'], status='error',
                                 error_type=type(exc).__name__, reason=str(exc))
            print(f'[{number}/{len(entries)}] ERROR {name}: {type(exc).__name__}: {exc}', flush=True)
        output.put_json('progress.json', dict(layout=LAYOUT, source_uri=source.root, recordings=list(results.values())))
    successful = [r for r in results.values() if r['status'] == 'success']
    aggregate = dict(layout=LAYOUT, source_uri=source.root, output_uri=output.root,
        updated_at_utc=datetime.now(timezone.utc).isoformat(), recordings=sorted(results.values(), key=lambda r: r['recording_name']),
        processed_recording_count=len(successful), n_windows=sum(r['n_windows'] for r in successful),
        qc_pass_count=sum(r['qc_pass_count'] for r in successful),
        qc_pass_emotion_count=sum(r['qc_pass_emotion_count'] for r in successful),
        status_counts=dict(Counter(r['status'] for r in results.values())))
    aggregate['status'] = 'pass' if successful and not aggregate['status_counts'].get('error') else 'fail'
    output.put_json('manifest.json', aggregate)
    output.put_json('summary.json', aggregate)
    return aggregate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-uri', default=DEFAULT_OUTPUT_URI)
    parser.add_argument('--output-uri', default=DEFAULT_LABRAM_URI)
    parser.add_argument('--profile', default='rustfs')
    parser.add_argument('--recording', help='Optional acquisition directory name')
    parser.add_argument('--no-resume', dest='resume', action='store_false', default=True)
    parser.add_argument('--work-dir', type=Path, help='Temporary HDF5 staging; automatically cleaned')
    args = parser.parse_args(argv)
    result = convert_dataset(**vars(args))
    print(json.dumps({key: result[key] for key in ['status', 'processed_recording_count', 'n_windows', 'qc_pass_emotion_count']}, ensure_ascii=False))
    if result['status'] != 'pass':
        parser.exit(1, 'Some LaBraM conversions failed; inspect the output manifest and retry.\n')


if __name__ == '__main__':
    main()
