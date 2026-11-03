"""Lazy S3 reader for EMOVID LaBraM HDF5 windows, with no disk waveform cache."""

from bisect import bisect_right
from collections import OrderedDict
import json
import operator
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from preprocessing.labram.labram_contract import AMPLITUDE_DIVISOR, CHANNEL_NAMES, INPUT_CHANS, LAYOUT, validate_hdf5
from preprocessing.common.emovid_segments import split_s3
from preprocessing.common.read_rustfs import client as s3_client
from preprocessing.common.s3_hdf5 import S3RangeFile, open_s3_hdf5


class S3LaBraMDataset(Dataset):
    """Default: QC-passing emotion windows; ``eeg`` is [59,3,200] uV/100.

    ``read_window`` returns unscaled physical uV for plots/PSD. ``metadata``
    exposes source video/session IDs for grouped train/test splitting. Use
    ``dataset.input_chans`` when calling the model; do not divide ``eeg`` again.
    """

    def __init__(self, uri='s3://personalized-fm/processed_labram/manifest.json', *,
                 profile='rustfs', qc_only=True, segment_type='emotion', client=None,
                 max_open_files=2):
        if segment_type not in ('emotion', 'rest', None) or max_open_files < 1:
            raise ValueError('Expected emotion/rest/None and a positive open-file limit')
        split_s3(uri)
        self.uri, self.profile = uri, profile
        self.qc_only, self.segment_type = qc_only, segment_type
        self.channel_names, self.input_chans = list(CHANNEL_NAMES), list(INPUT_CHANS)
        self.num_patches, self.patch_size, self.sampling_rate = 3, 200, 200
        self.max_open_files = max_open_files
        self._client = client
        self._pid = os.getpid()
        self._files = OrderedDict()
        self._records, self._ends = [], []
        if uri.endswith('.json'):
            bucket, key = split_s3(uri)
            response = self.client.get_object(Bucket=bucket, Key=key)
            try:
                manifest = json.loads(response['Body'].read())
            finally:
                response['Body'].close()
            if manifest.get('layout') != LAYOUT or manifest.get('status') != 'pass':
                raise ValueError('Expected a completely validated EMOVID LaBraM manifest')
            entries = manifest['recordings']
            if any(r['status'] != 'success' or r.get('validation_status') != 'pass' for r in entries):
                raise ValueError('LaBraM manifest contains unvalidated recordings')
        else:
            entries = [dict(hdf5_uri=uri)]
        for entry in entries:
            bucket, key = split_s3(entry['hdf5_uri'])
            head = self.client.head_object(Bucket=bucket, Key=key)
            if entry.get('etag') is not None and head['ETag'] != entry['etag']:
                raise ValueError('LaBraM file changed while building its index')
            with open_s3_hdf5(self.client, entry['hdf5_uri'], etag=head['ETag']) as remote:
                validate_hdf5(remote, n_windows=entry.get('n_windows'), signature=entry.get('signature'))
                labels = remote['labels'][:]
                emotion = remote['emotion_labels'][:]
                kinds = remote['segment_type'].asstr()[:]
                if (not np.isin(labels, [0, 1, 2, 3]).all() or not np.isin(kinds, ['emotion', 'rest']).all()
                        or not np.isin(emotion[kinds == 'emotion'], [0, 1, 2]).all()
                        or not np.array_equal(labels[kinds == 'emotion'], emotion[kinds == 'emotion'])
                        or not np.all(labels[kinds == 'rest'] == 3) or not np.all(emotion[kinds == 'rest'] == -1)):
                    raise ValueError('LaBraM emotion/rest labels disagree')
                mask = remote['qc_pass'][:].astype(bool) if qc_only else np.ones(len(labels), dtype=bool)
                if segment_type is not None:
                    mask &= kinds == segment_type
                indices = np.flatnonzero(mask)
                metadata = json.loads(remote.attrs['metadata_json'])
            if len(indices):
                self._records.append(dict(uri=entry['hdf5_uri'], etag=head['ETag'], indices=indices,
                    signature=metadata['signature'], recording_name=entry.get('recording_name', key.split('/')[-1])))
                self._ends.append((self._ends[-1] if self._ends else 0)+len(indices))

    @property
    def client(self):
        if self._pid != os.getpid():
            self.close()
            self._client, self._pid = None, os.getpid()
        if self._client is None:
            self._client = s3_client(self.profile)
        return self._client

    def __len__(self):
        return self._ends[-1] if self._ends else 0

    def _locate(self, index):
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        record_index = bisect_right(self._ends, index)
        start = self._ends[record_index-1] if record_index else 0
        record = self._records[record_index]
        return record_index, record, int(record['indices'][index-start])

    def _open(self, index):
        client = self.client  # also clears inherited clients/HDF5 handles after fork
        if index not in self._files:
            record = self._records[index]
            stream = S3RangeFile(client, record['uri'], etag=record['etag'])
            try:
                remote = h5py.File(stream, 'r')
                try:
                    validate_hdf5(remote, signature=record['signature'])
                except Exception:
                    remote.close()
                    raise
            except Exception:
                stream.close()
                raise
            self._files[index] = (remote, stream)
            if len(self._files) > self.max_open_files:
                _, (old_file, old_stream) = self._files.popitem(last=False)
                old_file.close()
                old_stream.close()
        self._files.move_to_end(index)
        return self._files[index][0]

    def read_window(self, index):
        """Read one [59,600] float32 uV window and its original EMOVID metadata."""
        file_index, record, row = self._locate(index)
        remote = self._open(file_index)
        values = remote['data'][row]
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite LaBraM waveform')
        metadata = json.loads(remote['rows_json'].asstr()[row])
        if (metadata['sample_id'] != remote['sample_ids'].asstr()[row]
                or metadata['label'] != int(remote['labels'][row])):
            raise ValueError('LaBraM row metadata is inconsistent')
        return dict(eeg_uv=values, metadata=metadata, recording_name=record['recording_name'],
                    hdf5_uri=record['uri'], channel_names=self.channel_names)

    def __getitem__(self, index):
        window = self.read_window(index)
        row = window['metadata']
        eeg = (window['eeg_uv']/np.float32(AMPLITUDE_DIVISOR)).reshape(len(self.channel_names), 3, 200)
        return dict(eeg=torch.from_numpy(eeg), label=torch.tensor(row['label'], dtype=torch.long),
                    emotion_label=torch.tensor(-1 if row['emotion_label'] is None else row['emotion_label'], dtype=torch.long),
                    subject_id=row['subject_id'], session_id=row['session_id'], sample_id=row['sample_id'],
                    video_id=row['video_id'] or '', events_uri=row['events_uri'], run_id=row['run_id'],
                    start_time=row['start_time'], split=row['split'], qc_pass=bool(row['qc_pass']),
                    recording_name=window['recording_name'])

    def metadata(self):
        """Read selected row metadata only, without loading waveform chunks."""
        rows = []
        for index, record in enumerate(self._records):
            remote = self._open(index)
            for text in remote['rows_json'].asstr()[record['indices']]:
                rows.append(dict(json.loads(text), recording_name=record['recording_name'], hdf5_uri=record['uri']))
        return rows

    def close(self):
        for remote, stream in self._files.values():
            remote.close()
            stream.close()
        self._files.clear()

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(_client=None, _files=OrderedDict(), _pid=None)
        return state

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
