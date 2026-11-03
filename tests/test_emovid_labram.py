"""Physical amplitude, label/QC preservation, S3 streaming and model input checks."""

from collections import OrderedDict
from io import BytesIO
import hashlib
import json
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from botocore.exceptions import ClientError
import h5py
import lance
import numpy as np
import torch

from preprocessing.labram import emovid_labram as labram
from preprocessing.common import preprocess_emovid_rules as prep
from preprocessing.common.emovid_pipeline import mapping_signature
from preprocessing.common.emovid_storage import OutputStore
from preprocessing.labram.labram_dataset import S3LaBraMDataset
from preprocessing.common.s3_hdf5 import S3RangeFile


class Body(BytesIO):
    def iter_chunks(self, size):
        while chunk := self.read(size):
            yield chunk


class MemoryS3:
    def __init__(self):
        self.objects, self.metadata = {}, {}
        self.reads, self.uploads = [], []

    def put_object(self, Bucket, Key, Body, Metadata=None, **kwargs):
        self.objects[Bucket, Key] = bytes(Body)
        self.metadata[Bucket, Key] = Metadata or {}

    def head_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise ClientError({'Error': {'Code': '404'}}, 'HeadObject')
        value = self.objects[Bucket, Key]
        return dict(ContentLength=len(value), ETag='"'+hashlib.md5(value).hexdigest()+'"',
                    Metadata=self.metadata[Bucket, Key])

    def get_object(self, Bucket, Key, Range=None, IfMatch=None, **kwargs):
        head = self.head_object(Bucket, Key)
        if IfMatch and IfMatch != head['ETag']:
            raise ClientError({'Error': {'Code': 'PreconditionFailed'}}, 'GetObject')
        value = self.objects[Bucket, Key]
        if Range:
            start, stop = map(int, Range.removeprefix('bytes=').split('-'))
            value = value[start:stop+1]
        self.reads.append(dict(bucket=Bucket, key=Key, range=Range, if_match=IfMatch, size=len(value)))
        return dict(Body=Body(value), **head)

    def upload_file(self, path, bucket, key, ExtraArgs):
        self.put_object(Bucket=bucket, Key=key, Body=Path(path).read_bytes(), **ExtraArgs)
        self.uploads.append((bucket, key))


class Signal:
    sfreq, n_times, duration, expected_size = 200., 12000, 60., 123
    units, dropped_channels = ['uV']*59, []

    def read_samples(self, start, stop):
        time = np.arange(start, stop)/self.sfreq
        amplitude = np.where((time >= 30) & (time < 36), 320., 20.)
        return np.stack([amplitude*np.sin(2*np.pi*10*time+i/59) for i in range(59)])


class LaBraMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='emovid-labram-test-')
        cls.root = Path(cls.temporary.name)
        options = SimpleNamespace(source_unit='uV', unit_status='confirmed',
                                  max_clip_fraction=.01, block_seconds=12., context_seconds=15.)
        cls.config = prep.preprocessing_contract(options)
        common = dict(usable=True, subject_id='00123', session_id=1, raw_run_index=0,
                      events_uri='s3://labels/events.json', start_time_basis='raw_trigger',
                      stop_time_basis='raw_trigger', trial_id=1, attempt_id=1, video_id='video1',
                      rest_type=None, duration_seconds=6.)
        cls.report = dict(raw_uri='s3://raw/recording.bdf', recording=dict(sfreq=200., duration_seconds=60.),
                          segments=[dict(common, segment_type=kind, emotion_label=label, emotion_label_id=label_id,
                              start_seconds=start, stop_seconds=start+6, start_event_index=i)
                              for i, (start, kind, label, label_id) in enumerate([
                                  (10.015, 'emotion', 'positive', 2), (20., 'rest', None, None),
                                  (30., 'emotion', 'negative', 0), (40., 'emotion', 'neutral', 1)])])
        cls.checkpoint = cls.root/'source'
        cls.source_result = prep.preprocess_recording(cls.report, Signal(), cls.config, cls.checkpoint)
        (cls.checkpoint/'report.json').write_text(json.dumps(cls.source_result))
        schema = prep.arrow_schema()
        cls.dataset = lance.write_dataset(prep.arrow_batches([cls.checkpoint], schema), str(cls.root/'source.lance'), schema=schema)
        cls.summary = dict(layout=labram.SOURCE_LAYOUT, status='success', raw_uri=cls.report['raw_uri'],
                           config=cls.config, signature='source-signature', lance_version=cls.dataset.version,
                           n_windows=len(cls.source_result['rows']), mapping_signature=mapping_signature(cls.report),
                           table_name='fixture.lance')
        cls.entry = dict(cls.summary, recording_name='fixture')

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def source(self):
        client = MemoryS3()
        with patch('preprocessing.common.emovid_storage.lance_storage_options', return_value={}):
            source = OutputStore(client, 's3://test/emovid')
            output = OutputStore(client, 's3://test/labram')
        source.put_json('fixture.summary.json', self.summary)
        source.put_json('fixture.segments.json', self.report)
        source.put_json('fixture.validation.json', dict(status='pass'))
        client.put_object(Bucket='raw', Key='recording.bdf', Body=b'0'*123)
        return source, output, client

    def export(self, source, output, reader=None, resume=True):
        with patch.object(labram.lance, 'dataset', return_value=self.dataset), \
             patch.object(labram, 'processing_hash', return_value='numeric-hash'), \
             patch.object(prep, 'SignalReader', return_value=reader or Signal()):
            return labram.export_recording(source, output, self.entry, resume=resume)

    def test_unclipped_uv_roundtrip_channel_order_qc_and_default_model_scale(self):
        source, output, client = self.source()
        result, resumed = self.export(source, output)
        self.assertFalse(resumed)
        self.assertEqual(result['verified_source_windows'], len(self.source_result['rows']))
        key = output.key(result['hdf5_key'])
        with h5py.File(BytesIO(client.objects[output.bucket, key]), 'r') as remote:
            self.assertGreater(abs(remote['data'][:]).max(), 300)
            self.assertLessEqual(abs(np.load(self.checkpoint/'data.npy')).max(), 1)
            self.assertEqual(remote['channel_names'].asstr()[:].tolist(), labram.CHANNEL_NAMES)
            self.assertEqual(remote['input_chans'][:].tolist(), labram.INPUT_CHANS)
            source_order = remote['data'][:][:, np.argsort(labram.ORDER), :]
            np.testing.assert_allclose(np.clip(source_order/200, -1, 1),
                                       np.load(self.checkpoint/'data.npy'), atol=1e-7, rtol=0)
            self.assertEqual(remote['subject_ids'].asstr()[0], '00123')
            self.assertEqual(remote['emotion_labels'][:].tolist(), [2, 2, -1, -1, 0, 0, 1, 1])
            stored_rows = [json.loads(value) for value in remote['rows_json'].asstr()[:]]
            self.assertEqual(stored_rows, self.source_result['rows'])
        with S3LaBraMDataset(result['hdf5_uri'], client=client) as dataset:
            expected = sum(r['qc_pass'] and r['segment_type'] == 'emotion' for r in stored_rows)
            self.assertEqual(len(dataset), expected)
            self.assertEqual(dataset[0]['eeg'].shape, (59, 3, 200))
            self.assertEqual(dataset[0]['label'].dtype, torch.long)
            np.testing.assert_array_equal(dataset[0]['eeg'].numpy().reshape(59, 600), dataset.read_window(0)['eeg_uv']/100)
            self.assertTrue(all(r['qc_pass'] and r['segment_type'] == 'emotion' for r in dataset.metadata()))
            self.assertEqual(dataset[-1]['sample_id'], dataset[len(dataset)-1]['sample_id'])
            with self.assertRaises(IndexError):
                dataset[len(dataset)]
        reads = [r for r in client.reads if r['key'] == key and r['range']]
        self.assertTrue(reads)
        self.assertTrue(all(r['if_match'] == result['etag'] for r in reads))

    def test_all_windows_and_rest_are_available_for_s3_analysis(self):
        source, output, client = self.source()
        result, _ = self.export(source, output)
        with S3LaBraMDataset(result['hdf5_uri'], qc_only=False, segment_type=None, client=client) as dataset:
            self.assertEqual(len(dataset), 8)
            self.assertEqual(dataset[2]['label'].item(), 3)
            self.assertEqual(dataset[2]['emotion_label'].item(), -1)
            self.assertGreater(dataset[4]['eeg'].abs().max().item(), 3.)
            self.assertFalse(dataset[4]['qc_pass'])
        with S3LaBraMDataset(result['hdf5_uri'], segment_type='rest', client=client) as dataset:
            self.assertTrue(all(r['segment_type'] == 'rest' for r in dataset.metadata()))

    def test_resume_checks_s3_identity_and_does_not_reconstruct(self):
        source, output, client = self.source()
        result, _ = self.export(source, output)
        class NeverRead(Signal):
            def read_samples(self, *args):
                raise AssertionError('Unexpected raw waveform read on resume')
        again, resumed = self.export(source, output, reader=NeverRead())
        self.assertTrue(resumed)
        self.assertEqual(result, again)
        self.assertEqual(len(client.uploads), 1)
        client.objects[output.bucket, output.key(result['hdf5_key'])] += b'changed'
        with self.assertRaisesRegex(ValueError, 'changed or is incomplete'):
            self.export(source, output)

    def test_source_disagreement_does_not_publish_a_ready_file(self):
        source, output, _ = self.source()
        class WrongSignal(Signal):
            def read_samples(self, start, stop):
                time = np.arange(start, stop)/self.sfreq
                return super().read_samples(start, stop)+5*np.sin(2*np.pi*13*time)
        with self.assertRaisesRegex(ValueError, 'differs from source Lance'):
            self.export(source, output, reader=WrongSignal())
        self.assertFalse(output.exists('fixture.summary.json'))
        self.assertEqual(output.client.uploads, [])

    def test_environment_and_unconfirmed_units_are_rejected(self):
        for key, value in [('window_seconds', 10), ('normalization', 'zscore'),
                           ('mne_version', 'another-version'), ('source_unit_status', 'assumed')]:
            config = dict(self.config, **{key: value})
            with self.assertRaises(ValueError):
                labram.validate_source_config(config)

    def test_manifest_loading_batch_collation_and_pickle(self):
        source, output, client = self.source()
        result, _ = self.export(source, output)
        output.put_json('manifest.json', dict(layout=labram.LAYOUT, status='pass', recordings=[result]))
        dataset = S3LaBraMDataset(output.uri('manifest.json'), client=client)
        try:
            batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=2)))
            self.assertEqual(batch['eeg'].shape, (2, 59, 3, 200))
            self.assertTrue(dataset._files)
            restored = pickle.loads(pickle.dumps(dataset))
            self.assertIsNone(restored._client)
            self.assertEqual(restored._files, OrderedDict())
            self.assertEqual(len(restored), len(dataset))
        finally:
            dataset.close()
        output.put_json('manifest.json', dict(layout=labram.LAYOUT, status='fail', recordings=[result]))
        with self.assertRaisesRegex(ValueError, 'completely validated'):
            S3LaBraMDataset(output.uri('manifest.json'), client=client)

    def test_changed_hdf5_is_rejected_after_metadata_was_indexed(self):
        source, output, client = self.source()
        result, _ = self.export(source, output)
        dataset = S3LaBraMDataset(result['hdf5_uri'], client=client)
        client.objects[output.bucket, output.key(result['hdf5_key'])] += b'changed'
        try:
            with self.assertRaisesRegex(ValueError, 'changed since'):
                dataset[0]
        finally:
            dataset.close()

    def test_multi_recording_index_and_file_cache(self):
        source, output, client = self.source()
        result, _ = self.export(source, output)
        # Use the same verified content under two index entries to exercise
        # concatenation and file-handle eviction independently of waveform math.
        second = dict(result, recording_name='second')
        output.put_json('manifest.json', dict(layout=labram.LAYOUT, status='pass', recordings=[result, second]))
        with S3LaBraMDataset(output.uri('manifest.json'), client=client, max_open_files=1) as dataset:
            boundary = len(dataset)//2
            self.assertEqual(dataset[boundary-1]['recording_name'], 'fixture')
            self.assertEqual(dataset[boundary]['recording_name'], 'second')
            self.assertEqual(dataset[-1]['recording_name'], 'second')
            self.assertEqual(len(dataset._files), 1)
            self.assertEqual(dataset[0]['recording_name'], 'fixture')
            self.assertEqual(len(dataset._files), 1)

    def test_rest_class_cannot_be_silently_loaded_as_an_emotion_target(self):
        source, output, client = self.source()
        result, _ = self.export(source, output)
        key = output.key(result['hdf5_key'])
        buffer = BytesIO(client.objects[output.bucket, key])
        with h5py.File(buffer, 'r+') as remote:
            remote['labels'][0] = 3
            remote['emotion_labels'][0] = 3
        client.put_object(Bucket=output.bucket, Key=key, Body=buffer.getvalue())
        with self.assertRaisesRegex(ValueError, 'emotion/rest labels disagree'):
            S3LaBraMDataset(result['hdf5_uri'], client=client)


class RangeTests(unittest.TestCase):
    def test_seek_cross_block_reads_and_bounded_cache(self):
        client = MemoryS3()
        value = bytes(range(256))*100
        client.put_object(Bucket='bucket', Key='file', Body=value)
        with S3RangeFile(client, 's3://bucket/file', block_size=100, cache_blocks=2) as stream:
            stream.seek(90)
            self.assertEqual(stream.read(25), value[90:115])
            stream.seek(-15, 2)
            self.assertEqual(stream.read(), value[-15:])
            self.assertEqual(stream.read(10), b'')
            stream.seek(0)
            buffer = bytearray(250)
            self.assertEqual(stream.readinto(buffer), 250)
            self.assertEqual(buffer, value[:250])
            self.assertLessEqual(len(stream.cache), 2)
            with self.assertRaises(ValueError):
                stream.seek(-1)
        self.assertTrue(all(r['range'] and r['if_match'] for r in client.reads))
        self.assertLessEqual(max(r['size'] for r in client.reads), 100)

    def test_mid_read_mutation_and_truncated_ranges_are_rejected(self):
        client = MemoryS3()
        client.put_object(Bucket='bucket', Key='file', Body=b'abcdefghij')
        with S3RangeFile(client, 's3://bucket/file', block_size=5) as stream:
            self.assertEqual(stream.read(5), b'abcde')
            client.objects['bucket', 'file'] = b'ABCDEFGHIJ'
            with self.assertRaises(ClientError):
                stream.read(5)
        head = client.head_object(Bucket='bucket', Key='file')
        store = labram.PinnedRawStore(client, 's3://bucket/file', head)
        self.assertEqual(store.read('s3://bucket/file', (1, 4)), b'BCDE')
        with self.assertRaisesRegex(OSError, 'Truncated'):
            store.read('s3://bucket/file', (1, 30))


if __name__ == '__main__':
    unittest.main()
