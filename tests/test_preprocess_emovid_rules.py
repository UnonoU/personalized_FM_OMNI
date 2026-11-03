"""Numerical checks for physical units, filter halos, complete windows and Lance metadata."""

import argparse
import json
from pathlib import Path
import tempfile
import unittest

import mne
import numpy as np

import preprocess_emovid_rules as prep
from emovid_segments import Store
from test_preprocess_lwh_labram import make_bdf


class PreprocessingTests(unittest.TestCase):
    def test_bdf_range_reader_matches_mne_microvolts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"test.bdf"
            make_bdf(path, "uV")
            reader = prep.SignalReader(Store(), str(path), channels=("C3", "FP1"))
            measured = reader.read_samples(1234, 15321)
            raw = mne.io.read_raw_bdf(str(path), preload=False, verbose=False)
            try:
                expected = raw.get_data(picks=["C3", "Fp1"], start=1234, stop=15321, units="uV")
            finally:
                raw.close()
            np.testing.assert_allclose(measured, expected, atol=1e-10, rtol=1e-12)
            self.assertEqual(reader.expected_size, path.stat().st_size)
            self.assertEqual(reader.channel_names, ("C3", "FP1"))

    def test_unknown_units_require_declaration_and_do_not_get_scaled_twice(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"test.bdf"
            make_bdf(path, "uV/mm")
            with self.assertRaisesRegex(ValueError, "Nonstandard unit"):
                prep.SignalReader(Store(), str(path), channels=("C3", "FP1"))
            reader = prep.SignalReader(Store(), str(path), source_unit="uV", channels=("C3", "FP1"))
            data = reader.read_samples(0, 1000)
            self.assertTrue(90 < data[0].mean() < 110)
            self.assertTrue(190 < data[1].mean() < 210)
            other = prep.SignalReader(Store(), str(path), source_unit="mV", channels=("C3", "FP1"))
            np.testing.assert_allclose(other.read_samples(0, 1000), data*1000)

    def test_edf_signed_sample_extremes_and_calibration(self):
        def field(value, width):
            return str(value).encode().ljust(width, b" ")
        fixed = b"0       " + field("test", 80) + field("test", 80)
        fixed += field("23.10.26", 8)+field("10.25.30", 8)+field(512, 8)+field("EDF+C", 44)
        fixed += field(1, 8)+field(1, 8)+field(1, 4)
        header = b"".join(field(v, w) for v, w in [("C3", 16), ("EEG", 80), ("uV", 8),
                            (-1000, 8), (1000, 8), (-32768, 8), (32767, 8), ("", 80), (1000, 8), ("", 32)])
        digital = np.resize(np.array([-32768, -1, 0, 32767], dtype="<i2"), 1000)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"test.edf"
            path.write_bytes(fixed+header+digital.tobytes())
            reader = prep.SignalReader(Store(), str(path), channels=("C3",))
            measured = reader.read_samples(0, 4)[0]
        np.testing.assert_allclose(measured, [-1000, -1000+32767*2000/65535,
                                              -1000+32768*2000/65535, 1000], atol=1e-10)

    def test_filter_context_matches_continuous_filter_and_preserves_grid(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(2, 100000))
        full = prep.filter_and_resample(x, 1000, 200)
        block = prep.filter_and_resample(x[:, 15000:65000], 1000, 200)
        np.testing.assert_allclose(full[:, 6000:10000], block[:, 3000:7000], atol=1e-10, rtol=1e-10)

    def test_passband_line_noise_and_high_frequency_rejection(self):
        t = np.arange(100000)/1000
        signal = np.sin(2*np.pi*10*t)+np.sin(2*np.pi*50*t)+np.sin(2*np.pi*110*t)
        output = prep.filter_and_resample(signal[None], 1000, 200)[0, 4000:16000]
        time = np.arange(4000, 16000)/200
        def amplitude(frequency):
            return abs(2*np.mean(output*np.exp(-2j*np.pi*frequency*time)))
        self.assertTrue(0.95 < amplitude(10) < 1.05)
        self.assertLess(amplitude(50), .02)
        self.assertLess(amplitude(90), .02)

    def test_window_plan_does_not_cross_emotion_boundaries(self):
        segments = [dict(usable=True, start_seconds=10.003, stop_seconds=35.103, duration_seconds=25.1),
                    dict(usable=True, start_seconds=50, stop_seconds=64.9, duration_seconds=14.9),
                    dict(usable=False, start_seconds=100, stop_seconds=120, duration_seconds=20)]
        windows, discarded = prep.window_plan(segments)
        self.assertEqual(len(windows), 12)
        self.assertEqual(windows[0]["start"], 2001)
        self.assertTrue(all(w["stop"]-w["start"] == 600 for w in windows))
        self.assertEqual(discarded, 800)
        self.assertTrue(all(w["stop"] <= round(segments[w["segment_index"]]["stop_seconds"]*200) for w in windows))

    def test_short_tails_and_adjacent_same_label_segments_are_not_combined(self):
        segments = [dict(usable=True, start_seconds=0, stop_seconds=5, duration_seconds=5),
                    dict(usable=True, start_seconds=5, stop_seconds=10, duration_seconds=5),
                    dict(usable=True, start_seconds=12, stop_seconds=14.99, duration_seconds=2.99)]
        windows, discarded = prep.window_plan(segments)
        self.assertEqual([(w["start"], w["stop"]) for w in windows], [(0, 600), (1000, 1600)])
        self.assertEqual(discarded, 1398)

    def test_normalization_never_pads_and_rejects_incomplete_windows(self):
        x = np.vstack([np.tile([-400, 400.], 300), np.tile([-20, 20.], 300)])
        data, qc = prep.encode_window(x)
        self.assertFalse(qc["qc_pass"])
        self.assertEqual(qc["clip_fraction"], .5)
        self.assertEqual(data.dtype, np.float32)
        np.testing.assert_array_equal(data[0, :4], [-1, 1, -1, 1])
        self.assertEqual(data.shape, (2, 600))
        with self.assertRaisesRegex(ValueError, "exactly 600"):
            prep.encode_window(x[:, :599])
        _, qc = prep.encode_window(np.vstack([np.zeros(600), np.tile([-1, 1.], 300)]))
        self.assertIn("flat_channel", qc["qc_reasons"])

    def test_toml_declares_every_arrow_column_and_nullable_rest_labels(self):
        import tomli
        schema = prep.arrow_schema()
        args = argparse.Namespace(source_unit="uV", unit_status="assumed", max_clip_fraction=.01,
                                  block_seconds=120, context_seconds=15)
        summary = dict(n_windows=10, n_subjects=1, created_at_utc="2026-11-02T00:00:00+00:00", lance_version=1)
        with tempfile.TemporaryDirectory() as directory:
            prep.write_metadata(Path(directory), schema, summary, prep.preprocessing_contract(args))
            metadata = tomli.loads((Path(directory)/"metadata.toml").read_text())
        self.assertEqual(metadata["profile"], "base")
        self.assertEqual({c["name"] for c in metadata["schema"]["columns"]}, set(schema.names))
        self.assertEqual(len(metadata["eeg"]["channels"]), 59)
        self.assertEqual(metadata["preprocessing"]["source_unit_status"], "assumed")
        self.assertFalse(next(c for c in metadata["schema"]["columns"] if c["name"]=="label")["nullable"])
        self.assertTrue(next(c for c in metadata["schema"]["columns"] if c["name"]=="emotion_label")["nullable"])
        self.assertEqual(metadata["label"]["num_classes"], 4)
        self.assertFalse(metadata["data"]["padding"])
        self.assertEqual(metadata["preprocessing"]["segment_samples"], 600)

    def test_real_lance_roundtrip_keeps_three_class_target_and_rest_state(self):
        import lance
        import pyarrow as pa
        from validate_emovid_rules import validate
        args = argparse.Namespace(source_unit="uV", unit_status="confirmed", max_clip_fraction=.01,
                                  block_seconds=120, context_seconds=15)
        config = prep.preprocessing_contract(args)
        class Reader:
            sfreq, n_times = 200, 6000
            units, dropped_channels = ["uV"]*59, ["ECG"]
            def read_samples(self, start, stop):
                time = np.arange(start, stop)/self.sfreq
                return np.stack([10*np.sin(2*np.pi*10*time+i/59) for i in range(59)])
        common = dict(usable=True, duration_seconds=5, subject_id="123", session_id=1, raw_run_index=0,
                      events_uri="s3://bucket/20261023_102630/events.json", start_time_basis="raw_trigger",
                      stop_time_basis="raw_trigger", trial_id=1, attempt_id=1, video_id="emotion:a")
        report = {"raw_uri": "s3://bucket/20261023102530_test/test.bdf", "segments": [
            dict(common, segment_type="emotion", rest_type=None, emotion_label_id=2, emotion_label="positive",
                 start_seconds=5, stop_seconds=10, start_event_index=1),
            dict(common, segment_type="rest", rest_type="long_rest", emotion_label_id=None, emotion_label=None,
                 start_seconds=15, stop_seconds=20, start_event_index=5)]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root/"checkpoint"
            result = prep.preprocess_recording(report, Reader(), config, folder)
            (folder/"report.json").write_text(json.dumps(result))
            schema = prep.arrow_schema()
            reader = pa.RecordBatchReader.from_batches(schema, prep.arrow_batches([folder], schema))
            dataset = lance.write_dataset(reader, root/"dataset.lance")
            summary = dict(n_windows=2, n_subjects=1, created_at_utc="2026-11-02T00:00:00+00:00", lance_version=dataset.version,
                           processed_recording_count=1, selected_recording_count=1, recordings=[dict(status="success")],
                           qc_pass_count=2, padded_window_count=0, emotion_counts={"positive": 1})
            prep.write_metadata(root, schema, summary, config)
            (root/"summary.json").write_text(json.dumps(summary))
            validation = validate(root)
            self.assertEqual(validation["status"], "pass")
            rows = dataset.to_table(columns=["label", "emotion_label", "segment_type"]).to_pylist()
            self.assertEqual([row["label"] for row in rows], [2, 3])
            self.assertEqual([row["emotion_label"] for row in rows], [2, None])


if __name__ == "__main__":
    unittest.main()
