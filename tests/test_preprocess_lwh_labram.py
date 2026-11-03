"""Numerical regression tests using small, known 24-bit BDF recordings."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import h5py
import mne
import numpy as np

from preprocessing.labram.preprocess_lwh_labram import LABRAM_CHANNELS, POSITION, preprocess


def make_bdf(path, unit):
    rate, records = 1000, 60
    labels = ["C3", "ECG", "Fp1", "HEOR", "BDF Annotations"]
    transducers = ["EEG", "HECG", "EEG", "EOG", ""]
    count = len(labels)

    def field(value, width):
        value = str(value).encode("ascii")
        assert len(value) <= width
        return value.ljust(width, b" ")

    header = b"\xffBIOSEMI" + field("test", 80) + field("test", 80)
    header += field("22.10.26", 8) + field("16.13.19", 8)
    header += field((count + 1) * 256, 8) + field("BDF+C", 44)
    header += field(records, 8) + field(1, 8) + field(count, 4)
    specifications = [
        (labels, 16), (transducers, 80), ([unit] * 4 + [""], 8),
        ([-1000] * 4 + [-1], 8), ([1000] * 4 + [1], 8),
        ([-8388608] * count, 8), ([8388607] * count, 8),
        ([""] * count, 80), ([rate] * 4 + [100], 8), ([""] * count, 32),
    ]
    for values, width in specifications:
        header += b"".join(field(value, width) for value in values)
    times = np.arange(rate * records) / rate
    c3 = 100 + 20 * np.sin(2 * np.pi * 10 * times)
    c3 += 8 * np.sin(2 * np.pi * 50 * times) + 5 * np.sin(2 * np.pi * 120 * times)
    fp1 = 200 + 30 * np.sin(2 * np.pi * 6 * times)
    physical = np.stack([c3, np.zeros_like(times), fp1, np.zeros_like(times)])
    digits = np.rint((physical + 1000) * (16777215 / 2000) - 8388608).astype(np.int32)
    digits = digits.astype("<i4").view(np.uint8).reshape(4, records, rate, 4)
    events = {8: (8.5, "250"), 12: (12.347, "132"), 45: (45.125, "133")}
    with path.open("wb") as output:
        output.write(header)
        for record in range(records):
            output.write(digits[:, record, :, :3].tobytes())
            tal = f"+{record}\x14\x14\x00".encode()
            if record in events:
                onset, description = events[record]
                tal += f"+{onset}\x150\x14{description}\x14\x00".encode()
            output.write(tal.ljust(300, b"\x00"))


class PreprocessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="labram-preprocess-test-")
        cls.root = Path(cls.temporary.name)
        cls.nonstandard = cls.root / "nonstandard.bdf"
        cls.standard = cls.root / "standard.bdf"
        make_bdf(cls.nonstandard, "uV/mm")
        make_bdf(cls.standard, "uV")
        cls.output = cls.root / "processed.hdf5"
        with contextlib.redirect_stdout(io.StringIO()):
            cls.metadata = preprocess(cls.nonstandard, cls.output, source_unit="uV")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_unknown_units_require_explicit_declaration(self):
        output = self.root / "should-not-exist.hdf5"
        with self.assertRaisesRegex(ValueError, "Nonstandard unit"):
            preprocess(self.nonstandard, output)
        self.assertFalse(output.exists())

    def test_units_match_official_mne_pipeline(self):
        raw = mne.io.read_raw_bdf(str(self.standard), preload=True, stim_channel=None,
                                  exclude=["ECG", "HEOR", "BDF Annotations"], verbose="error")
        raw.reorder_channels(["Fp1", "C3"])
        raw.filter(0.1, 75, n_jobs=1, verbose="error")
        raw.notch_filter(50, n_jobs=1, verbose="error")
        raw.resample(200, n_jobs=1, verbose="error")
        expected = raw.get_data(units="uV").astype(np.float32)
        with h5py.File(self.output, "r") as output:
            np.testing.assert_allclose(output["lwh/eeg"][:], expected, rtol=1e-6, atol=1e-5)
        raw.close()

    def test_filtering_and_resampling(self):
        with h5py.File(self.output, "r") as output:
            signal = output["lwh/eeg"][1, 20 * 200:40 * 200]
        frequencies = np.fft.rfftfreq(len(signal), 1 / 200)
        spectrum = np.abs(np.fft.rfft(signal)) * 2 / len(signal)
        amplitude = lambda freq: spectrum[np.argmin(abs(frequencies - freq))]
        self.assertGreater(amplitude(10), 18)
        self.assertLess(amplitude(50), 0.1)
        self.assertLess(amplitude(80), 0.1)  # 120 Hz would alias to 80 without rejection.

    def test_hdf5_layout_positions_and_events(self):
        with h5py.File(self.output, "r") as output:
            self.assertEqual(list(output), ["lwh"])
            self.assertEqual(output.attrs["processing_status"], "complete")
            eeg = output["lwh/eeg"]
            self.assertEqual(eeg.shape, (2, 12000))
            self.assertEqual(eeg.attrs["unit"], "uV")
            self.assertEqual(eeg.attrs["chOrder"].tolist(), ["FP1", "C3"])
            self.assertEqual(output["lwh/input_chans"][:].tolist(), [0, POSITION["FP1"] + 1, POSITION["C3"] + 1])
            self.assertEqual(output["lwh/events/sample_output"][:].tolist(), [1700, 2469, 9025])
            self.assertEqual(output["lwh/events/description"].asstr()[:].tolist(), ["250", "132", "133"])
            self.assertNotIn("labels", output["lwh"])
        self.assertEqual(self.metadata["event_count_in_output"], 3)
        self.assertEqual(LABRAM_CHANNELS[POSITION["FP1"]], "FP1")

    def test_crop_keeps_original_times_and_marks_outside_events(self):
        path = self.root / "cropped.hdf5"
        with contextlib.redirect_stdout(io.StringIO()):
            preprocess(self.nonstandard, path, source_unit="uV", start_seconds=10, duration_seconds=40)
        with h5py.File(path, "r") as output:
            self.assertEqual(output["lwh/eeg"].shape, (2, 8000))
            self.assertEqual(output["lwh/events/in_output"][:].tolist(), [False, True, True])
            self.assertEqual(output["lwh/events/sample_output"][:].tolist(), [-300, 469, 7025])
            np.testing.assert_allclose(output["lwh/events/onset_original_seconds"][:], [8.5, 12.347, 45.125])

    def test_does_not_overwrite_output(self):
        before = self.output.stat().st_size
        with self.assertRaises(FileExistsError):
            preprocess(self.nonstandard, self.output, source_unit="uV")
        self.assertEqual(self.output.stat().st_size, before)


if __name__ == "__main__":
    unittest.main()
