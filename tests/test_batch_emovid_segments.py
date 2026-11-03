"""Batch regressions: duplicate exports, multiple sessions and resumable failures."""

import argparse
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from preprocessing.common import batch_emovid_segments as batch
from test_emovid_segments import event


ROOT = "s3://oi-data/EMOVID_RAW/"
LABEL_ROOT = "s3://oi-data/EMOVID-261029/"


def obj(uri, size=100, etag="abc"):
    return {"uri": uri, "size": size, "etag": etag}


def recording(directory="20261023102530_test/test"):
    return {"raw_uri": ROOT + directory + "/test.bdf", "aliases": [],
            "signal_object": obj(ROOT + directory + "/test.bdf"),
            "annotation_object": obj(ROOT + directory + "/evt.bdf", 50, "evt")}


def config():
    return {"search_padding_seconds": 600, "timezone": "Asia/Shanghai", "wall_tolerance_seconds": 30,
            "max_residual_seconds": 0.5, "min_segment_seconds": 5, "output_sfreq": 200}


class BatchTests(unittest.TestCase):
    def test_export_components_and_duplicate_directories(self):
        original = ROOT + "20261023102530_test/export/test/"
        duplicate = ROOT + "20261023102530_test/export(1)/test/"
        objects = [obj(p + name, size, tag) for p in (duplicate, original)
                   for name, size, tag in (("test.bdf", 100, "combined"), ("evt.bdf", 10, "events"), ("data.bdf", 90, "data"))]
        recordings, ignored = batch.discover_recordings(objects)
        self.assertEqual(len(recordings), 1)
        self.assertEqual(recordings[0]["raw_uri"], original + "test.bdf")
        self.assertEqual(recordings[0]["aliases"], [duplicate + "test.bdf"])
        self.assertEqual(len(ignored), 3)

    def test_equal_size_without_etag_does_not_deduplicate(self):
        objects = [obj(ROOT + p + "/data.bdf", etag=None) for p in ("a", "b")]
        recordings, _ = batch.discover_recordings(objects)
        self.assertEqual(len(recordings), 2)

    def test_acquisition_runs_include_incomplete_final_run(self):
        events = [(5, 250), (10, 101), (20, 150), (30, 151), (40, 102), (50, 101), (60, 150)]
        self.assertEqual(batch.raw_run_intervals(events, 100), [(10, 40), (50, 100)])

    def test_ambiguous_validated_logs_are_not_selected(self):
        rows = [{"raw_run_index": 0, "events_uri": uri, "wall_clock_difference_seconds": diff}
                for uri, diff in (("a", 0.1), ("b", 0.2))]
        matches, ambiguous = batch.reject_overlapping_runs(rows)
        self.assertFalse(matches)
        self.assertEqual(ambiguous[0]["events_uris"], ["a", "b"])

    def test_two_logs_inside_one_bdf_and_short_rest_filter(self):
        logs = [obj(LABEL_ROOT + f"123/session_01/{stamp}/events.json")
                for stamp in ("20261023_102630", "20261023_103530")]
        documents = {}
        for log in logs:
            documents[log["uri"]] = [event("session_start", 0),
                                    event("EMOTION_VIDEO_ONSET", 1, three_class_label="positive"),
                                    event("EMOTION_VIDEO_OFFSET", 21),
                                    event("SHORT_REST_ONSET", 22), event("SHORT_REST_OFFSET", 24),
                                    event("session_end", 25)]
        raw = [(t+offset, code) for offset in (60, 600)
               for t, code in ((0, 101), (1, 150), (21, 151), (22, 156), (24, 157), (25, 102))]
        class FakeStore:
            def json(self, uri):
                return documents[uri]
        header = {"sfreq": 1000, "duration_seconds": 1000, "start_as_stored": "23.10.2610.25.30"}
        with patch.object(batch, "recording_header", return_value=header), \
             patch.object(batch, "load_annotations", return_value=(raw, "evt.bdf")):
            result = batch.process_recording(recording(), logs, FakeStore(), config())
        self.assertEqual(result["status"], "success")
        self.assertEqual(len(result["matches"]), 2)
        self.assertEqual(result["duration_filtered_count"], 2)
        self.assertEqual(result["usable_segment_count"], 2)
        emotion = [s for s in result["segments"] if s["segment_type"] == "emotion"]
        self.assertEqual([s["start_sample"] for s in emotion], [61000, 601000])
        self.assertTrue(all(s["emotion_label"] == "positive" for s in emotion))

    def test_failed_segment_stays_excluded_even_if_long(self):
        rows = [{"duration_seconds": 10, "usable": False, "exclusion_reasons": ["incomplete_video"]},
                {"duration_seconds": 4.999, "usable": True, "exclusion_reasons": []},
                {"duration_seconds": 5, "usable": True, "exclusion_reasons": []}]
        batch.apply_duration_filter(rows, 5)
        self.assertFalse(rows[0]["usable"])
        self.assertFalse(rows[1]["usable"])
        self.assertTrue(rows[2]["usable"])

    def test_batch_continues_after_error_and_resumes_only_valid_cache(self):
        one, two = recording("20261023102530_a/test"), recording("20261023103530_b/test")
        two["signal_object"]["etag"] = "different-signal"
        skipped = obj(ROOT + "20261023104530_c/only.ndf")
        objects = [one["signal_object"], one["annotation_object"],
                   two["signal_object"], two["annotation_object"], skipped]
        def process(rec, logs, store, options):
            if rec["raw_uri"] == two["raw_uri"]:
                raise ValueError("invalid recording header")
            return {"raw_uri": rec["raw_uri"], "status": "success", "segments": [], "usable_segment_count": 0}
        class FakeStore:
            s3 = object()
            def __init__(self, *args):
                pass
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(profile="rustfs", raw_root=ROOT, label_root=LABEL_ROOT,
                                      output_dir=Path(directory), workers=2, resume=False,
                                      **config())
            def listing(store, root):
                return objects if root == ROOT else []
            with patch.object(batch, "Store", FakeStore), patch.object(batch, "list_objects", side_effect=listing), \
                 patch.object(batch, "process_recording", side_effect=process) as mocked, \
                 contextlib.redirect_stdout(io.StringIO()):
                summary = batch.batch_run(args)
                self.assertEqual(summary["status_counts"], {"success": 1, "error": 1})
                self.assertEqual(summary["skipped_collections"][0]["collection"], "20261023104530_c")
                self.assertEqual(mocked.call_count, 2)
                args.resume = True
                batch.batch_run(args)
                self.assertEqual(mocked.call_count, 3)
                args.min_segment_seconds = 10
                batch.batch_run(args)
                self.assertEqual(mocked.call_count, 5)
            self.assertTrue((Path(directory) / "manifest.json").is_file())
            with (Path(directory) / "recordings.csv").open(encoding="utf-8-sig") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 2)


if __name__ == "__main__":
    unittest.main()
