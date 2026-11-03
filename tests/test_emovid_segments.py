"""Regression tests for identity matching, clock offsets and incomplete trials."""

import unittest

from preprocessing.common.emovid_segments import (
    ClockAlignment, normalize_events, rank_candidates, recording_header,
    segment_events, select_candidate, split_s3,
)


def event(name, seconds, **payload):
    base = dict(subject_id="13467638", session_id=1, eeg_part=1)
    base.update(payload)
    # Deliberately unusable event sample counts: mapping must use device triggers.
    return dict(name=name, relative_time_sec=seconds, sample_index=0, payload=base)


class MappingTests(unittest.TestCase):
    def test_user_example_and_region_uri(self):
        raw = "s3://oi-data@us-east-1/EMOVID_RAW/20261023102529_20261023_lwh/lwh/lwh.edf"
        root = "s3://oi-data/EMOVID-261029/13467638/session_01/"
        uris = [root + stamp + "/events.json" for stamp in
                ("20261023_102211", "20261023_102630", "20261023_121123", "20261022_102630")]
        selected = select_candidate(rank_candidates(raw, uris))
        self.assertEqual(selected.events_uri, uris[1])
        self.assertEqual(selected.delta_seconds, 61)
        self.assertEqual(split_s3(raw)[0], "oi-data")

    def test_ambiguous_subjects_and_parts_require_selection(self):
        raw = "/EMOVID_RAW/20261023102529_test/test.bdf"
        uris = [f"/EMOVID-261029/{subject}/session_01/20261023_102630/events.json"
                for subject in ("123", "456")]
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            select_candidate(rank_candidates(raw, uris))
        self.assertEqual(select_candidate(rank_candidates(raw, uris, subject_id="456")).subject_id, "456")
        uris += [uris[0].replace("events.json", "events_part_002.json")]
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            select_candidate(rank_candidates(raw, uris, subject_id="123"))

    def test_clock_alignment_emotion_and_local_rest(self):
        context = dict(video_id="emotion:a", trial_id=12, attempt_id=1, three_class_label="positive")
        events = [event("session_start", 0),
                  event("EMOTION_VIDEO_ONSET", 10, **context),
                  event("EMOTION_VIDEO_OFFSET", 30, completed_naturally=True, **context),
                  event("rest_start", 31, rest_index=2, **context),
                  event("rest_end", 40, rest_index=2, **context),
                  event("session_end", 50)]
        # Device timeline starts 65s earlier, with small linear clock drift.
        raw = [(65, 101), (75.001, 150), (95.003, 151), (115.005, 102)]
        alignment = ClockAlignment(events, raw)
        segments, issues = segment_events(events, alignment, sfreq=1000, duration_seconds=120, output_sfreq=200)
        self.assertFalse(issues)
        emotion, rest = segments
        self.assertEqual(emotion["start_sample"], 75001)
        self.assertEqual(emotion["stop_sample"], 95003)
        self.assertEqual(emotion["emotion_label"], "positive")
        self.assertEqual(emotion["output_start_sample"], 15000)
        self.assertEqual(rest["rest_type"], "long_rest")
        self.assertIsNone(rest["emotion_label"])
        self.assertIsNone(rest["video_id"])
        self.assertAlmostEqual(rest["start_seconds"], 96.0031)
        self.assertEqual(rest["start_time_basis"], "interpolated_log_clock")
        self.assertTrue(all(s["usable"] for s in segments))

    def test_all_rest_marker_pairs_are_separate_types(self):
        names = [("baseline_start", "baseline_end", 110, 111),
                 ("eyes_open_baseline_start", "eyes_open_baseline_end", 112, 113),
                 ("eyes_closed_baseline_start", "eyes_closed_baseline_end", 114, 115),
                 ("SHORT_REST_ONSET", "SHORT_REST_OFFSET", 156, 157),
                 ("break_start", "break_end", 136, 137)]
        events, raw = [], []
        for i, (start, stop, a, b) in enumerate(names):
            events += [event(start, i*10), event(stop, i*10+2)]
            raw += [(i*10+60, a), (i*10+62, b)]
        segments, issues = segment_events(events, ClockAlignment(events, raw), sfreq=1000, duration_seconds=110)
        self.assertFalse(issues)
        self.assertEqual([s["rest_type"] for s in segments],
                         ["baseline", "eyes_open_baseline", "eyes_closed_baseline", "short_rest", "short_rest"])
        self.assertTrue(all(s["emotion_label"] is None for s in segments))

    def test_missing_offset_does_not_pair_across_attempt_or_session(self):
        events = [event("session_start", 0),
                  event("EMOTION_VIDEO_ONSET", 1, video_id="a", trial_id=1, attempt_id=1),
                  event("session_end", 3), event("session_start", 10),
                  event("EMOTION_VIDEO_OFFSET", 12, video_id="a", trial_id=1, attempt_id=2),
                  event("session_end", 15)]
        raw = list(zip([60, 61, 63, 70, 72, 75], [101, 150, 102, 101, 151, 102]))
        segments, issues = segment_events(events, ClockAlignment(events, raw), sfreq=1000, duration_seconds=100)
        self.assertFalse(segments)
        self.assertEqual({x["issue"] for x in issues}, {"unclosed_at_session_boundary", "end_without_matching_start"})

    def test_skipped_missing_or_conflicting_labels_excluded(self):
        for label, ending, reason in [
            ("negative", {"skipped": True}, "incomplete_video"),
            ("", {}, "missing_or_unknown_emotion_label"),
            ("negative", {"three_class_label": "positive"}, "conflicting_emotion_labels"),
        ]:
            with self.subTest(reason=reason):
                events = [event("session_start", 0),
                          event("EMOTION_VIDEO_ONSET", 1, three_class_label=label),
                          event("EMOTION_VIDEO_OFFSET", 3, **ending), event("session_end", 5)]
                raw = [(60, 101), (61, 150), (63, 151), (65, 102)]
                rows, _ = segment_events(events, ClockAlignment(events, raw), sfreq=1000, duration_seconds=70)
                self.assertFalse(rows[0]["usable"])
                self.assertIn(reason, rows[0]["exclusion_reasons"])

    def test_missing_raw_boundary_is_not_a_usable_segment(self):
        events = [event("session_start", 0), event("fixation_on", 1), event("fixation_off", 2),
                  event("EMOTION_VIDEO_ONSET", 3, three_class_label="neutral"),
                  event("EMOTION_VIDEO_OFFSET", 5), event("session_end", 6)]
        raw = [(60, 101), (61, 130), (62, 131), (65, 151), (66, 102)]
        rows, _ = segment_events(events, ClockAlignment(events, raw), sfreq=1000, duration_seconds=70)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["exclusion_reasons"], ["missing_raw_boundary_trigger"])

    def test_wrong_log_and_clock_disagreement_rejected(self):
        events = [event("session_start", 0), event("video_on", 1), event("video_off", 3), event("session_end", 5)]
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            ClockAlignment(events, [(60, 250), (61, 251), (62, 252)])
        with self.assertRaisesRegex(ValueError, "clocks disagree"):
            ClockAlignment(events, [(60, 101), (61, 132), (80, 133), (85, 102)])

    def test_precise_callback_clock_handles_rounded_log_times(self):
        events = [event("session_start", 0), event("trial_start", 1),
                  event("fixation_on", 1), event("session_end", 3)]
        for e, t in zip(events, [100, 101, 101.015, 103]):
            e["payload"]["flip_callback_time_monotonic_sec"] = t
        alignment = ClockAlignment(events, [(60, 101), (61, 140), (61.015, 130), (63, 102)])
        self.assertEqual(alignment.report["source_clock"], "flip_callback_time_monotonic_sec")
        self.assertEqual(alignment.map_event(2, events[2]), (61.015, "raw_trigger"))

    def test_local_events_outside_anchor_range_are_not_extrapolated(self):
        events = [event("rest_start", 0, rest_index=1), event("rest_end", 1, rest_index=1),
                  event("session_start", 2), event("video_on", 3), event("session_end", 5)]
        alignment = ClockAlignment(events, [(62, 101), (63, 132), (65, 102)])
        rows, issues = segment_events(events, alignment, sfreq=1000, duration_seconds=70)
        self.assertFalse(rows)
        self.assertEqual(issues[0]["issue"], "unmapped_boundary")

    def test_diagnostic_error_without_callback_does_not_change_clock_basis(self):
        events = [event("session_start", 0), event("trial_start", 1),
                  event("fixation_on", 1), event("session_end", 3)]
        for e, t in zip(events, [100, 101, 101.015, 103]):
            e["payload"]["flip_callback_time_monotonic_sec"] = t
        events.append(event("eeg_acquisition_error", 8, code="no_samples_timeout"))
        alignment = ClockAlignment(events, [(60, 101), (61, 140), (61.015, 130), (63, 102)])
        self.assertEqual(alignment.report["source_clock"], "flip_callback_time_monotonic_sec")

    def test_attempt_identity_and_recording_bounds_are_checked(self):
        events = [event("session_start", 0),
                  event("EMOTION_VIDEO_ONSET", 1, attempt_id=1, three_class_label="negative"),
                  event("EMOTION_VIDEO_OFFSET", 3, attempt_id=2), event("session_end", 5)]
        alignment = ClockAlignment(events, [(60, 101), (61, 150), (63, 151), (65, 102)])
        rows, issues = segment_events(events, alignment, sfreq=1000, duration_seconds=70)
        self.assertFalse(rows)
        self.assertIn("end_without_matching_start", [i["issue"] for i in issues])
        events[2]["payload"]["attempt_id"] = 1
        rows, issues = segment_events(events, alignment, sfreq=1000, duration_seconds=62)
        self.assertFalse(rows)
        self.assertEqual(issues[0]["issue"], "invalid_or_out_of_recording_interval")

    def test_clock_resets_and_multiple_parts_rejected(self):
        with self.assertRaisesRegex(ValueError, "clock resets"):
            normalize_events([event("session_start", 5), event("session_end", 0)])
        with self.assertRaisesRegex(ValueError, "Multiple EEG parts"):
            normalize_events([event("session_start", 0), event("session_end", 5, eeg_part=2)])

    def test_header_uses_signal_rate_not_annotation_rate(self):
        from test_preprocess_lwh_labram import make_bdf
        from preprocessing.common.emovid_segments import Store
        from pathlib import Path
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.bdf"
            make_bdf(path, "uV")
            header = recording_header(Store(), str(path))
        self.assertEqual(header["sfreq"], 1000)
        self.assertEqual(header["duration_seconds"], 60)


if __name__ == "__main__":
    unittest.main()
