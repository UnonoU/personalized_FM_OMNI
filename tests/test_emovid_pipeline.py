"""Storage interface regressions: acquisition names, collisions and remote resume."""

from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from preprocessing.common import emovid_pipeline as pipeline
from preprocessing.common.emovid_storage import OutputStore, recording_path
from preprocessing.common import preprocess_emovid_rules as prep

RAW = ("s3://oi-data@us-east-1/EMOVID_RAW/20261022161317_20261022_lwh/"
       "20261022161317_20261022_lwh/lwh/lwh.bdf")
NAME = "20261022161317_20261022_lwh"


class PipelineTests(unittest.TestCase):
    def test_output_name_is_outer_acquisition_directory(self):
        self.assertEqual(str(recording_path(RAW)), NAME)
        self.assertEqual(str(recording_path(RAW.replace("lwh.bdf", "another.edf"))), NAME)

    def test_source_paths_cannot_escape_root(self):
        for uri in ["s3://another/EMOVID_RAW/a/a.bdf", "s3://oi-data/EMOVID_RAW/../a.bdf", "s3://oi-data/OTHER/a.bdf"]:
            with self.assertRaises(ValueError):
                recording_path(uri)

    def test_same_acquisition_multiple_distinct_bdfs_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            uris = [RAW.replace("@us-east-1", ""), RAW.replace("@us-east-1", "").replace("lwh.bdf", "other.bdf")]
            (root/"report.json").write_text('{}')
            (root/"manifest.json").write_text(json.dumps({"recordings": [
                {"raw_uri": uri, "report_path": str(root/"report.json")} for uri in uris]}))
            (root/"inventory.json").write_text(json.dumps({"raw_objects": [{"uri": uri} for uri in uris]}))
            options = pipeline.PreprocessingOptions(from_manifest=root/"manifest.json")
            with self.assertRaisesRegex(ValueError, "collision"):
                pipeline.mapping_jobs(options, Mock())

    def test_destination_uri_and_json_upload(self):
        client = Mock()
        with patch("preprocessing.common.emovid_storage.lance_storage_options", return_value={"allow_http": "true"}):
            output = OutputStore(client)
        self.assertEqual(output.uri(NAME+".lance"), f"s3://personalized-fm/processed/{NAME}.lance")
        output.put_json(NAME+".summary.json", {"n_windows": 273})
        kwargs = client.put_object.call_args.kwargs
        self.assertEqual(kwargs["Bucket"], "personalized-fm")
        self.assertEqual(kwargs["Key"], "processed/"+NAME+".summary.json")
        self.assertEqual(json.loads(kwargs["Body"]), {"n_windows": 273})
        with self.assertRaises(ValueError):
            output.key("../outside")

    def test_verified_remote_result_resumes_without_local_waveforms(self):
        options = pipeline.PreprocessingOptions()
        config = prep.preprocessing_contract(options)
        report = {"raw_uri": RAW, "segments": []}
        job = {"raw_uri": RAW, "name": NAME, "signal_object": {"etag": "test", "size": 100}}
        signature = prep.canonical_hash({"source": job["signal_object"], "mapping": pipeline.mapping_signature(report),
                                        "config": config, "processing_hash": pipeline.processing_hash()})
        previous = {"layout": pipeline.LAYOUT, "raw_uri": RAW, "signature": signature,
                    "lance_version": 1, "n_windows": 273}
        output = Mock()
        output.exists.side_effect = lambda key: key.endswith((".summary.json", ".validation.json"))
        output.read_json.side_effect = lambda key: previous if key.endswith(".summary.json") else {"status": "pass"}
        dataset = Mock(version=1)
        dataset.count_rows.return_value = 273
        with patch.object(pipeline.lance, "dataset", return_value=dataset):
            result, resumed = pipeline.export_recording(options, job, report, config, output, Mock())
        self.assertTrue(resumed)
        self.assertEqual(result, previous)
        output.put_json.assert_not_called()

    def test_remote_owner_mismatch_is_never_overwritten(self):
        options = pipeline.PreprocessingOptions()
        output = Mock()
        output.exists.return_value = True
        output.read_json.return_value = {"layout": pipeline.LAYOUT, "raw_uri": "s3://other/source.bdf"}
        with self.assertRaisesRegex(ValueError, "another source"):
            pipeline.export_recording(options, {"name": NAME, "raw_uri": RAW, "signal_object": {}},
                                      {}, prep.preprocessing_contract(options), output, Mock())
        output.put_json.assert_not_called()

    def test_legacy_executable_routes_to_the_unified_interface(self):
        with patch.object(pipeline, "main", return_value="unified") as entry:
            self.assertEqual(prep.main(), "unified")
        entry.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
