"""Offline tests for the reusable annotation CLI."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "annotate_jsonl.py"
spec = importlib.util.spec_from_file_location("annotate_jsonl", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class AnnotateJsonlTests(unittest.TestCase):
    def test_json_array_and_nested_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            path.write_text(json.dumps([{"data": {"text": "abc", "source": "web", "row": 3}}]))
            records = module.read_records(path)
            job = module.make_job(records[0], 0, "data.text", ["data.source", "data.row"], ["data.row"], "p", "s", "m")
            self.assertEqual(job["source_id"], '["web",3]')
            self.assertEqual(job["source"], {"data.row": 3})
            self.assertEqual(job["text_sha256"], module.digest("abc"))

    def test_jsonl_and_resume_key(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.jsonl"
            source.write_text('{"excerpt":"one","id":1}\n{"excerpt":"two","id":2}\n')
            records = module.read_records(source)
            self.assertEqual(len(records), 2)
            job = module.make_job(records[0], 0, "excerpt", ["id"], [], "p", "s", "m")
            output = Path(directory) / "output.jsonl"
            output.write_text(module.canonical_json({**job, "status": "ok"}) + "\n")
            self.assertIn(module.job_key(job), module.completed_keys(output))
            changed_prompt = {**job, "prompt_sha256": "new"}
            self.assertNotIn(module.job_key(changed_prompt), module.completed_keys(output))

    def test_missing_text_is_error(self):
        with self.assertRaises(KeyError):
            module.make_job({"other": "x"}, 0, "excerpt", [], [], "p", "s", "m")

    def test_temperature_changes_resume_key(self):
        record = {"excerpt": "sample", "id": 1}
        default = module.make_job(record, 0, "excerpt", ["id"], [], "p", "s", "m")
        zero = module.make_job(record, 0, "excerpt", ["id"], [], "p", "s", "m", 0.0)
        self.assertNotEqual(module.job_key(default), module.job_key(zero))


if __name__ == "__main__":
    unittest.main()
