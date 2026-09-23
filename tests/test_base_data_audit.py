import unittest

from experiments.base_data_audit import crawl_year, strip_python_metadata_prefix, summarize


class BaseDataAuditTests(unittest.TestCase):
    def test_strip_known_metadata_prefix(self):
        content = "<reponame>owner/repo<filename>x.py<gh_stars>10-100\nprint(1)\n"
        self.assertEqual(strip_python_metadata_prefix(content), "print(1)\n")
        self.assertEqual(strip_python_metadata_prefix("<gh_stars>1000+\npass\n"), "pass\n")

    def test_do_not_strip_real_code(self):
        content = "# <filename> is documentation\nprint(1)\n"
        self.assertEqual(strip_python_metadata_prefix(content), content)

    def test_python_parse_counts_after_prefix_cleaning(self):
        records = [
            {"row": {"content": "<gh_stars>0\nprint(1)\n", "int_score": 4, "score": 4.0}},
            {"row": {"content": "def broken(:\n", "int_score": 2, "score": 2.0}},
        ]
        result = summarize(records, "content")
        self.assertEqual(result["python_ast_parseable_raw"], 0)
        self.assertEqual(result["python_ast_parseable_after_prefix_strip"], 1)
        self.assertEqual(result["score_ge_4_python_ast_parseable_after_prefix_strip"], 1)

    def test_crawl_year_is_collection_time(self):
        self.assertEqual(crawl_year({"dump": "CC-MAIN-2022-49"}), 2022)
        self.assertEqual(crawl_year({"date": "2024-03-01T00:00:00Z"}), 2024)
        self.assertEqual(crawl_year({"fetch_time": 1725765787000000000}), 2024)
        self.assertIsNone(crawl_year({"content": "print(1)"}))


if __name__ == "__main__":
    unittest.main()
