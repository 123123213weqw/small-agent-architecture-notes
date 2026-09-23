import collections
import unittest

from experiments.base_data_review import digest, select_rows


class BaseDataReviewTests(unittest.TestCase):
    def test_normalized_digest_ignores_whitespace(self):
        self.assertEqual(digest("a  b\n"), digest(" a b "))

    def test_code_selection_balances_scores_and_lengths(self):
        records = [
            {"row_idx": i, "row": {"content": "x" * (i % 97 + 1), "int_score": i % 4 + 1}}
            for i in range(1000)
        ]
        selected = select_rows(records, "python_edu", 20260923)
        self.assertEqual(len(selected), 100)
        self.assertEqual(len({entry[0]["row_idx"] for entry in selected}), 100)
        counts = collections.Counter(entry[1] for entry in selected)
        self.assertEqual(counts, {"1": 25, "2": 25, "3": 25, "4plus": 25})
        self.assertEqual(selected, select_rows(records, "python_edu", 20260923))


if __name__ == "__main__":
    unittest.main()
