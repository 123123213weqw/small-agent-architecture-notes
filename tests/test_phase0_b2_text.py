import random
import unittest

from experiments.phase0_ab import make_episode
from experiments.phase0_b2_text import TextRenderer, encode_bytes, harden_episode


class Phase0B2Tests(unittest.TestCase):
    def test_model_input_contains_no_literal_structure_flags(self):
        episode = make_episode(10, 32, random.Random(4), "completed_intermediate")
        renderer = TextRenderer("train")
        candidate = episode.records[5]
        text = renderer.model_input(episode, candidate, episode.records[:6], 5)
        self.assertNotIn("consumed=", text)
        self.assertNotIn("unresolved=", text)
        self.assertNotIn("goal_match=", text)

    def test_paraphrase_templates_are_disjoint(self):
        episode = make_episode(11, 32, random.Random(5), "state_overwrite")
        record = episode.records[4]
        self.assertNotEqual(TextRenderer("train").record(record), TextRenderer("paraphrase").record(record))

    def test_byte_encoding_is_fixed_length(self):
        ids, mask = encode_bytes("目标：测试。", 32)
        self.assertEqual(tuple(ids.shape), (32,))
        self.assertEqual(tuple(mask.shape), (32,))
        self.assertGreater(int(mask.sum()), 1)

    def test_hard_negatives_do_not_replace_required_records(self):
        episode = make_episode(12, 48, random.Random(8), "unresolved_subgoal")
        original = {r.uid: r for r in episode.records}
        hardened = harden_episode(episode, 99)
        for uid in episode.required_ids:
            self.assertEqual(original[uid], next(r for r in hardened.records if r.uid == uid))
        overlaps = [r for r in hardened.records if r.uid not in episode.required_ids and r.key in episode.goal_tokens]
        self.assertGreater(len(overlaps), 0)


if __name__ == "__main__":
    unittest.main()
