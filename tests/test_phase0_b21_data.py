import json
import random
import tempfile
import unittest
from pathlib import Path

from experiments.phase0_ab import make_episode
from experiments.phase0_b21_data import (
    FORBIDDEN_INPUT_FIELDS,
    SplitSpec,
    build_dataset,
    collect_episode_groups,
    composition_unseen_characters,
    group_model_input,
    validate_group,
)


class Phase0B21DataTests(unittest.TestCase):
    def test_composition_family_uses_seen_character_vocabulary(self):
        self.assertEqual(composition_unseen_characters(), {})

    def test_group_is_complete_causal_decision(self):
        episode = make_episode(101, 32, random.Random(12), "state_overwrite")
        groups, _ = collect_episode_groups(
            episode,
            split="train",
            capacity=4,
            hard_negative_count=4,
            template_family="A",
            behavior_policy="fifo",
            seed=7,
        )
        self.assertEqual(len(groups), 12)
        for group in groups:
            self.assertEqual(validate_group(group), [])
            self.assertEqual(len(group["records"]), group["capacity"] + 1)
            self.assertEqual(len(group["utilities"]), len(group["records"]))
            self.assertTrue(group["records"][group["candidate_index"]]["is_candidate"])
            self.assertTrue(
                all(record["event_index"] <= group["decision_position"] for record in group["records"])
            )
            self.assertTrue(
                all(
                    record["event_index"] < group["decision_position"]
                    for record in group["recent_context"]
                )
            )

    def test_model_input_excludes_labels_and_structure_truth(self):
        episode = make_episode(102, 32, random.Random(13), "completed_intermediate")
        groups, _ = collect_episode_groups(
            episode,
            split="validation",
            capacity=8,
            hard_negative_count=8,
            template_family="C",
            behavior_policy="oracle",
            seed=8,
        )
        serialized = json.dumps(group_model_input(groups[0]), ensure_ascii=False)
        for field in FORBIDDEN_INPUT_FIELDS:
            self.assertNotIn(f'"{field}"', serialized)
        self.assertNotIn("decision_position", serialized)
        self.assertNotIn("behavior_policy", serialized)

    def test_generation_is_deterministic(self):
        episode = make_episode(103, 48, random.Random(14), "relation_chain")
        kwargs = dict(
            split="test_composition",
            capacity=12,
            hard_negative_count=14,
            template_family="D",
            behavior_policy="random",
            seed=9,
        )
        first, first_entities = collect_episode_groups(episode, **kwargs)
        second, second_entities = collect_episode_groups(episode, **kwargs)
        self.assertEqual(first, second)
        self.assertEqual(first_entities, second_entities)

    def test_split_files_and_entities_are_isolated(self):
        specs = (
            SplitSpec("train", 6, (32,), ("A", "B", "C")),
            SplitSpec("test_composition", 6, (32,), ("D",)),
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "dataset"
            manifest = build_dataset(output, "audit", seed=0, specs=specs)
            self.assertEqual(set(manifest["splits"]), {"train", "test_composition"})
            self.assertTrue((output / "manifest.json").is_file())
            self.assertTrue((output / "AUDIT.md").is_file())
            train = (output / "train.jsonl").read_text(encoding="utf-8")
            test = (output / "test_composition.jsonl").read_text(encoding="utf-8")
            self.assertIn("entity_e100000000_", train)
            self.assertNotIn("entity_e200000000_", train)
            self.assertIn("entity_e200000000_", test)
            self.assertNotIn('"template_family":"D"', train)


if __name__ == "__main__":
    unittest.main()
