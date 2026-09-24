from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from small_agent.data import StatefulDistributedSampler  # noqa: E402
from small_agent.training.checkpoint import (  # noqa: E402
    load_full_checkpoint,
    resolve_checkpoint,
    save_full_checkpoint,
)
from small_agent.training.scheduler import (  # noqa: E402
    TokenLRScheduler,
    WarmupCosineConfig,
)


class CheckpointTests(unittest.TestCase):
    def test_roundtrip_restores_all_training_state(self) -> None:
        device = torch.device("cpu")
        model = torch.nn.Linear(4, 3).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        schedule_config = WarmupCosineConfig(1e-3, 1e-4, 10, 100)
        scheduler = TokenLRScheduler(optimizer, schedule_config)
        sampler = StatefulDistributedSampler(20, seed=7)

        loss = model(torch.ones((2, 4), device=device)).sum()
        loss.backward()
        optimizer.step()
        scheduler.step(8)
        consumed = [next(iter(sampler)) for _ in range(3)]
        self.assertEqual(len(consumed), 3)
        expected = {name: value.detach().clone() for name, value in model.state_dict().items()}
        compatibility = {"model": "abc", "data": "def"}

        with tempfile.TemporaryDirectory() as temporary:
            checkpoints = Path(temporary)
            path, _ = save_full_checkpoint(
                checkpoints,
                step=1,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=sampler,
                compatibility=compatibility,
                keep=1,
            )
            self.assertEqual(resolve_checkpoint(checkpoints), path)
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
            scheduler.step(3)
            sampler.set_epoch(2)
            manifest = load_full_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=sampler,
                compatibility=compatibility,
                device=device,
            )
            self.assertEqual(manifest["step"], 1)
            self.assertEqual(scheduler.tokens_seen, 8)
            self.assertEqual(sampler.cursor, 3)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, expected[name], rtol=0, atol=0)

    def test_compatibility_drift_is_rejected(self) -> None:
        device = torch.device("cpu")
        model = torch.nn.Linear(2, 2).to(device)
        optimizer = torch.optim.AdamW(model.parameters())
        scheduler = TokenLRScheduler(
            optimizer, WarmupCosineConfig(1e-3, 1e-4, 1, 10)
        )
        sampler = StatefulDistributedSampler(4)
        with tempfile.TemporaryDirectory() as temporary:
            path, _ = save_full_checkpoint(
                Path(temporary),
                step=0,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=sampler,
                compatibility={"data": "one"},
            )
            with self.assertRaisesRegex(ValueError, "compatibility"):
                load_full_checkpoint(
                    path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    sampler=sampler,
                    compatibility={"data": "two"},
                    device=device,
                )

    def test_continuation_matches_uninterrupted_training_exactly(self) -> None:
        device = torch.device("cpu")
        torch.manual_seed(123)
        model = torch.nn.Linear(3, 2).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        config = WarmupCosineConfig(1e-3, 1e-4, 2, 20)
        scheduler = TokenLRScheduler(optimizer, config)
        sampler = StatefulDistributedSampler(8, seed=17)
        samples = [torch.tensor([[i, i + 1, i + 2]], dtype=torch.float32) for i in range(8)]

        def take_step(current_model, current_optimizer, current_scheduler, iterator):
            sample = samples[next(iterator)]
            current_optimizer.zero_grad(set_to_none=True)
            loss = current_model(sample).square().mean()
            loss.backward()
            current_scheduler.step(1)
            current_optimizer.step()

        iterator = iter(sampler)
        take_step(model, optimizer, scheduler, iterator)
        take_step(model, optimizer, scheduler, iterator)
        compatibility = {"test": "exact-continuation"}

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint, _ = save_full_checkpoint(
                Path(temporary),
                step=2,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=sampler,
                compatibility=compatibility,
            )
            take_step(model, optimizer, scheduler, iterator)
            take_step(model, optimizer, scheduler, iterator)
            expected = {
                name: value.detach().clone() for name, value in model.state_dict().items()
            }

            torch.manual_seed(999)
            resumed_model = torch.nn.Linear(3, 2).to(device)
            resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=1e-3)
            resumed_scheduler = TokenLRScheduler(resumed_optimizer, config)
            resumed_sampler = StatefulDistributedSampler(8, seed=17)
            load_full_checkpoint(
                checkpoint,
                model=resumed_model,
                optimizer=resumed_optimizer,
                scheduler=resumed_scheduler,
                sampler=resumed_sampler,
                compatibility=compatibility,
                device=device,
            )
            resumed_iterator = iter(resumed_sampler)
            take_step(
                resumed_model, resumed_optimizer, resumed_scheduler, resumed_iterator
            )
            take_step(
                resumed_model, resumed_optimizer, resumed_scheduler, resumed_iterator
            )
            for name, value in resumed_model.state_dict().items():
                torch.testing.assert_close(value, expected[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
