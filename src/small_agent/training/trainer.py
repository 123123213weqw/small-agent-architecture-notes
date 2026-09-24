"""Single-GPU M2 trainer used to validate the pretraining infrastructure."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import tempfile
import time
from typing import Any, Iterator

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from small_agent.data import IndexedTokenDataset, StatefulDistributedSampler
from small_agent.evaluation import evaluate_causal_lm
from small_agent.models import build_model, load_model_spec
from small_agent.training.checkpoint import (
    load_full_checkpoint,
    resolve_checkpoint,
    save_full_checkpoint,
)
from small_agent.training.scheduler import TokenLRScheduler, WarmupCosineConfig


VERSION = "small_agent_trainer_m2_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def find_repository_root(config_path: Path) -> Path:
    for candidate in (config_path.parent, *config_path.parents):
        if (candidate / "src" / "small_agent").is_dir() and (candidate / "configs").is_dir():
            return candidate
    raise ValueError(f"cannot locate repository root from {config_path}")


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def source_revision(root: Path) -> str:
    explicit = os.environ.get("SMALL_AGENT_SOURCE_COMMIT")
    if explicit:
        return explicit
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def infinite_train_batches(
    dataset: IndexedTokenDataset,
    sampler: StatefulDistributedSampler,
    batch_size: int,
    loader_generator: torch.Generator,
) -> Iterator[dict[str, torch.Tensor]]:
    while True:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
            generator=loader_generator,
        )
        yielded = False
        for batch in loader:
            yielded = True
            yield batch
        if not yielded and sampler.cursor == 0:
            raise ValueError("no full training batch can be formed")
        sampler.set_epoch(sampler.epoch + 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-final-checkpoint", action="store_true")
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("M2 trainer is single-GPU only; DDP is added in M4")

    config_path = args.run_config.resolve()
    run_config = json.loads(config_path.read_text(encoding="utf-8"))
    repository_root = find_repository_root(config_path)
    model_spec_path = resolve_path(repository_root, run_config["model_spec"])
    data_dir = resolve_path(repository_root, run_config["data_dir"])
    output_root = resolve_path(repository_root, run_config["output_root"])
    run_dir = output_root / args.run_id
    if args.resume:
        if not run_dir.is_dir():
            raise FileNotFoundError(run_dir)
    else:
        if run_dir.exists():
            raise FileExistsError(run_dir)
        run_dir.mkdir(parents=True)
    metrics_path = run_dir / "metrics.jsonl"
    status_path = run_dir / "status.json"
    writer = SummaryWriter(log_dir=run_dir / "tensorboard")

    max_steps = args.max_steps or int(run_config["max_steps"])
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    context_length = int(run_config["context_length"])
    micro_batch_size = int(run_config["micro_batch_size"])
    accumulation_steps = int(run_config["gradient_accumulation_steps"])
    seed = int(run_config["seed"])
    if micro_batch_size < 1 or accumulation_steps < 1:
        raise ValueError("batch size and accumulation must be positive")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)

    try:
        train_dataset = IndexedTokenDataset(
            data_dir, "train", context_length, tail_policy="drop"
        )
        validation_dataset = IndexedTokenDataset(
            data_dir, "validation", context_length, tail_policy="pad"
        )
        train_sampler = StatefulDistributedSampler(
            len(train_dataset), seed=seed, shuffle=True, drop_last=True
        )
        train_batches = infinite_train_batches(
            train_dataset,
            train_sampler,
            micro_batch_size,
            torch.Generator().manual_seed(seed + 10_000),
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=int(run_config["validation"]["batch_size"]),
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            generator=torch.Generator().manual_seed(seed + 20_000),
        )

        model_spec = load_model_spec(model_spec_path)
        model = build_model(model_spec).to(device)
        if run_config.get("gradient_checkpointing", False):
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        model.config.use_cache = False
        model.train()
        parameter_count = sum(parameter.numel() for parameter in model.parameters())

        optimizer_config = run_config["optimizer"]
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(optimizer_config["maximum_learning_rate"]),
            betas=tuple(optimizer_config["betas"]),
            weight_decay=float(optimizer_config["weight_decay"]),
            fused=True,
        )
        scheduler_config = WarmupCosineConfig(
            maximum_learning_rate=float(optimizer_config["maximum_learning_rate"]),
            minimum_learning_rate=float(optimizer_config["minimum_learning_rate"]),
            warmup_tokens=int(run_config["scheduler"]["warmup_tokens"]),
            total_tokens=int(run_config["scheduler"]["total_tokens"]),
        )
        scheduler = TokenLRScheduler(optimizer, scheduler_config)

        compatibility = {
            "run_config_sha256": sha256_file(config_path),
            "model_spec_sha256": sha256_file(model_spec_path),
            "data_manifest_sha256": sha256_file(data_dir / "manifest.json"),
            "seed": seed,
            "context_length": context_length,
            "micro_batch_size": micro_batch_size,
            "gradient_accumulation_steps": accumulation_steps,
            "gradient_checkpointing": bool(run_config.get("gradient_checkpointing", False)),
        }

        manifest = {
            "version": VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "run_id": args.run_id,
            "purpose": run_config["purpose"],
            "distributable": bool(run_config["distributable"]),
            "source_revision": source_revision(repository_root),
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "gpu": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "parameter_count": parameter_count,
            "max_steps": max_steps,
            "run_config": {"path": str(config_path), "sha256": sha256_file(config_path)},
            "model_spec": {
                "path": str(model_spec_path),
                "sha256": sha256_file(model_spec_path),
            },
            "data_manifest": {
                "path": str(data_dir / "manifest.json"),
                "sha256": sha256_file(data_dir / "manifest.json"),
            },
            "train_coverage": train_dataset.coverage_report(),
            "validation_coverage": validation_dataset.coverage_report(),
        }
        start_step = 0
        if args.resume:
            existing_manifest = json.loads(
                (run_dir / "run_manifest.json").read_text(encoding="utf-8")
            )
            for field in ("run_config", "model_spec", "data_manifest"):
                if existing_manifest[field]["sha256"] != manifest[field]["sha256"]:
                    raise ValueError(f"run manifest changed across resume: {field}")
            checkpoint_path = resolve_checkpoint(run_dir / "checkpoints")
            checkpoint_manifest = load_full_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=train_sampler,
                compatibility=compatibility,
                device=device,
            )
            start_step = int(checkpoint_manifest["step"])
            if max_steps <= start_step:
                raise ValueError("max_steps must be greater than the resumed step")
            append_jsonl(
                metrics_path,
                {
                    "event": "resume",
                    "step": start_step,
                    "tokens_seen": scheduler.tokens_seen,
                    "checkpoint": checkpoint_path.name,
                },
            )
        else:
            atomic_json(run_dir / "run_manifest.json", manifest)
            append_jsonl(metrics_path, {"event": "start", **manifest})
        atomic_json(
            status_path,
            {"state": "running", "step": start_step, "tokens_seen": scheduler.tokens_seen},
        )

        if not args.resume and run_config["validation"].get("at_start", True):
            validation = evaluate_causal_lm(model, validation_loader, device)
            row = {
                "event": "validation",
                "step": 0,
                "tokens_seen": 0,
                **validation.__dict__,
            }
            append_jsonl(metrics_path, row)
            writer.add_scalar("validation/loss", validation.loss, 0)
            writer.add_scalar("validation/perplexity", validation.perplexity, 0)
            print(json.dumps(row), flush=True)

        logging_interval = int(run_config["logging_interval_steps"])
        validation_interval = int(run_config["validation"]["interval_steps"])
        checkpoint_interval = int(
            run_config.get("checkpoint", {}).get("interval_steps", 0)
        )
        window_loss = torch.zeros((), device=device)
        window_tokens = 0
        window_steps = 0
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        window_started = time.perf_counter()
        window_elapsed = 0.0

        for step in range(start_step + 1, max_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            step_loss = torch.zeros((), device=device)
            step_tokens = 0
            for _ in range(accumulation_steps):
                batch = next(train_batches)
                predicted_tokens = int(
                    torch.count_nonzero(batch["labels"][:, 1:] != -100)
                )
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = model(
                        input_ids=input_ids,
                        labels=labels,
                        attention_mask=attention_mask,
                        use_cache=False,
                    )
                    loss = output.loss
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite training loss at step {step}")
                (loss / accumulation_steps).backward()
                step_loss += loss.detach() / accumulation_steps
                step_tokens += predicted_tokens

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(optimizer_config["gradient_clip"]),
                error_if_nonfinite=True,
            )
            learning_rate = scheduler.step(step_tokens)
            optimizer.step()
            window_loss += step_loss
            window_tokens += step_tokens
            window_steps += 1

            should_log = step % logging_interval == 0 or step == max_steps
            if should_log:
                torch.cuda.synchronize(device)
                elapsed = window_elapsed + time.perf_counter() - window_started
                row = {
                    "event": "train",
                    "step": step,
                    "epoch": train_sampler.epoch,
                    "sampler_cursor": train_sampler.cursor,
                    "tokens_seen": scheduler.tokens_seen,
                    "loss": float(window_loss / window_steps),
                    "learning_rate": learning_rate,
                    "grad_norm": float(grad_norm),
                    "tokens_per_second": window_tokens / elapsed,
                    "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                    "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
                }
                append_jsonl(metrics_path, row)
                print(json.dumps(row), flush=True)
                for key in ("loss", "learning_rate", "grad_norm", "tokens_per_second"):
                    writer.add_scalar(f"train/{key}", row[key], step)
                writer.add_scalar("train/tokens_seen", scheduler.tokens_seen, step)
                writer.add_scalar("system/gpu_allocated_gib", row["peak_allocated_gib"], step)
                writer.add_scalar("system/gpu_reserved_gib", row["peak_reserved_gib"], step)
                writer.flush()
                atomic_json(
                    status_path,
                    {"state": "running", "step": step, "tokens_seen": scheduler.tokens_seen},
                )
                window_loss.zero_()
                window_tokens = 0
                window_steps = 0
                window_started = time.perf_counter()
                window_elapsed = 0.0

            if checkpoint_interval and step % checkpoint_interval == 0:
                checkpoint_path, checkpoint_manifest = save_full_checkpoint(
                    run_dir / "checkpoints",
                    step=step,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    sampler=train_sampler,
                    compatibility=compatibility,
                    keep=int(run_config.get("checkpoint", {}).get("keep_full", 2)),
                )
                append_jsonl(
                    metrics_path,
                    {
                        "event": "checkpoint",
                        "step": step,
                        "tokens_seen": scheduler.tokens_seen,
                        "directory": checkpoint_path.name,
                        "elapsed_seconds": checkpoint_manifest["elapsed_seconds"],
                    },
                )

            if step % validation_interval == 0:
                if window_steps:
                    torch.cuda.synchronize(device)
                    window_elapsed += time.perf_counter() - window_started
                validation = evaluate_causal_lm(model, validation_loader, device)
                row = {
                    "event": "validation",
                    "step": step,
                    "tokens_seen": scheduler.tokens_seen,
                    **validation.__dict__,
                }
                append_jsonl(metrics_path, row)
                print(json.dumps(row), flush=True)
                writer.add_scalar("validation/loss", validation.loss, step)
                writer.add_scalar("validation/perplexity", validation.perplexity, step)
                writer.flush()
                torch.cuda.synchronize(device)
                window_started = time.perf_counter()

        final_already_checkpointed = bool(
            checkpoint_interval and max_steps % checkpoint_interval == 0
        )
        if args.save_final_checkpoint and not final_already_checkpointed:
            checkpoint_path, checkpoint_manifest = save_full_checkpoint(
                run_dir / "checkpoints",
                step=max_steps,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=train_sampler,
                compatibility=compatibility,
                keep=int(run_config.get("checkpoint", {}).get("keep_full", 2)),
            )
            append_jsonl(
                metrics_path,
                {
                    "event": "checkpoint",
                    "step": max_steps,
                    "tokens_seen": scheduler.tokens_seen,
                    "directory": checkpoint_path.name,
                    "elapsed_seconds": checkpoint_manifest["elapsed_seconds"],
                },
            )

        validation = evaluate_causal_lm(model, validation_loader, device)
        row = {
            "event": "validation",
            "step": max_steps,
            "tokens_seen": scheduler.tokens_seen,
            **validation.__dict__,
        }
        append_jsonl(metrics_path, row)
        print(json.dumps(row), flush=True)
        writer.add_scalar("validation/loss", validation.loss, max_steps)
        writer.add_scalar("validation/perplexity", validation.perplexity, max_steps)
        writer.flush()

        atomic_json(
            status_path,
            {"state": "completed", "step": max_steps, "tokens_seen": scheduler.tokens_seen},
        )
    except BaseException as error:
        atomic_json(
            status_path,
            {"state": "failed", "error_type": type(error).__name__, "error": str(error)},
        )
        raise
    finally:
        writer.close()


if __name__ == "__main__":
    main()
