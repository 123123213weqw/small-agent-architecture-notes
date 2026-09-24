"""Atomic, content-addressed full checkpoints for exact training resume."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any

import torch
from safetensors.torch import load_model, save_model

from small_agent.data import StatefulDistributedSampler
from small_agent.training.scheduler import TokenLRScheduler


VERSION = "small_agent_full_checkpoint_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
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


def _artifact(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def save_full_checkpoint(
    checkpoints_dir: Path,
    *,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: TokenLRScheduler,
    sampler: StatefulDistributedSampler,
    compatibility: dict[str, Any],
    keep: int = 2,
) -> tuple[Path, dict[str, Any]]:
    if step < 0 or keep < 1:
        raise ValueError("step must be non-negative and keep must be positive")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    final = checkpoints_dir / f"step_{step:08d}"
    if final.exists():
        raise FileExistsError(final)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{final.name}.tmp-", dir=checkpoints_dir)
    )
    started = time.time()
    try:
        model_path = temporary / "model.safetensors"
        optimizer_path = temporary / "optimizer.pt"
        runtime_path = temporary / "runtime.pt"
        scheduler_path = temporary / "scheduler.json"
        sampler_path = temporary / "sampler.json"

        save_model(
            model,
            str(model_path),
            metadata={"format": "pt", "checkpoint_version": VERSION},
        )
        torch.save(optimizer.state_dict(), optimizer_path)
        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            model_device = torch.device("cpu")
        torch.save(
            {
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": (
                    torch.cuda.get_rng_state(model_device)
                    if model_device.type == "cuda"
                    else None
                ),
            },
            runtime_path,
        )
        atomic_json(scheduler_path, scheduler.state_dict())
        atomic_json(sampler_path, sampler.state_dict())
        artifacts = {
            path.name: _artifact(path)
            for path in (
                model_path,
                optimizer_path,
                runtime_path,
                scheduler_path,
                sampler_path,
            )
        }
        manifest = {
            "version": VERSION,
            "stage": "full_training_checkpoint",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "step": step,
            "tokens_seen": scheduler.tokens_seen,
            "elapsed_seconds": round(time.time() - started, 3),
            "compatibility": compatibility,
            "artifacts": artifacts,
        }
        atomic_json(temporary / "manifest.json", manifest)
        os.replace(temporary, final)
        atomic_json(
            checkpoints_dir / "latest.json",
            {
                "version": VERSION,
                "step": step,
                "directory": final.name,
                "manifest_sha256": sha256_file(final / "manifest.json"),
            },
        )
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    completed = sorted(
        path
        for path in checkpoints_dir.glob("step_[0-9]*")
        if path.is_dir() and (path / "manifest.json").is_file()
    )
    for obsolete in completed[:-keep]:
        shutil.rmtree(obsolete)
    return final, manifest


def resolve_checkpoint(checkpoints_dir: Path, checkpoint: Path | None = None) -> Path:
    if checkpoint is not None:
        return checkpoint.resolve()
    latest_path = checkpoints_dir / "latest.json"
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    resolved = checkpoints_dir / latest["directory"]
    if sha256_file(resolved / "manifest.json") != latest["manifest_sha256"]:
        raise ValueError("latest checkpoint manifest hash mismatch")
    return resolved


def load_full_checkpoint(
    checkpoint: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: TokenLRScheduler,
    sampler: StatefulDistributedSampler,
    compatibility: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("version") != VERSION or manifest.get("stage") != "full_training_checkpoint":
        raise ValueError("unsupported checkpoint manifest")
    if manifest.get("compatibility") != compatibility:
        raise ValueError("checkpoint compatibility metadata changed")
    for name, expected in manifest["artifacts"].items():
        path = checkpoint / name
        if path.stat().st_size != expected["bytes"] or sha256_file(path) != expected["sha256"]:
            raise ValueError(f"checkpoint artifact integrity failure: {name}")

    missing, unexpected = load_model(
        model, checkpoint / "model.safetensors", strict=True, device=str(device)
    )
    if missing or unexpected:
        raise ValueError(f"model checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    optimizer.load_state_dict(
        torch.load(
            checkpoint / "optimizer.pt",
            map_location=device,
            weights_only=False,
        )
    )
    scheduler.load_state_dict(
        json.loads((checkpoint / "scheduler.json").read_text(encoding="utf-8"))
    )
    sampler.load_state_dict(
        json.loads((checkpoint / "sampler.json").read_text(encoding="utf-8"))
    )
    runtime = torch.load(
        checkpoint / "runtime.pt", map_location="cpu", weights_only=False
    )
    torch.set_rng_state(runtime["torch_rng_state"])
    if runtime["cuda_rng_state"] is not None:
        if device.type != "cuda":
            raise ValueError("checkpoint contains CUDA RNG state but target device is not CUDA")
        torch.cuda.set_rng_state(runtime["cuda_rng_state"], device=device)
    if scheduler.tokens_seen != manifest["tokens_seen"]:
        raise ValueError("checkpoint scheduler token count mismatch")
    return manifest
