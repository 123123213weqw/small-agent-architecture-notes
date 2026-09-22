#!/usr/bin/env python3
"""Aggregate the six B2.1 Stage-1 runs and apply the frozen decision gate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence


NAMES = (
    "A_pointwise",
    "B_joint_mse",
    "C_joint_rank",
    "D_joint_evict",
    "E_joint_full",
    "F_joint_partial",
)
SPLITS = ("test_id", "test_composition", "test_length", "test_semantic_stress")
TASKS = (
    "delayed_query",
    "state_overwrite",
    "long_instruction",
    "completed_intermediate",
    "unresolved_subgoal",
    "relation_chain",
)


def mean(values: Sequence[float]) -> float:
    return sum(values) / max(1, len(values))


def load_runs(run_dir: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for name in NAMES:
        path = run_dir / name / "summary.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        result[name] = json.loads(path.read_text(encoding="utf-8"))
    return result


def rollout(run: dict[str, Any], split: str, policy: str = "model") -> dict[str, float]:
    return run["rollout"][f"{split}/{policy}"]


def build_report(runs: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, object]], str]:
    rows: list[dict[str, object]] = []
    for name in NAMES:
        run = runs[name]
        for split in SPLITS:
            model = rollout(run, split)
            fifo = rollout(run, split, "fifo")
            oracle = rollout(run, split, "oracle")
            gap = oracle["success"] - fifo["success"]
            recovery = (model["success"] - fifo["success"]) / gap if gap > 1e-12 else 0.0
            offline = run["offline"][split]
            efficiency = run["offline_efficiency"][split]
            rows.append(
                {
                    "experiment": name,
                    "architecture": run["architecture"],
                    "parameters": run["parameters"],
                    "split": split,
                    "success": model["success"],
                    "fifo_success": fifo["success"],
                    "oracle_success": oracle["success"],
                    "gap_recovery": recovery,
                    "required_recall": model["required_recall"],
                    "eviction_regret": model["eviction_regret"],
                    "stale_value_rate": model["stale_value_rate"],
                    "offline_mse": offline["mse"],
                    "pairwise_accuracy": offline["pairwise_accuracy"],
                    "argmin_accuracy": offline["argmin_correct"],
                    "offline_regret": offline["regret"],
                    "groups_per_second": efficiency["groups_per_second"],
                    "peak_memory_mib": efficiency["peak_memory_mib"],
                }
            )

    a = runs["A_pointwise"]
    e = runs["E_joint_full"]
    f = runs["F_joint_partial"]
    composition_gain = rollout(e, "test_composition")["success"] - rollout(a, "test_composition")["success"]
    a_regret = mean([rollout(a, split)["eviction_regret"] for split in SPLITS])
    e_regret = mean([rollout(e, split)["eviction_regret"] for split in SPLITS])
    regret_reduction = (a_regret - e_regret) / a_regret if a_regret > 1e-12 else 0.0
    per_task = []
    for task in TASKS:
        key = f"test_composition/{task}"
        a_success = a["rollout_by_task"][key]["success"]
        e_success = e["rollout_by_task"][key]["success"]
        f_success = f["rollout_by_task"][key]["success"]
        per_task.append(
            {"task": task, "A": a_success, "E": e_success, "F": f_success, "E_minus_A": e_success - a_success}
        )
    non_decreasing_tasks = sum(item["E"] + 1e-12 >= item["A"] for item in per_task)
    e_comp = rollout(e, "test_composition")
    f_comp = rollout(f, "test_composition")
    full_set_success_gain = e_comp["success"] - f_comp["success"]
    full_set_regret_reduction = (
        (f_comp["eviction_regret"] - e_comp["eviction_regret"]) / f_comp["eviction_regret"]
        if f_comp["eviction_regret"] > 1e-12
        else 0.0
    )
    a_eff = a["offline_efficiency"]["test_composition"]
    e_eff = e["offline_efficiency"]["test_composition"]
    throughput_ratio = e_eff["groups_per_second"] / max(1e-12, a_eff["groups_per_second"])
    memory_ratio = e_eff["peak_memory_mib"] / max(1e-12, a_eff["peak_memory_mib"])

    gates = {
        "composition_success_gain_at_least_5pp": {
            "value": composition_gain,
            "pass": composition_gain >= 0.05,
        },
        "mean_regret_reduction_at_least_20pct": {
            "value": regret_reduction,
            "pass": regret_reduction >= 0.20,
        },
        "at_least_4_of_6_tasks_non_decreasing": {
            "value": non_decreasing_tasks,
            "pass": non_decreasing_tasks >= 4,
        },
        "full_set_beats_partial": {
            "success_gain": full_set_success_gain,
            "regret_reduction": full_set_regret_reduction,
            "pass": full_set_success_gain >= 0.02 or full_set_regret_reduction >= 0.10,
        },
        "efficiency_acceptable": {
            "throughput_ratio_E_over_A": throughput_ratio,
            "memory_ratio_E_over_A": memory_ratio,
            "pass": throughput_ratio >= 0.50 and memory_ratio <= 1.25,
        },
    }
    decision = all(item["pass"] for item in gates.values())
    summary = {
        "decision": "continue_to_stage2" if decision else "stop_and_diagnose",
        "pass": decision,
        "gates": gates,
        "per_task_composition": per_task,
        "mean_rollout_regret": {"A_pointwise": a_regret, "E_joint_full": e_regret},
    }

    lines = [
        "# Phase 0B2.1 第一阶段结果",
        "",
        "本报告只使用 Seed 0，用于选择是否进入多种子稳定性验证，不构成最终架构结论。",
        "",
        "## 连续 rollout 成功率",
        "",
        "| 实验 | ID | 组合 | 长度外推 | 语义压力 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in NAMES:
        lines.append(
            f"| {name} | "
            + " | ".join(f"{rollout(runs[name], split)['success']:.3f}" for split in SPLITS)
            + " |"
        )
    lines.extend(
        [
            "",
            "## 连续 rollout 淘汰 regret",
            "",
            "| 实验 | ID | 组合 | 长度外推 | 语义压力 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name in NAMES:
        lines.append(
            f"| {name} | "
            + " | ".join(f"{rollout(runs[name], split)['eviction_regret']:.5f}" for split in SPLITS)
            + " |"
        )
    lines.extend(
        [
            "",
            "## 判定门槛",
            "",
            "| 条件 | 数值 | 结论 |",
            "|---|---:|:---:|",
            f"| E 相对 A 的组合成功率增益 | {composition_gain:+.1%} | {'通过' if gates['composition_success_gain_at_least_5pp']['pass'] else '失败'} |",
            f"| E 相对 A 的四测试集平均 regret 降幅 | {regret_reduction:.1%} | {'通过' if gates['mean_regret_reduction_at_least_20pct']['pass'] else '失败'} |",
            f"| E 不下降的任务数 | {non_decreasing_tasks}/6 | {'通过' if gates['at_least_4_of_6_tasks_non_decreasing']['pass'] else '失败'} |",
            f"| E 相对 F 的组合成功率增益 | {full_set_success_gain:+.1%} | {'通过' if gates['full_set_beats_partial']['pass'] else '失败'} |",
            f"| E/A 吞吐比；显存比 | {throughput_ratio:.2f}；{memory_ratio:.2f} | {'通过' if gates['efficiency_acceptable']['pass'] else '失败'} |",
            "",
            f"**第一阶段决定：{'进入第二阶段多种子验证' if decision else '停止扩展，先诊断失败原因'}。**",
            "",
            "## 组合测试分任务",
            "",
            "| 任务 | A | E | F | E−A |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for item in per_task:
        lines.append(
            f"| {item['task']} | {item['A']:.3f} | {item['E']:.3f} | {item['F']:.3f} | {item['E_minus_A']:+.3f} |"
        )
    lines.append("")
    return summary, rows, "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("runs/b2_1_stage1"))
    parser.add_argument("--output", type=Path, default=Path("results/phase0_b21_stage1"))
    args = parser.parse_args()
    runs = load_runs(args.run_dir)
    summary, rows, markdown = build_report(runs)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output / "RESULTS.md").write_text(markdown, encoding="utf-8")
    with (args.output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
