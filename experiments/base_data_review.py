"""Prepare a content-free review queue; materialize text locally from Dataset Viewer."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import requests


TEXT_KEY = {
    "fineweb_edu": "text",
    "fineweb2_hq_zh": "text",
    "finemath_4plus": "text",
    "python_edu": "content",
}

QUEUE_FIELDS = [
    "source_id", "dataset", "config", "split", "revision_observed",
    "window_offset", "row_idx", "stratum", "stratum_population",
    "length_bin", "score", "text_chars", "normalized_sha256",
    "decision", "reason", "notes",
]


def digest(text: str) -> str:
    return hashlib.sha256(re.sub(r"\s+", " ", text).strip().encode()).hexdigest()


def select_rows(records: list[dict], source_id: str, seed: int) -> list[tuple[dict, str, int, int]]:
    """25 per score stratum, balanced across five within-stratum length bins."""
    rng = random.Random(f"review:{seed}:{source_id}")
    if source_id == "python_edu":
        groups = defaultdict(list)
        for record in records:
            score = record["row"]["int_score"]
            groups["4plus" if score >= 4 else str(max(1, score))].append(record)
        keys = ["1", "2", "3", "4plus"]
    else:
        ranked = sorted(records, key=lambda record: (
            float(record["row"].get("score", record["row"].get("quality_score", 0))),
            record["row_idx"],
        ))
        groups = {
            f"score_q{quartile + 1}": ranked[quartile * len(ranked) // 4 : (quartile + 1) * len(ranked) // 4]
            for quartile in range(4)
        }
        keys = list(groups)
    selected = []
    for key in keys:
        stratum = sorted(groups[key], key=lambda record: (
            len(record["row"][TEXT_KEY[source_id]]), record["row_idx"]
        ))
        if len(stratum) < 25:
            raise ValueError(f"{source_id} stratum {key} has only {len(stratum)} rows")
        for bin_idx in range(5):
            chunk = stratum[bin_idx * len(stratum) // 5 : (bin_idx + 1) * len(stratum) // 5]
            for record in rng.sample(chunk, 5):
                selected.append((record, key, len(stratum), bin_idx + 1))
    return sorted(selected, key=lambda entry: entry[0]["row_idx"])


def prepare(samples: Path, summary_path: Path, queue_path: Path, seed: int) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    groups = defaultdict(list)
    with samples.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            groups[record["source_id"]].append(record)
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    with queue_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=QUEUE_FIELDS, lineterminator="\n")
        writer.writeheader()
        for source_id, info in summary["sources"].items():
            if len(groups[source_id]) != info["sampled_records"]:
                raise ValueError(f"{source_id}: sample file and summary count disagree")
            for record, stratum, population, length_bin in select_rows(groups[source_id], source_id, seed):
                idx = record["row_idx"]
                windows = [offset for offset in info["offsets"] if offset <= idx < offset + summary["batch_size"]]
                if len(windows) != 1:
                    raise ValueError(f"{source_id} row {idx} has {len(windows)} matching windows")
                row = record["row"]
                content = row[TEXT_KEY[source_id]]
                writer.writerow({
                    "source_id": source_id,
                    "dataset": info["repo_id"],
                    "config": info["config"],
                    "split": "train",
                    "revision_observed": info["observed_revision"],
                    "window_offset": windows[0],
                    "row_idx": idx,
                    "stratum": stratum,
                    "stratum_population": population,
                    "length_bin": length_bin,
                    "score": row.get("score", row.get("quality_score")),
                    "text_chars": len(content),
                    "normalized_sha256": digest(content),
                    "decision": "",
                    "reason": "",
                    "notes": "",
                })


def get_rows(session: requests.Session, dataset: str, config: str, split: str, offset: int, length: int) -> list[dict]:
    params = {"dataset": dataset, "config": config, "split": split, "offset": offset, "length": length}
    for attempt in range(8):
        try:
            response = session.get("https://datasets-server.huggingface.co/rows", params=params, timeout=60)
        except requests.RequestException:
            time.sleep(min(90, 2 ** attempt + 3))
            continue
        if response.status_code in (429, 500, 502, 503, 504):
            time.sleep(min(90, 2 ** attempt + 3))
            continue
        response.raise_for_status()
        return response.json()["rows"]
    raise RuntimeError(f"Dataset Viewer unavailable: {dataset} {config} offset={offset}")


def materialize(queue_path: Path, output: Path, source: str | None, batch_size: int) -> None:
    with queue_path.open(encoding="utf-8", newline="") as stream:
        queue = [row for row in csv.DictReader(stream) if source is None or row["source_id"] == source]
    windows = defaultdict(list)
    for row in queue:
        windows[(row["source_id"], row["dataset"], row["config"], row["split"], int(row["window_offset"]))].append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    session = requests.Session()
    session.headers.update({"User-Agent": "small-agent-base-data-review/0.1"})
    try:
        with partial.open("w", encoding="utf-8") as stream:
            for (source_id, dataset, config, split, offset), entries in sorted(windows.items()):
                fetched = {item["row_idx"]: item["row"] for item in get_rows(session, dataset, config, split, offset, batch_size)}
                for entry in entries:
                    idx = int(entry["row_idx"])
                    if idx not in fetched:
                        raise ValueError(f"Missing row {source_id}:{idx}")
                    row = fetched[idx]
                    content = row[TEXT_KEY[source_id]]
                    if digest(content) != entry["normalized_sha256"]:
                        raise ValueError(f"Content hash changed at {source_id}:{idx}; do not review mismatched revision")
                    stream.write(json.dumps({
                        "source_id": source_id,
                        "row_idx": idx,
                        "stratum": entry["stratum"],
                        "length_bin": int(entry["length_bin"]),
                        "text": content,
                        "url": row.get("url"),
                        "repo_name": row.get("max_stars_repo_name"),
                        "repo_path": row.get("max_stars_repo_path"),
                    }, ensure_ascii=False) + "\n")
                print(f"{source_id} offset={offset}: {len(entries)} review rows", flush=True)
                time.sleep(2)
        partial.replace(output)
    finally:
        partial.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--samples", type=Path, required=True)
    prepare_parser.add_argument("--summary", type=Path, required=True)
    prepare_parser.add_argument("--queue", type=Path, required=True)
    prepare_parser.add_argument("--seed", type=int, default=20260923)
    materialize_parser = sub.add_parser("materialize")
    materialize_parser.add_argument("--queue", type=Path, required=True)
    materialize_parser.add_argument("--output", type=Path, required=True)
    materialize_parser.add_argument("--source", choices=list(TEXT_KEY))
    materialize_parser.add_argument("--batch-size", type=int, default=50)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.samples, args.summary, args.queue, args.seed)
    else:
        materialize(args.queue, args.output, args.source, args.batch_size)
