"""Small, reproducible HF Dataset Viewer audit; never downloads whole datasets."""

from __future__ import annotations

import argparse
import ast
import datetime
import hashlib
import json
import random
import re
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import requests


SOURCES = [
    ("fineweb_edu", "HuggingFaceFW/fineweb-edu", "sample-10BT", "text"),
    ("fineweb2_hq_zh", "epfml/FineWeb2-HQ", "cmn_Hani", "text"),
    ("finemath_4plus", "HuggingFaceTB/finemath", "finemath-4plus", "text"),
    ("python_edu", "jon-tow/starcoderdata-python-edu", "default", "content"),
]

ROW_FIELDS = {
    "fineweb_edu": ("text", "id", "dump", "url", "language", "score", "int_score"),
    "fineweb2_hq_zh": ("text", "id", "date", "dump", "url", "language", "language_script", "quality_score"),
    "finemath_4plus": ("text", "url", "fetch_time", "score", "int_score", "language"),
    "python_edu": ("content", "id", "max_stars_repo_name", "max_stars_repo_path", "score", "int_score"),
}


def strip_python_metadata_prefix(content: str) -> str:
    """Remove only a known one-line StarCoder metadata prefix, not code comments."""
    return re.sub(
        r"\A(?:<reponame>[^\r\n<>]*|<filename>[^\r\n<>]*|<gh_stars>\d+(?:-\d+)?\+?)+[ \t]*\r?\n",
        "",
        content,
        count=1,
    )


def get_json(session: requests.Session, url: str, params: dict | None = None) -> dict:
    for attempt in range(6):
        try:
            response = session.get(url, params=params, timeout=50)
            if response.status_code in (429, 500, 502, 503, 504):
                delay = min(30, 2 ** attempt + 1)
                print(f"retry HTTP {response.status_code} after {delay}s: {url}", flush=True)
                time.sleep(delay)
                continue
            response.raise_for_status()
            return response.json()
        except (requests.Timeout, requests.ConnectionError) as exc:
            if attempt == 5:
                raise
            delay = min(30, 2 ** attempt + 1)
            print(f"retry {type(exc).__name__} after {delay}s: {url}", flush=True)
            time.sleep(delay)
    raise RuntimeError(f"Dataset Viewer failed after retries: {url} {params}")


def percentiles(values: list[int | float]) -> dict:
    if not values:
        return {}
    values = sorted(values)

    def at(q: float) -> float:
        return round(values[int((len(values) - 1) * q)], 3)

    return {"min": at(0), "p10": at(0.1), "median": at(0.5), "p90": at(0.9), "max": at(1)}


def repeated_line_fraction(text: str) -> float:
    lines = [line.strip() for line in text.splitlines() if len(line.strip()) >= 20]
    if len(lines) < 3:
        return 0.0
    return 1 - len(set(lines)) / len(lines)


def crawl_year(row: dict) -> int | None:
    """Return collection/fetch year, not the document's original publication year."""
    fetch_time = row.get("fetch_time")
    if isinstance(fetch_time, (int, float)):
        try:
            return datetime.datetime.fromtimestamp(fetch_time / 1_000_000_000, datetime.timezone.utc).year
        except (OverflowError, OSError, ValueError):
            return None
    for key in ("date", "dump"):
        match = re.search(r"20\d{2}", str(row.get(key) or ""))
        if match:
            return int(match.group())
    return None


def summarize(records: list[dict], text_key: str) -> dict:
    texts = [record["row"].get(text_key, "") or "" for record in records]
    hashes = [hashlib.sha256(re.sub(r"\s+", " ", text).strip().encode()).hexdigest() for text in texts]
    scores = [record["row"].get("score", record["row"].get("quality_score")) for record in records]
    scores = [float(score) for score in scores if isinstance(score, (int, float))]
    domains = Counter(urlparse(record["row"].get("url", "")).netloc.lower() for record in records)
    domains.pop("", None)
    year_counts = Counter(crawl_year(record["row"]) for record in records)
    common = {
        "sampled_records": len(records),
        "total_text_characters": sum(map(len, texts)),
        "unique_normalized_hashes": len(set(hashes)),
        "text_char_lengths": percentiles([len(text) for text in texts]),
        "text_utf8_bytes": percentiles([len(text.encode()) for text in texts]),
        "score": percentiles(scores),
        "shorter_than_200_chars": sum(len(text) < 200 for text in texts),
        "longer_than_50000_chars": sum(len(text) > 50000 for text in texts),
        "high_repeated_line_fraction_over_0_2": sum(repeated_line_fraction(text) > 0.2 for text in texts),
        "html_or_boilerplate_marker": sum(bool(re.search(r"(?i)<(?:script|style|div|span)|cookie policy|all rights reserved", text)) for text in texts),
        "email_like_marker": sum(bool(re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text)) for text in texts),
        "high_confidence_secret_pattern": sum(
            bool(re.search(
                r"(?i)-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|\bAKIA[0-9A-Z]{16}\b|\bghp_[A-Za-z0-9]{30,}\b|\bhf_[A-Za-z0-9]{30,}\b",
                text,
            ))
            for text in texts
        ),
        "top_url_domains": domains.most_common(10),
        "crawl_year_counts": {str(year): year_counts[year] for year in sorted(year_counts) if year is not None},
        "crawl_year_missing": year_counts[None],
    }
    if text_key == "content":
        score_counts = Counter(str(record["row"].get("int_score")) for record in records)
        parsed_raw = 0
        parsed_cleaned = 0
        high_score = [record["row"] for record in records if (record["row"].get("int_score") or 0) >= 4]
        high_score_parsed = 0
        for text in texts:
            try:
                ast.parse(text)
                parsed_raw += 1
            except (SyntaxError, ValueError):
                pass
            try:
                ast.parse(strip_python_metadata_prefix(text))
                parsed_cleaned += 1
            except (SyntaxError, ValueError):
                pass
        for row in high_score:
            try:
                ast.parse(strip_python_metadata_prefix(row["content"]))
                high_score_parsed += 1
            except (SyntaxError, ValueError):
                pass
        common.update(
            int_score_counts=dict(sorted(score_counts.items())),
            score_ge_4_count=len(high_score),
            python_metadata_prefix_count=sum(strip_python_metadata_prefix(text) != text for text in texts),
            python_ast_parseable_raw=parsed_raw,
            python_ast_parseable_after_prefix_strip=parsed_cleaned,
            score_ge_4_python_ast_parseable_after_prefix_strip=high_score_parsed,
            distinct_repositories=len({record["row"].get("max_stars_repo_name") for record in records}),
        )
    else:
        common["han_character_fraction"] = round(
            sum(len(re.findall(r"[\u4e00-\u9fff]", text)) for text in texts)
            / max(1, sum(len(text) for text in texts)), 4
        )
        common["math_notation_marker"] = sum(
            bool(re.search(r"(?:\\\(|\\\[|\\frac|\$[^$]+\$|\b(?:equation|theorem|algebra)\b|[=±∑√])", text, re.I))
            for text in texts
        )
    return common


def run(output: Path, batches: int, batch_size: int, seed: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "small-agent-base-data-audit/0.1"})
    summary = {"seed": seed, "batches": batches, "batch_size": batch_size, "method": "uniform random row offsets; contiguous rows within each batch", "sources": {}}
    sample_path = output / "samples.jsonl"
    with sample_path.open("w", encoding="utf-8") as sample_file:
        for source_id, repo, config, text_key in SOURCES:
            info = get_json(session, f"https://huggingface.co/api/datasets/{repo}")
            base = {"dataset": repo, "config": config, "split": "train"}
            first = get_json(session, "https://datasets-server.huggingface.co/rows", {**base, "offset": 0, "length": 1})
            total = first["num_rows_total"]
            rng = random.Random(f"{seed}:{source_id}")
            offsets = sorted(rng.sample(range(total - batch_size + 1), batches))
            records = []
            print(f"{source_id}: {batches} x {batch_size} rows from {total:,}", flush=True)
            for offset in offsets:
                page = get_json(session, "https://datasets-server.huggingface.co/rows", {**base, "offset": offset, "length": batch_size})
                records.extend(page["rows"])
                print(f"  offset {offset}: {len(page['rows'])} rows", flush=True)
                time.sleep(1.5)
            for record in records:
                row = {key: record["row"].get(key) for key in ROW_FIELDS[source_id]}
                sample_file.write(json.dumps({"source_id": source_id, "row_idx": record["row_idx"], "row": row}, ensure_ascii=False) + "\n")
            summary["sources"][source_id] = {
                "repo_id": repo,
                "config": config,
                "observed_revision": info.get("sha"),
                "rows_total": total,
                "offsets": offsets,
                **summarize(records, text_key),
            }
            (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output / 'summary.json'} and local-only {sample_path}", flush=True)


def summarize_existing(output: Path) -> None:
    """Recompute metrics after changing audit logic, without fetching data again."""
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    groups: dict[str, list[dict]] = {source_id: [] for source_id, *_ in SOURCES}
    with (output / "samples.jsonl").open(encoding="utf-8") as sample_file:
        for line in sample_file:
            record = json.loads(line)
            groups[record["source_id"]].append(record)
    for source_id, _, _, text_key in SOURCES:
        if source_id in summary["sources"]:
            summary["sources"][source_id].update(summarize(groups[source_id], text_key))
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--summarize-existing", action="store_true")
    args = parser.parse_args()
    if args.batches < 1 or not 1 <= args.batch_size <= 100:
        parser.error("batches must be positive and batch-size must be 1..100")
    if args.summarize_existing:
        summarize_existing(args.output)
    else:
        run(args.output, args.batches, args.batch_size, args.seed)
