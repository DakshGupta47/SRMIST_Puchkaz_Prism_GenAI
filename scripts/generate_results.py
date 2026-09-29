"""
generate_results.py

Runs troubleshooting queries through Pipeline directly and writes one JSON
object per line to results.jsonl in the required schema.

Input sources:
  - data/input.txt: 20 reference complaints paired with data/siis_responses.json
  - data/unseen_scenarios.txt: unseen complaints (if present) evaluated with empty
    SIIS context

Error handling:
  - If a row fails or raises an exception, logs the error and emits a valid fallback
    record instead of crashing.

Output:
  - results.jsonl (one JSON object per line)
  - Summary printed to stdout: row count, no_match count, average latency, cache hits.

Usage:
  python scripts/generate_results.py [--output results.jsonl]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

# Ensure src/ is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
DATA_DIR = ROOT_DIR / "data"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from src.pipeline import Pipeline
    from src.stage2_deeplinks import map_deeplinks
    from src.structure_extraction import structure_extraction
except ImportError:
    from pipeline import Pipeline  # type: ignore[no-redef]  # noqa: E402
    from stage2_deeplinks import map_deeplinks  # type: ignore[no-redef]  # noqa: E402
    from structure_extraction import structure_extraction  # type: ignore[no-redef]  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("generate_results")


def _load_lines(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_siis_responses(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("responses", [])
    except Exception as exc:
        logger.warning("Failed to load SIIS responses from %s: %s", path, exc)
        return []


def _create_fallback_record(
    query: str,
    siis_response: dict | None,
    latency_ms: float,
    model_name: str = "fallback",
) -> dict[str, Any]:
    has_siis = bool(
        siis_response
        and (
            not isinstance(siis_response, dict)
            or any(str(v).strip() for v in siis_response.values())
        )
    )
    fallback_reason = "no_match" if has_siis else "no_siis_context"
    return {
        "query": query,
        "query_variations": [query],
        "response": {
            "contexts": [],
            "fallback": fallback_reason,
        },
        "meta": {
            "cache_hit": False,
            "similarity": None,
            "latency_ms": round(latency_ms, 2),
            "model": model_name,
            "cost_usd": 0.0,
        },
        "enrichment": {
            "device": "unknown",
            "symptom_category": "unclassified",
            "device_confidence": 0.0,
            "symptom_confidence": 0.0,
            "overall_confidence": 0.0,
            "is_low_confidence": True,
            "classification_source": "fallback",
        },
    }


def generate_results(
    input_file: Path = DATA_DIR / "input.txt",
    unseen_file: Path = DATA_DIR / "unseen_scenarios.txt",
    siis_file: Path = DATA_DIR / "siis_responses.json",
    output_file: Path = ROOT_DIR / "results.jsonl",
) -> dict[str, Any]:
    pipeline = Pipeline(
        stage1_fn=structure_extraction,
        stage2_fn=map_deeplinks,
    )

    input_queries = _load_lines(input_file)
    unseen_queries = _load_lines(unseen_file)
    siis_records = _load_siis_responses(siis_file)

    total_rows = 0
    no_match_count = 0
    no_siis_context_count = 0
    cache_hits = 0
    total_latency_ms = 0.0
    records: list[dict[str, Any]] = []

    # 1. Process data/input.txt queries paired with SIIS responses
    for idx, query in enumerate(input_queries):
        siis_payload = (
            siis_records[idx].get("siis_response", {})
            if idx < len(siis_records)
            else {}
        )
        t0 = time.perf_counter()
        try:
            res = pipeline.run(query, siis_payload)
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.exception("Error processing input.txt row %d ('%s'): %s", idx + 1, query[:50], exc)
            model_name = getattr(pipeline.llm_client, "model_name", "fallback")
            res = _create_fallback_record(query, siis_payload, elapsed_ms, model_name=model_name)

        records.append(res)
        total_rows += 1
        latency = res.get("meta", {}).get("latency_ms", 0.0)
        total_latency_ms += latency
        if res.get("meta", {}).get("cache_hit"):
            cache_hits += 1

        fallback = res.get("response", {}).get("fallback")
        if fallback == "no_match":
            no_match_count += 1
        elif fallback == "no_siis_context":
            no_siis_context_count += 1

    # 2. Process data/unseen_scenarios.txt queries (if present)
    for idx, query in enumerate(unseen_queries):
        siis_payload: dict[str, Any] = {}
        t0 = time.perf_counter()
        try:
            res = pipeline.run(query, siis_payload)
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.exception("Error processing unseen row %d ('%s'): %s", idx + 1, query[:50], exc)
            model_name = getattr(pipeline.llm_client, "model_name", "fallback")
            res = _create_fallback_record(query, siis_payload, elapsed_ms, model_name=model_name)

        records.append(res)
        total_rows += 1
        latency = res.get("meta", {}).get("latency_ms", 0.0)
        total_latency_ms += latency
        if res.get("meta", {}).get("cache_hit"):
            cache_hits += 1

        fallback = res.get("response", {}).get("fallback")
        if fallback == "no_match":
            no_match_count += 1
        elif fallback == "no_siis_context":
            no_siis_context_count += 1

    # Write output to results.jsonl
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    avg_latency = (total_latency_ms / total_rows) if total_rows > 0 else 0.0

    summary = {
        "row_count": total_rows,
        "input_rows": len(input_queries),
        "unseen_rows": len(unseen_queries),
        "no_match_count": no_match_count,
        "no_siis_context_count": no_siis_context_count,
        "total_fallback_count": no_match_count + no_siis_context_count,
        "average_latency_ms": round(avg_latency, 2),
        "cache_hits": cache_hits,
        "cache_hit_rate": round(cache_hits / total_rows, 4) if total_rows > 0 else 0.0,
        "output_file": str(output_file.name),
    }

    print("\n" + "=" * 60)
    print("RESULTS GENERATION SUMMARY")
    print("=" * 60)
    print(f"Row count              : {summary['row_count']} ({summary['input_rows']} input, {summary['unseen_rows']} unseen)")
    print(f"No-match count         : {summary['no_match_count']}")
    print(f"No-SIIS-context count  : {summary['no_siis_context_count']}")
    print(f"Total fallback count   : {summary['total_fallback_count']}")
    print(f"Average latency        : {summary['average_latency_ms']} ms")
    print(f"Cache hits             : {summary['cache_hits']} ({summary['cache_hit_rate']:.1%})")
    print(f"Output file            : {summary['output_file']}")
    print("=" * 60 + "\n")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate results.jsonl via Pipeline")
    parser.add_argument("--output", type=Path, default=ROOT_DIR / "results.jsonl", help="Path to output results.jsonl")
    parser.add_argument("--input", type=Path, default=DATA_DIR / "input.txt", help="Path to input.txt")
    parser.add_argument("--unseen", type=Path, default=DATA_DIR / "unseen_scenarios.txt", help="Path to unseen_scenarios.txt")
    parser.add_argument("--siis", type=Path, default=DATA_DIR / "siis_responses.json", help="Path to siis_responses.json")
    args = parser.parse_args()

    generate_results(
        input_file=args.input,
        unseen_file=args.unseen,
        siis_file=args.siis,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
