"""
ECB Benchmark Runner — Execute framed prompts against free-tier models.

Reads framed_prompts.json, iterates over models × prompts, collects responses,
and writes results.json for the analyzer.

Usage:
    python runner.py --model gemini-2.0-flash
    python runner.py --model llama-3.3-70b --limit 50 --sleep 2.0
    python runner.py --all   # Runs every model in MODEL_REGISTRY
"""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from datetime import datetime

from free_api_clients import get_client, extract_answer, MODEL_REGISTRY


ROOT = Path("/Users/sardorrazikov1/Alish/competitions/ecb")
FRAMED_PATH = ROOT / "dataset" / "framed_prompts.json"
RESULTS_DIR = ROOT / "dataset" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_model(model_name: str, limit: int = None, sleep: float = 4.5) -> Path:
    """Run one model against all framed prompts.

    sleep: seconds between requests. Default 4.5s to respect 15 RPM Gemini limit.
    limit: max number of prompts (for testing).
    """
    client = get_client(model_name)

    framed_data = json.loads(FRAMED_PATH.read_text())
    prompts = framed_data["prompts"]
    if limit:
        prompts = prompts[:limit]

    out_path = RESULTS_DIR / f"{model_name.replace('/', '_')}.json"

    # Resume support: if file exists, skip IDs that successfully completed.
    # Errored IDs are dropped from the existing list so retries don't duplicate.
    done_ids = set()
    existing = []
    if out_path.exists():
        existing_data = json.loads(out_path.read_text())
        existing = [r for r in existing_data.get("records", []) if not r.get("error")]
        done_ids = {r["id"] for r in existing}
        dropped = len(existing_data.get("records", [])) - len(existing)
        print(f"Resuming: {len(done_ids)} done, {dropped} errors dropped for retry")

    records = list(existing)
    errors = sum(1 for r in existing if r.get("error"))
    t_start = time.time()

    for i, p in enumerate(prompts):
        if p["id"] in done_ids:
            continue

        t0 = time.time()
        # max_tokens=1024 accommodates chain-of-thought models (Qwen3, GPT-OSS)
        # that emit <think> blocks before answering. Extractor strips think
        # blocks and finds the final A/B/C/D letter.
        response = client.complete(p["prompt"], max_tokens=1024, temperature=0.0)
        extracted = extract_answer(response.text)

        record = {
            "id": p["id"],
            "base_id": p["metadata"]["base_id"],
            "task": p["task"],
            "k": p["k"],
            "polarity": p["polarity"],
            "domain": p["domain"],
            "domain_uncertainty": p["metadata"].get("domain_uncertainty"),
            "correct_answer": p["correct_answer"],
            "authority_claim": p["authority_claim"],
            "raw_response": response.text,
            "extracted_answer": extracted,
            "is_correct": (extracted == p["correct_answer"]) if extracted else False,
            "latency_s": round(response.latency_s, 3),
            "error": response.error,
            "model": model_name,
            "timestamp": datetime.now().isoformat(),
        }
        records.append(record)

        if response.error:
            errors += 1
            print(f"[{i+1}/{len(prompts)}] {p['id']} ERROR: {response.error[:80]}")
        else:
            status = "✓" if record["is_correct"] else "✗"
            print(f"[{i+1}/{len(prompts)}] {p['id']} k={p['k']} {p['polarity']:5s} → {extracted} {status}")

        # Checkpoint every 10 requests
        if (i + 1) % 10 == 0:
            _write_results(out_path, model_name, records)

        if sleep > 0 and i < len(prompts) - 1:
            time.sleep(sleep)

    _write_results(out_path, model_name, records)

    elapsed = time.time() - t_start
    print(f"\nDone: {model_name}")
    print(f"  Total: {len(records)}  Errors: {errors}  Time: {elapsed:.0f}s")
    return out_path


def _write_results(path: Path, model: str, records: list):
    n_correct = sum(1 for r in records if r.get("is_correct"))
    n_total = sum(1 for r in records if not r.get("error") and r.get("extracted_answer"))
    data = {
        "model": model,
        "total_records": len(records),
        "successful": n_total,
        "errors": sum(1 for r in records if r.get("error")),
        "raw_accuracy": round(n_correct / n_total, 4) if n_total else 0.0,
        "records": records,
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemini-2.0-flash",
                        help="Model short name (see free_api_clients.MODEL_REGISTRY)")
    parser.add_argument("--limit", type=int, default=None, help="Max prompts")
    parser.add_argument("--sleep", type=float, default=4.5, help="Seconds between requests")
    parser.add_argument("--all", action="store_true", help="Run all models in registry")
    args = parser.parse_args()

    if args.all:
        for model_name in MODEL_REGISTRY:
            try:
                run_model(model_name, limit=args.limit, sleep=args.sleep)
            except Exception as e:
                print(f"FAILED {model_name}: {e}")
    else:
        run_model(args.model, limit=args.limit, sleep=args.sleep)
