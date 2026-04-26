"""
Run extended probe range (k=1.25, k=1.5) through available models.
Uses the same runner infrastructure but reads framed_prompts_extended.json.
"""

import json
import time
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from free_api_clients import get_client, extract_answer

ROOT = Path(__file__).parent.parent
EXTENDED_PATH = ROOT / "dataset" / "framed_prompts_extended.json"
RESULTS_DIR = ROOT / "dataset" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Models available with current API keys
# Groq models need GROQ_API_KEY, Gemini models need GOOGLE_API_KEY
MODELS_TO_RUN = [
    # Google AI Studio (GOOGLE_API_KEY set)
    "gemma-3-27b",
    # Groq (need GROQ_API_KEY)
    # "llama-3.3-70b",
    # "llama-3.1-8b",
    # "llama-4-scout",
    # "qwen-3-32b",
    # "gpt-oss-120b",
    # "kimi-k2",
]


def run_extended(model_name: str, sleep: float = 4.5):
    """Run extended prompts for one model."""
    client = get_client(model_name)
    prompts = json.loads(EXTENDED_PATH.read_text())

    out_path = RESULTS_DIR / f"{model_name.replace('/', '_')}_extended.json"

    # Resume support
    done_ids = set()
    existing = []
    if out_path.exists():
        existing_data = json.loads(out_path.read_text())
        existing = [r for r in existing_data.get("records", []) if not r.get("error")]
        done_ids = {r["id"] for r in existing}
        print(f"Resuming: {len(done_ids)} done")

    records = list(existing)
    errors = 0

    for i, p in enumerate(prompts):
        if p["id"] in done_ids:
            continue

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
            status = "V" if record["is_correct"] else "X"
            print(f"[{i+1}/{len(prompts)}] {p['id']} k={p['k']} {p['polarity']:5s} -> {extracted} {status}")

        # Checkpoint every 10
        if (i + 1) % 10 == 0:
            _save(out_path, model_name, records)

        if sleep > 0:
            time.sleep(sleep)

    _save(out_path, model_name, records)
    print(f"\nDone: {model_name} — {len(records)} records, {errors} errors")


def _save(path, model, records):
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Specific model to run")
    parser.add_argument("--sleep", type=float, default=4.5)
    args = parser.parse_args()

    if args.model:
        run_extended(args.model, sleep=args.sleep)
    else:
        for m in MODELS_TO_RUN:
            try:
                print(f"\n{'='*50}\nRunning {m}\n{'='*50}")
                run_extended(m, sleep=args.sleep)
            except Exception as e:
                print(f"FAILED {m}: {e}")
