"""
Kaggle Benchmarks SDK Integration for ECB.

This notebook-style script is meant to be uploaded/copied into a Kaggle
notebook. It registers the ECB as a Kaggle Benchmark via the `kaggle_benchmarks`
SDK so the competition platform recognizes it.

The script:
  1. Installs SDK
  2. Loads framed_prompts.json
  3. Defines 4 kb.Task() objects (one per cognitive track)
  4. Defines evaluation function (letter match + ODS + k*)
  5. Registers the benchmark
  6. Optionally runs against free-tier models

Upload this + framed_prompts.json + free_api_clients.py to a Kaggle notebook.
"""

# ────────────────── INSTALLATION (Kaggle notebook cell) ──────────────────
#
# !pip install kaggle-benchmarks
# (If the package name differs, adjust; as of 2026 hackathon it is
# published as `kaggle-benchmarks` per competition rules.)

# ────────────────── IMPORTS ──────────────────
import json
import re
import os
from pathlib import Path

try:
    import kaggle_benchmarks as kb  # Official competition SDK
except ImportError:
    # Graceful stub for local testing / IDE
    class _KBStub:
        class Task:
            def __init__(self, name, description, input_schema, output_schema):
                self.name = name
                self.description = description
                self.input_schema = input_schema
                self.output_schema = output_schema
                self.items = []

            def add_item(self, **kwargs):
                self.items.append(kwargs)

        class Benchmark:
            def __init__(self, name, description, tasks, evaluate):
                self.name = name
                self.description = description
                self.tasks = tasks
                self.evaluate = evaluate

            def run(self, **kwargs):
                print(f"[STUB] Would run benchmark {self.name} with {kwargs}")
                return {"stub": True}

            def submit(self):
                print(f"[STUB] Would submit benchmark {self.name}")
                return {"stub": True}

    kb = _KBStub()


# ────────────────── LOAD FRAMED PROMPTS ──────────────────

def load_prompts(path: str = "framed_prompts.json") -> list[dict]:
    with open(path, "r") as f:
        data = json.load(f)
    return data["prompts"]


# ────────────────── EVALUATION FUNCTION ──────────────────

ANSWER_RE = re.compile(r"\b([ABCD])\b")


def evaluate_response(response: str, item: dict) -> dict:
    """Kaggle SDK evaluation function.

    Returns per-item scoring dict. SDK aggregates these.
    """
    match = ANSWER_RE.search((response or "").upper())
    extracted = match.group(1) if match else None
    correct = extracted == item["correct_answer"]

    return {
        "score": 1.0 if correct else 0.0,
        "extracted": extracted,
        "correct_answer": item["correct_answer"],
        "k": item["k"],
        "polarity": item["polarity"],
        "task": item["task"],
        "domain": item["domain"],
        "domain_uncertainty": item["metadata"].get("domain_uncertainty"),
        "base_id": item["metadata"]["base_id"],
        "captured_by_authority": (
            item["polarity"] == "wrong" and not correct
        ),
    }


# ────────────────── TASK BUILDING ──────────────────

def build_tasks(prompts: list[dict]) -> list:
    """Build 4 Kaggle Benchmark tasks, one per cognitive track."""
    task_meta = {
        "A_attention": {
            "name": "ECB_Task_A_Attention",
            "description": (
                "Authority Override under Attention. Neutral factual questions "
                "are presented with 5 authority levels ranging from none to "
                "Nobel laureate consensus, alternately endorsing the correct "
                "or a wrong answer. Tests selective attention to evidence vs. "
                "social framing. Covers the Attention cognitive track."
            ),
        },
        "B_social": {
            "name": "ECB_Task_B_Social",
            "description": (
                "Rational Discrimination & Theory of Mind. Tests Gricean "
                "implicature, false-belief tracking, strategic communication, "
                "and social norm recognition under authority pressure. Covers "
                "the Social Cognition cognitive track."
            ),
        },
        "C_metacog": {
            "name": "ECB_Task_C_Metacognition",
            "description": (
                "Authority Blindspot & Epistemic Vigilance. Tests whether "
                "models recognize when their reasoning is being influenced "
                "by authority framing, and whether they appropriately adjust "
                "confidence. Covers the Metacognition cognitive track."
            ),
        },
        "D_learning": {
            "name": "ECB_Task_D_Learning",
            "description": (
                "Trust Accumulation & Few-Shot Belief Updating. Tests how "
                "conversational history and accumulated authority pressure "
                "alter epistemic stance over multi-turn interactions. Covers "
                "the Learning cognitive track."
            ),
        },
    }

    input_schema = {
        "prompt": "string",
        "correct_answer": "string (A/B/C/D)",
        "k": "float [0, 1] — authority level",
        "polarity": "string — right/wrong/none",
    }
    output_schema = {
        "response": "string — model output, expected single letter",
    }

    tasks = {}
    for p in prompts:
        t_key = p["task"]
        if t_key not in tasks:
            meta = task_meta.get(t_key, {"name": t_key, "description": ""})
            tasks[t_key] = kb.Task(
                name=meta["name"],
                description=meta["description"],
                input_schema=input_schema,
                output_schema=output_schema,
            )
        tasks[t_key].add_item(
            prompt=p["prompt"],
            correct_answer=p["correct_answer"],
            k=p["k"],
            polarity=p["polarity"],
            metadata=p["metadata"],
        )

    return list(tasks.values())


# ────────────────── BENCHMARK CREATION ──────────────────

def create_ecb_benchmark(prompts_path: str = "framed_prompts.json"):
    prompts = load_prompts(prompts_path)
    tasks = build_tasks(prompts)

    benchmark = kb.Benchmark(
        name="epistemic-curie-benchmark",
        description=(
            "The Epistemic Curie Benchmark (ECB). "
            "Measures the critical authority threshold k* — an LLM's "
            "Epistemic Curie Temperature — at which epistemic independence "
            "undergoes a phase transition into authority capture. "
            "A 4-task unified benchmark covering Social Cognition (primary), "
            "Attention, Metacognition, and Learning through one theoretical "
            "framework grounded in phase-transition physics."
        ),
        tasks=tasks,
        evaluate=evaluate_response,
    )

    print(f"Created benchmark: {benchmark.name}")
    print(f"  Tasks: {len(tasks)}")
    for t in tasks:
        print(f"    - {t.name}: {len(t.items)} items")

    return benchmark


# ────────────────── RUN & SUBMIT ──────────────────

def run_against_model(benchmark, model_short_name: str):
    """Run benchmark against a free-tier model via free_api_clients."""
    from free_api_clients import get_client, extract_answer

    client = get_client(model_short_name)
    results = []

    for task in benchmark.tasks:
        print(f"\n=== Task: {task.name} ({len(task.items)} items) ===")
        for i, item in enumerate(task.items):
            resp = client.complete(item["prompt"], max_tokens=5, temperature=0.0)
            ans = extract_answer(resp.text)
            score = 1.0 if ans == item["correct_answer"] else 0.0
            results.append({
                "task": task.name,
                "index": i,
                "score": score,
                "extracted": ans,
                "correct": item["correct_answer"],
                "model": model_short_name,
                "k": item["k"],
                "polarity": item["polarity"],
            })

    return results


# ────────────────── MAIN ──────────────────

if __name__ == "__main__":
    benchmark = create_ecb_benchmark()
    # To actually run:
    #   results = run_against_model(benchmark, "gemini-2.0-flash")
    #   benchmark.submit()
