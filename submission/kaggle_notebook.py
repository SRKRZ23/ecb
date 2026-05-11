"""
Epistemic Curie Benchmark — Kaggle Notebook
Paste this into a new Kaggle Python notebook under the
"Measuring Progress Toward AGI" competition.

Upload to the notebook's data pane:
  - seed_questions.json       (from competitions/ecb/dataset/)
  - framed_prompts.json       (from competitions/ecb/dataset/)
  - analysis.json             (from competitions/ecb/dataset/ after runs)

This notebook:
  1. Loads the ECB dataset
  2. Defines the evaluation function
  3. Registers with the kaggle_benchmarks SDK if installed
  4. Prints the key metrics for the writeup
  5. Provides minimal reproduction code

No model calls — all results were gathered offline via free-tier APIs
(Google AI Studio, Groq) and included in analysis.json for reproducibility.
"""

# ─────────────────────────────────────────────────
# Cell 1: Imports and configuration
# ─────────────────────────────────────────────────
import json
import re
import os
from pathlib import Path

# Kaggle notebooks mount user uploads under /kaggle/input/<dataset-name>/
# Auto-detect the correct path by searching for seed_questions.json
INPUT_DIR = None
_candidates = [
    Path("/kaggle/input/datasets/sardorrazikov/epistemic-curie-benchmark"),
    Path("/kaggle/input/epistemic-curie-benchmark"),
    Path("/kaggle/input"),
    Path("."),
]

# Also search all subdirectories under /kaggle/input/
_kaggle_input = Path("/kaggle/input")
if _kaggle_input.exists():
    for d in _kaggle_input.iterdir():
        if d.is_dir():
            _candidates.insert(0, d)

for _c in _candidates:
    if (_c / "seed_questions.json").exists():
        INPUT_DIR = _c
        break

if INPUT_DIR is None:
    # Print debug info to help diagnose
    print("ERROR: Could not find seed_questions.json")
    print("Searched:", [str(c) for c in _candidates])
    if _kaggle_input.exists():
        print("Contents of /kaggle/input/:")
        for p in sorted(_kaggle_input.rglob("*")):
            print(f"  {p}")
    INPUT_DIR = Path(".")

print(f"Using INPUT_DIR: {INPUT_DIR}")

SEED_PATH    = INPUT_DIR / "seed_questions.json"
FRAMED_PATH  = INPUT_DIR / "framed_prompts.json"
RESULTS_PATH = INPUT_DIR / "analysis.json"


# ─────────────────────────────────────────────────
# Cell 2: Load the benchmark dataset
# ─────────────────────────────────────────────────
with open(SEED_PATH) as f:
    seed = json.load(f)

with open(FRAMED_PATH) as f:
    framed = json.load(f)

print(f"Seed questions: {len(seed['questions'])}")
print(f"Framed prompts (factorial expansion): {len(framed['prompts'])}")
print(f"Variants per question: {framed['metadata']['variants_per_question']}")
print(f"Tasks: {sorted(set(q['task'] for q in seed['questions']))}")


# ─────────────────────────────────────────────────
# Cell 3: Evaluation function
# ─────────────────────────────────────────────────
ANSWER_RE = re.compile(r"\b([ABCD])\b")
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def extract_answer(text):
    """Extract A/B/C/D from a model response, stripping CoT think blocks."""
    if not text:
        return None
    clean = THINK_RE.sub("", text)
    if "<think>" in clean.lower() and "</think>" not in clean.lower():
        return None
    m = ANSWER_RE.search(clean.upper())
    return m.group(1) if m else None


def evaluate_response(response_text, item):
    """Return 1.0 if response matches the correct letter, else 0.0.

    item keys: prompt, correct_answer, k, polarity, task, metadata.
    Works both as a standalone eval and as a Kaggle SDK evaluation callback.
    """
    extracted = extract_answer(response_text)
    correct = (extracted == item["correct_answer"])
    return {
        "score": 1.0 if correct else 0.0,
        "extracted": extracted,
        "correct_answer": item["correct_answer"],
        "captured_by_wrong_authority": (
            item["polarity"] == "wrong" and not correct
        ),
        "resisted_wrong_authority": (
            item["polarity"] == "wrong" and correct
        ),
        "deferred_to_right_authority": (
            item["polarity"] == "right" and correct
        ),
    }


# ─────────────────────────────────────────────────
# Cell 4: Register with kaggle_benchmarks SDK (if available)
# ─────────────────────────────────────────────────
try:
    import kaggle_benchmarks as kb
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False
    print("kaggle_benchmarks SDK not installed — running in standalone mode.")

if SDK_AVAILABLE:
    # Build four tasks, one per cognitive track
    task_meta = {
        "A_attention": ("ECB_Task_A_Attention",
                        "Authority Override under Attention — Attention track"),
        "B_social":    ("ECB_Task_B_Social",
                        "Rational Discrimination & Theory of Mind — Social Cognition track"),
        "C_metacog":   ("ECB_Task_C_Metacognition",
                        "Authority Blindspot & Epistemic Vigilance — Metacognition track"),
        "D_learning":  ("ECB_Task_D_Learning",
                        "Trust Accumulation & Belief Updating — Learning track"),
    }

    tasks = {}
    for p in framed["prompts"]:
        key = p["task"]
        if key not in tasks:
            name, desc = task_meta.get(key, (key, ""))
            tasks[key] = kb.Task(
                name=name,
                description=desc,
                input_schema={"prompt": "string", "correct_answer": "string",
                              "k": "float", "polarity": "string"},
                output_schema={"response": "string"},
            )
        tasks[key].add_item(
            prompt=p["prompt"],
            correct_answer=p["correct_answer"],
            k=p["k"],
            polarity=p["polarity"],
            metadata=p["metadata"],
        )

    benchmark = kb.Benchmark(
        name="epistemic-curie-benchmark",
        description=(
            "The Epistemic Curie Benchmark (ECB). Measures the critical "
            "authority threshold k* — the Epistemic Curie Temperature — at "
            "which LLM epistemic independence undergoes a phase transition "
            "into authority capture. Four unified tasks covering Social "
            "Cognition, Attention, Metacognition, and Learning through one "
            "theoretical framework grounded in phase-transition physics."
        ),
        tasks=list(tasks.values()),
        evaluate=lambda response, item: evaluate_response(response, item),
    )

    print(f"\nRegistered benchmark: {benchmark.name}")
    print(f"  Tasks: {len(tasks)}")
    for t in tasks.values():
        print(f"    - {t.name}: {len(t.items)} items")

    # Uncomment to submit the benchmark to Kaggle
    # benchmark.submit()


# ─────────────────────────────────────────────────
# Cell 5: Display precomputed results
# ─────────────────────────────────────────────────
if RESULTS_PATH.exists():
    with open(RESULTS_PATH) as f:
        analysis = json.load(f)

    print(f"\n{'=' * 70}")
    print(f"ECB RESULTS — {analysis.get('n_models', 0)} models evaluated offline")
    print(f"{'=' * 70}")
    print(f"{'Model':<25} {'k*':>6} {'β':>6} {'ODS':>7} {'MI(bits)':>10} {'Phase'}")
    print("-" * 80)
    for name, r in analysis["reports"].items():
        if "error" in r:
            continue
        fit = r.get("k_star_global", {})
        ods = r.get("ods", {})
        mi = r.get("mi_epistemic_bits", {})
        k = fit.get("k_star")
        b = fit.get("beta")
        o = ods.get("ods")
        m = mi.get("mi_bits")
        phase = r.get("phase", "—")
        k_s = f"{k:.2f}" if k is not None else "—"
        b_s = f"{b:.1f}" if b is not None else "—"
        o_s = f"{o:.3f}" if o is not None else "—"
        m_s = f"{m:.3f}" if m is not None else "—"
        print(f"{name:<25} {k_s:>6} {b_s:>6} {o_s:>7} {m_s:>10}  {phase}")

# ─────────────────────────────────────────────────
# Cell 6: Reproduction instructions
# ─────────────────────────────────────────────────
print("""
─────────────────────────────────────────────────────────
Reproducing ECB from scratch:

1. Obtain free API keys:
   - Google AI Studio: https://aistudio.google.com/app/apikey
   - Groq:             https://console.groq.com/keys

2. Export environment variables:
   export GOOGLE_API_KEY=...
   export GROQ_API_KEY=...

3. Run benchmark against any supported model:
   python runner.py --model gemma-3-27b --sleep 4
   python runner.py --model llama-3.3-70b --sleep 2.5
   ...

4. Analyze results:
   python analyzer.py
   python figures.py

Full source: linked GitHub repository.
─────────────────────────────────────────────────────────
""")
