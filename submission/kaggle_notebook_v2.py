"""
Epistemic Curie Benchmark — Kaggle Notebook (v2 — Extended Probe Range)

Upload the dataset "epistemic-curie-benchmark" (v2) to this notebook's data pane.

This notebook:
  1. Loads the ECB dataset (original + extended probe range)
  2. Defines the evaluation function
  3. Registers with the kaggle_benchmarks SDK if installed
  4. Displays precomputed results for both standard and extended range
  5. Generates figures: compliance curves, phase diagram, ODS bar chart
  6. Provides reproduction instructions

No model calls — all 3,640 results gathered offline via free-tier APIs.
"""

# ─────────────────────────────────────────────────
# Cell 1: Imports and configuration
# ─────────────────────────────────────────────────
import json
import re
import os
import math
from pathlib import Path
from collections import defaultdict

INPUT_DIR = None
_candidates = [
    Path("/kaggle/input/datasets/sardorrazikov/epistemic-curie-benchmark"),
    Path("/kaggle/input/epistemic-curie-benchmark"),
    Path("/kaggle/input"),
    Path("."),
]

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
    print("ERROR: Could not find seed_questions.json")
    print("Searched:", [str(c) for c in _candidates])
    if _kaggle_input.exists():
        for p in sorted(_kaggle_input.rglob("*")):
            print(f"  {p}")
    INPUT_DIR = Path(".")

print(f"Using INPUT_DIR: {INPUT_DIR}")

SEED_PATH = INPUT_DIR / "seed_questions.json"
FRAMED_PATH = INPUT_DIR / "framed_prompts.json"
FRAMED_FULL_PATH = INPUT_DIR / "framed_prompts_full.json"
RESULTS_PATH = INPUT_DIR / "analysis.json"
RESULTS_EXT_PATH = INPUT_DIR / "analysis_extended.json"


# ─────────────────────────────────────────────────
# Cell 2: Load the benchmark dataset
# ─────────────────────────────────────────────────
with open(SEED_PATH) as f:
    seed = json.load(f)

# Use full prompts if available, otherwise standard
if FRAMED_FULL_PATH.exists():
    with open(FRAMED_FULL_PATH) as f:
        framed_list = json.load(f)
    print(f"Loaded FULL framed prompts (standard + extended): {len(framed_list)}")
else:
    with open(FRAMED_PATH) as f:
        framed_data = json.load(f)
    framed_list = framed_data.get("prompts", framed_data)
    print(f"Loaded standard framed prompts: {len(framed_list)}")

print(f"Seed questions: {len(seed['questions'])}")
print(f"Total prompts: {len(framed_list)}")
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
    """Return 1.0 if response matches the correct letter, else 0.0."""
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
# Cell 4: Register with kaggle_benchmarks SDK
# ─────────────────────────────────────────────────
try:
    import kaggle_benchmarks as kb
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False
    print("kaggle_benchmarks SDK not installed — running in standalone mode.")

if SDK_AVAILABLE:
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
    for p in framed_list:
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
            metadata=p.get("metadata", {}),
        )

    benchmark = kb.Benchmark(
        name="epistemic-curie-benchmark",
        description=(
            "The Epistemic Curie Benchmark (ECB). Measures the critical "
            "authority threshold k* — the Epistemic Curie Temperature — at "
            "which LLM epistemic independence undergoes a phase transition "
            "into authority capture. Extended probe range to k=1.5 "
            "(universal institutional consensus). 3,640 measurements across "
            "7 frontier models from 5 laboratories."
        ),
        tasks=list(tasks.values()),
        evaluate=lambda response, item: evaluate_response(response, item),
    )

    print(f"\nRegistered benchmark: {benchmark.name}")
    print(f"  Tasks: {len(tasks)}")
    for t in tasks.values():
        print(f"    - {t.name}: {len(t.items)} items")


# ─────────────────────────────────────────────────
# Cell 5: Display precomputed results (standard range)
# ─────────────────────────────────────────────────
if RESULTS_PATH.exists():
    with open(RESULTS_PATH) as f:
        analysis = json.load(f)

    print(f"\n{'=' * 70}")
    print(f"ECB RESULTS — Standard Range (k ≤ 1.0)")
    print(f"{'=' * 70}")
    print(f"{'Model':<25} {'k*':>6} {'β':>6} {'ODS':>7} {'MI(bits)':>10} {'Phase'}")
    print("-" * 70)
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
# Cell 6: Display extended range results (k=1.25, 1.5)
# ─────────────────────────────────────────────────
if RESULTS_EXT_PATH.exists():
    with open(RESULTS_EXT_PATH) as f:
        ext_analysis = json.load(f)

    print(f"\n{'=' * 70}")
    print(f"ECB RESULTS — Extended Range (k ≤ 1.5) — 520 measurements/model")
    print(f"{'=' * 70}")
    print(f"{'Model':<25} {'k*':>7} {'β':>6} {'ODS':>7} {'Phase'}")
    print("-" * 60)
    for name, a in sorted(ext_analysis.items(), key=lambda x: -(x[1].get("ods") or 0)):
        k = a.get("k_star")
        b = a.get("beta")
        o = a.get("ods", 0)
        phase = a.get("phase", "—")
        k_s = f"{k:.3f}" if k is not None and abs(k) < 100 else ">2.0"
        b_s = f"{b:.2f}" if b is not None and abs(b) < 100 else "flat"
        print(f"{name:<25} {k_s:>7} {b_s:>6} {o:>7.3f}  {phase}")

    # Show compliance by k for each model
    print(f"\n{'=' * 70}")
    print("Wrong-authority compliance rate by k level")
    print(f"{'=' * 70}")
    header = f"{'Model':<20}"
    k_levels = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    for k in k_levels:
        header += f" {'k='+str(k):>8}"
    print(header)
    print("-" * 75)

    for name, a in sorted(ext_analysis.items(), key=lambda x: -(x[1].get("ods") or 0)):
        row = f"{name:<20}"
        compliance = a.get("compliance_by_k", {})
        for k in k_levels:
            d = compliance.get(str(k), {})
            rate = d.get("wrong_comply_rate", None)
            if rate is not None:
                row += f" {rate:>8.1%}"
            else:
                row += f" {'—':>8}"
        print(row)

else:
    print("\nNo extended analysis found (analysis_extended.json)")


# ─────────────────────────────────────────────────
# Cell 7: Generate figures
# ─────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not available — skipping figures")

if HAS_MPL and RESULTS_EXT_PATH.exists():
    with open(RESULTS_EXT_PATH) as f:
        ext = json.load(f)

    # --- Figure 1: Wrong-authority compliance curves ---
    fig, ax = plt.subplots(figsize=(12, 7))
    k_levels = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    colors = {
        "gpt-oss-120b": "#1f77b4",
        "llama-3.3-70b": "#ff7f0e",
        "kimi-k2": "#2ca02c",
        "qwen-3-32b": "#d62728",
        "gemma-3-27b": "#9467bd",
        "llama-3.1-8b": "#8c564b",
        "llama-4-scout": "#e377c2",
    }

    for name, a in ext.items():
        compliance = a.get("compliance_by_k", {})
        rates = []
        ks = []
        for k in k_levels:
            d = compliance.get(str(k), {})
            r = d.get("wrong_comply_rate")
            if r is not None:
                ks.append(k)
                rates.append(r)
        if ks:
            ax.plot(ks, rates, 'o-', label=name, color=colors.get(name, None),
                    linewidth=2, markersize=6)

    ax.set_xlabel("Authority Level (k)", fontsize=12)
    ax.set_ylabel("Wrong-Authority Compliance Rate", fontsize=12)
    ax.set_title("ECB: Authority Compliance Curves (Extended Range)", fontsize=14)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlim(0.1, 1.6)
    ax.set_ylim(-0.05, 0.85)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% threshold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("compliance_curves_extended.png", dpi=200)
    plt.close()
    print("Saved: compliance_curves_extended.png")

    # --- Figure 2: ODS Bar Chart ---
    fig, ax = plt.subplots(figsize=(12, 7))
    models_sorted = sorted(ext.items(), key=lambda x: x[1].get("ods", 0), reverse=True)
    names = [m[0] for m in models_sorted]
    ods_vals = [m[1].get("ods", 0) for m in models_sorted]
    bar_colors = [colors.get(n, "#333333") for n in names]

    bars = ax.barh(names, ods_vals, color=bar_colors, edgecolor='white')
    ax.set_xlabel("Optimal Discrimination Score (ODS)", fontsize=12)
    ax.set_title("ECB: Epistemic Wisdom Ranking (Extended Range)", fontsize=14)
    ax.set_xlim(0, 1.0)
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5)
    for bar, val in zip(bars, ods_vals):
        ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va='center', fontsize=10)
    plt.tight_layout()
    plt.savefig("ods_bar_extended.png", dpi=200)
    plt.close()
    print("Saved: ods_bar_extended.png")

    # --- Figure 3: Phase Diagram ---
    fig, ax = plt.subplots(figsize=(12, 7))
    phase_colors = {
        "superconducting": "#2196F3",
        "near_superconducting": "#4CAF50",
        "ferromagnetic": "#FF9800",
        "paramagnetic": "#F44336",
        "transitioning": "#9C27B0",
    }

    for name, a in ext.items():
        k_star = a.get("k_star")
        ods = a.get("ods", 0)
        phase = a.get("phase", "unknown")
        if k_star is None or abs(k_star) > 100:
            k_star = 2.5  # Plot off-scale models at 2.5
        c = phase_colors.get(phase, "#666666")
        ax.scatter(k_star, ods, s=200, c=c, edgecolors='black', zorder=5)
        ax.annotate(name.replace("-", "\n"), (k_star, ods),
                    textcoords="offset points", xytext=(10, 5), fontsize=8)

    ax.axvline(x=0.85, color='orange', linestyle='--', alpha=0.5, label='Ferromagnetic boundary')
    ax.axvline(x=1.5, color='blue', linestyle='--', alpha=0.5, label='Extended probe limit')
    ax.set_xlabel("k* (Epistemic Curie Temperature)", fontsize=12)
    ax.set_ylabel("ODS (Epistemic Wisdom)", fontsize=12)
    ax.set_title("ECB Cognitive Phase Diagram", fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("phase_diagram_extended.png", dpi=200)
    plt.close()
    print("Saved: phase_diagram_extended.png")


# ─────────────────────────────────────────────────
# Cell 8: Summary statistics
# ─────────────────────────────────────────────────
print(f"""
{'=' * 60}
ECB BENCHMARK SUMMARY
{'=' * 60}
Seed questions:       40 (4 tasks × 10 each)
Standard variants:    360 (40 × 9: 1 neutral + 4 right + 4 wrong)
Extended variants:    160 (40 × 4: 2 k-levels × 2 polarities)
Total prompts:        520

Models tested:        7 (from 5 laboratories)
Standard measurements: 2,520 (7 models × 360 prompts)
Extended measurements: 1,120 (7 models × 160 prompts)
Total measurements:   3,640

Key findings:
  - Llama-4-Scout: first measured k* in ferromagnetic phase (k*=0.85)
  - Qwen-3-32B: phase transition at k=1.5 (52% wrong compliance)
  - Gemma-3-27B: phase transition at k=1.5 (55% wrong compliance)
  - GPT-OSS-120B: genuinely superconducting (k=1.5: <15% compliance)
  - ODS gap: 0.909 (GPT-OSS) vs 0.307 (Llama-4-Scout)
{'=' * 60}
""")


# ─────────────────────────────────────────────────
# Cell 9: Reproduction instructions
# ─────────────────────────────────────────────────
print("""
Reproducing ECB from scratch:

1. Obtain free API keys:
   - Google AI Studio: https://aistudio.google.com/app/apikey
   - Groq:             https://console.groq.com/keys

2. Export environment variables:
   export GOOGLE_API_KEY=...
   export GROQ_API_KEY=...

3. Run benchmark (standard range):
   python runner.py --model gemma-3-27b --sleep 4
   python runner.py --model llama-3.3-70b --sleep 2.5

4. Run extended range:
   python run_extended.py --model gemma-3-27b
   python run_extended.py --model llama-3.3-70b

5. Analyze results:
   python analyze_extended.py

Full source: linked GitHub repository.
""")
