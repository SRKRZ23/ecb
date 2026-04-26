"""
ECB Model Extension Runner.

Runs ECB against new models not in the original 7-model paper set,
then re-runs analysis to produce updated results.

Target new models (priority order):
  1. claude-haiku   — Anthropic, requires ANTHROPIC_API_KEY
  2. deepseek-v3    — DeepSeek, requires DEEPSEEK_API_KEY   (~$0.003/run)
  3. mistral-small  — Mistral,  requires MISTRAL_API_KEY
  4. deepseek-r1    — chain-of-thought variant (for comparison)

Usage:
    # Run all new models (will skip if results already exist)
    python extend_models.py

    # Run single model
    python extend_models.py --model claude-haiku

    # Force re-run even if results exist
    python extend_models.py --model deepseek-v3 --force

    # Check which models have results / are missing
    python extend_models.py --status

    # Re-run full analysis after adding models
    python extend_models.py --analyze-only
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path("/Users/sardorrazikov1/Alish/competitions/ecb")
RESULTS_DIR = ROOT / "dataset" / "results"
ANALYSIS_OUT = ROOT / "dataset" / "analysis_extended.json"

# Models to add for paper extension.
# Format: (short_name, display_name, env_var_needed, estimated_cost_usd)
NEW_MODELS = [
    ("claude-haiku",  "Claude Haiku 4.5 (Anthropic)",  "ANTHROPIC_API_KEY", 0.01),
    ("deepseek-v3",   "DeepSeek-V3 (DeepSeek)",        "DEEPSEEK_API_KEY",  0.003),
    ("mistral-small", "Mistral Small (Mistral AI)",     "MISTRAL_API_KEY",   0.01),
    ("deepseek-r1",   "DeepSeek-R1 (chain-of-thought)", "DEEPSEEK_API_KEY",  0.02),
]

# Sleep between requests per model (adjust based on rate limits)
SLEEP_PER_MODEL = {
    "claude-haiku":  1.0,   # Anthropic: generous rate limits
    "deepseek-v3":   0.5,   # DeepSeek: fast, generous
    "mistral-small": 1.0,   # Mistral: standard
    "deepseek-r1":   2.0,   # R1 is slower (chain-of-thought)
}


def check_api_key(env_var: str) -> bool:
    import os
    return bool(os.environ.get(env_var, ""))


def results_exist(model_name: str) -> bool:
    path = RESULTS_DIR / f"{model_name.replace('/', '_')}.json"
    if not path.exists():
        return False
    data = json.loads(path.read_text())
    n = data.get("successful", 0)
    return n >= 300  # require at least 300 successful completions


def print_status():
    print("\n=== ECB Model Coverage Status ===\n")
    print(f"{'Model':<40} {'Status':<15} {'Env var':<25} {'Est. cost'}")
    print("─" * 90)

    # Original 7 models
    original = [
        "llama-3.1-8b", "llama-3.3-70b", "llama-4-scout",
        "gemma-3-27b", "qwen-3-32b", "kimi-k2", "gpt-oss-120b",
    ]
    for m in original:
        exists = results_exist(m)
        status = "✓ DONE" if exists else "✗ MISSING"
        print(f"  {m:<38} {status:<15} {'(original set)':<25} —")

    print()
    for model_name, display, env_var, cost in NEW_MODELS:
        exists = results_exist(model_name)
        has_key = check_api_key(env_var)
        if exists:
            status = "✓ DONE"
        elif has_key:
            status = "→ READY"
        else:
            status = "⚠ NO KEY"
        print(f"  {display:<38} {status:<15} {env_var:<25} ~${cost:.3f}")

    total_original = sum(1 for m in original if results_exist(m))
    total_new = sum(1 for m, _, _, _ in NEW_MODELS if results_exist(m))
    print(f"\nOriginal: {total_original}/7  |  New: {total_new}/{len(NEW_MODELS)}  |  "
          f"Total: {total_original + total_new}/{7 + len(NEW_MODELS)}")


def run_new_models(force: bool = False, target: str = None):
    sys.path.insert(0, str(ROOT / "code"))
    from runner import run_model

    to_run = NEW_MODELS if target is None else [
        (m, d, e, c) for m, d, e, c in NEW_MODELS if m == target
    ]

    if not to_run:
        print(f"Unknown model: {target}. Available: {[m for m, *_ in NEW_MODELS]}")
        sys.exit(1)

    for model_name, display, env_var, cost in to_run:
        print(f"\n{'='*60}")
        print(f"Model: {display}")
        print(f"Env:   {env_var}")
        print(f"Cost:  ~${cost:.3f}")
        print(f"{'='*60}")

        if not check_api_key(env_var):
            print(f"SKIP: {env_var} not set. Set it with:")
            print(f"  export {env_var}=your_api_key_here")
            continue

        if results_exist(model_name) and not force:
            print(f"SKIP: Results already exist. Use --force to re-run.")
            continue

        sleep = SLEEP_PER_MODEL.get(model_name, 2.0)
        try:
            run_model(model_name, sleep=sleep)
        except Exception as e:
            print(f"ERROR running {model_name}: {e}")


def run_analysis():
    sys.path.insert(0, str(ROOT / "code"))
    from analyzer import analyze_all

    print(f"\nRunning analysis on all results in {RESULTS_DIR}")
    result = analyze_all(RESULTS_DIR, ANALYSIS_OUT)
    print(f"\nAnalysis written to: {ANALYSIS_OUT}")
    print(f"Total models analyzed: {result['n_models']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extend ECB to new models")
    parser.add_argument("--model", default=None, help="Run single model by short name")
    parser.add_argument("--force", action="store_true", help="Re-run even if results exist")
    parser.add_argument("--status", action="store_true", help="Show model coverage status")
    parser.add_argument("--analyze-only", action="store_true", help="Re-run analysis only")
    args = parser.parse_args()

    if args.status:
        print_status()
    elif args.analyze_only:
        run_analysis()
    else:
        run_new_models(force=args.force, target=args.model)
        print("\n" + "="*60)
        run_analysis()
        print("\nDone. Check analysis_extended.json for updated results.")
        print_status()
