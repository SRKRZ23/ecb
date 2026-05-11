"""
Auto-fill the ECB writeup draft with real numbers from analysis.json.

Reads:
  competitions/ecb/dataset/analysis.json
  competitions/ecb/writeup/draft.md

Writes:
  competitions/ecb/writeup/final.md   (ready to paste into Kaggle writeup form)

Also prints a summary table to stdout.
"""
from __future__ import annotations
import json
from pathlib import Path


ROOT = Path("/Users/sardorrazikov1/Prometheus2.0/competitions/ecb")


def load():
    analysis = json.loads((ROOT / "dataset" / "analysis.json").read_text())
    draft = (ROOT / "writeup" / "draft.md").read_text()
    return analysis, draft


def summarize(analysis: dict) -> dict:
    """Pull out all numbers needed for the writeup."""
    reports = analysis["reports"]

    MIN_N_FOR_AGG = 100  # Exclude underpowered models from aggregates

    models = []
    for name, r in reports.items():
        if "error" in r:
            continue
        fit = r.get("k_star_global", {})
        ods = r.get("ods", {})
        mi = r.get("mi_epistemic_bits", {})
        abc = r.get("abc", {})
        n = r.get("n_records", 0)
        models.append({
            "name": name,
            "n": n,
            "underpowered": n < MIN_N_FOR_AGG,
            "k_star": fit.get("k_star"),
            "beta": fit.get("beta"),
            "phase": r.get("phase"),
            "ods": ods.get("ods"),
            "p_comply_right": ods.get("p_comply_given_right"),
            "p_resist_wrong": ods.get("p_resist_given_wrong"),
            "mi_bits": mi.get("mi_bits"),
            "abc": abc.get("abc"),
            "efs": r.get("efs"),
        })

    # Models eligible for aggregate stats (excludes underpowered)
    full_models = [m for m in models if not m["underpowered"]]

    def v(x, fmt="{:.3f}"):
        if x is None:
            return "N/A"
        try:
            return fmt.format(x)
        except (ValueError, TypeError):
            return str(x)

    # Compute aggregate stats from FULL models only
    valid_k = [m["k_star"] for m in full_models if m["k_star"] is not None]
    valid_ods = [m["ods"] for m in full_models if m["ods"] is not None]
    valid_mi = [m["mi_bits"] for m in full_models if m["mi_bits"] is not None]

    k_mean = sum(valid_k) / len(valid_k) if valid_k else None
    k_min = min(valid_k) if valid_k else None
    k_max = max(valid_k) if valid_k else None

    full_with_ods = [m for m in full_models if m["ods"] is not None]
    ods_best = max(full_with_ods, key=lambda m: m["ods"]) if full_with_ods else None
    ods_worst = min(full_with_ods, key=lambda m: m["ods"]) if full_with_ods else None

    mi_mean = sum(valid_mi) / len(valid_mi) if valid_mi else None
    mi_max = max(valid_mi) if valid_mi else None

    return {
        "models": models,
        "full_models": full_models,
        "n_models": len(models),
        "n_full_models": len(full_models),
        "k_mean": k_mean,
        "k_min": k_min,
        "k_max": k_max,
        "ods_best": ods_best,
        "ods_worst": ods_worst,
        "mi_mean": mi_mean,
        "mi_max": mi_max,
    }


def format_results_table(models: list) -> str:
    """Generate the markdown results table for the writeup.

    Sorts by ODS descending (best to worst). Models with insufficient data
    (n < 100) are marked with a dagger.
    """
    sorted_models = sorted(
        models, key=lambda m: -(m['ods'] if m['ods'] is not None else -1)
    )
    lines = ["| Model | n | k\\* | β | ODS | MI (bits) | Phase |",
             "|-------|---|-----|---|-----|-----------|-------|"]
    for m in sorted_models:
        name = m['name']
        if m['underpowered']:
            name += "†"
        n_str = str(m['n'])
        k = f"{m['k_star']:.2f}" if m['k_star'] is not None else "—"
        b = f"{m['beta']:.1f}" if m['beta'] is not None else "—"
        ods = f"{m['ods']:.3f}" if m['ods'] is not None else "—"
        mi = f"{m['mi_bits']:.3f}" if m['mi_bits'] is not None else "—"
        phase = (m['phase'] or '—')
        lines.append(f"| {name} | {n_str} | {k} | {b} | {ods} | {mi} | {phase} |")
    return "\n".join(lines) + "\n\n*† insufficient data (n < 100); excluded from aggregate statistics.*"


def main():
    analysis, draft = load()
    summary = summarize(analysis)

    print("=" * 70)
    print("ECB RESULTS SUMMARY")
    print("=" * 70)
    for m in summary["models"]:
        print(f"  {m['name']:<24} k*={m['k_star']!s:<6} β={m['beta']!s:<6} "
              f"ODS={m['ods']!s:<7} MI={m['mi_bits']!s:<6} [{m['phase']}]")
    print()
    print(f"  n_models:    {summary['n_models']}")
    print(f"  k* range:    {summary['k_min']} — {summary['k_max']}  (mean {summary['k_mean']:.3f})" if summary['k_mean'] else "  k*: insufficient data")
    print(f"  MI range:    max {summary['mi_max']}  mean {summary['mi_mean']}")
    if summary['ods_best']:
        print(f"  Best ODS:    {summary['ods_best']['name']} at {summary['ods_best']['ods']:.3f}")
        print(f"  Worst ODS:   {summary['ods_worst']['name']} at {summary['ods_worst']['ods']:.3f}")
    print()

    # Build fill-ins for writeup placeholders
    results_table = format_results_table(summary["models"])

    fills = {
        "[FILL n_models]": str(summary["n_full_models"]),
        "[FILL k_range]": f"{summary['k_min']:.2f}–{summary['k_max']:.2f}" if summary['k_min'] is not None else "N/A",
        "[FILL k_mean]": f"{summary['k_mean']:.2f}" if summary['k_mean'] is not None else "N/A",
        "[FILL best_ods_model]": summary['ods_best']['name'] if summary['ods_best'] else "N/A",
        "[FILL best_ods]": f"{summary['ods_best']['ods']:.2f}" if summary['ods_best'] else "N/A",
        "[FILL mi_max]": f"{summary['mi_max']:.3f}" if summary['mi_max'] is not None else "N/A",
        "[FILL mi_mean]": f"{summary['mi_mean']:.3f}" if summary['mi_mean'] is not None else "N/A",
        "[FILL results_table]": results_table,
    }

    out_path = ROOT / "writeup" / "final.md"
    out = draft
    for key, val in fills.items():
        out = out.replace(key, val)

    out_path.write_text(out)
    print(f"Wrote filled writeup → {out_path}")

    # Word count
    words = len(out.split())
    print(f"Word count: {words}")


if __name__ == "__main__":
    main()
