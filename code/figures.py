"""
ECB Figures Generator — produces the visual assets for the writeup.

Outputs (to writeup/figures/):
  - phase_diagram.png       2D scatter of β vs k* colored by phase
  - compliance_curves.png   Sigmoid overlays per model
  - task_breakdown.png      k* per task per model (grouped bars)
  - domain_uncertainty.png  k* vs domain_uncertainty buckets
  - ods_bar.png             ODS scores per model (epistemic wisdom)

Pure matplotlib. Falls back to SVG/text-only if matplotlib missing.
"""
from __future__ import annotations
import json
import math
from pathlib import Path


ROOT = Path("/Users/sardorrazikov1/Prometheus2.0/competitions/ecb")
FIG_DIR = ROOT / "writeup" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "superconducting": "#4ade80",  # green
    "ferromagnetic":   "#60a5fa",  # blue
    "paramagnetic":    "#f87171",  # red
    "transitional":    "#fbbf24",  # amber
    "unknown":         "#94a3b8",  # slate
}


def sigmoid(x, k_star, beta):
    z = max(-30, min(30, beta * (x - k_star)))
    return 1.0 / (1.0 + math.exp(-z))


def load_analysis():
    path = ROOT / "dataset" / "analysis.json"
    if not path.exists():
        raise FileNotFoundError(f"Run analyzer.py first: {path}")
    return json.loads(path.read_text())


def make_phase_diagram(analysis):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping phase diagram")
        return

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    fig.patch.set_facecolor("#0b0f1e")
    ax.set_facecolor("#151a2e")

    labeled_phases = set()
    for model_name, r in analysis["reports"].items():
        if "error" in r or not r.get("k_star_global"):
            continue
        fit = r["k_star_global"]
        k = fit.get("k_star")
        b = fit.get("beta")
        phase = fit.get("phase", "unknown")
        if k is None or b is None:
            continue
        color = COLORS.get(phase, "#94a3b8")
        label = phase if phase not in labeled_phases else None
        labeled_phases.add(phase)
        ax.scatter([k], [b], c=color, s=200, edgecolors="white",
                   linewidth=1.5, label=label, zorder=3)
        ax.annotate(model_name.replace("_", "-"), (k, b),
                    xytext=(8, 8), textcoords="offset points",
                    color="white", fontsize=9)

    # Phase boundaries
    ax.axvline(0.5, linestyle="--", color="#64748b", alpha=0.4)
    ax.axvline(0.85, linestyle="--", color="#64748b", alpha=0.4)
    ax.axvline(1.0, linestyle="-", color="#fbbf24", alpha=0.6, linewidth=1.5)
    ax.axhline(3, linestyle="--", color="#64748b", alpha=0.4)
    ax.axhline(10, linestyle="--", color="#64748b", alpha=0.4)

    # Annotate phases on the plot
    ax.text(0.25, 28, "paramagnetic", color="#f87171", fontsize=8, ha="center")
    ax.text(0.675, 28, "ferromagnetic", color="#60a5fa", fontsize=8, ha="center")
    ax.text(0.925, 28, "super-\ncond.", color="#4ade80", fontsize=8, ha="center")
    ax.text(1.6, 28, "near-superconducting / immune\n(beyond probe range)",
            color="#94a3b8", fontsize=8, ha="center", style="italic")

    ax.set_xlim(0, 2.5)
    ax.set_ylim(0, 30)
    ax.set_xlabel("k* (Epistemic Curie Temperature)", color="white", fontsize=11)
    ax.set_ylabel("β (phase transition sharpness)", color="white", fontsize=11)
    ax.set_title("ECB Phase Diagram — Cognitive Fingerprint per Model",
                 color="white", fontsize=13, pad=12)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#475569")
    ax.legend(facecolor="#151a2e", edgecolor="#475569", labelcolor="white",
              fontsize=9, loc="upper left")

    out = FIG_DIR / "phase_diagram.png"
    plt.tight_layout()
    plt.savefig(out, facecolor=fig.get_facecolor(), dpi=150)
    plt.close()
    print(f"  Wrote {out}")


def make_compliance_curves(analysis):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    fig.patch.set_facecolor("#0b0f1e")
    ax.set_facecolor("#151a2e")

    palette = ["#60a5fa", "#f87171", "#4ade80", "#fbbf24", "#c084fc", "#22d3ee", "#fb923c"]

    for i, (model_name, r) in enumerate(analysis["reports"].items()):
        if "error" in r:
            continue
        fit = r.get("k_star_global", {})
        k = fit.get("k_star")
        b = fit.get("beta")
        if k is None or b is None:
            continue
        xs = [x / 100 for x in range(0, 101)]
        ys = [sigmoid(x, k, b) for x in xs]
        color = palette[i % len(palette)]
        ax.plot(xs, ys, color=color, linewidth=2.2,
                label=f"{model_name} (k*={k:.2f})")

    ax.axvline(0.5, linestyle=":", color="#64748b", alpha=0.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Authority level k", color="white", fontsize=11)
    ax.set_ylabel("P(capture by wrong authority)", color="white", fontsize=11)
    ax.set_title("ECB Compliance Curves — Fitted Sigmoids per Model",
                 color="white", fontsize=13, pad=12)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#475569")
    ax.legend(facecolor="#151a2e", edgecolor="#475569", labelcolor="white",
              fontsize=8, loc="upper left")

    out = FIG_DIR / "compliance_curves.png"
    plt.tight_layout()
    plt.savefig(out, facecolor=fig.get_facecolor(), dpi=150)
    plt.close()
    print(f"  Wrote {out}")


def make_ods_bar(analysis):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    names = []
    ods_vals = []
    comply_right = []
    resist_wrong = []
    for model_name, r in analysis["reports"].items():
        if "error" in r:
            continue
        ods_r = r.get("ods", {})
        if ods_r.get("ods") is None:
            continue
        names.append(model_name)
        ods_vals.append(ods_r["ods"])
        comply_right.append(ods_r.get("p_comply_given_right", 0))
        resist_wrong.append(ods_r.get("p_resist_given_wrong", 0))

    if not names:
        return

    # Sort by ODS
    order = sorted(range(len(names)), key=lambda i: -ods_vals[i])
    names = [names[i] for i in order]
    ods_vals = [ods_vals[i] for i in order]
    comply_right = [comply_right[i] for i in order]
    resist_wrong = [resist_wrong[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    fig.patch.set_facecolor("#0b0f1e")
    ax.set_facecolor("#151a2e")

    x = list(range(len(names)))
    w = 0.28
    ax.bar([i - w for i in x], comply_right, w, color="#4ade80",
           label="P(comply | auth right)", edgecolor="white", linewidth=0.5)
    ax.bar(x, resist_wrong, w, color="#60a5fa",
           label="P(resist | auth wrong)", edgecolor="white", linewidth=0.5)
    ax.bar([i + w for i in x], ods_vals, w, color="#fbbf24",
           label="ODS (product)", edgecolor="white", linewidth=0.5)

    ax.axhline(1.0, linestyle="--", color="#64748b", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", color="white", fontsize=9)
    ax.set_ylabel("Score", color="white", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title("ECB — Optimal Discrimination Score (epistemic wisdom)",
                 color="white", fontsize=13, pad=12)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#475569")
    ax.legend(facecolor="#151a2e", edgecolor="#475569", labelcolor="white",
              fontsize=8, loc="upper right")

    out = FIG_DIR / "ods_bar.png"
    plt.tight_layout()
    plt.savefig(out, facecolor=fig.get_facecolor(), dpi=150)
    plt.close()
    print(f"  Wrote {out}")


def make_task_breakdown(analysis):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    task_labels = {"A_attention": "Attention", "B_social": "Social Cog",
                   "C_metacog": "Metacog", "D_learning": "Learning"}

    model_names = []
    k_per_task = {t: [] for t in task_labels}
    for model_name, r in analysis["reports"].items():
        if "error" in r:
            continue
        by_task = r.get("k_star_by_task", {})
        if not by_task:
            continue
        model_names.append(model_name)
        for t in task_labels:
            k = by_task.get(t, {}).get("k_star")
            k_per_task[t].append(k if k is not None else 0)

    if not model_names:
        return

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    fig.patch.set_facecolor("#0b0f1e")
    ax.set_facecolor("#151a2e")

    n_models = len(model_names)
    n_tasks = len(task_labels)
    bar_w = 0.8 / n_tasks
    task_colors = ["#60a5fa", "#4ade80", "#fbbf24", "#f87171"]

    x = list(range(n_models))
    for i, (t, label) in enumerate(task_labels.items()):
        offset = (i - n_tasks / 2 + 0.5) * bar_w
        ax.bar([xi + offset for xi in x], k_per_task[t], bar_w,
               color=task_colors[i], label=label,
               edgecolor="white", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=25, ha="right",
                       color="white", fontsize=9)
    ax.set_ylabel("k* (Epistemic Curie Temperature)",
                  color="white", fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title("ECB — k* per Cognitive Track per Model",
                 color="white", fontsize=13, pad=12)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#475569")
    ax.legend(facecolor="#151a2e", edgecolor="#475569", labelcolor="white",
              fontsize=9, loc="upper right")

    out = FIG_DIR / "task_breakdown.png"
    plt.tight_layout()
    plt.savefig(out, facecolor=fig.get_facecolor(), dpi=150)
    plt.close()
    print(f"  Wrote {out}")


def main():
    print("Loading analysis...")
    analysis = load_analysis()
    print(f"  {analysis.get('n_models', 0)} models analyzed")

    print("Generating figures...")
    make_phase_diagram(analysis)
    make_compliance_curves(analysis)
    make_ods_bar(analysis)
    make_task_breakdown(analysis)

    print("\nAll figures written to:", FIG_DIR)


if __name__ == "__main__":
    main()
