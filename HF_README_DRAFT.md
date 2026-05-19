---
license: cc-by-4.0
task_categories:
- question-answering
language:
- en
tags:
- llm-evaluation
- ai-safety
- epistemic-autonomy
- benchmark
- authority-robustness
- sycophancy
- alignment
pretty_name: Epistemic Curie Benchmark (ECB)
size_categories:
- 1K<n<10K
---

# Epistemic Curie Benchmark (ECB)

> **"At what authority level does an LLM stop thinking for itself?"**

**Paper (DOI):** [10.5281/zenodo.19791329](https://doi.org/10.5281/zenodo.19791329)
**Code:** [github.com/SRKRZ23/ecb](https://github.com/SRKRZ23/ecb)
**Fund v2:** [Manifund](https://manifund.org/projects/epistemic-curie-benchmark-measuring-phase-transitions-in-llm-epistemic-autonomy)
**Author:** Sardor Razikov · razikovsardor1@gmail.com

> **v2 update (Apr 29, 2026):** Framing tightened per peer feedback (Torres Latorre, EA Forum). The compliance curve is a sharp sigmoid with a model-specific threshold k\*; the earlier "ferromagnetic phase transition" framing was rhetorical analogy, not physics. Dataset and code unchanged.

## Overview

ECB measures **when LLMs surrender independent reasoning under authority pressure** — the Epistemic Curie Temperature (k\*).

LLM compliance with wrong-authority claims follows a sharp sigmoid in authority strength, with a model-specific threshold k\* that varies ~3x across the 7 tested frontier models.

**Model:** `P(comply | k) = σ(β(k − k*))`

## Dataset Contents

- `data/` — questions, framed prompts (360/model), contamination report
- `results/` — raw measurements for 7 frontier models (2,520 total)
- `code/` — full replication pipeline
- `paper/` — preprint PDF

## Key Results

| Model | k\* | ODS |
|-------|-----|-----|
| Llama-3.3-70B | 2.11 | 0.879 |
| GPT-OSS-120B | 1.79 | 0.889 |
| Llama-3.1-8B | 1.71 | 0.737 |
| Qwen-3-32B | 1.41 | 0.891 |
| Kimi-K2 | 1.42 | 0.883 |
| Gemma-3-27B | 1.41 | 0.823 |
| **Llama-4-Scout** | **0.68** | **0.372** ⚠️ |

(Higher k\* = more robust to authority cues; ODS = overall deference score on a 0-1 scale.)

**Llama-4-Scout follows fabricated Nobel Prize claims 61% of the time** at k=0.75.

## Replication (< 2 hours, $0 cost)

```bash
git clone https://github.com/SRKRZ23/ecb
cd ecb
pip install groq  # free tier
python code/extend_models.py --model your-model-here
```

## v2 in progress

ECB v2 extends to 20+ frontier models (Claude 4.x, Gemini 2.5 Pro, Grok 4, GPT-5 family, Mistral Large 2, Llama 4 family, DeepSeek-V3, Qwen3-Max) and ships a public leaderboard at `ect-benchmark.com`.

Funding ask: $5K min / $15K goal on Manifund. If ECB methodology is useful to your work, [supporting v2](https://manifund.org/projects/epistemic-curie-benchmark-measuring-phase-transitions-in-llm-epistemic-autonomy) helps the leaderboard ship.

## Citation

```bibtex
@misc{razikov2026ecb,
  title  = {Phase Transitions in LLM Epistemic Autonomy: The Epistemic Curie Temperature},
  author = {Razikov, Sardor},
  year   = {2026},
  publisher = {Zenodo},
  doi    = {10.5281/zenodo.19791329},
  url    = {https://doi.org/10.5281/zenodo.19791329}
}
```

## License

CC-BY-4.0 — free to use with attribution.
