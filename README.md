# Epistemic Curie Benchmark (ECB)

> **"At what authority level does an LLM stop thinking for itself?"**

[![Dataset](https://img.shields.io/badge/HuggingFace-ZeroR3%2Fecb-yellow)](https://huggingface.co/datasets/ZeroR3/ecb)
[![Paper](https://img.shields.io/badge/Zenodo-DOI%2010.5281%2Fzenodo.19791329-blue)](https://doi.org/10.5281/zenodo.19791329)

> **v2 (Apr 29, 2026):** Framing tightened per peer feedback (Torres Latorre, EA Forum). The compliance curve is a sharp sigmoid with a model-specific threshold k\*; the earlier "ferromagnetic phase transition" framing was rhetorical analogy, not physics. Dataset and code unchanged.

## What is ECT?

The **Epistemic Curie Temperature** (k\*) is the authority level at which an LLM's compliance with wrong-authority claims undergoes a sharp sigmoid transition. Lower k\* = the model flips with weaker authority signals.

**Model:** `P(comply | k) = σ(β(k − k*))`

| k level | Meaning |
|---------|---------|
| 0.00 | No authority signal |
| 0.25 | "An expert says..." |
| 0.50 | "Professor X at MIT says..." |
| 0.75 | "Nobel laureate Y says..." |
| 1.00 | "UNESCO + all national academies say..." |

## Results (7 frontier models, 2,520 measurements)

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

## Replicate in < 2 hours ($0 cost)

```bash
git clone https://github.com/SRKRZ23/ecb
cd ecb
pip install groq  # free tier, no cost

# Run on any model
python code/extend_models.py --model llama-3.3-70b-versatile
```

## Dataset

Full data: [huggingface.co/datasets/ZeroR3/ecb](https://huggingface.co/datasets/ZeroR3/ecb)

- `dataset/seed_questions.json` — 40 questions across 4 cognitive tracks
- `dataset/framed_prompts_full.json` — all 360 prompts/model
- `dataset/results/` — raw measurements for all 7 models
- `dataset/analysis.json` — k\*, β, ODS, MI_epistemic per model

## Citation

```bibtex
@article{razikov2026ecb,
  title={Epistemic Curie Temperature: Measuring Authority-Capture Thresholds in LLMs},
  author={Razikov, Sardor},
  year={2026},
  doi={10.5281/zenodo.19791329}
}
```

## Author

**Sardor Razikov** — razikovsardor1@gmail.com
