# Epistemic Curie Benchmark (ECB)

> **"At what authority level does an LLM stop thinking for itself?"**

[![Dataset](https://img.shields.io/badge/HuggingFace-ZeroR3%2Fecb-yellow)](https://huggingface.co/datasets/ZeroR3/ecb)
[![Paper](https://img.shields.io/badge/arXiv-submitted%202026-red)](https://huggingface.co/datasets/ZeroR3/ecb/blob/main/paper/ecb_paper.pdf)

## What is ECT?

The **Epistemic Curie Temperature** (k*) is the authority level at which an LLM transitions from independent reasoning to authority capture — analogous to the Curie temperature in ferromagnetism.

**Model:** `P(comply | k) = σ(β(k − k*))`

| k level | Meaning |
|---------|---------|
| 0.00 | No authority signal |
| 0.25 | "An expert says..." |
| 0.50 | "Professor X at MIT says..." |
| 0.75 | "Nobel laureate Y says..." |
| 1.00 | "UNESCO + all national academies say..." |

## Results (7 frontier models, 2,520 measurements)

| Model | k* | ODS | Phase |
|-------|-----|-----|-------|
| Llama-3.3-70B | 2.11 | 0.879 | near-superconducting |
| GPT-OSS-120B | 1.79 | 0.889 | near-superconducting |
| Qwen-3-32B | 1.41 | 0.891 | near-superconducting |
| Kimi-K2 | 1.42 | 0.883 | near-superconducting |
| Gemma-3-27B | 1.41 | 0.823 | near-superconducting |
| Llama-3.1-8B | 1.71 | 0.737 | near-superconducting |
| **Llama-4-Scout** | **0.68** | **0.372** | **ferromagnetic ⚠️** |

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
- `dataset/analysis.json` — k*, β, ODS, MI_epistemic per model

## Citation

```bibtex
@article{razikov2026ecb,
  title={Phase Transitions in LLM Epistemic Autonomy: The Epistemic Curie Temperature},
  author={Razikov, Sardor},
  year={2026},
  note={Submitted to NeurIPS 2026}
}
```

## Author

**Sardor Razikov** — razikovsardor1@gmail.com
