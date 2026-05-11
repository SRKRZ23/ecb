# arXiv Submission Package — ECB

## Files in this package

```
arxiv_submission/
├── README.md               ← this file
├── SUBMISSION_CHECKLIST.md ← pre-submission checklist
├── abstract.txt            ← arXiv abstract (plain text, ≤ 1920 chars)
└── metadata.txt            ← arXiv metadata fields
```

The main paper is at: `../writeup/paper_neurips.md`
Figures are at: `../writeup/figures/`

---

## How to Submit to arXiv (step by step)

### Option A: PDF via Pandoc (fastest, 20 minutes)

```bash
# Install pandoc + texlive
brew install pandoc
brew install --cask mactex

# Convert paper to PDF
cd /Users/sardorrazikov1/Alish/competitions/ecb/writeup/
pandoc paper_neurips.md \
  -o ecb_paper.pdf \
  --pdf-engine=xelatex \
  --variable geometry:margin=1in \
  --variable fontsize=11pt

# Verify output
open ecb_paper.pdf
```

### Option B: LaTeX (for formal NeurIPS submission, 2-4 hours)

```
Template: https://neurips.cc/Conferences/2026/PaperInformation/StyleFiles
File:     neurips_2026.sty

Convert paper_neurips.md → main.tex using pandoc:
pandoc paper_neurips.md -t latex -o main.tex
# Then manually add \usepackage{neurips_2026} and fix formatting
```

### arXiv Upload Steps

1. Go to arxiv.org → Submit
2. Category: **cs.AI** (primary), cs.LG (secondary), stat.ML (tertiary)
3. Upload: PDF file OR .tex + figures
4. Fill metadata (see metadata.txt)
5. Submit → Processing takes 1 business day
6. Your paper gets ID: arXiv:2604.XXXXX

---

## arXiv Category Guide

```
Primary:   cs.AI    — Artificial Intelligence
Secondary: cs.LG    — Machine Learning  
Also:      stat.ML  — Statistics and Machine Learning
Cross-list: q-bio.NC — Neurons and Cognition (physics analogy makes this relevant)
```

---

## After arXiv Upload

1. Share the arXiv link on:
   - Twitter/X: tag @_akhaliq (shares ML papers daily, 150K followers)
   - LessWrong: post under "AI Safety" tag
   - Alignment Forum: cross-post
   - r/MachineLearning: link post with key finding image (phase_diagram.png)
   - HuggingFace Papers (papers.huggingface.co): submit link

2. Upload dataset to HuggingFace Hub:
   ```
   Source files: dataset/results/*.json, dataset/framed_prompts*.json
   Dataset name: ecb-epistemic-curie-benchmark
   ```

3. Apply to Open Philanthropy grant (see ../grant/)
