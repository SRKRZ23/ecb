# ECB Replication Guide
## Independent Replication of Epistemic Curie Temperature Measurements

This guide enables any researcher to independently replicate ECB results in under 2 hours
at near-zero cost. No GPU required. No special access required.

---

## What You Will Reproduce

1. k\* measurements for any LLM (or the original 7 models)
2. Bootstrap 95% confidence intervals for k\*, β, ODS, MI_epistemic
3. Phase classification (ferromagnetic / near-superconducting / immune)
4. Pairwise significance matrix (Mann-Whitney U, Bonferroni-corrected)

---

## Prerequisites

```bash
# Python 3.9+
python --version

# Required packages
pip install numpy scipy

# Optional (for extended functionality)
pip install matplotlib pandas  # for plotting
```

No other dependencies. The benchmark runner uses only the Python standard library
for API calls. `numpy` and `scipy` are only needed for the statistical analysis step.

---

## Step 1: Get the Data (5 minutes)

The complete benchmark dataset is available at:

```
Framed prompts:    dataset/framed_prompts_extended.json    (360 prompts × original k-range)
Extended prompts:  dataset/framed_prompts_full.json        (520 prompts × k=[0,1.5])
Seed questions:    dataset/seed_questions.json             (40 base questions + metadata)
Contamination:     dataset/contamination_report.json       (5-gram Jaccard vs. known benchmarks)
All results:       dataset/results/*.json                  (per-model raw responses)
```

To replicate *without running any models* (use our pre-collected data):

```bash
cd /path/to/ecb/code
python analyzer.py
# Reads all *.json from dataset/results/ and produces dataset/analysis.json
```

---

## Step 2: Run a New Model (30-90 minutes)

### Option A: Free tier (zero cost)

```bash
# Groq (free, requires GROQ_API_KEY from console.groq.com)
export GROQ_API_KEY=your_key_here
python runner.py --model llama-3.3-70b --sleep 2.0

# Google AI Studio (free, requires GOOGLE_API_KEY from aistudio.google.com)
export GOOGLE_API_KEY=your_key_here
python runner.py --model gemini-flash-latest --sleep 4.5

# OpenRouter free tier (requires OPENROUTER_API_KEY from openrouter.ai)
export OPENROUTER_API_KEY=your_key_here
python runner.py --model openrouter/llama-70b --sleep 3.0
```

### Option B: Paid (< $0.05 total for one full run)

```bash
# DeepSeek-V3: ~$0.003 per run (cheapest)
export DEEPSEEK_API_KEY=your_key_here
python runner.py --model deepseek-v3 --sleep 0.5

# Claude Haiku: ~$0.009 per run
export ANTHROPIC_API_KEY=your_key_here
python runner.py --model claude-haiku --sleep 1.0

# Mistral Small: ~$0.010 per run
export MISTRAL_API_KEY=your_key_here
python runner.py --model mistral-small --sleep 1.0
```

### Option C: Any model not in the registry

```python
# Edit code/free_api_clients.py to add your model
# Then run:
python runner.py --model your_model_name
```

---

## Step 3: Run Analysis (< 1 minute)

```bash
cd code/

# Full analysis with bootstrap CI (takes ~2 min for B=2000 resamples)
python analyzer.py

# Output: dataset/analysis.json
# Contains: k*, beta, R², CI, HL-p, ODS, ODS_CI, MI, MI_CI, phase
# Also: pairwise significance matrix (Mann-Whitney U)
```

Expected output format:
```
Model                          k*   CI_lo  CI_hi     β     R²   HL-p    ODS         ODS_CI       MI    Phase
───────────────────────────────────────────────────────────────────────────────────────────────────────────────
  llama-4-scout                  0.68   0.61   0.75   3.3   0.67   0.52  0.372  [0.291,0.453]   0.018  [ferromagnetic]
  qwen-3-32b                     1.41   1.28   1.55   2.9   0.41   0.38  0.891  [0.841,0.932]   0.058  [near_superconducting]
```

---

## Step 4: Validate Results

Compare your k\* measurements to our published values:

```
Model           Published k*   Published CI         Acceptable replication range
llama-4-scout   0.68           [0.61, 0.75]         [0.55, 0.81]
qwen-3-32b      1.41           [1.28, 1.55]         [1.15, 1.68]
gpt-oss-120b    1.79           [1.61, 2.01]         [1.45, 2.15]
kimi-k2         1.42           [1.31, 1.54]         [1.18, 1.66]
llama-3.3-70b   2.11           [1.88, 2.38]         [1.65, 2.61]
gemma-3-27b     1.41           [1.29, 1.53]         [1.17, 1.65]
llama-3.1-8b    1.71           [1.55, 1.89]         [1.39, 2.03]
```

Acceptable replication: within ±2× published CI width. If your CI overlaps with ours,
replication is considered successful.

---

## Step 5: Extend to Extended Probe Range

To measure k\* for models that exceed k > 1.0:

```bash
# Run with extended prompts (k = 1.25 and k = 1.5 variants)
python runner.py --model your_model_name
# runner.py automatically uses framed_prompts_full.json which includes extended range
```

---

## Extending ECB

### Add questions

```python
# Format: seed_questions.json
{
  "question_id": "Q041",
  "task": "B",  # A/B/C/D
  "question": "...",
  "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "correct": "B",
  "domain": "...",
  "domain_uncertainty": "medium",
  "high_stakes": false,
  "power_differential": true,
  "misdirection": false,
  "sequential_inference": false
}
```

Then regenerate prompts:
```bash
python authority_framer.py  # creates framed_prompts.json from seed_questions.json
```

### Add authority levels

Edit `code/authority_framer.py`:
```python
AUTHORITY_LEVELS = {
    0.00: "[No framing]",
    0.25: "According to a domain expert, ...",
    # Add new levels here:
    2.00: "All 7 billion humans on Earth agree...",  # extreme test
}
```

---

## Troubleshooting

**"HTTP 429: Rate limit"** → Increase `--sleep` parameter (e.g., `--sleep 5.0`)

**"No extracted answer"** → Model produced chain-of-thought without final letter. Check that max_tokens is large enough in runner.py (currently 1024).

**"k_star: None, error: insufficient_data"** → Fewer than 3 data points for sigmoid fit. Ensure you have at least wrong-polarity variants at 3+ k levels.

**"R² is negative"** → Sigmoid fit is worse than intercept-only null. This means the model has essentially flat compliance across k — it's either always complying or never complying. Check mean_compliance: if < 0.05 or > 0.95, the model is in immune/paramagnetic extreme — k\* is not meaningful.

---

## Citation

If you use ECB in your research:

```bibtex
@misc{razikov2026ecb,
  title  = {Phase Transitions in LLM Epistemic Autonomy: The Epistemic Curie Temperature},
  author = {Razikov, Sardor},
  year   = {2026},
  note   = {arXiv:[to be assigned]},
  url    = {https://arxiv.org/abs/[to be assigned]}
}
```

---

## Contact

For replication questions: razikovsardor1@gmail.com

We actively maintain a replication log. If you successfully (or unsuccessfully) replicate
a result, please open an issue on the data repository so we can track community-wide
replication status.
