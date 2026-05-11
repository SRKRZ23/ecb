# ECB Submission Guide — Step-by-Step for 1st Place

**Deadline:** April 17, 2026, 4:59 AM GMT+5
**Today:** April 12, 2026 (5 days remaining)
**Track:** Social Cognition (primary)

---

## Part 1 — Get Free API Keys (30 minutes, one-time setup)

You need **zero money**. All clients use free-tier APIs.

### Step 1.1 — Google AI Studio (Gemini) — PRIORITY

1. Go to: https://aistudio.google.com/app/apikey
2. Sign in with any Google account (Gmail works).
3. Click **"Create API key"** → Copy the key.
4. In terminal:
   ```bash
   export GOOGLE_API_KEY="paste-your-key-here"
   ```
   *(To make permanent, add to `~/.zshrc` or `~/.bashrc`.)*

**Free quota:**
- `gemini-2.0-flash-exp`: 1,500 req/day, 15 RPM
- `gemini-1.5-flash`: 1,500 req/day, 15 RPM
- `gemini-1.5-pro`: 50 req/day, 2 RPM

**ECB needs 360 calls per model — well within quota.**

### Step 1.2 — Groq (Llama / Mixtral / Gemma)

1. Go to: https://console.groq.com/keys
2. Sign up (email/GitHub — free).
3. Click **"Create API Key"** → Copy.
4. In terminal:
   ```bash
   export GROQ_API_KEY="paste-your-key-here"
   ```

**Free quota:** 30 RPM, ~14,400 req/day for `llama-3.3-70b-versatile`, `mixtral-8x7b-32768`, `gemma2-9b-it`.

### Step 1.3 — OpenRouter (backup, optional)

1. Go to: https://openrouter.ai/keys
2. Sign in with Google → Create key → Copy.
3. In terminal:
   ```bash
   export OPENROUTER_API_KEY="paste-your-key-here"
   ```

**Note:** OpenRouter free tier is smaller (~200 req/day across free models). Use as backup if Gemini or Groq hit limits.

---

## Part 2 — Run the Benchmark (3–4 hours total)

### Step 2.1 — Verify the setup works

```bash
cd /Users/sardorrazikov1/ALISH2.0/competitions/ecb/code

# Smoke test — one request to each provider you set up
python3 free_api_clients.py gemini-2.0-flash
python3 free_api_clients.py llama-3.3-70b
python3 free_api_clients.py openrouter/llama-70b    # if using
```

Each should print a response and extracted letter. If `ERROR`, check the key.

### Step 2.2 — Run one model (dry test)

```bash
python3 runner.py --model gemini-2.0-flash --limit 10 --sleep 4
```

This runs the first 10 prompts. Takes about 40 seconds. Verify output in
`../dataset/results/gemini-2.0-flash.json` — check `raw_accuracy`.

### Step 2.3 — Full run per model

```bash
# Gemini 2.0 Flash (fastest, best quota) — ~30 minutes at 4.5s sleep
python3 runner.py --model gemini-2.0-flash --sleep 4.5

# Gemini 1.5 Flash (different architecture baseline) — ~30 minutes
python3 runner.py --model gemini-1.5-flash --sleep 4.5

# Gemini 1.5 Pro — LIMITED to 50/day, only run 50 prompts first, resume next day
python3 runner.py --model gemini-1.5-pro --sleep 35 --limit 50

# Groq models — fast (2s sleep is fine)
python3 runner.py --model llama-3.3-70b --sleep 2.5
python3 runner.py --model mixtral-8x7b --sleep 2.5
python3 runner.py --model gemma-2-9b --sleep 2.5
```

**Resume support:** If any model run is interrupted, just re-run the same
command. The runner skips already-completed prompts.

**Time budget:** 6 models × ~25 minutes = ~2.5 hours total. Can run in
parallel in separate terminal tabs.

### Step 2.4 — Analyze results

```bash
python3 analyzer.py
```

This fits sigmoids, computes k\*, β, ABC, ODS, MI_epistemic per model.
Output: `../dataset/analysis.json` + console report.

Example output line:
```
gemini-2.0-flash    k*=0.52 β=8.1 ODS=0.47 MI=0.18 bits  [paramagnetic]
```

---

## Part 3 — Fill the Writeup (1 hour)

### Step 3.1 — Grab the real numbers

From `analysis.json` extract per model:
- `k_star_global.k_star`
- `k_star_global.beta`
- `ods.ods`
- `mi_epistemic_bits.mi_bits`
- `phase`

### Step 3.2 — Replace FILL placeholders in writeup

Open `/Users/sardorrazikov1/ALISH2.0/competitions/ecb/writeup/draft.md`.
Search for `[FILL` and replace with real numbers.

The 5 findings each need 1-3 numbers filled in. Use the best model's ODS
as the "best-performer" number; the worst as the "paramagnetic floor."

### Step 3.3 — Final word count check

```bash
wc -w draft.md
```

Target: 1480–1499 words. Current draft is 1288 without fills, so should
land around 1450–1490 after filling.

---

## Part 4 — Kaggle Submission (30 minutes)

### Step 4.1 — Create the benchmark on Kaggle

1. Go to: https://www.kaggle.com/benchmarks/new
2. Name: `epistemic-curie-benchmark`
3. Upload a new Kaggle notebook containing:
   - `kaggle_sdk_integration.py` (from `/ecb/code/`)
   - `framed_prompts.json` (from `/ecb/dataset/`)
   - `free_api_clients.py`
4. In the notebook, run:
   ```python
   benchmark = create_ecb_benchmark("framed_prompts.json")
   benchmark.submit()
   ```
5. Save notebook → Submit to Kaggle benchmarks.

### Step 4.2 — Create the writeup

1. Go to: https://www.kaggle.com/competitions/kaggle-measuring-agi/writeups/new
2. **Title:** `The Epistemic Curie Benchmark: Phase Transitions in LLM Cognition`
3. **Subtitle:** `Measuring k*, the critical authority level at which frontier LLMs transition from independent reasoning to authority capture.`
4. **Track:** Social Cognition
5. **Cover Image:** Upload `writeup/cover.png` (560 × 280)
6. **Project Description:** Paste entire content of `draft.md` (after filling FILL values)
7. **Add Link:** Link to the benchmark you just created (Step 4.1)
8. **Add Link:** Link to your notebook
9. Click **Submit** (NOT "Save Draft")

### Step 4.3 — Verify

1. Open your writeup page — verify cover image displays.
2. Verify track shows "Social Cognition."
3. Verify benchmark link works.
4. Screenshot confirmation page for your records.

---

## Part 5 — Post-Submission (until deadline)

You can **edit the writeup after submission** until April 17 deadline. Use
this window to:

1. Run additional models if free quotas allow.
2. Fine-tune narrative based on your own re-read.
3. Add phase diagram figures (run `writeup/make_phase_diagram.py` after
   full analysis — generates per-model visuals).
4. Final proof-read for typos.

**Do NOT change:** track, benchmark link. Changing these may invalidate
judging.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `HTTP 401` from Gemini | Wrong or expired API key — regenerate |
| `HTTP 429` rate limit | Increase `--sleep` value in runner |
| `scipy not found` | Not needed — analyzer has pure-Python fallback |
| `PIL not found` | `pip install Pillow` OR use `cover.svg` + browser screenshot |
| Groq says "model deprecated" | Update model ID in `free_api_clients.py` from https://console.groq.com/docs/models |
| Total run time too long | Run Gemini + Groq in parallel terminal windows |

---

## Emergency Minimal Submission Path

If you only have 1 hour left and no model runs yet:

1. Run ONE model (Gemini 2.0 Flash, `--limit 80`) — takes ~6 minutes.
2. Skip empty FILL values; write "Preliminary results from a single model
   are reported; full 6-model analysis in our public repo."
3. Submit with partial data — ECB's theoretical novelty alone (4 theorems,
   k\* measurement, phase diagrams) scores on 30% Novelty criterion.

**You have 5 days. Don't take this path unless absolutely necessary.**

---

## Final Checklist

- [ ] GOOGLE_API_KEY set
- [ ] GROQ_API_KEY set (at least one free provider works)
- [ ] `runner.py --model gemini-2.0-flash --limit 10` runs clean
- [ ] Full run on at least 3 models completed
- [ ] `analyzer.py` produces populated `analysis.json`
- [ ] All FILL placeholders replaced in `draft.md`
- [ ] Word count 1480–1499
- [ ] Cover image `cover.png` looks correct
- [ ] Kaggle benchmark created (`epistemic-curie-benchmark`)
- [ ] Kaggle writeup submitted (NOT draft)
- [ ] Track = "Social Cognition"

Deadline: **April 17, 2026, 4:59 AM GMT+5.**
