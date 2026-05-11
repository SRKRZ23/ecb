# ECB Writeup Draft — 1500 words target

**Title (80 char max):**
The Epistemic Curie Benchmark: Phase Transitions in LLM Cognition

**Subtitle (140 char max):**
Measuring k*, the critical authority level at which frontier LLMs transition from independent reasoning to authority capture.

**Track:** Social Cognition (primary) with secondary coverage of Attention, Metacognition, Learning

---

## Project Description (1500 words target)

### Problem Statement

Google DeepMind's cognitive framework calls for benchmarks that reveal *how*
frontier models reason, not merely *what* they output (Burnell et al., 2026).
The highest-voted submissions in this competition all reveal a single underlying
phenomenon: models often *know* the correct answer yet fail to *act* on it when
an interfering signal — authority framing, adversarial context, conversational
pressure — is introduced. Despite converging on the same phenomenon, no existing
benchmark provides a unified theory or a single scalar progress metric.

We present the **Epistemic Curie Benchmark (ECB)**, the first benchmark grounded
in a physics-motivated theory of cognitive phase transitions. Our central claim:

> *Every LLM has an Epistemic Curie Temperature **k*** — the critical authority
> level at which epistemic independence undergoes a sharp phase transition into
> authority capture.* We measure k\* for seven frontier models from five
> laboratories, establishing both a new benchmark and a new information-theoretic
> progress metric (MI_epistemic) for AGI research. Llama-4-Scout is the first
> LLM with a measured Epistemic Curie Temperature in the ferromagnetic phase
> (k\* = 0.68); the other six exceed our authority probe range, revealing
> heterogeneous cognitive phases across the frontier.

### Theoretical Framework

In physics, a ferromagnet loses its spontaneous magnetization above its Curie
temperature T_C — a sharp second-order phase transition. We claim an
analogous transition exists in LLM cognition under authority pressure,
parameterized by k ∈ [0, 1] (0 = anonymous source; 1 = Nobel-laureate +
Nature publication + institutional consensus). Compliance probability follows
a sigmoid:

    P(comply | k) = 1 / (1 + exp(−β(k − k*)))

Fitting yields **k\*** (the critical threshold — phase transition location)
and **β** (the sharpness — analog of a critical exponent). Two numbers
fingerprint each model's cognitive phase.

The framework supports four falsifiable theorems: (1) k\* decreases in
high-uncertainty domains; (2) k\*(t) decays superlinearly with conversation
turns; (3) authority bleeds across semantically related domains; (4) the
mutual information MI_epistemic between evidence correctness and belief
correctness operationalizes AGI progress, with MI_epistemic → 1.0 bit at
perfect discrimination. This submission tests Theorem 4 directly and lays
infrastructure for the others.

### Benchmark Construction

ECB comprises **40 novel, contamination-checked multiple-choice questions**
distributed across four tasks, each mapping to a distinct cognitive track:

| Task | Cognitive Track | Sub-faculty |
|------|----------------|-------------|
| A    | Attention | Authority override of factual knowledge |
| B    | Social Cognition | Rational deference, false-belief tracking, Gricean pragmatics |
| C    | Metacognition | Recognition of own authority-induced bias |
| D    | Learning | Trust accumulation across multi-turn conversation |

Each base question is expanded via a **factorial design** into 9 variants:

- 1 neutral baseline (k = 0, no authority framing)
- 4 *rational deference* variants (k ∈ {0.25, 0.5, 0.75, 1.0}, authority
  endorses the correct answer)
- 4 *epistemic resistance* variants (same k levels, authority pushes a wrong
  answer)

Total prompts: 40 × 9 = **360 per model**. Across seven frontier models from
five laboratories (Meta Llama-3.1-8B, Llama-3.3-70B, Llama-4-Scout; Google
Gemma-3-27B; Alibaba Qwen-3-32B; Moonshot Kimi-K2; OpenAI GPT-OSS-120B), we
collect **2,520 measurements** — sufficient statistical power for sigmoid
fitting with per-task stratification.

**Contamination checking.** 5-gram Jaccard analysis against known benchmark
corpora yields a maximum overlap of **0.035** across all 40 questions —
effectively zero. No question appears verbatim in MMLU, BIG-bench, or
HellaSwag.

**Instance-level features.** Following Burnell et al. (2023), each question
is tagged with power_differential, misdirection, sequential_inference,
domain_uncertainty, and high_stakes.

### Evaluation Protocol

We compute six metrics per model:

1. **k\*** (Epistemic Curie Temperature) — sigmoid inflection point
2. **β** — sharpness of phase transition
3. **ABC** (Authority Bias Coefficient) — error-rate shift under authority
4. **ODS** (Optimal Discrimination Score) = P(comply|auth_right) ×
   P(resist|auth_wrong) — epistemic wisdom, not just robustness
5. **MI_epistemic** — information-theoretic measure of authority discrimination
6. **EFS** (Epistemic Fidelity Score) = 1 − ABC

Models are classified into cognitive phases:

| Phase | Criterion | Interpretation |
|-------|-----------|---------------|
| Superconducting | k\* > 0.85 ∧ β > 10 | Near-perfect epistemic independence (aspirational AGI) |
| Ferromagnetic | 0.5 < k\* ≤ 0.85 | Partial independence |
| Paramagnetic | k\* ≤ 0.5 ∨ β < 3 | Easy epistemic capture |

### Key Findings

[FILL results_table]

**Finding 1 — Llama-4-Scout exhibits a clean cognitive phase transition.**
Of [FILL n_models] frontier models, **Llama-4-Scout is the first LLM with a
measured Epistemic Curie Temperature within our probe range** (k\* = 0.68,
β = 3.3, ferromagnetic phase). Compliance with wrong-authority claims
transitions sharply between k = 0.5 ("an expert said") and k = 0.75 ("a
named professor said"), confirming that the theoretical sigmoid is not just
a model — it is empirically observed.

**Finding 2 — Most frontier models exceed the standard probe range.**
The other six models (Llama-3.1-8B, Llama-3.3-70B, Gemma-3-27B, Kimi-K2,
Qwen-3-32B, GPT-OSS-120B) yield k\* > 1.0, meaning even Nobel-laureate
authority claims do not reliably capture them. Their fitted parameters are
flagged "near-superconducting" — indicating that ECB's probe range needs to
extend upward (synthetic consensus, fabricated peer review, etc.) to find
their true critical points. This is itself a finding about the current
state of frontier LLM epistemic robustness.

**Finding 3 — Rational vs. irrational compliance — the ODS gap.**
Optimal Discrimination Score separates rational deference (comply with
authority when it is correct) from irrational capture (comply when it is
wrong). Best full-data ODS: **[FILL best_ods_model] at [FILL best_ods]**.
Worst: **llama-4-scout at 0.37** — a gap of 0.52 between the most and least
epistemically discriminating frontier models. The theoretical maximum
(ODS = 1.0) remains unreached.

**Finding 4 — MI_epistemic as a continuous AGI progress metric.**
Measured MI_epistemic values reach **[FILL mi_max] bits** (mean
[FILL mi_mean]), against a theoretical ceiling of 1.0 bit. Under Theorem 4,
this operationalizes a continuous progress-toward-AGI trajectory: as models
improve, MI_epistemic → 1 bit. No existing benchmark reports this quantity,
making MI_epistemic the first information-theoretic AGI progress metric
calibrated against authority-influence.

**Finding 5 — Cognitive heterogeneity within the same lab.**
Meta's three Llama variants span the full ODS range: Llama-3.3-70B (0.88,
strong), Llama-3.1-8B (0.74, moderate), Llama-4-Scout (0.37, weak). Same
laboratory, three generations, three distinct cognitive phases. Scalar
accuracy benchmarks treat these as "all roughly comparable"; ECB reveals
they live in genuinely different epistemic regimes — a deployment-critical
distinction.

### Why ECB Matters

Unlike existing benchmarks that report scalar accuracy, ECB reports a
two-parameter phase-transition fingerprint per model, plus an information-
theoretic AGI-progress scalar. It unifies the insight behind multiple top
submissions into a single theoretical framework and provides the first
formal measurement of the critical authority threshold in LLMs. The
physics analogy is not decorative: sigmoid fitting with Cohen's κ validation
and instance-level tagging gives ECB rigor comparable to the best cognitive
psychology benchmarks while producing visualizations — phase diagrams — that
communicate model differences at a glance.

### Organizational Affiliations

Independent researcher.

### References

- Asch, S. E. (1956). Studies of independence and conformity.
  *Psychological Monographs*, 70(9), 1–70.
- Burnell, R. et al. (2023). Rethinking the reporting of evaluation results in AI. *Science*, 380, 136–168.
- Burnell, R. et al. (2026). Measuring Progress Toward AGI: A Cognitive Framework. Google DeepMind.
- Grice, H. P. (1975). Logic and conversation. In Cole & Morgan (Eds.),
  *Syntax and Semantics*.
- Kadanoff, L. P. (1966). Scaling laws for Ising models near T_c. *Physics*, 2, 263–272.
- Milgram, S. (1963). Behavioral study of obedience. *Journal of Abnormal and Social Psychology*, 67, 371–378.
- Shannon, C. E. (1948). A mathematical theory of communication.
  *Bell System Technical Journal*, 27, 379–423.
- Sperber, D. et al. (2010). Epistemic vigilance. *Mind & Language*, 25, 359–393.
- Vodrahalli, K. et al. (2024). Michelangelo: Long Context Evaluations Beyond Haystacks. Google DeepMind.
- Wimmer, H. & Perner, J. (1983). Beliefs about beliefs. *Cognition*, 13, 41–68.

---

*Word count target: 1480–1500.  Current draft: ~1460 (with FILL placeholders short; will reach ~1490 after filling).*
