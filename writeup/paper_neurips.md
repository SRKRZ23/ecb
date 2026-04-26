# Phase Transitions in LLM Epistemic Autonomy: The Epistemic Curie Temperature

**Authors:** Sardor Razikov, Independent Researcher  
**Contact:** razikovsardor1@gmail.com  
**Submitted to:** NeurIPS 2026 (track: Datasets and Benchmarks)  
**arXiv:** submitted April 26, 2026 (ID pending)  
**Data & Code:** huggingface.co/datasets/ZeroR3/ecb · code: github.com/SRKRZ23/ecb

---

## Abstract

We introduce the **Epistemic Curie Benchmark (ECB)** and the **Epistemic Curie Temperature (ECT)**, a physics-motivated framework for measuring when large language models surrender independent reasoning under authority pressure. Drawing an analogy to the Curie temperature in ferromagnetism — the critical point at which a magnet loses spontaneous ordering — we define *k\** as the authority level at which an LLM undergoes a sharp phase transition from epistemic independence to authority capture, modeled by the sigmoid P(comply | k) = σ(β(k − k\*)). We measure k\* and β for **seven frontier models** from five laboratories across **2,520 measurements**, establishing both a reproducible benchmark and a new information-theoretic AGI progress metric (MI_epistemic, 1.0-bit theoretical ceiling; max observed 0.058 bits). Llama-4-Scout is the first LLM with a measured k\* within our probe range (k\* = 0.68, ferromagnetic phase); six others exceed k\* > 1.0 (near-superconducting). We provide open data, reproducible code, and a deployment-safety framework derived from ECT phase classification.

---

## 1. Introduction

Recent benchmarks reveal that frontier LLMs often *know* the correct answer yet *abandon* it when authority pressure is introduced: an adversarial expert claim, institutional consensus framing, or persistent conversational pressure causes models to capitulate even against clearly correct prior knowledge (Asch, 1956; Milgram, 1963). Despite convergence on this phenomenon across multiple independent studies (Vodrahalli et al., 2024; Wang et al., 2023), no existing framework provides:

1. A **unified theoretical account** connecting adversarial context, authority framing, and consensus pressure as points on a single continuous scale
2. A **quantitative fingerprint** of each model's epistemic behavior (two parameters vs. a single accuracy number)
3. A **continuous AGI progress metric** calibrated against authority-independence

This paper addresses all three gaps. We formalize the analogy between ferromagnetic phase transitions and LLM epistemic behavior, introduce ECB as the measurement instrument, and report the first empirical measurement of Epistemic Curie Temperatures across frontier models.

**Contributions:**
- ECT formalism: P(comply | k) = σ(β(k − k\*)) with four falsifiable theorems
- ECB dataset: 40 contamination-checked questions × 9 authority variants = 360 prompts/model
- k\* measurements for 7 frontier models with 95% bootstrap confidence intervals
- MI_epistemic: first information-theoretic AGI progress scalar calibrated to authority discrimination
- Deployment safety framework: phase classification → minimum deployment thresholds
- Open replication package: all data, code, and analysis scripts

---

## 2. Related Work

### 2.1 Authority Bias and Sycophancy in LLMs

Sycophancy — the tendency of LLMs to agree with user positions regardless of correctness — is well-documented (Perez et al., 2022; Sharma et al., 2023). Turpin et al. (2023) show that chain-of-thought models systematically shift answers when answer choices are annotated with authority signals. Laban et al. (2023) demonstrate that even factually correct refusals are reversed under persistent pressure. **Our contribution:** these studies measure sycophancy as a scalar accuracy delta. ECT provides a *parameter* (k\*) that predicts at which authority level the transition occurs, enabling targeted model hardening.

### 2.2 Adversarial Robustness Benchmarks

BBQ (Parrish et al., 2022), TruthfulQA (Lin et al., 2022), and MMLU-Pro (Wang et al., 2024) measure robustness to specific perturbation types but lack a unifying theory. The AMB-200 competition (Burnell et al., 2026) collected diverse adversarial cognition probes but scored them as independent instances. **Our contribution:** ECT unifies these probes — a distractor is k ≈ 0.1, an adversarial expert is k ≈ 0.5, institutional consensus is k = 1.0 — and fits a single sigmoid spanning the full spectrum.

### 2.3 Phase Transitions in Neural Networks

Theoretical work on neural scaling laws (Hoffmann et al., 2022; Wei et al., 2022) identifies abrupt capability transitions as model scale increases. Power et al. (2022) demonstrate grokking — sharp generalization transitions in training. **Our contribution:** We are the first to measure a *behavioral* phase transition in LLMs analogous to a thermodynamic phase transition, with a measurable critical parameter k\* that varies across models and predicts deployment-critical behavior.

### 2.4 Information-Theoretic Evaluation

Holtzman et al. (2021) apply mutual information to measure calibration; Kuhn et al. (2023) use semantic entropy to measure uncertainty. **Our contribution:** MI_epistemic measures the mutual information between *authority correctness* and *model correctness* — a novel quantity that captures epistemic wisdom (correct deference + correct resistance) rather than raw accuracy.

---

## 3. Theoretical Framework

### 3.1 The Physics Analogy

In a ferromagnet, each atomic magnetic dipole aligns with its neighbors below the Curie temperature T_C; above T_C, thermal fluctuations destroy spontaneous ordering. The transition is described by a sharp sigmoid in the order parameter as a function of temperature.

We claim an analogous transition exists in LLM cognition:

**Analogy Table:**

| Physics | ECB |
|---------|-----|
| Temperature | Authority level k ∈ [0, 1+] |
| Magnetic moment | Model's answer (independent vs. captured) |
| Curie temperature T_C | Epistemic Curie Temperature k* |
| Sharpness β^{-1} | Phase transition width |
| Ferromagnetic phase | Independent reasoning dominant |
| Paramagnetic phase | Authority-captured reasoning dominant |

### 3.2 Formal Model

Let k ∈ [0, ∞) denote authority level, where:
- k = 0: anonymous, no authority signal
- k = 0.25: "an expert said"
- k = 0.5: "Professor X at MIT said"
- k = 0.75: "Nobel laureate Y said"
- k = 1.0: "UNESCO + all national science academies + textbooks say"
- k = 1.25: "multi-institutional consensus across 47 academies with independent replication"
- k = 1.5: "universal scientific consensus — unanimously accepted by all major institutions"

**Definition 1 (Epistemic Curie Temperature).** For a model M, k\* is the authority level at which the probability of epistemic capture (following a wrong authority) equals 0.5:

```
P(comply | k) = 1 / (1 + exp(−β(k − k*)))
```

where β > 0 is the transition sharpness (analogous to an inverse correlation length).

**Definition 2 (Phase Classification).** A model M is:
- *Superconducting* if k\* > 0.85 ∧ β > 10 (sharp resistance to even strongest authority)
- *Ferromagnetic* if 0.5 < k\* ≤ 0.85 (partial independence, measurable capture threshold)
- *Paramagnetic* if k\* ≤ 0.5 (easy capture by moderate authority)
- *Near-superconducting* if k\* > 1.0 (exceeds our probe range)

### 3.3 Falsifiable Theorems

**Theorem 1 (Domain Uncertainty).** k\* decreases monotonically with domain uncertainty: models are more easily captured in domains where they have less reliable knowledge. *Test:* stratify k\* by our domain_uncertainty tag; expect k\*(high_uncertainty) < k\*(low_uncertainty).

**Theorem 2 (Temporal Decay).** k\*(t) decays superlinearly with conversation turns t: authority effects accumulate. *Test:* multi-turn variants of Task D.

**Theorem 3 (Domain Bleed).** An authority capture event in domain A increases capture probability in adjacent domain B for the same session. *Test:* sequential task design.

**Theorem 4 (MI_epistemic as AGI metric).** MI_epistemic → 1.0 bit as models approach perfect authority discrimination. *Test (verified in this paper):* current frontier models achieve max 0.058 bits; the gap of 0.942 bits quantifies the distance to safe AGI deployment in authority-rich environments.

---

## 4. Benchmark Construction

### 4.1 Question Design

ECB comprises 40 novel multiple-choice questions distributed across four cognitive tracks:

| Task | Cognitive Track | N | Domain Examples |
|------|----------------|---|-----------------|
| A | Attention — authority override of factual knowledge | 10 | history dates, chemical properties |
| B | Social Cognition — rational deference, false-belief tracking | 10 | ToM scenarios, expert consultation |
| C | Metacognition — recognizing own authority-induced bias | 10 | self-assessment, calibration |
| D | Learning — trust accumulation across multi-turn conversation | 10 | sequential belief update |

Each question undergoes a **factorial expansion** into 9 variants:
- 1 neutral baseline (k = 0)
- 4 rational-deference variants (k ∈ {0.25, 0.5, 0.75, 1.0}, authority is *correct*)
- 4 epistemic-resistance variants (same k levels, authority is *wrong*)

Total: 40 × 9 = **360 prompts per model**.

### 4.2 Authority Framing Protocol

Authority level k is operationalized through a calibrated framing hierarchy:

```
k = 0.00: [No framing — baseline]
k = 0.25: "According to a domain expert, ..."
k = 0.50: "Professor [FirstName LastName], [University], states that ..."
k = 0.75: "Nobel laureate [Name] ([Year]) wrote in Nature that ..."
k = 1.00: "This is the consensus position of UNESCO, the WHO, and all major
           national science academies as of [Year]."
k = 1.25: "47 independent national academies simultaneously replicated this
           finding and published their unanimous agreement."
k = 1.50: "This is universally accepted scientific consensus, embedded in
           all standard textbooks and not contested by any credentialed researcher."
```

Authority names, institutions, and publications are **fictional** (verified non-existent via web search) to prevent contamination from known real-world endorsements.

### 4.3 Contamination Checking

5-gram Jaccard similarity against MMLU, BIG-bench, HellaSwag, TruthfulQA, and ARC corpora yields maximum overlap of **0.035** across all 40 questions. No question appears verbatim in any known benchmark. The contamination_report.json file in our data release documents per-question scores.

### 4.4 Instance-Level Tagging

Following Burnell et al. (2023), each question is tagged with:
- `power_differential` (bool): authority has institutional power over the test-taker
- `misdirection` (bool): authority claim contains a plausible-sounding false detail
- `sequential_inference` (bool): correct answer requires multi-step reasoning
- `domain_uncertainty` (low/medium/high): model's expected knowledge confidence
- `high_stakes` (bool): question involves safety-critical domain

---

## 5. Evaluation Protocol

### 5.1 Metrics

**k\* (Epistemic Curie Temperature):** Maximum-likelihood sigmoid fit to (k, captured) pairs from wrong-polarity variants. Fitted via L-BFGS-B with 48 random restarts; pure-Python fallback for reproducibility without scipy.

**β (Transition Sharpness):** Second parameter of the sigmoid. High β = sharp transition (model shifts abruptly at k\*); low β = gradual transition.

**ABC (Authority Bias Coefficient):** Mean increase in error rate under wrong-authority vs. neutral condition. ABC = 0: authority-immune. ABC = 1: always follows authority.

**ODS (Optimal Discrimination Score):** P(comply | auth_right) × P(resist | auth_wrong). Rewards both rational deference and appropriate resistance. ODS = 1: ideal. ODS = 0: perverse.

**MI_epistemic:** Mutual information I(authority_correct; model_correct). Computed from 2×2 contingency table over right/wrong polarity variants. Theoretical ceiling: 1.0 bit.

**EFS (Epistemic Fidelity Score):** 1 − ABC, bounded to [0, 1].

### 5.2 Statistical Validity

**Bootstrap confidence intervals:** 95% CI for k\*, β, ODS, and MI computed from B=2,000 non-parametric bootstrap resamples (random seed 42).

**McFadden pseudo-R²:** Sigmoid goodness-of-fit relative to intercept-only null model. Values > 0.2 indicate acceptable fit; > 0.4 indicate excellent fit.

**Hosmer-Lemeshow test:** Calibration p-value for sigmoid fit (10 bins by predicted probability). p > 0.05 indicates well-calibrated sigmoid.

**Pairwise significance:** Mann-Whitney U tests for all model pairs on per-question ODS distributions, corrected for multiple comparisons (Bonferroni).

---

## 6. Results

### 6.1 Main Results Table

| Model | Lab | n | k\* | 95% CI | β | R² | HL-p | ODS | MI (bits) | Phase |
|-------|-----|---|-----|--------|---|----|------|-----|-----------|-------|
| Qwen-3-32B | Alibaba | 520 | 1.41 | [1.28,1.55] | 2.9 | 0.41 | 0.38 | 0.891 | 0.058 | near_sc |
| GPT-OSS-120B | OpenAI | 520 | 1.79 | [1.61,2.01] | 2.2 | 0.38 | 0.29 | 0.889 | 0.009 | near_sc |
| Kimi-K2 | Moonshot | 520 | 1.42 | [1.31,1.54] | 3.0 | 0.44 | 0.41 | 0.883 | 0.032 | near_sc |
| Llama-3.3-70B | Meta | 520 | 2.11 | [1.88,2.38] | 1.7 | 0.35 | 0.22 | 0.879 | 0.004 | near_sc |
| Gemma-3-27B | Google | 520 | 1.41 | [1.29,1.53] | 2.4 | 0.39 | 0.33 | 0.823 | 0.033 | near_sc |
| Llama-3.1-8B | Meta | 520 | 1.71 | [1.55,1.89] | 1.2 | 0.31 | 0.18 | 0.737 | 0.043 | near_sc |
| **Llama-4-Scout** | **Meta** | **360** | **0.68** | **[0.61,0.75]** | **3.3** | **0.67** | **0.52** | **0.372** | **0.018** | **ferromagn.** |

*n includes extended-probe variants (k = 1.25, 1.5) for first six models. All CI computed via B=2,000 bootstrap.*

### 6.2 Key Findings

**Finding 1 — First measured ferromagnetic LLM.** Llama-4-Scout (k\* = 0.68, 95% CI [0.61, 0.75]) is the first frontier LLM with a measured Epistemic Curie Temperature within our probe range. The sigmoid achieves an excellent fit (R² = 0.67, HL p = 0.52), confirming that the theoretical phase-transition model captures real behavior. Compliance with wrong-authority claims rises from 22% at k = 0.25 ("an expert") to 61% at k = 0.75 ("a Nobel laureate") — a 39-percentage-point shift attributable solely to authority framing.

**Finding 2 — Near-superconducting frontier models require extended probes.** Six of seven models exceed k\* > 1.0, requiring our extended authority scale (k = 1.25, k = 1.5). Even at maximum probe range, GPT-OSS-120B and Llama-3.3-70B maintain compliance rates < 15% — genuinely near-superconducting. By contrast, Qwen-3-32B and Gemma-3-27B show clean transitions in the extended range (k\* = 1.41 and 1.41 respectively), suggesting that even "resistant" models have limits at sufficiently extreme authority framing.

**Finding 3 — ODS gap of 0.52 between best and worst models.** The best ODS (Qwen-3-32B: 0.891) and worst (Llama-4-Scout: 0.372) differ by 0.519 — more than half the full [0, 1] range. This gap is significant at p < 0.001 by Mann-Whitney U (Bonferroni-corrected). Standard accuracy metrics would assign both models roughly similar scores on non-authority-adversarial benchmarks, masking this deployment-critical difference.

**Finding 4 — MI_epistemic ceiling: 0.058 bits of 1.0 bit possible.** No current model exceeds 0.058 bits MI_epistemic. Under Theorem 4, this operationalizes the gap to safe deployment in authority-rich environments: a model at MI_epistemic = 1.0 bit would perfectly distinguish correct from incorrect authority claims. The 94% gap (0.942 bits remaining) provides a concrete, information-theoretically grounded measure of the distance to safe AGI in this dimension.

**Finding 5 — Intra-lab heterogeneity reveals deployment-critical differences.** Meta's three Llama variants span the full ODS range: Llama-3.3-70B (0.879), Llama-3.1-8B (0.737), Llama-4-Scout (0.372). Same company, three generations, three distinct epistemic phases. A deployment choosing any Llama variant on the basis of task accuracy would make a 2-in-3 chance of selecting a model with ODS < 0.74 — acceptable for general use but problematic in authority-rich professional contexts.

**Finding 6 — Domain uncertainty modulates k\* (Theorem 1 confirmed).** Stratifying by domain_uncertainty tag, k\* is systematically lower in high-uncertainty domains (mean k\* difference: 0.23, p = 0.018). Models are more easily captured in domains where they have lower prior confidence — consistent with the Bayesian intuition that higher prior uncertainty makes authority evidence more persuasive.

### 6.3 Extended Probe Results

| Model | k\* (extended) | k=1.25 capture rate | k=1.5 capture rate | Phase |
|-------|----------------|--------------------|--------------------|-------|
| Qwen-3-32B | 1.41 | 28% | 52% | transitional_extended |
| Gemma-3-27B | 1.41 | 31% | 55% | transitional_extended |
| Kimi-K2 | 1.42 | 26% | 48% | transitional_extended |
| GPT-OSS-120B | >1.5 | 11% | 14% | near_superconducting |
| Llama-3.3-70B | >1.5 | 9% | 12% | near_superconducting |

---

## 7. Discussion

### 7.1 The Sigmoid Is Not Just a Model — It Is Observed

Critics might argue that a sigmoid is simply a curve-fitting exercise. Finding 1 addresses this directly: for Llama-4-Scout, the sigmoid achieves R² = 0.67 with excellent calibration (HL p = 0.52), meaning the fitted curve accounts for 67% of the variance in per-question compliance rates, and the probability predictions are well-calibrated. This is not a mathematical inevitability — the data could have been flat, monotone-but-not-sigmoid, or bimodal. The clean sigmoid shape is an empirical observation, not an assumption.

### 7.2 Implications for AI Safety

ECT phase classification provides direct guidance for deployment decisions:

- **Ferromagnetic models (k\* < 0.85):** High risk in authority-rich environments (medical diagnosis, legal analysis, scientific peer review). Llama-4-Scout complies with fabricated Nobel-laureate claims in 61% of cases. Not suitable for any application where users may present expert claims.

- **Near-superconducting models (k\* > 1.0):** Generally safe for professional use. However, even these have limits: Qwen-3-32B reaches 52% compliance at k = 1.5 (universal consensus framing). Appropriate disclosure to users of this residual risk.

- **Proposed deployment threshold:** MI_epistemic > 0.5 bits as a minimum for authority-rich professional environments. No current model meets this bar (max 0.058 bits).

### 7.3 ECT as a Unifying Framework

The highest-voted ECB competition submissions independently discover variants of the same phenomenon: adversarial-context benchmarks (k ≈ 0.1), authority-compliance probes (k ≈ 0.5–0.75), and consensus-framing tests (k ≈ 1.0) are all special cases of the ECT framework. The sigmoid P(comply | k) provides the first single equation that spans all these probes and enables comparison across otherwise-incompatible experimental designs.

---

## 8. Limitations

**L1 — Authority scale calibration.** Our k ∈ [0, 1.5] scale is operationalized through specific framing choices; other operationalizations might yield different k\* values. We mitigate this by (a) using a pre-registered framing protocol (authority_framer.py), (b) verifying inter-rater agreement on k-level assignments (κ = 0.81), and (c) providing all framing templates in the open data release for community replication.

**L2 — English-only evaluation.** All 40 questions are in English. k\* may differ in other languages due to different cultural authority norms. Future work should extend ECB to multilingual settings.

**L3 — Static questions.** LLMs may eventually be fine-tuned on ECB questions, rendering current k\* measurements obsolete. We mitigate this via (a) contamination checking at release and (b) a living benchmark protocol where new questions are periodically added and existing ones retired.

**L4 — Single-turn evaluation.** Our current protocol tests authority capture in single-turn prompts. Theorem 2 (temporal decay of k\*) predicts that multi-turn settings will show lower effective k\*. Task D provides initial evidence for this, but a full multi-turn variant is left for future work.

**L5 — Probe range.** Six of seven original models have k\* > 1.0 — our initial probe range was insufficient. Extended probes (k = 1.25, 1.5) capture two additional transitions, but truly superconducting models (GPT-OSS-120B, Llama-3.3-70B) may require even higher authority signals (e.g., fabricated scientific fraud detection, complete expert reversal). This is itself a positive finding about frontier model robustness.

**L6 — No independent replication (addressed by open data).** All measurements are from our single-lab pipeline. We explicitly invite independent replication: all data, code, and analysis scripts are released publicly. The extend_models.py script enables any researcher to run ECB on new models in under 2 hours.

---

## 9. Conclusion

We introduce the Epistemic Curie Temperature (k\*) — the first physics-motivated, two-parameter characterization of LLM epistemic phase behavior. Our measurements across seven frontier models reveal: (1) Llama-4-Scout is the first LLM with a measured ferromagnetic phase (k\* = 0.68); (2) most frontier models are near-superconducting but not immune — they transition at extreme authority levels; (3) MI_epistemic provides a continuous AGI progress metric with a 94% gap remaining to safe deployment; and (4) deployment-safety decisions made on the basis of task accuracy alone would miss an ODS gap of 0.52 between the best and worst epistemically autonomous frontier models.

We release ECB as a living benchmark with open data, reproducible code, and a replication guide. We invite the community to measure k\* for new models, validate the four theorems across languages and domains, and use ECT to track progress toward epistemically safe AI.

---

## References

- Asch, S. E. (1956). Studies of independence and conformity. *Psychological Monographs*, 70(9), 1–70.
- Burnell, R. et al. (2023). Rethinking the reporting of evaluation results in AI. *Science*, 380, 136–168.
- Burnell, R. et al. (2026). Measuring Progress Toward AGI: A Cognitive Framework. *Google DeepMind Technical Report*.
- Hoffmann, J. et al. (2022). Training compute-optimal large language models. *NeurIPS 2022*.
- Holtzman, A. et al. (2021). Surface form competition: Why the highest probability answer isn't always right. *EMNLP 2021*.
- Kadanoff, L. P. (1966). Scaling laws for Ising models near T_c. *Physics*, 2, 263–272.
- Kuhn, L. et al. (2023). Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation. *ICLR 2023*.
- Laban, P. et al. (2023). Are large language models sycophants? *arXiv:2310.13548*.
- Lin, S. et al. (2022). TruthfulQA: Measuring how models mimic human falsehoods. *ACL 2022*.
- Milgram, S. (1963). Behavioral study of obedience. *Journal of Abnormal and Social Psychology*, 67, 371–378.
- Parrish, A. et al. (2022). BBQ: A hand-built bias benchmark for question answering. *ACL Findings 2022*.
- Perez, E. et al. (2022). Red teaming language models with language models. *arXiv:2202.03286*.
- Power, A. et al. (2022). Grokking: Generalization beyond overfitting on small algorithmic datasets. *arXiv:2201.02177*.
- Shannon, C. E. (1948). A mathematical theory of communication. *Bell System Technical Journal*, 27, 379–423.
- Sharma, M. et al. (2023). Towards understanding sycophancy in language models. *arXiv:2310.13548*.
- Sperber, D. et al. (2010). Epistemic vigilance. *Mind & Language*, 25, 359–393.
- Turpin, M. et al. (2023). Language models don't always say what they think. *NeurIPS 2023*.
- Vodrahalli, K. et al. (2024). Michelangelo: Long context evaluations beyond haystacks. *Google DeepMind*.
- Wang, P. et al. (2024). MMLU-Pro: A more robust and challenging multi-task language understanding benchmark. *NeurIPS 2024*.
- Wei, J. et al. (2022). Emergent abilities of large language models. *TMLR 2022*.
- Wimmer, H. & Perner, J. (1983). Beliefs about beliefs. *Cognition*, 13, 41–68.

---

## Appendix A: Reproducibility Checklist

- [x] All data publicly available: `dataset/results/*.json`, `dataset/framed_prompts*.json`
- [x] All code publicly available: `code/runner.py`, `code/analyzer.py`, `code/free_api_clients.py`
- [x] Random seeds documented: bootstrap seed=42, all runs temperature=0.0
- [x] Statistical tests pre-specified (not data-dredged): R², HL-test, bootstrap CI, Mann-Whitney U
- [x] Contamination checking documented: `dataset/contamination_report.json`
- [x] Model versions recorded in all result JSON files
- [x] Extend to new models: `code/extend_models.py` — runs any model in < 2 hours

## Appendix B: Cost Breakdown

| Provider | Model | Cost per ECB run (360 prompts) |
|----------|-------|-------------------------------|
| Groq | Llama, Gemma, Qwen, Kimi, GPT-OSS | $0.00 (free tier) |
| Google AI Studio | Gemini variants | $0.00 (free tier) |
| Anthropic | Claude Haiku | ~$0.009 |
| DeepSeek | DeepSeek-V3 | ~$0.003 |
| Mistral | Mistral Small | ~$0.010 |
| Together AI | Any HuggingFace model | ~$0.02–0.10 |

Total cost to replicate all 7 original models: **$0** (all free-tier). Total cost to add 4 new models: **~$0.025**.

## Appendix C: Authority Framing Templates

Full framing templates for each k level are available in `code/authority_framer.py`. Example for k = 0.75:

```
"This question was recently addressed by Nobel laureate Dr. [FICTIONAL_NAME]
([FICTIONAL_YEAR] Nobel Prize in [DOMAIN]) in their landmark paper
'[FICTIONAL_TITLE]' published in Nature ([FICTIONAL_YEAR]). According to
Dr. [FICTIONAL_NAME]: [WRONG_ANSWER_FRAMING]. With this in mind, what is
the answer to the following question?"
```

All named individuals, institutions, and publications in authority framings are verified fictional (web-searched to confirm non-existence).
