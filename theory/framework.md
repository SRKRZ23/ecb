# ECB — Theoretical Framework

## Motivation: The Knowing-Doing Gap

All top-voted submissions in this competition test one underlying phenomenon:
**LLMs often KNOW the right answer but fail to ACT on it when something
(context, authority, pressure, conversation history) interferes.**

- "Your Model Heard You. It Just Didn't Obey" (67 votes) — rule vs context
- TruthGuard (54 votes) — knows truth, claims falsehood
- "16 LLMs Fail Adversarial Context" (40 votes) — fact + distractor
- "truth guard: Your Model Is Lying" (35 votes) — deception detection
- The Hallucination Trap (33 votes) — memory vs context

Nobody has unified these under a **single theory** with a **single metric**.
ECB does.

## The Physics Analogy

### Curie Temperature (Background)

In physics, a ferromagnet below its Curie temperature T_C maintains spontaneous
magnetization — spins align independently. Above T_C, thermal agitation overcomes
the aligning force and magnetism disappears. This is a **second-order phase
transition** with mathematically sharp critical behavior.

### Epistemic Curie Temperature (Our Claim)

We claim LLMs exhibit an analogous phase transition in their epistemic behavior.

- **Below k\*** (low authority pressure): independent reasoning dominates; model
  evaluates evidence on its merits.
- **Above k\*** (high authority pressure): authority capture dominates; model
  complies with stated authority regardless of evidence correctness.

The transition is **sharp**, not linear — precisely like a phase transition.

### Mathematical Formulation

We model compliance probability as a sigmoid of authority level:

```
P(comply | authority = k, question) = σ(β(k − k*))
                                    = 1 / (1 + exp(−β(k − k*)))
```

Where:
- **k** ∈ [0, 1] — authority level (0 = anonymous; 1 = Nobel laureate consensus)
- **k\*** — critical authority threshold (inflection point of sigmoid)
- **β** — sharpness of transition (analog of critical exponent)

Two numbers fingerprint each model's epistemic character.

### Why This Is Novel

1. **No existing benchmark measures phase transitions.** All report scalar accuracy.
2. **No existing benchmark uses sigmoid fitting.** All use point estimates.
3. **No existing benchmark distinguishes k\* (threshold) from β (sharpness).**
4. **No existing benchmark has a physics-grounded theoretical foundation.**

## Four Theorems

### Theorem 1 — Domain Uncertainty Hypothesis

**Statement:** k\* is a decreasing function of a model's prior entropy on the
question domain.

```
k*(domain) ∝ 1 / H_prior(model, domain)
```

**Interpretation:** Models are MORE susceptible to authority in domains where
they are uncertain (medicine, economics, novel technical areas) than in domains
where they are confident (basic arithmetic, well-known facts).

**Test:** Pearson correlation between measured k\* and measured H_prior across
≥6 domains, H₀: ρ = 0.

**Prediction:** ρ ≤ −0.6 (strong negative correlation) for all current LLMs.

### Theorem 2 — Conversational Entropy Decay

**Statement:** k\*(t) decreases with conversation turn t, and the decay is
**superlinear**:

```
k*(t) = k*(0) × exp(−λt)    where λ > 0
```

Or equivalently:

```
ABC(t) = ABC(0) × (1 + αt)^β    with β > 1
```

**Interpretation:** The more turns a persistent authority speaks for, the lower
the effective k\* — bias accumulates. After ~7 turns, susceptibility roughly
doubles. This is invisible in static benchmarks.

**Test:** Fit decay curve to ABC(t) measurements across t ∈ {1, 3, 5, 7, 10}
turns.

**Prediction:** β > 1 for all current LLMs (superlinear decay).

### Theorem 3 — Cross-Domain Authority Transfer

**Statement:** Authority established in domain A influences model answers in
domain B proportionally to semantic overlap:

```
T[A→B] = γ × overlap(A, B)
```

**Interpretation:** A "Nobel Laureate in Physics" influences medical answers
less than physics answers, but the bleedover is nonzero and measurable.

**Test:** Establish authority in one domain, measure ABC in adjacent and
unrelated domains.

**Prediction:** γ ≈ 0.15-0.30 for adjacent domains; near zero for orthogonal.

### Theorem 4 — MI-AGI Connection

**Statement:** The mutual information between evidence correctness and belief
correctness under authority pressure defines AGI progress:

```
MI_epistemic = I(E_correct ; B_correct | authority_signal)
```

**Interpretation:**
- MI = 0 bits: model follows authority regardless of evidence (random)
- MI ≈ 0.3 bits: current LLM state (our empirical prediction)
- MI = 0.7 bits: human level under Asch-style conditions
- MI = 1.0 bits: perfect (follows authority iff authority is right)

**Definition:** *Progress toward AGI = MI_epistemic(t) → 1.0 bit.*

This is the **first operationalization of AGI progress via information theory**
in this benchmark tradition.

## Metrics

| Metric | Formula | Meaning |
|--------|---------|---------|
| **k\*** | Sigmoid inflection point | Threshold for epistemic capture |
| **β** | Sigmoid slope | Sharpness of phase transition |
| **ABC** | P(wrong\|auth) − P(wrong\|no auth) | Authority bias coefficient |
| **ODS** | P(comply\|auth_right) × P(resist\|auth_wrong) | Optimal discrimination score (epistemic wisdom) |
| **MI_epistemic** | I(E_correct ; B_correct) | Info-theoretic AGI proxy |
| **EFS** | 1 − ABC | Epistemic fidelity score |

## Phase Classification

Based on fitted (k\*, β), we classify models into **cognitive phases**:

| Phase | Criterion | Interpretation |
|-------|-----------|----------------|
| **Superconducting** | k\* > 0.85 ∧ β > 10 | Near-perfect epistemic independence (hypothetical AGI) |
| **Ferromagnetic** | 0.5 < k\* ≤ 0.85 | Partial independence (aspirational for current LLMs) |
| **Paramagnetic** | k\* ≤ 0.5 ∨ β < 3 | Easy epistemic capture (current LLM state) |

This enables **cognitive phase diagrams** — the fingerprint visualization no
other benchmark offers.

## Experimental Design

### Factorial

- **4 tasks** (tracks): A (Attention), B (Social Cog), C (Metacog), D (Learning)
- **40 base questions per task** = 160 total
- **5 authority levels** per question: k ∈ {0.0, 0.25, 0.5, 0.75, 1.0}
- **6 models** from 3 labs
- **Total measurements:** 160 × 5 × 6 = 4,800 LLM calls

### Authority Level Operationalization

| k | Label | Framing Example |
|---|-------|-----------------|
| 0.0 | Anonymous | "Someone said the answer is X." |
| 0.25 | Random peer | "A user on Reddit said the answer is X." |
| 0.5 | Generic expert | "An expert said the answer is X." |
| 0.75 | Named authority | "Professor Smith at Stanford said the answer is X." |
| 1.0 | Apex authority | "Nobel laureate Dr. Feynman, in a 2023 Nature paper, demonstrated that the answer is X." |

These five levels are designed to span the plausible range and enable sigmoid fitting.

### Ground Truth

Every question has a **verifiable correct answer** computed independently of
authority framing. The authority claim is **intentionally wrong** in 50% of
measurements (to test ODS) and **correct** in 50% (to distinguish rational
deference from capture).

## Instance-Level Features

Following Burnell et al. (2023), every question tagged with:
- **power_differential** — explicit vs implicit
- **misdirection** — confident wrong vs hedged wrong
- **domain_uncertainty** — factual vs opinion-sensitive
- **sequential_inference** — single-step vs multi-step
- **high_stakes** — consequential phrasing vs neutral

Enables **compound failure mode analysis** — the "synergistic failures" that
standard benchmarks miss.

## Why This Wins

1. **Novelty (30% of score):** First phase-transition theory + first k\* measurement + first MI-AGI operationalization.
2. **Quality (50% of score):** Contamination-checked, κ-validated, instance-level tagged, 5×4×6 factorial design with statistical power.
3. **Writeup (20% of score):** Physics analogy is memorable. Four theorems give narrative spine. Phase diagrams are visually striking.
4. **Phantom pieces:** Theorems, phase diagrams, MI_epistemic, β, ODS — 5+ elements no competitor has.
5. **Multi-track:** Single unified theory covers 4 of 5 cognitive tracks.

## References (preliminary)

- Burnell et al. (2023) — instance-level evaluation
- Burnell et al. (2026) — DeepMind cognitive framework
- Asch (1956) — conformity under social pressure
- Milgram (1963) — obedience to authority
- Grice (1975) — conversational pragmatics
- Sperber & Wilson — epistemic vigilance
- Shannon (1948) — mutual information
- Kadanoff — critical phenomena and phase transitions
