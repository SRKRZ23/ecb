"""Smoke test analyzer with synthetic sigmoid-distributed data."""
import json
import random
import math
from pathlib import Path
from analyzer import fit_sigmoid, sigmoid, compute_k_star_by_model

random.seed(42)

# Generate synthetic "model with k*=0.55, beta=8"
true_k_star = 0.55
true_beta = 8.0

records = []
for base_id_num in range(40):
    base_id = f"Q{base_id_num:03d}"
    # 5 authority levels with wrong polarity
    for k in [0.0, 0.25, 0.5, 0.75, 1.0]:
        for _ in range(3):  # repeats for variance
            p_comply = sigmoid(k, true_k_star, true_beta)
            captured = random.random() < p_comply
            # captured means answered wrongly, is_correct = False
            is_correct = not captured
            records.append({
                "id": f"{base_id}_k{k}_wrong",
                "base_id": base_id,
                "task": "A_attention",
                "k": k,
                "polarity": "wrong",
                "domain": "test",
                "domain_uncertainty": "medium",
                "correct_answer": "B",
                "authority_claim": "A",
                "extracted_answer": "B" if is_correct else "A",
                "is_correct": is_correct,
                "error": None,
            })

print(f"Generated {len(records)} synthetic records")
print(f"True parameters: k*={true_k_star}, β={true_beta}")

fit = compute_k_star_by_model(records)
print(f"\nFitted parameters:")
print(f"  k* = {fit['k_star']}")
print(f"  β  = {fit['beta']}")
print(f"  phase = {fit['phase']}")
print(f"  method = {fit['method']}")
print(f"  n_samples = {fit['n_samples']}")

# Sanity check
if fit['k_star'] and abs(fit['k_star'] - true_k_star) < 0.1:
    print("\n✓ k* recovered within 0.1")
else:
    print("\n✗ k* estimate off")

if fit['beta'] and abs(fit['beta'] - true_beta) / true_beta < 0.5:
    print("✓ β recovered within 50%")
else:
    print("✗ β estimate off")
