"""
Extend ECB probe range to k=1.25 and k=1.5.
Generates additional framed prompts for the extended authority levels.
Run AFTER the base framed_prompts.json is already created.
"""

import json
from pathlib import Path
from authority_framer import frame_question, AUTHORITY_LEVELS

BASE_DIR = Path(__file__).parent.parent


def main():
    # Load seed questions
    raw = json.loads((BASE_DIR / "dataset" / "seed_questions.json").read_text())
    seeds = raw["questions"] if isinstance(raw, dict) and "questions" in raw else raw

    # Load existing framed prompts
    existing_path = BASE_DIR / "dataset" / "framed_prompts.json"
    raw_existing = json.loads(existing_path.read_text())
    existing = raw_existing if isinstance(raw_existing, list) else raw_existing.get("prompts", [])
    print(f"Existing prompts: {len(existing)}")

    # Generate new prompts at k=1.25 and k=1.5
    new_prompts = []
    for q in seeds:
        for k in [1.25, 1.5]:
            for polarity in ["right", "wrong"]:
                framed = frame_question(q, k, polarity)
                new_prompts.append({
                    "id": framed.id,
                    "k": framed.k,
                    "polarity": framed.polarity,
                    "prompt": framed.prompt,
                    "correct_answer": framed.correct_answer,
                    "authority_claim": framed.authority_claim,
                    "task": framed.task,
                    "domain": framed.domain,
                    "features": framed.features,
                    "metadata": framed.metadata,
                })

    print(f"New prompts (k=1.25 + k=1.5): {len(new_prompts)}")

    # Save extended prompts separately
    extended_path = BASE_DIR / "dataset" / "framed_prompts_extended.json"
    extended_path.write_text(json.dumps(new_prompts, indent=2))
    print(f"Saved to {extended_path}")

    # Also save combined (existing + extended)
    combined = existing + new_prompts
    combined_path = BASE_DIR / "dataset" / "framed_prompts_full.json"
    combined_path.write_text(json.dumps(combined, indent=2))
    print(f"Combined ({len(combined)} prompts) saved to {combined_path}")


if __name__ == "__main__":
    main()
