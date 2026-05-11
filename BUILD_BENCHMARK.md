# ECB Benchmark — Build 4 Tasks

## PATH: /kaggle/input/datasets/sardorrazikov/epistemic-curie-benchmark/framed_prompts_full.json
## Each task = separate notebook. 3 blocks each. Build Task after each.

---

## TASK A: Attention

### Block 1:
```
import kaggle_benchmarks as kbench
import json
import re

data = json.load(open("/kaggle/input/datasets/sardorrazikov/epistemic-curie-benchmark/framed_prompts_full.json"))
items_a = [item for item in data if item["task"] == "A_attention"]
print(f"Task A items: {len(items_a)}")
```

### Block 2:
```
@kbench.task(name="ECB_A_Attention", description="Authority override of factual knowledge. Tests whether LLMs change correct answers when a fabricated authority endorses a wrong answer. 130 prompts at 7 authority levels (k=0 to k=1.5).")
def ecb_attention(llm) -> None:
    for item in items_a:
        response = llm.prompt(item["prompt"])
        m = re.search(r'\b([ABCD])\b', response.strip().upper())
        extracted = m.group(1) if m else None
        kbench.assertions.assert_equal(extracted, item["correct_answer"])

ecb_attention.run(kbench.llm)
```

### Block 3:
```
%choose ECB_A_Attention
```

### → Build Task

---

## TASK B: Social Cognition (new task notebook)

### Block 1:
```
import kaggle_benchmarks as kbench
import json
import re

data = json.load(open("/kaggle/input/datasets/sardorrazikov/epistemic-curie-benchmark/framed_prompts_full.json"))
items_b = [item for item in data if item["task"] == "B_social"]
print(f"Task B items: {len(items_b)}")
```

### Block 2:
```
@kbench.task(name="ECB_B_Social", description="Rational deference and false-belief tracking. Tests whether LLMs can distinguish legitimate authority from fabricated authority. 130 prompts at 7 authority levels.")
def ecb_social(llm) -> None:
    for item in items_b:
        response = llm.prompt(item["prompt"])
        m = re.search(r'\b([ABCD])\b', response.strip().upper())
        extracted = m.group(1) if m else None
        kbench.assertions.assert_equal(extracted, item["correct_answer"])

ecb_social.run(kbench.llm)
```

### Block 3:
```
%choose ECB_B_Social
```

### → Build Task

---

## TASK C: Metacognition (new task notebook)

### Block 1:
```
import kaggle_benchmarks as kbench
import json
import re

data = json.load(open("/kaggle/input/datasets/sardorrazikov/epistemic-curie-benchmark/framed_prompts_full.json"))
items_c = [item for item in data if item["task"] == "C_metacog"]
print(f"Task C items: {len(items_c)}")
```

### Block 2:
```
@kbench.task(name="ECB_C_Metacognition", description="Recognition of authority-induced bias. Tests whether LLMs can detect when their reasoning is influenced by authority framing. 130 prompts at 7 authority levels.")
def ecb_metacog(llm) -> None:
    for item in items_c:
        response = llm.prompt(item["prompt"])
        m = re.search(r'\b([ABCD])\b', response.strip().upper())
        extracted = m.group(1) if m else None
        kbench.assertions.assert_equal(extracted, item["correct_answer"])

ecb_metacog.run(kbench.llm)
```

### Block 3:
```
%choose ECB_C_Metacognition
```

### → Build Task

---

## TASK D: Learning (new task notebook)

### Block 1:
```
import kaggle_benchmarks as kbench
import json
import re

data = json.load(open("/kaggle/input/datasets/sardorrazikov/epistemic-curie-benchmark/framed_prompts_full.json"))
items_d = [item for item in data if item["task"] == "D_learning"]
print(f"Task D items: {len(items_d)}")
```

### Block 2:
```
@kbench.task(name="ECB_D_Learning", description="Trust accumulation across authority levels. Tests whether LLMs appropriately adjust trust based on authority credibility. 130 prompts at 7 authority levels.")
def ecb_learning(llm) -> None:
    for item in items_d:
        response = llm.prompt(item["prompt"])
        m = re.search(r'\b([ABCD])\b', response.strip().upper())
        extracted = m.group(1) if m else None
        kbench.assertions.assert_equal(extracted, item["correct_answer"])

ecb_learning.run(kbench.llm)
```

### Block 3:
```
%choose ECB_D_Learning
```

### → Build Task

---

## AFTER ALL 4 TASKS:
## 1. Go to benchmark page
## 2. All 4 tasks should appear
## 3. Add models to run
## 4. Link benchmark URL in writeup
