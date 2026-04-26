"""
Free-Tier Model Clients for ECB.

All providers below offer FREE API access sufficient for ECB's 4800 calls:

1. Google AI Studio (Gemini)        — 1500 req/day, 15 RPM (Flash), 50/day (Pro)
   https://aistudio.google.com/app/apikey
   Models: gemini-2.0-flash-exp, gemini-1.5-pro, gemini-1.5-flash

2. Groq                              — ~30 RPM, ~14k RPD free
   https://console.groq.com/keys
   Models: llama-3.3-70b-versatile, mixtral-8x7b-32768, gemma2-9b-it

3. OpenRouter (free models)          — ~200 req/day across free-tier routes
   https://openrouter.ai/keys
   Models: google/gemini-2.0-flash-exp:free, meta-llama/llama-3.3-70b-instruct:free,
           mistralai/mistral-nemo:free, qwen/qwen-2.5-72b-instruct:free

4. Kaggle Notebook (local inference) — 30h/week T4 GPU, 20h/week TPU free
   Models: any HuggingFace model that fits in memory
           Recommended: google/gemma-2-9b-it, Qwen/Qwen2.5-7B-Instruct

Usage:
    client = get_client("gemini-2.0-flash")
    response = client.complete(prompt, max_tokens=5, temperature=0.0)
"""

from __future__ import annotations
import os
import time
import json
import re
from dataclasses import dataclass
from typing import Optional


# ───────────────────── BASE CLIENT ─────────────────────

@dataclass
class ModelResponse:
    text: str
    model: str
    latency_s: float
    raw: Optional[dict] = None
    error: Optional[str] = None


class FreeModelClient:
    """Base class for all free-tier clients."""
    provider: str = "base"
    model: str = ""

    def complete(self, prompt: str, max_tokens: int = 5,
                 temperature: float = 0.0) -> ModelResponse:
        raise NotImplementedError

    def __repr__(self):
        return f"{self.provider}/{self.model}"


# ───────────────────── GOOGLE AI STUDIO (GEMINI) ─────────────────────

class GeminiClient(FreeModelClient):
    """Google AI Studio free tier.

    Get free API key: https://aistudio.google.com/app/apikey
    Set env: export GOOGLE_API_KEY=...

    Free quotas:
    - gemini-2.0-flash-exp: 1500 req/day, 15 RPM, 1M TPM
    - gemini-1.5-flash:    1500 req/day, 15 RPM, 1M TPM
    - gemini-1.5-pro:      50 req/day, 2 RPM, 32K TPM
    """
    provider = "google"

    def __init__(self, model: str = "gemini-2.0-flash-exp"):
        self.model = model
        self.api_key = os.environ.get("GOOGLE_API_KEY", "")

    def complete(self, prompt: str, max_tokens: int = 5,
                 temperature: float = 0.0) -> ModelResponse:
        import urllib.request, urllib.error

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )
        body = json.dumps({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }).encode("utf-8")

        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            return ModelResponse(
                text=text.strip(),
                model=self.model,
                latency_s=time.time() - t0,
                raw=data,
            )
        except urllib.error.HTTPError as e:
            return ModelResponse(
                text="", model=self.model, latency_s=time.time() - t0,
                error=f"HTTP {e.code}: {e.read().decode('utf-8', errors='ignore')[:200]}",
            )
        except Exception as e:
            return ModelResponse(
                text="", model=self.model, latency_s=time.time() - t0,
                error=f"{type(e).__name__}: {e}",
            )


# ───────────────────── GROQ (LLAMA / MIXTRAL / GEMMA) ─────────────────────

class GroqClient(FreeModelClient):
    """Groq free tier.

    Get free API key: https://console.groq.com/keys
    Set env: export GROQ_API_KEY=...

    Free quotas (as of 2026):
    - llama-3.3-70b-versatile:  30 RPM, 6K TPM, 14400 RPD
    - mixtral-8x7b-32768:       30 RPM, ~15K TPM, 14400 RPD
    - gemma2-9b-it:             30 RPM, 15K TPM, 14400 RPD
    """
    provider = "groq"

    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        self.model = model
        self.api_key = os.environ.get("GROQ_API_KEY", "")

    def complete(self, prompt: str, max_tokens: int = 5,
                 temperature: float = 0.0) -> ModelResponse:
        import urllib.request, urllib.error

        url = "https://api.groq.com/openai/v1/chat/completions"
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }).encode("utf-8")

        req = urllib.request.Request(
            url, data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "ecb-benchmark/1.0 (Kaggle submission)",
                "Accept": "application/json",
            },
            method="POST",
        )

        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            text = data["choices"][0]["message"]["content"]
            return ModelResponse(
                text=text.strip(),
                model=self.model,
                latency_s=time.time() - t0,
                raw=data,
            )
        except urllib.error.HTTPError as e:
            return ModelResponse(
                text="", model=self.model, latency_s=time.time() - t0,
                error=f"HTTP {e.code}: {e.read().decode('utf-8', errors='ignore')[:200]}",
            )
        except Exception as e:
            return ModelResponse(
                text="", model=self.model, latency_s=time.time() - t0,
                error=f"{type(e).__name__}: {e}",
            )


# ───────────────────── OPENROUTER (FREE ROUTES) ─────────────────────

class OpenRouterClient(FreeModelClient):
    """OpenRouter free tier.

    Get free API key: https://openrouter.ai/keys
    Set env: export OPENROUTER_API_KEY=...

    Free models (suffix :free, updated regularly):
    - google/gemini-2.0-flash-exp:free
    - meta-llama/llama-3.3-70b-instruct:free
    - mistralai/mistral-nemo:free
    - qwen/qwen-2.5-72b-instruct:free
    - google/gemma-2-9b-it:free

    Free tier: ~200 req/day across all free models.
    """
    provider = "openrouter"

    def __init__(self, model: str = "meta-llama/llama-3.3-70b-instruct:free"):
        self.model = model
        self.api_key = os.environ.get("OPENROUTER_API_KEY", "")

    def complete(self, prompt: str, max_tokens: int = 5,
                 temperature: float = 0.0) -> ModelResponse:
        import urllib.request, urllib.error

        url = "https://openrouter.ai/api/v1/chat/completions"
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }).encode("utf-8")

        req = urllib.request.Request(
            url, data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": "https://kaggle.com",
                "X-Title": "Epistemic Curie Benchmark",
            },
            method="POST",
        )

        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            text = data["choices"][0]["message"]["content"]
            return ModelResponse(
                text=text.strip(),
                model=self.model,
                latency_s=time.time() - t0,
                raw=data,
            )
        except urllib.error.HTTPError as e:
            return ModelResponse(
                text="", model=self.model, latency_s=time.time() - t0,
                error=f"HTTP {e.code}: {e.read().decode('utf-8', errors='ignore')[:200]}",
            )
        except Exception as e:
            return ModelResponse(
                text="", model=self.model, latency_s=time.time() - t0,
                error=f"{type(e).__name__}: {e}",
            )


# ───────────────────── KAGGLE NOTEBOOK (LOCAL HF) ─────────────────────

class KaggleHFClient(FreeModelClient):
    """Local HuggingFace inference inside a Kaggle notebook.

    This works ONLY inside a Kaggle notebook with GPU/TPU enabled.
    Models are loaded once and reused; no API key required.

    Recommended models (fit on free T4 16GB):
    - google/gemma-2-9b-it          (9B, fits comfortably)
    - Qwen/Qwen2.5-7B-Instruct      (7B, fast)
    - microsoft/Phi-3-medium-4k-instruct

    Usage inside Kaggle notebook:
        !pip install transformers accelerate bitsandbytes
        client = KaggleHFClient("google/gemma-2-9b-it")
        response = client.complete(prompt)
    """
    provider = "kaggle_hf"
    _loaded_models = {}

    def __init__(self, model: str = "google/gemma-2-9b-it"):
        self.model = model

    def _ensure_loaded(self):
        if self.model in KaggleHFClient._loaded_models:
            return KaggleHFClient._loaded_models[self.model]

        try:
            from transformers import (
                AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
            )
            import torch
        except ImportError as e:
            raise RuntimeError(
                "KaggleHFClient requires transformers, accelerate, bitsandbytes, torch. "
                "Install inside a Kaggle notebook: !pip install transformers accelerate bitsandbytes"
            ) from e

        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(self.model)
        model = AutoModelForCausalLM.from_pretrained(
            self.model, quantization_config=bnb, device_map="auto",
        )
        KaggleHFClient._loaded_models[self.model] = (model, tokenizer)
        return model, tokenizer

    def complete(self, prompt: str, max_tokens: int = 5,
                 temperature: float = 0.0) -> ModelResponse:
        import torch

        t0 = time.time()
        try:
            model, tokenizer = self._ensure_loaded()

            # Format as chat if chat_template available
            try:
                messages = [{"role": "user", "content": prompt}]
                inputs = tokenizer.apply_chat_template(
                    messages, return_tensors="pt", add_generation_prompt=True,
                ).to(model.device)
            except Exception:
                inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature if temperature > 0 else 1.0,
                    do_sample=temperature > 0,
                    pad_token_id=tokenizer.eos_token_id,
                )

            new_tokens = outputs[0, inputs.shape[1]:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            return ModelResponse(
                text=text.strip(),
                model=self.model,
                latency_s=time.time() - t0,
            )
        except Exception as e:
            return ModelResponse(
                text="", model=self.model, latency_s=time.time() - t0,
                error=f"{type(e).__name__}: {e}",
            )


# ───────────────────── ANTHROPIC CLAUDE ─────────────────────

class AnthropicClient(FreeModelClient):
    """Anthropic Claude API.

    Get API key: https://console.anthropic.com/
    Set env: export ANTHROPIC_API_KEY=...

    Models (cheapest first for ECB scale):
    - claude-haiku-4-5-20251001   — fastest, cheapest (~$0.25/M input)
    - claude-sonnet-4-6           — balanced
    - claude-opus-4-7             — most capable

    ECB cost estimate (360 prompts × ~100 tokens each):
    - Haiku:   ~$0.009 per model run (near-free)
    - Sonnet:  ~$0.12 per model run
    """
    provider = "anthropic"

    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        self.model = model
        self.api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    def complete(self, prompt: str, max_tokens: int = 5,
                 temperature: float = 0.0) -> ModelResponse:
        import urllib.request, urllib.error

        url = "https://api.anthropic.com/v1/messages"
        body = json.dumps({
            "model": self.model,
            "max_tokens": max(max_tokens, 1024),
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }).encode("utf-8")

        req = urllib.request.Request(
            url, data=body,
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )

        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            text = data["content"][0]["text"]
            return ModelResponse(
                text=text.strip(),
                model=self.model,
                latency_s=time.time() - t0,
                raw=data,
            )
        except urllib.error.HTTPError as e:
            return ModelResponse(
                text="", model=self.model, latency_s=time.time() - t0,
                error=f"HTTP {e.code}: {e.read().decode('utf-8', errors='ignore')[:200]}",
            )
        except Exception as e:
            return ModelResponse(
                text="", model=self.model, latency_s=time.time() - t0,
                error=f"{type(e).__name__}: {e}",
            )


# ───────────────────── MISTRAL AI ─────────────────────

class MistralClient(FreeModelClient):
    """Mistral AI API.

    Get API key (includes free trial credits): https://console.mistral.ai/
    Set env: export MISTRAL_API_KEY=...

    Models:
    - mistral-small-latest   — fast, cheap (~$0.20/M input)
    - mistral-medium-latest  — balanced
    - mistral-large-latest   — most capable

    ECB cost estimate: ~$0.01-0.05 per model run.
    """
    provider = "mistral"

    def __init__(self, model: str = "mistral-small-latest"):
        self.model = model
        self.api_key = os.environ.get("MISTRAL_API_KEY", "")

    def complete(self, prompt: str, max_tokens: int = 5,
                 temperature: float = 0.0) -> ModelResponse:
        import urllib.request, urllib.error

        url = "https://api.mistral.ai/v1/chat/completions"
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max(max_tokens, 1024),
            "temperature": temperature,
        }).encode("utf-8")

        req = urllib.request.Request(
            url, data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            text = data["choices"][0]["message"]["content"]
            return ModelResponse(
                text=text.strip(),
                model=self.model,
                latency_s=time.time() - t0,
                raw=data,
            )
        except urllib.error.HTTPError as e:
            return ModelResponse(
                text="", model=self.model, latency_s=time.time() - t0,
                error=f"HTTP {e.code}: {e.read().decode('utf-8', errors='ignore')[:200]}",
            )
        except Exception as e:
            return ModelResponse(
                text="", model=self.model, latency_s=time.time() - t0,
                error=f"{type(e).__name__}: {e}",
            )


# ───────────────────── DEEPSEEK ─────────────────────

class DeepSeekClient(FreeModelClient):
    """DeepSeek API — very cheap, strong reasoning.

    Get API key: https://platform.deepseek.com/
    Set env: export DEEPSEEK_API_KEY=...

    Models:
    - deepseek-chat     — DeepSeek-V3, ~$0.07/M input (extremely cheap)
    - deepseek-reasoner — DeepSeek-R1, chain-of-thought (~$0.55/M input)

    ECB cost estimate: ~$0.003 per model run with deepseek-chat (near-free).
    """
    provider = "deepseek"

    def __init__(self, model: str = "deepseek-chat"):
        self.model = model
        self.api_key = os.environ.get("DEEPSEEK_API_KEY", "")

    def complete(self, prompt: str, max_tokens: int = 5,
                 temperature: float = 0.0) -> ModelResponse:
        import urllib.request, urllib.error

        url = "https://api.deepseek.com/chat/completions"
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max(max_tokens, 1024),
            "temperature": temperature,
        }).encode("utf-8")

        req = urllib.request.Request(
            url, data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            text = data["choices"][0]["message"]["content"]
            return ModelResponse(
                text=text.strip(),
                model=self.model,
                latency_s=time.time() - t0,
                raw=data,
            )
        except urllib.error.HTTPError as e:
            return ModelResponse(
                text="", model=self.model, latency_s=time.time() - t0,
                error=f"HTTP {e.code}: {e.read().decode('utf-8', errors='ignore')[:200]}",
            )
        except Exception as e:
            return ModelResponse(
                text="", model=self.model, latency_s=time.time() - t0,
                error=f"{type(e).__name__}: {e}",
            )


# ───────────────────── TOGETHER AI ─────────────────────

class TogetherClient(FreeModelClient):
    """Together AI API — wide model selection, competitive pricing.

    Get API key (includes $1 free credit): https://api.together.xyz/
    Set env: export TOGETHER_API_KEY=...

    Key models for ECB replication:
    - meta-llama/Llama-3.3-70B-Instruct-Turbo  (~$0.88/M)
    - mistralai/Mistral-7B-Instruct-v0.3        (~$0.20/M)
    - google/gemma-2-9b-it                       (~$0.30/M)
    - Qwen/Qwen2.5-72B-Instruct-Turbo           (~$1.20/M)
    - deepseek-ai/DeepSeek-R1-Distill-Llama-70B (~$0.75/M)
    """
    provider = "together"

    def __init__(self, model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo"):
        self.model = model
        self.api_key = os.environ.get("TOGETHER_API_KEY", "")

    def complete(self, prompt: str, max_tokens: int = 5,
                 temperature: float = 0.0) -> ModelResponse:
        import urllib.request, urllib.error

        url = "https://api.together.xyz/v1/chat/completions"
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max(max_tokens, 1024),
            "temperature": temperature,
        }).encode("utf-8")

        req = urllib.request.Request(
            url, data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            text = data["choices"][0]["message"]["content"]
            return ModelResponse(
                text=text.strip(),
                model=self.model,
                latency_s=time.time() - t0,
                raw=data,
            )
        except urllib.error.HTTPError as e:
            return ModelResponse(
                text="", model=self.model, latency_s=time.time() - t0,
                error=f"HTTP {e.code}: {e.read().decode('utf-8', errors='ignore')[:200]}",
            )
        except Exception as e:
            return ModelResponse(
                text="", model=self.model, latency_s=time.time() - t0,
                error=f"{type(e).__name__}: {e}",
            )


# ───────────────────── MODEL REGISTRY ─────────────────────

MODEL_REGISTRY = {
    # Google AI Studio (free tier) — confirmed working April 2026
    "gemini-flash-latest":     lambda: GeminiClient("gemini-flash-latest"),
    "gemini-flash-lite-latest":lambda: GeminiClient("gemini-flash-lite-latest"),
    "gemini-3-flash-preview":  lambda: GeminiClient("gemini-3-flash-preview"),
    "gemma-3-27b":             lambda: GeminiClient("gemma-3-27b-it"),
    "gemma-3-12b":             lambda: GeminiClient("gemma-3-12b-it"),
    # (exhausted today: gemini-2.5-flash, gemini-2.5-pro, gemini-2.0-flash)

    # Groq (free tier) — April 2026 model lineup
    "llama-3.3-70b": lambda: GroqClient("llama-3.3-70b-versatile"),
    "llama-3.1-8b":  lambda: GroqClient("llama-3.1-8b-instant"),
    "llama-4-scout": lambda: GroqClient("meta-llama/llama-4-scout-17b-16e-instruct"),
    "qwen-3-32b":    lambda: GroqClient("qwen/qwen3-32b"),
    "gpt-oss-120b":  lambda: GroqClient("openai/gpt-oss-120b"),
    "gpt-oss-20b":   lambda: GroqClient("openai/gpt-oss-20b"),
    "kimi-k2":       lambda: GroqClient("moonshotai/kimi-k2-instruct-0905"),

    # OpenRouter (free routes)
    "openrouter/llama-70b":    lambda: OpenRouterClient("meta-llama/llama-3.3-70b-instruct:free"),
    "openrouter/qwen-72b":     lambda: OpenRouterClient("qwen/qwen-2.5-72b-instruct:free"),
    "openrouter/mistral-nemo": lambda: OpenRouterClient("mistralai/mistral-nemo:free"),

    # Kaggle notebook local (free GPU)
    "kaggle/gemma-9b": lambda: KaggleHFClient("google/gemma-2-9b-it"),
    "kaggle/qwen-7b":  lambda: KaggleHFClient("Qwen/Qwen2.5-7B-Instruct"),
    "kaggle/phi-3":    lambda: KaggleHFClient("microsoft/Phi-3-medium-4k-instruct"),

    # Anthropic Claude (ANTHROPIC_API_KEY)
    "claude-haiku":    lambda: AnthropicClient("claude-haiku-4-5-20251001"),
    "claude-sonnet":   lambda: AnthropicClient("claude-sonnet-4-6"),
    "claude-opus":     lambda: AnthropicClient("claude-opus-4-7"),

    # Mistral AI (MISTRAL_API_KEY)
    "mistral-small":   lambda: MistralClient("mistral-small-latest"),
    "mistral-large":   lambda: MistralClient("mistral-large-latest"),

    # DeepSeek (DEEPSEEK_API_KEY)
    "deepseek-v3":     lambda: DeepSeekClient("deepseek-chat"),
    "deepseek-r1":     lambda: DeepSeekClient("deepseek-reasoner"),

    # Together AI (TOGETHER_API_KEY)
    "together/llama-70b":   lambda: TogetherClient("meta-llama/Llama-3.3-70B-Instruct-Turbo"),
    "together/gemma-9b":    lambda: TogetherClient("google/gemma-2-9b-it"),
    "together/qwen-72b":    lambda: TogetherClient("Qwen/Qwen2.5-72B-Instruct-Turbo"),
    "together/deepseek-r1": lambda: TogetherClient("deepseek-ai/DeepSeek-R1-Distill-Llama-70B"),
}


def get_client(model_name: str) -> FreeModelClient:
    """Get a free-tier client by short name."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name]()


# ───────────────────── ANSWER EXTRACTION ─────────────────────

ANSWER_PATTERN = re.compile(r"\b([ABCD])\b")
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def extract_answer(response_text: str) -> Optional[str]:
    """Extract the first A/B/C/D letter from a model response.

    Handles chain-of-thought models (Qwen3, GPT-OSS) that wrap reasoning in
    <think>...</think> tags. We strip completed think blocks first; if the
    block is unclosed (hit max_tokens), we use the text AFTER the last
    opening <think> for a best-effort extraction of the final answer.
    """
    if not response_text:
        return None

    text = response_text

    # Strip completed think blocks
    text = THINK_BLOCK_RE.sub("", text)

    # If there's still an unclosed <think>, everything after it is reasoning —
    # no final answer was produced. Return None rather than picking a letter
    # from the reasoning chain.
    if "<think>" in text.lower() and "</think>" not in text.lower():
        return None

    match = ANSWER_PATTERN.search(text.upper())
    return match.group(1) if match else None


# ───────────────────── SMOKE TEST ─────────────────────

if __name__ == "__main__":
    import sys
    model_name = sys.argv[1] if len(sys.argv) > 1 else "gemini-2.0-flash"

    print(f"Smoke test: {model_name}")
    try:
        client = get_client(model_name)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    prompt = "What is 7 * 8? Answer with only the single digit letter A=54, B=56, C=58, D=64."
    response = client.complete(prompt, max_tokens=5)
    print(f"Model:     {response.model}")
    print(f"Latency:   {response.latency_s:.2f}s")
    if response.error:
        print(f"ERROR:     {response.error}")
    else:
        print(f"Response:  {response.text!r}")
        print(f"Extracted: {extract_answer(response.text)}")
