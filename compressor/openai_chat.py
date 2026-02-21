"""
OpenAI Chat Completions client for Reflector and Judge.
Returns (List[str], usage) from .chat(messages).
See docs/phase0.md: Reflector and judge use same client pattern.
"""

import os
from typing import Any, Dict, List

from openai import OpenAI


class Usage:
    """Minimal usage (prompt_tokens, completion_tokens, __add__). Compatible with minions.usage.Usage."""

    def __init__(self, prompt_tokens: int = 0, completion_tokens: int = 0):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )


class OpenAIChatClient:
    """
    OpenAI Chat Completions client. .chat(messages) -> (List[str], Usage).
    Used for both Reflector and Judge (same API; model/params from kconfig).
    """

    def __init__(
        self,
        api_base: str,
        api_key_env: str,
        model_id: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int,
        usage_class: type = Usage,
    ):
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"Environment variable {api_key_env} is not set")
        self._client = OpenAI(api_key=api_key, base_url=api_base.rstrip("/"))
        self.model_id = model_id
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.seed = seed
        self._usage_class = usage_class

    def chat(self, messages: List[Dict[str, Any]]) -> tuple:
        """Returns (List[str], usage). usage has prompt_tokens, completion_tokens, __add__."""
        r = self._client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            seed=self.seed,
        )
        text = r.choices[0].message.content if r.choices else ""
        pt = getattr(r.usage, "prompt_tokens", 0) or 0
        ct = getattr(r.usage, "completion_tokens", 0) or 0
        usage = self._usage_class(prompt_tokens=pt, completion_tokens=ct)
        return ([text], usage)
