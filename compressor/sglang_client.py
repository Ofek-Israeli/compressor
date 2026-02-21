"""
SGLang client for Target (local) generation only.
Single-prompt endpoint; no chat/messages wrapper.
See docs/phase0.md §4.4a.
"""

import logging
import time
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)


class SGLangClient:
    """
    Thin client for local SGLang server. All parameters from kconfig; no defaults.
    """

    def __init__(
        self,
        base_url: str,
        model_id: str,
        timeout_s: float,
        max_retries: int,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        # OpenAI-compatible completions endpoint (single prompt)
        if self.base_url.endswith("/v1"):
            self._url = f"{self.base_url}/completions"
        else:
            self._url = f"{self.base_url}/v1/completions"
        logger.info("SGLang endpoint: %s (model=%s, timeout=%ss)", self._url, model_id, timeout_s)

    def health_check(self) -> bool:
        """Quick check that the SGLang server is reachable."""
        url = self.base_url.rstrip("/")
        if not url.endswith("/v1"):
            url += "/v1"
        url += "/models"
        try:
            r = requests.get(url, timeout=5)
            r.raise_for_status()
            logger.info("SGLang health check OK: %s", url)
            return True
        except requests.exceptions.RequestException as e:
            logger.error("SGLang health check FAILED (%s): %s", url, e)
            return False

    def generate(
        self,
        prompt: str,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        seed: int,
        stop_token_ids: Optional[List[int]] = None,
    ) -> str:
        """
        Send one prompt to the SGLang server; return generated text only.
        seed: always from kconfig (target.seed); mandatory, no fallback.
        stop_token_ids: token IDs at which to stop (e.g. EOS/EOT); prevents over-decoding.
        """
        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_new_tokens,
            "seed": seed,
        }
        if stop_token_ids:
            payload["stop_token_ids"] = stop_token_ids
        prompt_len = len(prompt)
        logger.debug("SGLang request: prompt_len=%s max_tokens=%s → %s", prompt_len, max_new_tokens, self._url)
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                t0 = time.time()
                r = requests.post(
                    self._url,
                    json=payload,
                    timeout=self.timeout_s,
                    headers={"Content-Type": "application/json"},
                )
                logger.debug("SGLang response: status=%s elapsed=%.1fs", r.status_code, time.time() - t0)
                r.raise_for_status()
                data = r.json()
                # OpenAI completions: choices[0].text
                choices = data.get("choices", [])
                if not choices:
                    return ""
                text = choices[0].get("text", "")
                if isinstance(text, str):
                    return text.strip()
                return str(text).strip()
            except requests.exceptions.RequestException as e:
                last_exc = e
                logger.warning("SGLang request attempt %s failed: %s", attempt + 1, e)
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"SGLang generate failed after {self.max_retries} retries") from last_exc
