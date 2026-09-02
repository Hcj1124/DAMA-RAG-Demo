"""Local generation through Ollama -- stage 12."""

from __future__ import annotations

import logging

from engine.config import GenerationSettings
from engine.errors import LanguageModelError

logger = logging.getLogger(__name__)


class OllamaLanguageModel:
    """Implements :class:`engine.ports.LanguageModel` against a local Ollama.

    ``num_ctx`` is set explicitly rather than left to the server default: a
    grounded DMBOK prompt carries several full sections, and silently
    truncating the context yields a confident answer built on half the
    evidence. ``think`` is off by default because the reasoning trace costs
    latency and is discarded anyway.
    """

    def __init__(self, settings: GenerationSettings) -> None:
        self._settings = settings
        self._client = None

    @property
    def name(self) -> str:
        return f"ollama:{self._settings.model}"

    def _ensure_client(self):
        if self._client is None:
            import ollama

            self._client = (
                ollama.Client(host=self._settings.host)
                if self._settings.host
                else ollama.Client()
            )
        return self._client

    def complete(self, prompt: str) -> str:
        client = self._ensure_client()
        try:
            response = client.chat(
                model=self._settings.model,
                messages=[{"role": "user", "content": prompt}],
                think=self._settings.think,
                options={
                    "temperature": self._settings.temperature,
                    "num_ctx": self._settings.num_ctx,
                },
            )
        except Exception as error:
            raise LanguageModelError(
                f"Ollama request to '{self._settings.model}' failed: {error}\n"
                f"Check that Ollama is running (`ollama ps`) and that the "
                f"model is pulled (`ollama pull {self._settings.model}`)."
            ) from error

        content = response["message"]["content"]
        if not content.strip():
            raise LanguageModelError(
                f"'{self._settings.model}' returned an empty answer. If this "
                f"is a thinking model, the reasoning budget may have consumed "
                f"the whole response; try DAMA_THINK=0."
            )
        return content
