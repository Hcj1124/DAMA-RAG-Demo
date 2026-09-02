"""第 12 階段：透過 Ollama 執行本機回答生成。"""

from __future__ import annotations

import logging

from engine.config import GenerationSettings
from engine.errors import LanguageModelError

logger = logging.getLogger(__name__)


class OllamaLanguageModel:
    """以本機 Ollama 實作 :class:`engine.ports.LanguageModel`。

    明確設定 ``num_ctx``，避免包含多個完整來源的 DMBOK Prompt 被伺服器無聲截斷。
    ``think`` 預設關閉，因為目前不保存推理軌跡，只會增加回應延遲。
    """

    def __init__(self, settings: GenerationSettings) -> None:
        self._settings = settings
        self._client = None

    @property
    def name(self) -> str:
        return f"ollama:{self._settings.model}"

    def _ensure_client(self):
        """首次生成時才建立 Ollama client。"""
        if self._client is None:
            import ollama

            self._client = (
                ollama.Client(host=self._settings.host)
                if self._settings.host
                else ollama.Client()
            )
        return self._client

    def complete(self, prompt: str) -> str:
        """送出完整 Prompt，並將連線失敗或空回答轉為引擎錯誤。"""
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
