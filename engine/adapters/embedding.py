"""第 9 階段的 Sentence Transformers bi-encoder 實作。

權重延後到第一次使用時才載入，讓 CLI、測試與工具單純匯入套件時不必等待模型初始化。
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Sequence

from engine.config import EmbeddingSettings
from engine.device import resolve_device

logger = logging.getLogger(__name__)


class SentenceTransformerEmbedder:
    """以 ``SentenceTransformer`` 實作 :class:`engine.ports.Embedder`。

    文件與查詢仍保留不同入口，讓未來改用需要指令前綴的模型時只需調整設定。
    """

    def __init__(
        self, settings: EmbeddingSettings, *, device: str | None = None
    ) -> None:
        self._settings = settings
        self._device_preference = device
        self._model = None
        self._device: str | None = None

    @property
    def name(self) -> str:
        return self._settings.model

    @property
    def index_fingerprint(self) -> str:
        """雜湊所有會改變已儲存文件向量的設定。"""
        payload = {
            "model": self._settings.model,
            "normalize": self._settings.normalize,
            "max_seq_length": self._settings.max_seq_length,
            "document_prompt": self._settings.document_prompt,
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]

    @property
    def device(self) -> str:
        self._ensure_model()
        assert self._device is not None
        return self._device

    @property
    def dimension(self) -> int:
        """取得模型輸出的向量維度。"""
        model = self._ensure_model()
        # sentence-transformers 6 更改了方法名稱，因此同時相容新舊版本。
        getter = (
            getattr(model, "get_embedding_dimension", None)
            or model.get_sentence_embedding_dimension
        )
        return int(getter())

    def _ensure_model(self):
        """首次使用時才載入模型並選擇可用裝置。"""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._device = resolve_device(self._device_preference)
            logger.info(
                "Loading embedding model %s on %s",
                self._settings.model,
                self._device,
            )
            model = SentenceTransformer(
                self._settings.model, device=self._device
            )
            if self._settings.max_seq_length:
                model.max_seq_length = min(
                    self._settings.max_seq_length, model.max_seq_length
                )
            self._model = model
        return self._model

    def embed_documents(
        self, texts: Sequence[str], *, show_progress: bool = False
    ) -> list[list[float]]:
        """批次產生文件向量，可選擇顯示進度。"""
        return self._encode(
            [
                self._decorate(text, self._settings.document_prompt)
                for text in texts
            ],
            show_progress=show_progress,
        )

    def embed_query(self, text: str) -> list[float]:
        """產生單一查詢向量。"""
        return self._encode(
            [self._decorate(text, self._settings.query_prompt)],
            show_progress=False,
        )[0]

    def _encode(
        self, texts: Sequence[str], *, show_progress: bool
    ) -> list[list[float]]:
        """呼叫底層模型，以統一設定完成實際編碼。"""
        if not texts:
            return []
        model = self._ensure_model()
        vectors = model.encode(
            list(texts),
            batch_size=self._settings.batch_size,
            normalize_embeddings=self._settings.normalize,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        return vectors.tolist()

    @staticmethod
    def _decorate(text: str, prompt: str | None) -> str:
        """需要時在文字前加入模型指令前綴。"""
        return f"{prompt}{text}" if prompt else text
