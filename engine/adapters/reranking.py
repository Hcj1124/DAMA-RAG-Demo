"""第 10 階段的 cross-encoder reranker 實作。

Cross-encoder 同時讀取查詢與段落，可辨識 bi-encoder 壓縮後遺失的細部對應；
它只處理短候選清單，而不對整個語料逐筆運算。
"""

from __future__ import annotations

import logging
from typing import Sequence

from engine.config import RerankSettings
from engine.device import resolve_device

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """以 ``CrossEncoder`` 實作 :class:`engine.ports.Reranker`。

    分數是模型專屬的原始 logits，只能判斷同次結果的高低；未經此語料校準，不應跨模型
    比較或寫死判斷門檻。
    """

    def __init__(
        self, settings: RerankSettings, *, device: str | None = None
    ) -> None:
        self._settings = settings
        self._device_preference = device
        self._model = None
        self._device: str | None = None

    @property
    def name(self) -> str:
        return self._settings.model

    @property
    def device(self) -> str:
        self._ensure_model()
        assert self._device is not None
        return self._device

    def _ensure_model(self):
        """首次評分時才載入 reranker 與選擇執行裝置。"""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._device = resolve_device(self._device_preference)
            logger.info(
                "Loading reranker %s on %s", self._settings.model, self._device
            )
            self._model = CrossEncoder(
                self._settings.model,
                device=self._device,
                max_length=self._settings.max_length,
            )
        return self._model

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """批次評估查詢與每個候選段落的相關性。"""
        if not passages:
            return []
        model = self._ensure_model()
        scores = model.predict(
            [[query, passage] for passage in passages],
            batch_size=self._settings.batch_size,
            show_progress_bar=False,
        )
        return [float(score) for score in scores]
