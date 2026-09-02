"""Cross-encoder reranker -- stage 10.

The cross-encoder reads query and passage together, so it can see the
term-level agreement a bi-encoder has already compressed away. It is only
affordable because it runs on the shortlist, not the corpus.
"""

from __future__ import annotations

import logging
from typing import Sequence

from engine.config import RerankSettings
from engine.device import resolve_device

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Implements :class:`engine.ports.Reranker` on ``CrossEncoder``.

    Scores are raw logits: higher is better, but the scale is model-specific.
    Do not compare them across models or hard-code a threshold without
    recalibrating on this corpus.
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
        if not passages:
            return []
        model = self._ensure_model()
        scores = model.predict(
            [[query, passage] for passage in passages],
            batch_size=self._settings.batch_size,
            show_progress_bar=False,
        )
        return [float(score) for score in scores]
