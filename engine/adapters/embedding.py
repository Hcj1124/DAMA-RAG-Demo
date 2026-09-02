"""Sentence Transformers bi-encoder -- stage 9's embedding half.

Weights load on first use, not on import, so the CLI, the tests and any
tooling can import the package without paying a multi-second model load they
may never need.
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
    """Implements :class:`engine.ports.Embedder` on ``SentenceTransformer``.

    ``bge-m3`` needs no instruction prefix, so both prompts default to
    ``None`` and the document and query paths are identical. They stay
    separate methods anyway: the moment the model changes to an
    instruction-aware one, only the settings have to move.
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
        model = self._ensure_model()
        # sentence-transformers 6 renamed this; support both spellings.
        getter = (
            getattr(model, "get_embedding_dimension", None)
            or model.get_sentence_embedding_dimension
        )
        return int(getter())

    def _ensure_model(self):
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
        return self._encode(
            [
                self._decorate(text, self._settings.document_prompt)
                for text in texts
            ],
            show_progress=show_progress,
        )

    def embed_query(self, text: str) -> list[float]:
        return self._encode(
            [self._decorate(text, self._settings.query_prompt)],
            show_progress=False,
        )[0]

    def _encode(
        self, texts: Sequence[str], *, show_progress: bool
    ) -> list[list[float]]:
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
        return f"{prompt}{text}" if prompt else text
