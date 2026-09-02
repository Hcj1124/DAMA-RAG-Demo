"""Torch device selection.

Isolated so that the only ``import torch`` in the package happens inside a
function, at call time -- importing the engine must stay cheap.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_VALID = {"mps", "cuda", "cpu"}


def resolve_device(preference: str | None = None) -> str:
    """Return the torch device for local models.

    An explicit preference wins when it is usable; otherwise the best
    available backend is chosen. An unusable preference warns and falls back
    rather than crashing, because "slower" beats "does not start".
    """

    import torch

    available = {"cpu"}
    if torch.backends.mps.is_available():
        available.add("mps")
    if torch.cuda.is_available():
        available.add("cuda")

    if preference:
        wanted = preference.strip().lower()
        if wanted not in _VALID:
            logger.warning(
                "Unknown device %r; falling back to automatic selection",
                preference,
            )
        elif wanted in available:
            return wanted
        else:
            logger.warning(
                "Device %r is not available here; selecting automatically",
                wanted,
            )

    for candidate in ("mps", "cuda"):
        if candidate in available:
            return candidate
    return "cpu"
