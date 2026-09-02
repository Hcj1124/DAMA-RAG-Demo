"""選擇 Torch 執行裝置。

將 ``torch`` 延後到函式執行時才匯入，避免單純匯入引擎套件就付出初始化成本。
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# 引擎明確支援的運算後端；其他輸入會回退到自動選擇。
_VALID = {"mps", "cuda", "cpu"}


def resolve_device(preference: str | None = None) -> str:
    """回傳本機模型使用的 Torch 裝置。

    可用時優先採用明確指定的裝置，否則依 MPS、CUDA、CPU 的順序選擇。
    指定值不可用時只警告並回退，避免應用程式因此無法啟動。
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
