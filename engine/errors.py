"""Errors that carry a fix, not just a stack trace."""

from __future__ import annotations


class EngineError(Exception):
    """Base class for every error this package raises deliberately."""


class ConfigurationError(EngineError):
    """A setting is missing or malformed."""


class CorpusError(EngineError):
    """The chunk files are missing, unreadable, or internally inconsistent."""


class IndexNotBuiltError(EngineError):
    def __init__(self, collection: str, path: object) -> None:
        super().__init__(
            f"Collection '{collection}' does not exist under {path}.\n"
            f"Build it first:  dama-rag index"
        )


class IndexStaleError(EngineError):
    """The index was written by a different embedding configuration."""

    def __init__(self, *, indexed_with: str, querying_with: str) -> None:
        super().__init__(
            f"The index was built with '{indexed_with}' but you are querying "
            f"with '{querying_with}'. Comparing vectors from incompatible "
            f"embedding configurations returns plausible-looking nonsense, "
            f"so this is refused.\n"
            f"Rebuild:  dama-rag index --rebuild"
        )


class LanguageModelError(EngineError):
    """Generation failed or returned nothing usable."""
