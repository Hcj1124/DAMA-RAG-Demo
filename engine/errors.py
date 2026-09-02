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
            f"Build it first:  uv run dama-rag index"
        )


class IndexStaleError(EngineError):
    """The index was written by a different embedding model than the query."""

    def __init__(self, *, indexed_with: str, querying_with: str) -> None:
        super().__init__(
            f"The index was built with '{indexed_with}' but you are querying "
            f"with '{querying_with}'. Comparing vectors from two different "
            f"models returns plausible-looking nonsense rather than an error, "
            f"so this is refused.\n"
            f"Rebuild:  uv run dama-rag index --rebuild"
        )


class LanguageModelError(EngineError):
    """Generation failed or returned nothing usable."""
