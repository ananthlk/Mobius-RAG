"""Fillers — Step 3 of the answer engine. Pure chunk-to-slot assignment orchestrator."""

from app.services.retriever.fillers.contracts import (
    FilledChunk,
    FilledSlot,
    FilledShape,
)
from app.services.retriever.fillers.filler_a import fill_shape_bm25

__all__ = [
    "FilledChunk",
    "FilledSlot",
    "FilledShape",
    "fill_shape_bm25",
]
