"""Data access primitives for Small Agent pretraining."""

from .indexed_dataset import IndexedTokenDataset, SplitReport, load_manifest, validate_split
from .sampler import StatefulDistributedSampler

__all__ = [
    "IndexedTokenDataset",
    "SplitReport",
    "StatefulDistributedSampler",
    "load_manifest",
    "validate_split",
]
