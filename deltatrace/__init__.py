"""DeltaTrace: git-style version control for tabular data."""

from .repo import CommitResult, DeltaRepo, DiffResult
from .storage import StoreError

__version__ = "0.1.0"
__all__ = ["DeltaRepo", "CommitResult", "DiffResult", "StoreError", "__version__"]
