"""DeltaTrace: git-style version control for tabular data."""

from .identity import ContentMatchIdentity, IdentityStrategy, PrimaryKeyIdentity
from .repo import CommitResult, DeltaRepo, DiffResult
from .storage import StoreError

__version__ = "0.2.0"
__all__ = [
    "DeltaRepo",
    "CommitResult",
    "DiffResult",
    "StoreError",
    "ContentMatchIdentity",
    "PrimaryKeyIdentity",
    "IdentityStrategy",
    "__version__",
]
