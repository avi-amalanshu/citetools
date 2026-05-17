"""pluggable search strategies for citation bridges.

v1 ships an exhaustive N-way intersection strategy (find_bridges).
future siblings will be added here sharing the (oracle, groups, params) -> result signature.
"""

from .intersection import find_bridges

__all__ = ["find_bridges"]
