"""pluggable search strategies for citation bridges.

v1 ships an exhaustive N-way intersection strategy. Bidirectional variants (find_bridges_bidir, find_bridges_bidir_nway) are also available.
"""

from .intersection import find_bridges
from .bidir import find_bridges_bidir, find_bridges_bidir_nway

__all__ = ["find_bridges", "find_bridges_bidir", "find_bridges_bidir_nway"]
