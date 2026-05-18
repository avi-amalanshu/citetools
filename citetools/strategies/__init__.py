"""pluggable search strategies for citation bridges.

v1 ships an exhaustive N-way intersection strategy. Bidirectional variants (find_bridges_bidir, find_bridges_bidir_nway) are also available. A graph-walk variant (find_bridges_walk) is also provided. An iterative refinement variant (find_bridges_refine) is also provided.
"""

from .intersection import find_bridges
from .bidir import find_bridges_bidir, find_bridges_bidir_nway
from .walk import find_bridges_walk
from .refine import find_bridges_refine

__all__ = ["find_bridges", "find_bridges_bidir", "find_bridges_bidir_nway", "find_bridges_walk", "find_bridges_refine"]
