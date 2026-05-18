"""thin literature-review tools; v1: citation bridge finder over the openalex graph."""

__version__ = "0.1.0"

from .errors import OpenAlexError, BadId
from .openalex import OpenAlex
from .graph import CiteGraph
from .strategies import find_bridges, find_bridges_bidir, find_bridges_bidir_nway, find_bridges_walk, find_bridges_refine

__all__ = [
    "OpenAlex",
    "CiteGraph",
    "find_bridges",
    "find_bridges_bidir",
    "find_bridges_bidir_nway",
    "find_bridges_walk",
    "find_bridges_refine",
    "OpenAlexError",
    "BadId",
]
