"""thin literature-review tools; v1: citation bridge finder over the openalex graph."""

__version__ = "0.1.0"

from .errors import OpenAlexError, BadId
from .openalex import OpenAlex
from .graph import CiteGraph
from .strategies import find_bridges

__all__ = [
    "OpenAlex",
    "CiteGraph",
    "find_bridges",
    "OpenAlexError",
    "BadId",
]
