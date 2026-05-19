"""
pure generator engine protocol and determinism contract.

an engine is a Generator[Req, Resp, SearchResult] that holds no instance state;
all algorithm state is generator-frame local (single-owner sequential state, safe
under the one-thread invariant of the runtime).

DETERMINISM CONTRACT:
  for fixed injected dependencies (embedding model, RNG) and a fixed response
  transcript, an engine emits a deterministic request sequence and a deterministic
  SearchResult. the runtime advances each engine from exactly one thread.
  usage: prime with next(), never send(non-None) -- raises TypeError on an
  unstarted generator.

ENFORCEABLE INVARIANTS:
  - engine state that influences a Req must be set-typed (no list-order
    leakage into the request).
  - any DiGraph an engine builds inserts edges in sorted order (networkx
    shortest-path ties break on insertion order).
  - frontier cap/prune tie-breaks use an explicit key, never raw order
    (walk top-cap key = (sim, id)).
"""

from dataclasses import dataclass
from typing import Generator, Any
import numpy as np


@dataclass(frozen=True)
class Req:
    """
    batch request emitted by an engine.

    nodes: frozenset of work W-ids the engine needs fetched (may be empty).
    want_emb: if True, also fetch and attach title embeddings to the response
      for each node in nodes.
    """
    nodes: frozenset[str]
    want_emb: bool = False


Resp = dict[str, dict | Exception]
"""
response to a Req: per-node data dict or Exception if fetch failed.

per-node dict schema (present iff fetch succeeded):
  {
    "refs": list[str],              # W-ids of works this node cites
    "citers": list[str],            # W-ids of works that cite this node
    "title": str | None,            # work title
    "year": int | None,             # publication year
    "doi": str | None,              # DOI string
    "cited_by_count": int | None,   # count of citing works
    "emb": np.ndarray,              # (optional) title embedding vector;
                                    # present iff originating Req had want_emb=True
  }

if fetch fails (HTTP error, parse error, etc), the value is an Exception.
an engine treats Exception as "no neighbors" and continues.
"""


@dataclass(frozen=True)
class SearchResult:
    """
    final result returned by an engine (via StopIteration.value).

    bridges: dict[str, dict]. key = node W-id (meeting point or bridge candidate);
      value = per-group distance/scoring inputs (shape and content defined by
      each engine; consumed by core.score).
    meta: dict. diagnostics, strategy-specific. examples: radii (bidir),
      rounds (refine), recurrence flags, frontier sizes, etc.
      path-carrying convention: when an engine is built with want_paths=True,
      bidir and walk surface predecessor maps in
      meta["pred"] = {"a": {node: parent, ...}, "b": {node: parent, ...}}.
      a generator frame dies on return, so any state a downstream engine needs
      (e.g. refine reconstructing meeting paths) must travel here in meta,
      never be read off a finished engine object.
    """
    bridges: dict[str, dict]
    meta: dict


Engine = Generator[Req, Resp, SearchResult]
"""
pure generator engine type alias.

lifecycle (enforced by runtime):
  1. prime: engine = build(...); Req_0 = next(engine)
  2. loop: while live:
       resp = fetch(Req_i.nodes, want_emb=Req_i.want_emb)
       Req_{i+1} = engine.send(resp)
  3. return: StopIteration.value is SearchResult

invariants:
  - engine.send() is called exactly once per Req yielded (except the priming
    next() call receives no Resp).
  - Req.nodes are always frozenset (no list-order mutation).
  - if Req.nodes is empty, runtime sends Resp = {} (empty dict).
  - the RNG and model are immutable from the engine's frame perspective
    (injected once at construction; no state mutation).
"""
