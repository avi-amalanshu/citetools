"""Interpreter for pure generator engines.

The impure runtime that drives pure generator engines. All concurrency and
effects live here. This module implements lockstep-union execution: one
interpreter drives all live generators, unions every live request into one
batched fetch, distributes responses back. Engines are pure Python generators
that yield I/O requirements and return a SearchResult; the runtime handles all
fetching, embedding, and control flow.

Key invariant: every engine advances from exactly one thread (sequential state
safety). The lockstep loop ensures deterministic request ordering via sorted
fetch.
"""

from contextlib import nullcontext
from typing import Any, Generator
from joblib import Parallel

from citetools.core.engines.protocol import Req, Resp, SearchResult
from citetools.io.parallel import fetch_all
from citetools.io.models import embed_titles, Embedder


def _mk_parallel(n_jobs: int):
    """Build a context manager for parallel execution or serial mode.

    Args:
      n_jobs: Number of workers. If > 1, returns Parallel(n_jobs, backend="threading");
        else returns contextlib.nullcontext(None).

    Returns:
      A context manager suitable for `with _mk_parallel(n_jobs) as parallel:`.
      On entry, yields either a bound Parallel object (n_jobs > 1) or None
      (n_jobs <= 1). Pass this value to fetch_all, which checks `parallel is None`
      for serial dispatch. Ensures a single Parallel object (or None) is used
      across all requests within the context, avoiding thread pool leaks.
    """
    if n_jobs > 1:
        return Parallel(n_jobs=n_jobs, backend="threading")
    else:
        return nullcontext(None)


def run(
    engine: Generator[Req, Resp, SearchResult],
    oracle: Any,
    embedder: Any = None,
    *,
    n_jobs: int = 8,
    verbose: bool = False,
) -> SearchResult:
    """Drive a single engine to completion.

    Args:
      engine: Generator[Req, Resp, SearchResult].
      oracle: Object with an `nbrs(node_id: str) -> dict` method (e.g., OpenAlex client).
      embedder: None or an `io.models.Embedder` dataclass with `model` and `cache`
        attributes. If None, embedding requests are silently ignored.
      n_jobs: Number of parallel workers for node fetching. If > 1, uses
        joblib.Parallel with backend="threading"; otherwise serial.
      verbose: If True, pass progress=True to fetch_all (logs per-node completion).

    Returns:
      SearchResult with bridges dict and meta dict.

    Raises:
      TypeError: If engine is not a generator.
      StopIteration (should not reach caller): Engine bug (non-StopIteration exception).

    Procedure: delegates all work to run_ensemble with a singleton dict.
    """
    result = run_ensemble(
        {"_": engine},
        oracle,
        embedder,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    return result["_"]


def run_ensemble(
    engines: dict[Any, Generator[Req, Resp, SearchResult]],
    oracle: Any,
    embedder: Any = None,
    *,
    n_jobs: int = 8,
    verbose: bool = False,
) -> dict[Any, SearchResult]:
    """Drive multiple engines in lockstep-union.

    Proc: prime all engines, catch zero-I/O completions; enter Parallel context;
    loop while live: union all live requests; batch-fetch sorted nodes; embed if
    requested; route each engine its slice; advance; catch completions; exit
    Parallel context.

    Args:
      engines: dict mapping tag (any hashable, typically str or int) to
        Generator[Req, Resp, SearchResult].
      oracle: Object with an `nbrs(node_id: str) -> dict` method.
      embedder: None or an `io.models.Embedder` dataclass. If None, embedding
        requests in Req are silently ignored.
      n_jobs: Workers for fetch_all. > 1 -> joblib.Parallel; else serial.
      verbose: If True, fetch_all logs per-node completion.

    Returns:
      dict[tag, SearchResult] for all input tags; tags mapping to engines that
      finished during prime have their StopIteration.value cached.

    Raises:
      TypeError: If engine is not a generator.
      StopIteration (should not reach caller): Engine bug (non-StopIteration exception).
    """
    # step 1: prime all engines, detect zero-I/O completions
    live = {}  # tag -> (engine, req)
    results = {}  # tag -> SearchResult
    for tag, engine in engines.items():
        try:
            req = next(engine)  # NEVER send(None) -- raises TypeError
            live[tag] = (engine, req)
        except StopIteration as e:
            results[tag] = e.value

    # step 2: enter Parallel context (or nullcontext if n_jobs <= 1)
    with _mk_parallel(n_jobs) as parallel:
        # step 3: main loop -- while any engine live
        while live:
            # 3a. union all live requests
            union_nodes = frozenset().union(
                *(req.nodes for _, req in live.values())
            )
            union_want_emb = any(req.want_emb for _, req in live.values())

            # 3b. deterministic fetch: sorted nodes
            sorted_nodes = sorted(union_nodes) if union_nodes else []
            resp_dict = fetch_all(
                oracle, sorted_nodes, parallel, progress=verbose
            )

            # 3c. embed if needed
            if union_want_emb and embedder:
                titles = {
                    nid: resp_dict[nid].get("title", "")
                    for nid in sorted_nodes
                    if nid in resp_dict
                    and not isinstance(resp_dict[nid], Exception)
                }
                if titles:
                    embs = embed_titles(
                        embedder.model, embedder.cache, titles
                    )
                    for nid, emb in embs.items():
                        if nid in resp_dict:
                            resp_dict[nid]["emb"] = emb

            # 3d. route each engine its slice, collect completions
            done = []
            for tag, (engine, req) in live.items():
                # build response slice for this engine (dict of only its requested nodes)
                slice_resp = {
                    nid: resp_dict[nid]
                    for nid in req.nodes
                    if nid in resp_dict
                }
                # handle empty slice: send {} (engine still advances)
                try:
                    next_req = engine.send(slice_resp)
                    live[tag] = (engine, next_req)
                except StopIteration as e:
                    results[tag] = e.value
                    done.append(tag)

            # drop completed engines
            for tag in done:
                del live[tag]

    return results
