"""
Concurrent nbrs fetching via joblib threading.

Wraps oracle.nbrs calls for thread-pool dispatch. Isolates per-node
failures so one HTTP error does not abort the batch. Preserves input
node order in results via joblib's default list semantics.
"""

import logging

from joblib import delayed

log = logging.getLogger("citetools.io.parallel")


def _nbrs_safe(oracle, node: str, progress: bool = False) -> dict | Exception:
    """
    Safe nbrs call: returns result dict or caught Exception.

    Proc:
      - call oracle.nbrs(node)
      - if any exception is raised, catch it and return it as-is
      - if progress: log the node on completion (runs in the worker thread,
        so the log line marks true per-node completion)
      - otherwise return the result dict

    Args:
      oracle: OpenAlex instance (impure; rate-limited HTTP client)
      node: canonical W-id string
      progress: if True, emit an info log line when this node completes

    Returns:
      dict with keys "refs", "citers", "title", "year", "doi", "cited_by_count"
      on success; Exception instance on failure (never re-raised).
    """
    try:
        res = oracle.nbrs(node)
    except Exception as e:
        res = e
    if progress:
        log.info("    fetched %s", node)
    return res


def fetch_all(
    oracle, nodes: list[str], parallel=None, *, progress: bool = False
) -> dict[str, dict | Exception]:
    """
    Fetch nbrs for multiple nodes, serially or concurrently.

    Proc:
      - if nodes is empty: return {}
      - if parallel is None: run serially
          result_dict = {n: _nbrs_safe(oracle, n, progress) for n in nodes}
      - if parallel is a Parallel object: dispatch concurrently
          results = parallel(delayed(_nbrs_safe)(oracle, n, progress) for n in nodes)
          result_dict = dict(zip(nodes, results))
      - return result_dict

    Args:
      oracle: OpenAlex instance (impure; rate-limited HTTP client)
      nodes: ordered sequence of canonical W-id strings; may be empty
      parallel: None (serial mode) or a bound Parallel object from
                joblib.Parallel(..., backend="threading") (concurrent mode).
                the Parallel object is obtained by the caller via a
                context manager: `with Parallel(...) as p: fetch_all(..., p)`
      progress: if True, each worker logs its node on completion (verbose)

    Returns:
      dict mapping each input node id to its nbrs result or Exception.
      {node: {"refs": [...], "citers": [...], ...} | Exception for each node in input order}

    Order preservation:
      joblib.Parallel's default return_as="list" guarantees that the returned
      list order matches the input iterable order. Thus dict(zip(nodes, results))
      correctly pairs each node with its own result, never mixing them up.

    Edge cases:
      - empty nodes: returns {} (no calls dispatched)
      - per-node exception: returned in the result dict; does NOT abort the batch
      - parallel is None: serial execution (testing, single-threaded contexts)
    """
    if not nodes:
        return {}

    if parallel is None:
        # serial: simple dict comprehension
        return {n: _nbrs_safe(oracle, n, progress) for n in nodes}
    else:
        # concurrent: dispatch to joblib thread pool
        results = parallel(delayed(_nbrs_safe)(oracle, n, progress) for n in nodes)
        return dict(zip(nodes, results))
