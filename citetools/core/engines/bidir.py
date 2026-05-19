"""Bidirectional meet-in-the-middle bridge search engine.

Pure generator: yields node batches to fetch, ingests neighbor sets,
detects meetings (nodes reachable from both seed groups), and returns
a deterministic SearchResult. Modes: "budget" (first meeting), "exact"
(complete same-depth layer). No internal state mutation outside the
generator frame.
"""

from typing import Generator

from .protocol import Req, Resp, SearchResult


def bidir_engine(
    group_a: set[str],
    group_b: set[str],
    *,
    mode: str = "budget",
    max_depth: int = 6,
    frontier_cap: int = 50,
    want_paths: bool = False,
) -> Generator[Req, Resp, SearchResult]:
    """Meet-in-the-middle generator between two seed groups.

    Lazily expand frontiers on both sides, yielding batches of nodes to fetch,
    ingesting refs|citers as neighbors, relaxing distances, and detecting
    meetings (nodes reached from both sides). On termination, return
    SearchResult with bridges and diagnostics.

    Modes:
      - "budget": stop at first meeting (frontier-capped).
      - "exact": uncapped expansion; stop after a full same-depth layer
        containing a meeting.

    Args:
      group_a, group_b: non-empty seed sets (strings, canonicalized by caller).
      mode: "budget" or "exact" (default "budget").
      max_depth: maximum hop depth on either side (default 6).
      frontier_cap: max neighbors to expand per hop in budget mode; None
                    disables (default 50).
      want_paths: if True, build predecessor maps to reconstruct paths
                  (default False). [Note: paths not used in 2-way engine,
                  reserved for future use.]

    Yields:
      Req(nodes: frozenset[str], want_emb: bool) -- nodes to fetch;
      want_emb always False here.

    Receives (via .send(resp)):
      Resp = dict[str, dict | Exception] -- per-node neighbors+metadata or error.
      Each dict entry: node_id -> {"refs": [...], "citers": [...],
      "title": str, "year": int, "doi": str, "cited_by_count": int}
      Missing node or Exception value -> treat as no neighbors (deterministic).

    Returns (via StopIteration.value):
      SearchResult(bridges: dict, meta: dict)
        bridges: {node_id: {"a": dist_a, "b": dist_b}} for nodes reached
                 from both sides.
        meta: {"depth_a": int, "depth_b": int, "pred": {...} (only if want_paths)}
          "depth_a", "depth_b": final radii on both sides.
          "pred" (only if want_paths=True): predecessor maps shaped
            {"a": {node: parent, ...}, "b": {node: parent, ...}},
            enabling path reconstruction; omitted if want_paths=False.

    Raises:
      ValueError if group_a or group_b is empty.

    Procedure (per-hop loop):
      1. On priming (first next() call):
         - Initialize state: dist[side] = {seed: 0 for seed in group_side}.
         - frontier[side] = set(group_side).
         - If want_paths: pred[side] = {} for both sides (empty, seeds have no parent).
         - Record shared seeds as meetings with (dist_a=0, dist_b=0).
         - expanding = None (no frontier emitted yet).
         - Yield Req(frozenset(group_a), want_emb=False) [first frontier].
      2. On each .send(resp):
         - If expanding is not None (not first call), ingest the frontier:
           * side = expanding; other = "b" if side=="a" else "a".
           * discovered = empty set.
           * For each node in frontier[side]:
             - neighbors = refs | citers (union) if resp[node] is a valid dict;
               else empty set.
             - d = dist[side][node] + 1.
             - For each m in neighbors not yet in dist[side]:
               * Set dist[side][m] = d.
               * If want_paths: pred[side][m] = node.
               * Add m to discovered.
               * If m in dist[other] and m not in meeting:
                 meeting[m] = {"a": dist["a"][m], "b": dist["b"][m]}.
           * If mode=="budget" and discovered exceeds frontier_cap:
             - Cap discovered to frontier_cap via sorted(discovered)[:cap]
               (deterministic tie-break by ID).
           * frontier[side] = discovered.
         - Termination checks (in order):
           * If mode=="budget" and meeting is non-empty: done = True.
           * Elif mode=="exact" and meeting is non-empty:
             - best_total = min(d["a"] + d["b"] for d in meeting.values()).
             - If depth("a") + depth("b") >= best_total: done = True.
           * Elif frontier["a"] and frontier["b"] both empty: done = True.
           * Elif max(depth("a"), depth("b")) >= max_depth: done = True.
           * Else: done = False.
         - If done:
           * meta = {"depth_a": depth("a"), "depth_b": depth("b")}.
           * If want_paths: append meta["pred"] = pred (both sides' maps).
           * Return SearchResult(bridges=dict(meeting), meta=meta).
         - Else (not done):
           * Pick next side to expand:
             - len_a = len(frontier["a"]), len_b = len(frontier["b"]).
             - If len_a == 0 and len_b > 0: expanding = "b".
             - Elif len_b == 0 and len_a > 0: expanding = "a".
             - Elif len_a <= len_b: expanding = "a" (tie-break "a").
             - Else: expanding = "b".
           * Yield Req(frozenset(frontier[expanding]), want_emb=False).

    Helper: depth(side: str) -> int
      Return max(dist[side].values()) if dist[side] non-empty; else 0.
      (Hop distance of the most distant node on that side.)
    """
    # validate non-empty groups
    if not group_a or not group_b:
        raise ValueError("group must be non-empty")

    # initialize generator state
    dist = {
        "a": {seed: 0 for seed in group_a},
        "b": {seed: 0 for seed in group_b},
    }
    frontier = {
        "a": set(group_a),
        "b": set(group_b),
    }
    meeting = {}
    for node in group_a & group_b:
        meeting[node] = {"a": 0, "b": 0}

    expanding = None

    if want_paths:
        pred = {"a": {}, "b": {}}
    else:
        pred = None

    def depth(side: str) -> int:
        """current hop depth: max distance on side, or 0 if only seeds."""
        return max(dist[side].values()) if dist[side] else 0

    # yield initial frontier (group_a)
    expanding = "a"
    resp = yield Req(frozenset(frontier["a"]), want_emb=False)

    # main loop
    while True:
        # ingest the fetched frontier
        side = expanding
        other = "b" if side == "a" else "a"
        discovered = set()

        for node in frontier[side]:
            res = resp.get(node)
            if res is None or isinstance(res, Exception):
                neighbors = set()
            else:
                neighbors = set(res.get("refs", [])) | set(res.get("citers", []))

            d = dist[side][node] + 1
            for m in neighbors:
                if m not in dist[side]:
                    dist[side][m] = d
                    if want_paths:
                        pred[side][m] = node
                    discovered.add(m)
                    if m in dist[other] and m not in meeting:
                        meeting[m] = {
                            "a": dist["a"][m],
                            "b": dist["b"][m],
                        }

        # cap discovered set (budget mode)
        if mode == "budget" and frontier_cap is not None and len(discovered) > frontier_cap:
            discovered = set(sorted(discovered)[: frontier_cap])

        frontier[side] = discovered

        # termination checks
        done = False
        if mode == "budget" and meeting:
            done = True
        elif mode == "exact" and meeting:
            best_total = min(d["a"] + d["b"] for d in meeting.values())
            if depth("a") + depth("b") >= best_total:
                done = True
        elif not frontier["a"] and not frontier["b"]:
            done = True
        elif max(depth("a"), depth("b")) >= max_depth:
            done = True

        # return if done
        if done:
            meta = {"depth_a": depth("a"), "depth_b": depth("b")}
            if want_paths:
                meta["pred"] = pred
            return SearchResult(bridges=dict(meeting), meta=meta)

        # pick next side to expand
        len_a = len(frontier["a"])
        len_b = len(frontier["b"])

        if len_a == 0 and len_b > 0:
            expanding = "b"
        elif len_b == 0 and len_a > 0:
            expanding = "a"
        elif len_a <= len_b:
            expanding = "a"
        else:
            expanding = "b"

        # yield next frontier
        resp = yield Req(frozenset(frontier[expanding]), want_emb=False)
