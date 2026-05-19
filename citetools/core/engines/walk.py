"""Embedding-pruned bidir engine. Yields batch Req for frontier nodes with want_emb=True.
Prunes each hop's frontier by L2 embedding similarity to the partner group's seed embeddings:
deterministic top-cap (tie-break on (similarity, id)) when deterministic=True; stochastic
Bernoulli (temp T, floor p) using injected rng when deterministic=False, then trim to cap.
Preserves the exact prune algorithm from strategies/walk._keep. Returns SearchResult carrying
meeting/distance maps and algorithm metadata (radii, hop count)."""

from __future__ import annotations

import numpy as np
from typing import Generator

from .protocol import Req, Resp, SearchResult
from ..embed import sim_to_set


def walk_engine(
    group_a: set[str],
    group_b: set[str],
    *,
    rng: np.random.Generator,
    cap: int = 60,
    deterministic: bool = True,
    p: float = 0.15,
    T: float = 0.1,
    partner_seed_embs: dict[int, np.ndarray],
    want_paths: bool = False,
) -> Generator[Req, Resp, SearchResult]:
    """Embedding-pruned bidir meet-in-the-middle BFS engine.

    Yields Req(frontier, want_emb=True) each hop. Prunes frontier by embedding
    similarity to partner group's seed vectors: top-cap (tie-break (sim desc, id asc))
    when deterministic; Bernoulli (temp T, floor p) + trim when not. Returns SearchResult
    with meeting nodes {node: {"a": dist_a, "b": dist_b}} and radii/hop metadata.

    Args:
        group_a, group_b: seed sets (non-empty).
        rng: numpy.random.Generator, frame-local; consumed only in stochastic mode.
        cap: frontier cap per hop (must be > 0).
        deterministic: if True, top-cap prune; if False, Bernoulli + cap.
        p: floor probability for Bernoulli (0 < p < 1).
        T: sigmoid temperature (T > 0).
        partner_seed_embs: {group_idx: (k, dim) embedding array}.
        want_paths: if True, track predecessor chains.

    Yields:
        Req(frozenset(frontier), want_emb=True) each hop.

    Receives:
        Resp: dict[str, dict | Exception] with keys {"refs", "citers", "emb"?, ...}.
        "emb" present iff want_emb=True in the request; absent for titleless nodes.

    Returns:
        SearchResult with bridges={node: {"a": dist_a, "b": dist_b}} (meeting nodes only)
        and meta={"radii": {...}, "hops": int, "deterministic": bool, "cap": int, "pred"?: {...}}.
        When want_paths=True, meta includes "pred": {"a": {node: parent, ...}, "b": {node: parent, ...}}.
        When want_paths=False, "pred" is omitted.

    pseudocode:
      init phase:
        _dist = {"a": {s: 0 for s in group_a}, "b": {s: 0 for s in group_b}}
        _frontier = {"a": set(group_a), "b": set(group_b)}
        _meeting = {s: {"a": 0, "b": 0} for s in (group_a & group_b)}
        _expanding = None
        _done = False
        if want_paths: _pred = {"a": {}, "b": {}} else _pred = None

      prime (first hop):
        pick side to expand: smaller frontier (or "a" on tie); set _expanding
        yield first frontier

      hop loop (while not _done):
        frontier = _frontier[_expanding]
        state_out = build_state_and_radii()
        if _done:
          return SearchResult(...)

        yield Req(frozenset(frontier), want_emb=True)
        resp = receive

        ingest phase:
          side = _expanding
          other = "b" if side == "a" else "a"
          discovered = set()

          for node in frontier:
            res = resp.get(node, {})
            if isinstance(res, Exception) or ("refs" not in res and "citers" not in res):
              neighbours = set()
            else:
              neighbours = set(res.get("refs", [])) | set(res.get("citers", []))

            d = _dist[side][node] + 1
            for m in neighbours:
              if m not in _dist[side]:
                _dist[side][m] = d
                if want_paths: _pred[side][m] = node
                discovered.add(m)
                if m in _dist[other] and m not in _meeting:
                  _meeting[m] = {"a": _dist["a"][m], "b": _dist["b"][m]}

        prune frontier:
          frontier_sorted = sorted(discovered)

          node_embs_dict = {n: resp[n].get("emb") for n in frontier_sorted if n in resp}
          partner_group_idx = 1 if side == "a" else 0
          partner_seed_array = partner_seed_embs.get(partner_group_idx)

          if partner_seed_array is None or len(partner_seed_array) == 0:
            similarities = np.zeros(len(frontier_sorted), dtype=np.float32)
          else:
            similarities = np.array([
              (sim_to_set(node_embs_dict[n], partner_seed_array, agg="max")
               if n in node_embs_dict and node_embs_dict[n] is not None else 0.0)
              for n in frontier_sorted
            ], dtype=np.float32)

          if deterministic:
            if len(frontier_sorted) <= cap:
              kept = set(frontier_sorted)
            else:
              pairs = [(similarities[i], frontier_sorted[i]) for i in range(len(frontier_sorted))]
              pairs_sorted = sorted(pairs, key=lambda x: (-x[0], x[1]))
              kept = set(p[1] for p in pairs_sorted[:cap])
          else:
            h_med = np.median(similarities) if len(similarities) > 0 else 0.0
            denom = np.maximum(T, 1e-8)
            kappa = p + (1 - p) / (1 + np.exp(-(similarities - h_med) / denom))

            keep_flags = rng.binomial(1, kappa)
            kept = {frontier_sorted[i] for i in range(len(frontier_sorted)) if keep_flags[i] == 1}

            if len(kept) > cap:
              kept_sorted = sorted(kept)
              kept_sims = {frontier_sorted[i]: similarities[i] for i in range(len(frontier_sorted))}
              kept_pairs = [(kept_sims[n], n) for n in kept_sorted]
              kept_pairs_trimmed = sorted(kept_pairs, key=lambda x: (-x[0], x[1]))[:cap]
              kept = set(p[1] for p in kept_pairs_trimmed)

          _frontier[side] = kept

        terminate check:
          if _meeting is non-empty:
            _done = True
          elif both _frontier["a"] and _frontier["b"] are empty:
            _done = True
          else:
            pick next side: smaller frontier, tie-break "a"
            if len(_frontier["a"]) == 0:
              _expanding = "b"
            elif len(_frontier["b"]) == 0:
              _expanding = "a"
            elif len(_frontier["a"]) <= len(_frontier["b"]):
              _expanding = "a"
            else:
              _expanding = "b"
    """
    # init
    _dist = {
        "a": {s: 0 for s in group_a},
        "b": {s: 0 for s in group_b},
    }
    _frontier = {"a": set(group_a), "b": set(group_b)}
    _meeting = {s: {"a": 0, "b": 0} for s in (group_a & group_b)}
    _pred = {"a": {}, "b": {}} if want_paths else None
    hop = 0

    # prime: pick smaller frontier first
    _expanding = "a" if len(_frontier["a"]) <= len(_frontier["b"]) else "b"
    frontier = _frontier[_expanding]

    # combined prime yield+receive
    resp = yield Req(frozenset(frontier), want_emb=True)

    while True:
        # ingest
        side = _expanding
        other = "b" if side == "a" else "a"
        discovered = set()

        for node in frontier:
            res = resp.get(node, {})
            if isinstance(res, Exception):
                neighbours = set()
            elif "refs" not in res and "citers" not in res:
                neighbours = set()
            else:
                neighbours = set(res.get("refs", [])) | set(res.get("citers", []))

            d = _dist[side][node] + 1
            for m in neighbours:
                if m not in _dist[side]:
                    _dist[side][m] = d
                    if want_paths:
                        _pred[side][m] = node
                    discovered.add(m)
                    if m in _dist[other] and m not in _meeting:
                        _meeting[m] = {"a": _dist["a"][m], "b": _dist["b"][m]}

        # prune frontier
        frontier_sorted = sorted(discovered)
        node_embs_dict = {n: resp[n].get("emb") for n in frontier_sorted if n in resp}
        partner_group_idx = 1 if side == "a" else 0
        partner_seed_array = partner_seed_embs.get(partner_group_idx)

        if partner_seed_array is None or len(partner_seed_array) == 0:
            similarities = np.zeros(len(frontier_sorted), dtype=np.float32)
        else:
            similarities = np.array([
                (sim_to_set(node_embs_dict[n], partner_seed_array, agg="max")
                 if n in node_embs_dict and node_embs_dict[n] is not None else 0.0)
                for n in frontier_sorted
            ], dtype=np.float32)

        if deterministic:
            if len(frontier_sorted) <= cap:
                kept = set(frontier_sorted)
            else:
                spairs = sorted(
                    [(similarities[i], frontier_sorted[i]) for i in range(len(frontier_sorted))],
                    key=lambda x: (-x[0], x[1]),
                )
                kept = {sp[1] for sp in spairs[:cap]}
        else:
            h_med = np.median(similarities) if len(similarities) > 0 else 0.0
            denom = np.maximum(T, 1e-8)
            kappa = p + (1 - p) / (1 + np.exp(-(similarities - h_med) / denom))
            keep_flags = rng.binomial(1, kappa)
            kept = {frontier_sorted[i] for i in range(len(frontier_sorted)) if keep_flags[i] == 1}
            if len(kept) > cap:
                kept_sorted = sorted(kept)
                kept_sims = {frontier_sorted[i]: similarities[i] for i in range(len(frontier_sorted))}
                kept_pairs = sorted(
                    [(kept_sims[n], n) for n in kept_sorted],
                    key=lambda x: (-x[0], x[1]),
                )[:cap]
                kept = {kp[1] for kp in kept_pairs}

        _frontier[side] = kept
        hop += 1

        # terminate?
        if _meeting or (not _frontier["a"] and not _frontier["b"]):
            break

        # pick next side: smaller frontier, tie-break "a"
        if not _frontier["a"]:
            _expanding = "b"
        elif not _frontier["b"]:
            _expanding = "a"
        elif len(_frontier["a"]) <= len(_frontier["b"]):
            _expanding = "a"
        else:
            _expanding = "b"

        # combined yield+receive for next hop
        frontier = _frontier[_expanding]
        resp = yield Req(frozenset(frontier), want_emb=True)

    # build result
    depth_a = max(_dist["a"].values()) if _dist["a"] else 0
    depth_b = max(_dist["b"].values()) if _dist["b"] else 0
    _meta = {
        "radii": {"a": depth_a, "b": depth_b},
        "hops": hop,
        "deterministic": deterministic,
        "cap": cap,
    }
    if want_paths:
        _meta["pred"] = _pred

    return SearchResult(
        bridges={nid: {"a": _meeting[nid]["a"], "b": _meeting[nid]["b"]} for nid in _meeting},
        meta=_meta,
    )
