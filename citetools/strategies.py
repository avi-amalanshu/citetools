"""
Thin impure public API wrappers around pure generator engines.

Each function canonicalizes inputs, constructs the appropriate engine(s),
drives them via io.runtime, shapes results via core.score, and returns
a list of dicts (exact format matching the no-behavior-change contract).

Functions:
  - find_bridges: N-way intersection via one generator.
  - find_bridges_bidir: pairwise meet-in-the-middle (N=2).
  - find_bridges_bidir_nway: N-way via C(N,2) pairwise decomposition.
  - find_bridges_walk: pairwise bidir + embedding-guided frontier prune + fusion score.
  - find_bridges_refine: multi-round nested-generator refinement.
"""

import itertools
import logging
from typing import Optional

import numpy as np
from numpy.random import SeedSequence

from citetools.core import score, parse
from citetools.core.engines import intersection_engine, bidir_engine, walk_engine, refine_engine
from citetools.core.embed import sim_to_set
from citetools.io import runtime
from citetools.io import models

log = logging.getLogger("citetools.strategies")

__all__ = [
    "find_bridges",
    "find_bridges_bidir",
    "find_bridges_bidir_nway",
    "find_bridges_walk",
    "find_bridges_refine",
]


def find_bridges(
    oracle,
    groups: list[list[str]],
    *,
    depth: int = 2,
    n_jobs: int = 8,
    min_groups: Optional[int] = None,
    top_k: int = 25,
    verbose: bool = False,
) -> list[dict]:
    """N-way intersection via one generator.

    Proc:
      1. validate all groups non-empty; raise ValueError if any is empty
      2. default min_groups to len(groups) if not provided
      3. canonicalize groups: {parse.key(s) for s in g} for each group
      4. instantiate intersection_engine(canon, depth=depth)
      5. drive via runtime.run(eng, oracle, n_jobs, verbose)
      6. shape via core.score: per-group distances, candidate filter, sort, top_k
      7. finally: oracle.close()
    """
    # validate
    for g in groups:
        if not g:
            raise ValueError("group must be non-empty")

    if min_groups is None:
        min_groups = len(groups)

    # canonicalize
    canon = [{parse.key(s) for s in g} for g in groups]

    try:
        # build engine
        engine = intersection_engine(canon, depth=depth)

        # drive
        result = runtime.run(engine, oracle, n_jobs=n_jobs, verbose=verbose)

        # shape via score
        all_seeds = set().union(*canon)
        candidates = score.keep(result.bridges.keys(), result.bridges, all_seeds, min_groups)
        scored = score.agg(candidates, result.bridges)

        # build records: meta carries node metadata and degree per candidate
        records = []
        for cand in candidates:
            agg = scored[cand]
            work = result.meta.get(cand, {})
            rec = score.record(
                cand,
                agg["dists"],
                agg["groups_hit"],
                agg["score"],
                work,
                degree=work.get("degree"),
            )
            records.append(rec)

        # sort and trim
        records = score.topk(
            records,
            lambda r: (-r["groups_hit"], r["score"], -(r.get("cited_by_count") or 0), r["id"]),
            top_k,
        )

        log.info("find_bridges: done, %d bridge(s)", len(records))
        return records

    finally:
        oracle.close()


def find_bridges_bidir(
    oracle,
    group_a,
    group_b,
    *,
    mode: str = "budget",
    max_depth: int = 6,
    frontier_cap: int = 50,
    top_k: int = 25,
    n_jobs: int = 8,
    verbose: bool = False,
) -> list[dict]:
    """Pairwise meet-in-the-middle (N=2).

    Proc:
      1. validate groups non-empty
      2. canonicalize: {parse.key(s) for s in g} for each group
      3. instantiate bidir_engine(canon_a, canon_b, mode=..., max_depth=..., frontier_cap=...)
      4. drive via runtime.run
      5. filter seed nodes, backfill metadata
      6. build records: id, score=a+b, groups_hit=2, dists, title, year, doi, cited_by_count
      7. sort by (score asc, cited_by_count desc, id asc)
      8. finally: oracle.close()
    """
    if not group_a or not group_b:
        raise ValueError("group must be non-empty")

    # canonicalize
    canon_a = {parse.key(s) for s in group_a}
    canon_b = {parse.key(s) for s in group_b}

    try:
        # build engine
        engine = bidir_engine(
            canon_a, canon_b, mode=mode, max_depth=max_depth, frontier_cap=frontier_cap
        )

        # drive
        result = runtime.run(engine, oracle, n_jobs=n_jobs, verbose=verbose)

        # filter and backfill
        seeds = canon_a | canon_b
        bridges = result.bridges
        to_fetch = [n for n in bridges if n not in seeds]
        meta = {}
        if to_fetch and hasattr(oracle, 'works'):
            meta_fetched = oracle.works(to_fetch)
            if meta_fetched:
                meta = meta_fetched

        # build records
        records = []
        for node, d in bridges.items():
            if node in seeds:
                continue
            work = meta.get(node, {})
            rec = score.record(
                node,
                {"a": d["a"], "b": d["b"]},
                2,
                d["a"] + d["b"],
                work,
            )
            records.append(rec)

        # sort and trim
        records = score.topk(
            records,
            lambda r: (r["score"], -(r.get("cited_by_count") or 0), r["id"]),
            top_k,
        )

        log.info("find_bridges_bidir: done, %d bridge(s)", len(records))
        return records

    finally:
        oracle.close()


def find_bridges_bidir_nway(
    oracle,
    groups,
    *,
    min_groups: Optional[int] = None,
    mode: str = "budget",
    max_depth: int = 6,
    frontier_cap: int = 50,
    top_k: int = 25,
    n_jobs: int = 8,
    verbose: bool = False,
) -> list[dict]:
    """N-way via C(N,2) pairwise decomposition.

    Proc:
      1. validate groups non-empty, all groups non-empty
      2. N = len(groups)
      3. if N == 2: early delegation to find_bridges_bidir
      4. default min_groups to N
      5. canonicalize each group
      6. build one engine per pair: {(i,j): bidir_engine(...)}
      7. drive ensemble via runtime.run_ensemble
      8. aggregate: per_group[node][gi]=min_dist, recurrence[node]=pair_count
      9. filter candidates: non-seed, meets min_groups
      10. backfill metadata
      11. build records with aggregated distances
      12. sort and trim
      13. finally: oracle.close()
    """
    if not groups:
        raise ValueError("groups must be non-empty")
    for g in groups:
        if not g:
            raise ValueError("group must be non-empty")

    N = len(groups)
    if N == 2:
        # early delegation avoids duplication
        return find_bridges_bidir(
            oracle,
            groups[0],
            groups[1],
            mode=mode,
            max_depth=max_depth,
            frontier_cap=frontier_cap,
            top_k=top_k,
            n_jobs=n_jobs,
            verbose=verbose,
        )

    if min_groups is None:
        min_groups = N

    # canonicalize
    canon = [{parse.key(s) for s in g} for g in groups]

    try:
        # build engines for each pair
        pairs = list(itertools.combinations(range(N), 2))
        engines = {
            (i, j): bidir_engine(
                canon[i], canon[j], mode=mode, max_depth=max_depth, frontier_cap=frontier_cap
            )
            for i, j in pairs
        }

        # drive ensemble
        results = runtime.run_ensemble(engines, oracle, n_jobs=n_jobs, verbose=verbose)

        # aggregate distances and recurrence
        per_group = {}  # node_id -> {group_idx: min_dist}
        recurrence = {}  # node_id -> pair_count

        for (i, j), result in results.items():
            bridges = result.bridges
            for node, d in bridges.items():
                if node not in per_group:
                    per_group[node] = {}
                per_group[node][i] = min(per_group[node].get(i, float("inf")), d["a"])
                per_group[node][j] = min(per_group[node].get(j, float("inf")), d["b"])
                recurrence[node] = recurrence.get(node, 0) + 1

        # filter candidates
        seeds = set().union(*canon)
        candidates = score.keep(per_group.keys(), per_group, seeds, min_groups)

        # backfill metadata from oracle (result.meta carries engine diagnostics, not node data)
        meta = {}
        if candidates and hasattr(oracle, "works"):
            fetched = oracle.works(list(candidates))
            if fetched:
                meta = fetched

        # build records
        records = []
        for cand in candidates:
            pg = per_group[cand]
            work = meta.get(cand, {})
            rec = score.record(
                cand,
                dict(pg),
                len(pg),
                sum(pg.values()),
                work,
                recurrence=recurrence[cand],
            )
            records.append(rec)

        # sort and trim
        records = score.topk(
            records,
            lambda r: (-r["groups_hit"], r["score"], -(r.get("cited_by_count") or 0), r["id"]),
            top_k,
        )

        log.info("find_bridges_bidir_nway: done, %d bridge(s)", len(records))
        return records

    finally:
        oracle.close()


def find_bridges_walk(
    oracle,
    groups: list[list[str]],
    *,
    depth: int = 2,
    min_groups: Optional[int] = None,
    top_k: int = 25,
    ensemble: bool = False,
    M: int = 5,
    p: float = 0.15,
    T: float = 0.1,
    cap: int = 60,
    alpha: float = 0.7,
    seed: Optional[int] = None,
    embedder=None,
    n_jobs: int = 8,
    verbose: bool = False,
) -> list[dict]:
    """Pairwise bidir + embedding-guided frontier prune + fusion score.

    Proc:
      1. validate inputs (groups, N, depth, top_k, p, T, M, alpha, cap, min_groups)
      2. default min_groups to N
      3. canonicalize groups
      4. load embedder if None via models.make_embedder
      5. precompute seed embeddings via embed_titles (stacked per group)
      6. build RNGs: SeedSequence.spawn(M) if seed else unseeded
      7. build walk_engine per (run, i, j)
      8. drive ensemble via runtime.run_ensemble
      9. aggregate: per_run -> min_dist[node][gi], recurrence[node]
      10. filter candidates: non-seed, meets min_groups
      11. backfill metadata and embeddings
      12. score: fusion of struct_norm and embedding similarity
      13. sort and trim
      14. finally: oracle.close()
    """
    # validate
    if not groups:
        raise ValueError("groups must be non-empty")
    for g in groups:
        if not g:
            raise ValueError("group must be non-empty")
    N = len(groups)
    if N < 2:
        raise ValueError("groups must have at least 2 groups")
    if depth < 1:
        raise ValueError("depth must be >= 1")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if not (0 < p < 1):
        raise ValueError("p must be in (0, 1); 0 < p < 1")
    if T <= 0:
        raise ValueError("T must be > 0")
    if M < 1:
        raise ValueError("M must be >= 1")
    if not (0 <= alpha <= 1):
        raise ValueError("alpha must be in [0, 1]; 0 <= alpha <= 1")
    if cap < 1:
        raise ValueError("cap must be > 0")

    if min_groups is None:
        min_groups = N
    if min_groups < 1:
        raise ValueError("min_groups must be >= 1")

    # canonicalize
    canon = [{parse.key(s) for s in g} for g in groups]

    # load embedder
    if embedder is None:
        embedder = models.make_embedder()

    # ensemble config
    runs = M if ensemble else 1
    deterministic = not ensemble

    try:
        # precompute seed embeddings (stacked arrays per group)
        seed_emb = _precompute_seed_emb(oracle, canon, embedder)

        # build RNGs
        if seed is not None:
            ss = SeedSequence(seed)
            seeds_list = ss.spawn(runs)
            rng_list = [np.random.default_rng(s) for s in seeds_list]
        else:
            rng_list = [np.random.default_rng() for _ in range(runs)]

        # build engines per (run, i, j)
        engines = {}
        pairs = list(itertools.combinations(range(N), 2))
        for run_idx in range(runs):
            for i, j in pairs:
                engines[(run_idx, i, j)] = walk_engine(
                    canon[i],
                    canon[j],
                    partner_seed_embs={0: seed_emb[i], 1: seed_emb[j]},
                    rng=rng_list[run_idx],
                    deterministic=deterministic,
                    cap=cap,
                    p=p,
                    T=T,
                )

        # drive ensemble
        results = runtime.run_ensemble(
            engines, oracle, embedder=embedder, n_jobs=n_jobs, verbose=verbose
        )

        # aggregate per-run distances -> min_dist
        per_run = {}
        for run_idx in range(runs):
            per_run[run_idx] = {gi: {} for gi in range(N)}

        for (run_idx, i, j), result in results.items():
            bridges = result.bridges
            for node, d in bridges.items():
                per_run[run_idx][i][node] = min(per_run[run_idx][i].get(node, float("inf")), d["a"])
                per_run[run_idx][j][node] = min(per_run[run_idx][j].get(node, float("inf")), d["b"])

        # min_dist across all runs
        min_dist = {}
        for run_idx in range(runs):
            for gi in range(N):
                for node, dist in per_run[run_idx][gi].items():
                    if node not in min_dist:
                        min_dist[node] = {}
                    min_dist[node][gi] = min(min_dist[node].get(gi, float("inf")), dist)

        # recurrence: runs where node reached >= min_groups groups
        recur = {}
        for run_idx in range(runs):
            seen = set()
            for gi in range(N):
                seen.update(per_run[run_idx][gi].keys())
            for node in seen:
                hit = sum(1 for gi in range(N) if node in per_run[run_idx][gi])
                if hit >= min_groups:
                    recur[node] = recur.get(node, 0) + 1

        # filter candidates
        all_seeds = set().union(*canon)
        candidates = score.keep(min_dist.keys(), min_dist, all_seeds, min_groups)

        # backfill metadata and embeddings
        meta = oracle.works(list(candidates)) if candidates else {}

        # embed candidates
        cand_titles = {c: meta.get(c, {}).get("title", "") for c in candidates}
        cand_emb = models.embed_titles(embedder.model, embedder.cache, cand_titles)

        # score: fusion of struct_norm and embedding similarity
        records = []
        for cand in candidates:
            pg = min_dist[cand]
            groups_hit = len(pg)
            struct = sum(pg.values())
            struct_norm = struct / (groups_hit * depth) if groups_hit > 0 else float("inf")

            # embedding similarity: min over reached groups
            embsim_vals = []
            if cand in cand_emb:
                for g in pg:
                    seed_vecs = seed_emb[g]
                    if isinstance(seed_vecs, dict) and seed_vecs:
                        target = np.vstack(list(seed_vecs.values()))
                    elif isinstance(seed_vecs, np.ndarray) and seed_vecs.shape[0] > 0:
                        target = seed_vecs
                    else:
                        target = np.empty((0, cand_emb[cand].shape[0] if cand_emb[cand].ndim == 1 else cand_emb[cand].shape[1]))
                    embsim_vals.append(sim_to_set(cand_emb[cand], target, agg="max"))

            if embsim_vals:
                embsim = min(embsim_vals)
                final_score = alpha * struct_norm + (1 - alpha) * (1 - embsim)
            else:
                final_score = struct_norm

            work = meta.get(cand, {})
            rec = score.record(
                cand,
                dict(pg),
                groups_hit,
                final_score,
                work,
                recurrence=recur.get(cand, 0),
            )
            records.append(rec)

        # sort and trim
        records = score.topk(
            records,
            lambda r: (-r["groups_hit"], r["score"], -r.get("recurrence", 0), r["id"]),
            top_k,
        )

        log.info("find_bridges_walk: done, %d bridge(s)", len(records))
        return records

    finally:
        oracle.close()


def find_bridges_refine(
    oracle,
    groups: list[list[str]],
    *,
    depth: int = 2,
    iters: int = 3,
    top_k: int = 25,
    min_groups: Optional[int] = None,
    cap: int = 60,
    ensemble: bool = False,
    M: int = 5,
    p: float = 0.15,
    T: float = 0.1,
    seed: Optional[int] = None,
    embedder=None,
    n_jobs: int = 8,
    verbose: bool = False,
) -> list[dict]:
    """Multi-round nested-generator refinement via refine_engine.

    Proc:
      1. validate inputs (groups, N >= 3, depth, iters, top_k, cap, p, T, M)
      2. default min_groups to N
      3. canonicalize groups
      4. load embedder if None
      5. precompute seed_embs as {gi: {wid: vec}} for refine_engine
      6. build rng from seed or system entropy
      7. construct refine_engine and drive via runtime.run
      8. shape SearchResult bridges into records via core.score
      9. finally: oracle.close()
    """
    # validate
    if not groups:
        raise ValueError("groups must be non-empty")
    for g in groups:
        if not g:
            raise ValueError("group must be non-empty")
    N = len(groups)
    if N < 3:
        raise ValueError("groups must have at least 3 groups")
    if depth < 1:
        raise ValueError("depth must be >= 1")
    if iters < 1:
        raise ValueError("iters must be >= 1")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if cap < 1:
        raise ValueError("cap must be > 0")
    if not (0 < p < 1):
        raise ValueError("p must be in (0, 1); 0 < p < 1")
    if T <= 0:
        raise ValueError("T must be > 0")
    if M < 1:
        raise ValueError("M must be >= 1")

    if min_groups is None:
        min_groups = N

    # canonicalize
    canon = [{parse.key(s) for s in g} for g in groups]

    # load embedder
    if embedder is None:
        embedder = models.make_embedder()

    try:
        # precompute {gi: {wid: vec}} for refine_engine
        seed_embs = _seed_embs_dict(oracle, canon, embedder)

        # build rng
        if seed is not None:
            rng = np.random.default_rng(np.random.SeedSequence(seed))
        else:
            rng = np.random.default_rng()

        # construct and drive engine
        engine = refine_engine(
            canon,
            depth=depth,
            iters=iters,
            min_groups=min_groups,
            ensemble=ensemble,
            M=M,
            p=p,
            T=T,
            cap=cap,
            rng=rng,
            seed_embs=seed_embs,
        )
        result = runtime.run(engine, oracle, embedder=embedder, n_jobs=n_jobs, verbose=verbose)

        # shape result: bridges = {wid: {orig_group_idx: dist}}
        all_seeds = set().union(*canon)
        D = result.bridges
        cands = [w for w in D if w not in all_seeds and len(D[w]) >= min_groups]
        records = _emit(cands, D, oracle, embedder, seed_embs, canon, top_k, depth, min_groups)

        log.info("find_bridges_refine: done, %d bridge(s)", len(records))
        return records

    finally:
        oracle.close()


def _emit(cands, D, oracle, embedder, orig_seed_emb, canon, top_k, depth, min_groups):
    """score and rank candidates for return."""
    if not cands:
        return []

    # backfill metadata
    meta = {}
    if hasattr(oracle, 'works'):
        meta_fetched = oracle.works(cands)
        if meta_fetched:
            meta = meta_fetched

    # embed candidates
    cand_titles = {c: meta.get(c, {}).get("title", "") for c in cands}
    cand_emb = models.embed_titles(embedder.model, embedder.cache, cand_titles)

    # score
    records = []
    for cand in cands:
        pg = D.get(cand, {})
        groups_hit = len(pg)
        struct = sum(pg.values())
        struct_norm = struct / (groups_hit * depth) if groups_hit > 0 else float("inf")

        # fusion score with default alpha=0.7
        embsim_vals = []
        if cand in cand_emb:
            for g in pg:
                seed_vecs = orig_seed_emb[g]
                if isinstance(seed_vecs, dict) and seed_vecs:
                    target = np.vstack(list(seed_vecs.values()))
                elif isinstance(seed_vecs, np.ndarray) and seed_vecs.shape[0] > 0:
                    target = seed_vecs
                else:
                    continue
                embsim_vals.append(sim_to_set(cand_emb[cand], target, agg="max"))

        if embsim_vals:
            embsim = min(embsim_vals)
            final_score = 0.7 * struct_norm + 0.3 * (1 - embsim)
        else:
            final_score = struct_norm

        work = meta.get(cand, {})
        rec = score.record(
            cand,
            dict(pg),
            groups_hit,
            final_score,
            work,
            recurrence=0,
        )
        records.append(rec)

    # sort and trim
    records = score.topk(
        records,
        lambda r: (-r["groups_hit"], r["score"], -r.get("recurrence", 0), r["id"]),
        top_k,
    )
    return records


def _precompute_seed_emb(oracle, canon, embedder):
    """precompute seed embeddings stacked per group as (k, dim) arrays.

    canon: list[set[str]] of canonical seeds per group.
    embedder: io.models.Embedder instance with .model and .cache.
    returns: {group_idx: (k, dim) stacked numpy array}.
    """
    result = {}
    for gi, seeds in enumerate(canon):
        seed_list = sorted(seeds)
        if not seed_list:
            dim = embedder.model.get_embedding_dimension()
            result[gi] = np.empty((0, dim), dtype=np.float32)
            continue

        # fetch titles
        seed_titles = {}
        if hasattr(oracle, 'works'):
            works_fetched = oracle.works(seed_list)
            if works_fetched:
                seed_titles = {s: works_fetched.get(s, {}).get("title", "") for s in seed_list}

        # embed
        vecs = models.embed_titles(embedder.model, embedder.cache, seed_titles)
        if vecs:
            # stack into (k, dim) array
            stacked = np.vstack([vecs.get(s, np.zeros(embedder.model.get_embedding_dimension())) for s in seed_list])
            result[gi] = stacked
        else:
            dim = embedder.model.get_embedding_dimension()
            result[gi] = np.empty((0, dim), dtype=np.float32)

    return result


def _precompute_seed_emb_from_sets(oracle, current, embedder):
    """precompute seed embeddings from list of sets (for current groups in refine)."""
    result = {}
    for gi, seeds_set in enumerate(current):
        seed_list = sorted(seeds_set)
        if not seed_list:
            dim = embedder.model.get_embedding_dimension()
            result[gi] = np.empty((0, dim), dtype=np.float32)
            continue

        # fetch titles
        seed_titles = {}
        if hasattr(oracle, 'works'):
            works_fetched = oracle.works(seed_list)
            if works_fetched:
                seed_titles = {s: works_fetched.get(s, {}).get("title", "") for s in seed_list}

        # embed
        vecs = models.embed_titles(embedder.model, embedder.cache, seed_titles)
        if vecs:
            stacked = np.vstack([vecs.get(s, np.zeros(embedder.model.get_embedding_dimension())) for s in seed_list])
            result[gi] = stacked
        else:
            dim = embedder.model.get_embedding_dimension()
            result[gi] = np.empty((0, dim), dtype=np.float32)

    return result


def _seed_embs_dict(oracle, canon, embedder):
    """precompute seed embeddings as {gi: {wid: vec}} for refine_engine.

    canon: list[set[str]] of canonical seeds per group.
    embedder: io.models.Embedder instance.
    returns: {group_idx: {wid: (dim,) vector}}.
    """
    result = {}
    for gi, seeds in enumerate(canon):
        seed_list = sorted(seeds)

        # fetch titles
        seed_titles = {}
        if hasattr(oracle, 'works') and seed_list:
            works_fetched = oracle.works(seed_list)
            if works_fetched:
                seed_titles = {s: works_fetched.get(s, {}).get("title", "") for s in seed_list}

        # embed and store per-wid
        vecs = models.embed_titles(embedder.model, embedder.cache, seed_titles)
        result[gi] = {wid: vecs[wid] for wid in seed_list if wid in vecs}

    return result
