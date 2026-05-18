"""Semantic-ensemble bridge search: reuses BiDirEngine, adds smart embedding prune and fusion scoring."""

from __future__ import annotations

import itertools
import logging

import numpy as np
from joblib import Parallel, delayed

from .bidir import BiDirEngine, StepIn
from ..parallel import fetch_all
from ..embed import TitleEmbedder, sim_to_set

log = logging.getLogger("citetools.strategies")

__all__ = ["find_bridges_walk"]


def find_bridges_walk(
    oracle,
    groups: list[list[str]],
    *,
    depth: int = 2,
    min_groups: int | None = None,
    top_k: int = 25,
    ensemble: bool = False,
    M: int = 5,
    p: float = 0.15,
    T: float = 0.1,
    cap: int = 60,
    alpha: float = 0.7,
    seed: int | None = None,
    embedder: TitleEmbedder | None = None,
    n_jobs: int = 8,
    verbose: bool = False,
) -> list[dict]:
	"""semantic-ensemble bridge finder via reused BiDirEngine + embedding prune + fusion scoring.

	finds papers bridging multiple seed groups by running bidirectional search (budget mode)
	with embedding-guided pruning and optional ensemble. the engine is BiDirEngine
	(pairwise C(N,2) decomposition, same as find_bridges_bidir_nway); the driver adds:
	(a) smart embedding prune (deterministic or stochastic) that runs per-round before
	    fetch_all; (b) optional ensemble (M runs, aggregated by min-over-runs distance);
	(c) fusion scorer merging structural distance and title-embedding similarity.

	semantics:
	  - layer (a) discovery: per-group shortest-path distance d_g(c), estimated by
	    min over ensemble runs; discovered but not expanded (pruned) are recorded at
	    their hop distance and never re-fetched.
	  - layer (b) scoring: fusion of struct_norm (distance/depth) and embedding similarity;
	    candidates must reach >= min_groups groups.

	Args:
	  oracle: CiteGraph with oracle.client (works, key, close) and oracle.grow (unused).
	  groups: list of lists of paper ids (W-ids, DOIs, OpenAlex URLs).
	          Each group must be non-empty; len(groups) >= 2.
	  depth: max hop depth per search. Default 2. Must be >= 1.
	  min_groups: min groups a candidate must bridge. Defaults to len(groups).
	  top_k: number of bridges to return. Default 25. Must be >= 1.
	  ensemble: if False (default), one deterministic run; if True, M runs.
	  M: ensemble size. Ignored if ensemble=False. Default 5. Must be >= 1.
	  p: floor probability for stochastic prune. Default 0.15. Must be 0 < p < 1.
	  T: temperature for sigmoid (prune median-centred). Default 0.1. Must be > 0.
	  cap: frontier cap per engine per round. Default 60. Must be > 0.
	  alpha: fusion weight; alpha*struct_norm + (1-alpha)*(1-embsim). Default 0.7.
	         Range [0,1]; 1.0=pure-structural, 0.0=pure-semantic.
	  seed: numpy random seed for ensemble; None = system-seeded.
	  embedder: TitleEmbedder instance. If None, builds a new one (lazy model load).
	  n_jobs: total I/O thread budget. With an M-run ensemble it is split
	          max(1, n_jobs // M) per run; each run also gets one driver thread.
	          Default 8.
	  verbose: if True, joblib prints per-fetch progress to stderr. Default False.

	Returns:
	  list of dicts (sorted by (-groups_hit, score, -recurrence, -(cited_by_count or 0), id))
	  trimmed to top_k. Each dict:
	    id: paper W-id.
	    score: alpha*struct_norm + (1-alpha)*(1-embsim), lower=better.
	    groups_hit: number of groups this candidate bridges (>= min_groups).
	    dists: {group_idx: min_dist_over_runs} for each hit group.
	    recurrence: number of ensemble runs in which this candidate reached >= min_groups groups.
	    title: paper title or None.
	    year: publication year or None.
	    doi: paper DOI or None.
	    cited_by_count: citation count or None.

	Raises:
	  ValueError: if groups is empty, any group is empty, N < 2, depth < 1, top_k < 1,
	              not (0 < p < 1), T <= 0, M < 1, not (0 <= alpha <= 1), or min_groups < 1.
	"""
	# validate inputs
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
	if min_groups is None:
		min_groups = N
	if min_groups < 1:
		raise ValueError("min_groups must be >= 1")
	if cap < 1:
		raise ValueError("cap must be > 0")

	# canonicalize seed ids
	canon = [{oracle.client.key(s) for s in g} for g in groups]

	# build embedder if needed
	if embedder is None:
		embedder = TitleEmbedder()

	# ensemble config
	runs = 1 if not ensemble else M
	deterministic = not ensemble
	log.info(
		"walk: %d group(s), %d pairs, ensemble=%s (M=%d), depth=%d",
		N, len(list(itertools.combinations(range(N), 2))), ensemble, runs, depth
	)

	try:
		# precompute seed-title embeddings, indexed by group.
		# work() resolves any id form (w-id, doi, arxiv) -- works() skips
		# doi: keys, so per-seed work() calls are used (seeds are few).
		seed_wids = sorted(set().union(*canon))
		seed_titles = {
			w: (oracle.client.work(w).get("title") or "") for w in seed_wids
		}
		seed_vecs = embedder.embed(seed_titles)
		seed_emb = {gi: {w: seed_vecs[w] for w in g} for gi, g in enumerate(canon)}

		# spawn runs with seeded RNGs
		if seed is not None:
			ss = np.random.SeedSequence(seed)
			seeds = ss.spawn(runs)
			rngs = [np.random.default_rng(s) for s in seeds]
		else:
			rngs = [np.random.default_rng() for _ in range(runs)]

		# build engines: one per (run_idx, i, j) pair
		engines = {}
		pairs = list(itertools.combinations(range(N), 2))
		for run_idx in range(runs):
			for i, j in pairs:
				tag = (run_idx, i, j)
				engines[tag] = (
					BiDirEngine(canon[i], canon[j], mode="budget", max_depth=depth, frontier_cap=cap),
					rngs[run_idx],
					seed_emb,
				)

		# drive the ensemble: one thread per run, all sharing the oracle, its
		# client cache + rate-limiter, and the embedder. the i/o thread budget
		# is split across runs so the total stays near n_jobs.
		runs_engines = {r: {} for r in range(runs)}
		for tag, val in engines.items():
			runs_engines[tag[0]][tag] = val
		inner_jobs = max(1, n_jobs // runs)
		log.info("walk: driving %d run(s), %d i/o thread(s)/run", runs, inner_jobs)
		with Parallel(n_jobs=runs, backend="threading") as outer:
			outer(
				delayed(_drive_run)(
					oracle, runs_engines[r], deterministic, cap, p, T,
					embedder, inner_jobs, verbose, r
				)
				for r in range(runs)
			)

		# aggregate: min_dist[node][group_idx], recur[node]
		per_run = {}  # {run_idx: {group_idx: {node: dist}}}
		for run_idx in range(runs):
			per_run[run_idx] = {gi: {} for gi in range(N)}

		for (run_idx, i, j), (engine, _, _) in engines.items():
			for node, d in engine._meeting.items():
				per_run[run_idx][i][node] = min(per_run[run_idx][i].get(node, float("inf")), d["a"])
				per_run[run_idx][j][node] = min(per_run[run_idx][j].get(node, float("inf")), d["b"])

		# min_dist: {node: {group_idx: min dist over all runs}}
		min_dist = {}
		for run_idx in range(runs):
			for gi, by_node in per_run[run_idx].items():
				for node, dist in by_node.items():
					if node not in min_dist:
						min_dist[node] = {}
					min_dist[node][gi] = min(min_dist[node].get(gi, float("inf")), dist)

		# recurrence: {node: count of runs where node reached >= min_groups groups}
		recur = {}
		for run_idx in range(runs):
			seen = set()
			for gi, by_node in per_run[run_idx].items():
				for node in by_node.keys():
					seen.add(node)
			for node in seen:
				hit = sum(1 for gi in range(N) if node in per_run[run_idx][gi])
				if hit >= min_groups:
					recur[node] = recur.get(node, 0) + 1

		# filter candidates: non-seed, meets min_groups threshold
		all_seeds = set().union(*canon)
		candidates = [
			c for c in min_dist
			if c not in all_seeds and len(min_dist[c]) >= min_groups
		]

		# backfill metadata and embeddings
		meta = oracle.client.works(candidates) if candidates else {}
		cand_titles = {c: meta.get(c, {}).get("title", "") for c in candidates}
		cand_emb = embedder.embed(cand_titles)

		# score
		records = _score(
			candidates, min_dist, cand_emb, seed_emb, canon, recur, meta,
			alpha, depth, min_groups
		)

		# sort and trim
		records.sort(
			key=lambda r: (
				-r["groups_hit"],
				r["score"],
				-r["recurrence"],
				-(r.get("cited_by_count") or 0),
				r["id"]
			)
		)
		log.info("walk: done, %d bridge(s)", len(records))
		return records[:top_k]

	finally:
		oracle.client.close()


def _drive_run(
    oracle,
    engines: dict,
    deterministic: bool,
    cap: int,
    p: float,
    T: float,
    embedder: TitleEmbedder,
    n_jobs: int,
    verbose: bool,
    run_idx: int,
) -> None:
	"""round-robin drive one ensemble run's engines in lockstep with embedding prune.

	mirrors find_bridges_bidir_nway's loop, inserting the embedding-guided prune
	between frontier emission and fetch_all. the prune operates per-engine,
	pairwise: engine (i,j) expanding side "a" keeps frontier nodes close
	(in embedding space) to group j's seed embeddings; side "b" -> toward i.
	nodes pruned out are simply not fetched -> engine sees them as dead-ends
	(recorded in its _dist at their hop, never re-expanded). this is the
	layer-(a) "discovered but not expanded" semantics.

	stochastic prune (deterministic=False): for each node, Bernoulli with
	  kappa(n) = p + (1-p)*sigmoid((h_a(n) - h_med) / T),
	where h_a(n) = max-similarity to partner group's seed embeddings. if
	|kept| > cap after Bernoulli, keep top-cap by h_a.

	deterministic (deterministic=True): keep top-cap by h_a outright.

	Args:
	  oracle: CiteGraph with oracle.client.works.
	  engines: {(run_idx, i, j): (BiDirEngine, rng, seed_emb)} for ONE ensemble run.
	  deterministic: if True, _keep uses deterministic top-cap; if False,
	    stochastic Bernoulli prune. the caller sets this to (not ensemble).
	  cap: frontier cap per engine per round.
	  p: floor probability for Bernoulli.
	  T: temperature for sigmoid.
	  embedder: TitleEmbedder for embedding frontier nodes.
	  n_jobs: joblib thread count for this run's fetch_all pool.
	  verbose: if True, joblib prints per-fetch progress.
	  run_idx: ensemble run index, used to prefix log lines.
	"""
	# prime: advance each engine with empty nbrs
	live = {}  # {tag: (engine, rng, seed_emb, step)}
	for tag, (engine, rng, seed_emb) in engines.items():
		step = engine.advance(StepIn(nbrs={}))
		if not step.done:
			live[tag] = (engine, rng, seed_emb, step)

	# round-robin with prune
	with Parallel(n_jobs=n_jobs, backend="threading", verbose=10 if verbose else 0) as parallel:
		hop = 0
		while live:
			hop += 1

			# phase 1: union frontiers, fetch titles, embed
			raw = set().union(*(step.frontier for _, _, _, step in live.values()))
			if raw:
				works = oracle.client.works(list(raw))
				titles = {wid: works.get(wid, {}).get("title", "") for wid in raw}
				node_embs = embedder.embed(titles)
			else:
				node_embs = {}
			log.info("  run %d hop %d: %d live engine(s), %d frontier node(s), %d title(s) fetched",
					 run_idx, hop, len(live), len(raw), len(works) if raw else 0)

			# phase 2: prune per engine
			kept = {}  # {tag: set of kept frontier nodes}
			for tag, (engine, rng, seed_emb, step) in live.items():
				run_idx, i, j = tag
				side = engine._expanding
				partner = j if side == "a" else i
				target_embs_dict = seed_emb[partner]
				if target_embs_dict:
					target_embs = np.vstack(list(target_embs_dict.values()))
				else:
					target_embs = np.empty((0, embedder.dim))

				kept[tag] = _keep(
					step.frontier, node_embs, target_embs, deterministic, rng, p, T, cap
				)

			# phase 3: batch fetch union of kept frontiers
			union_kept = sorted(set().union(*kept.values()))
			if union_kept:
				nbrs = fetch_all(oracle, union_kept, parallel, progress=verbose)
				log.info("  run %d hop %d: fetching %d kept frontier node(s)", run_idx, hop, len(union_kept))
			else:
				nbrs = {}

			# phase 4: advance engines
			next_live = {}
			for tag, (engine, rng, seed_emb, step) in live.items():
				sub = {n: nbrs[n] for n in kept[tag] if n in nbrs}
				nxt = engine.advance(StepIn(nbrs=sub))
				if not nxt.done:
					next_live[tag] = (engine, rng, seed_emb, nxt)

			live = next_live
			log.info("  run %d hop %d: %d meeting node(s) so far",
					 run_idx, hop, sum(len(e._meeting) for e, _, _, _ in live.values()) if live else 0)


def _keep(
    frontier: set[str],
    node_embs: dict[str, np.ndarray],
    target_embs: np.ndarray,
    deterministic: bool,
    rng: np.random.Generator,
    p: float,
    T: float,
    cap: int,
) -> set[str]:
	"""prune frontier via embedding similarity to partner group.

	computes h_a(n) = max-similarity of frontier node n to partner group's
	seed embeddings. then:
	  - deterministic=True: return top-cap nodes by h_a.
	  - deterministic=False: for each node, Bernoulli with kappa(n) =
	    p + (1-p)*sigmoid((h_a(n) - h_med) / T). if |kept| > cap, keep
	    top-cap by h_a.

	a node with no embedding gets h_a = 0.0.

	Args:
	  frontier: set of node ids.
	  node_embs: {wid: (dim,) embedding} (may not include all frontier nodes).
	  target_embs: (k, dim) L2-normalized array (may be empty).
	  deterministic: if True, top-cap policy; if False, Bernoulli+cap.
	  rng: numpy.random.Generator for stochastic sampling.
	  p: floor probability (0 < p < 1).
	  T: temperature > 0.
	  cap: frontier cap.

	Returns:
	  set of frontier nodes to keep (size <= cap).
	"""
	frontier_list = sorted(frontier)
	h_a = np.array([
		sim_to_set(node_embs[n], target_embs, agg="max") if n in node_embs else 0.0
		for n in frontier_list
	], dtype=np.float32)

	if deterministic:
		# top-cap deterministic
		if len(frontier_list) <= cap:
			return set(frontier_list)
		# sort by h_a descending, keep top cap
		sorted_by_h = sorted(enumerate(frontier_list), key=lambda x: -h_a[x[0]])
		return set(n for _, n in sorted_by_h[:cap])
	else:
		# stochastic: Bernoulli + cap
		h_med = np.median(h_a)
		denom = np.maximum(T, 1e-8)
		kappa = p + (1 - p) / (1 + np.exp(-(h_a - h_med) / denom))
		keep = rng.binomial(1, kappa)
		kept = {frontier_list[i] for i in range(len(frontier_list)) if keep[i] == 1}
		if len(kept) > cap:
			# trim to top-cap by h_a
			kept_list = sorted(kept)
			h_a_dict = {frontier_list[i]: h_a[i] for i in range(len(frontier_list))}
			sorted_by_h = sorted(kept_list, key=lambda n: -h_a_dict[n])
			kept = set(sorted_by_h[:cap])
		return kept


def _score(
    candidates: list[str],
    min_dist: dict,
    cand_emb: dict,
    seed_emb: dict,
    canon: list[set[str]],
    recur: dict,
    meta: dict,
    alpha: float,
    depth: int,
    min_groups: int,
) -> list[dict]:
	"""score candidates via fusion of structural distance and embedding similarity.

	layer (b) scoring: for each candidate c with >= min_groups entries in min_dist,
	  - struct = sum over hit groups of min_dist[c][group_idx].
	  - struct_norm = struct / (groups_hit * depth).
	  - embsim(c) = min over REACHED groups of max-similarity of cand_emb[c]
	    to seed_emb[group_idx].values(). Clipped to [0,1].
	  - score = alpha * struct_norm + (1 - alpha) * (1 - embsim).

	node with no embedding gets embsim = 0.0 (worst similarity).

	Args:
	  candidates: list of non-seed node ids to score.
	  min_dist: {cand: {group_idx: min_dist}}.
	  cand_emb: {cand: embedding (dim,)}.
	  seed_emb: {group_idx: {wid: embedding (dim,)}}.
	  canon: list of canonical seed sets per group (for hit group extraction).
	  recur: {cand: count}.
	  meta: {cand: {title, year, doi, cited_by_count, ...}}.
	  alpha: fusion weight (0 <= alpha <= 1).
	  depth: max hop depth.
	  min_groups: min groups threshold (already filtered by caller).

	Returns:
	  list of record dicts, unsorted.
	"""
	records = []
	for cand in candidates:
		pg = min_dist[cand]  # {group_idx: min_dist}
		groups_hit = len(pg)
		struct = sum(pg.values())
		struct_norm = struct / (groups_hit * depth) if groups_hit > 0 else float("inf")

		# embedding similarity: min over hit groups
		hit_groups = set(pg.keys())
		embsim_vals = []
		for g in hit_groups:
			if cand in cand_emb:
				seed_embs_g = seed_emb.get(g, {})
				if seed_embs_g:
					target = np.vstack(list(seed_embs_g.values()))
					sim = sim_to_set(cand_emb[cand], target, agg="max")
				else:
					sim = 0.0
			else:
				sim = 0.0
			embsim_vals.append(sim)

		embsim = min(embsim_vals) if embsim_vals else 0.0
		embsim = np.clip(embsim, 0.0, 1.0)

		score = alpha * struct_norm + (1 - alpha) * (1 - embsim)

		w = meta.get(cand, {})
		records.append({
			"id": cand,
			"score": score,
			"groups_hit": groups_hit,
			"dists": dict(pg),
			"recurrence": recur.get(cand, 0),
			"title": w.get("title"),
			"year": w.get("publication_year"),
			"doi": w.get("doi"),
			"cited_by_count": w.get("cited_by_count"),
		})

	return records
