"""bounded greedy refinement heuristic over walk: iterate pairwise bidirectional search."""

from __future__ import annotations

import itertools
import logging

import numpy as np

from .walk import _run_pairs, _seed_emb, _score
from ..embed import TitleEmbedder, sim_to_set

log = logging.getLogger("citetools.strategies")

__all__ = ["find_bridges_refine"]


def reconstruct_paths(engine) -> dict[str, tuple[list[str], dict[str, tuple[int, int]]]]:
	"""walk engine._pred to extract meeting-node paths with distance tuples.

	input: a finished BiDirEngine with want_paths=True (has _meeting, _pred["a"],
	_pred["b"] populated and _dist).

	for each meeting node m in engine._meeting, walk engine._pred["a"] from m
	backward to a group-A seed (seed has no _pred entry), then reverse. likewise
	walk engine._pred["b"] from m backward to a group-B seed and reverse. splice
	A-segment, m, B-segment into one [A-seed, ..., m, ..., B-seed] path.

	return {m: (path_wid_list, dists)} where path_wid_list is the full list, and
	dists maps every wid on the path to (dist_to_group_a, dist_to_group_b):
	  - for a wid on the A-side (including m): dist_a = engine._dist["a"][wid],
	    dist_b = (hops from wid to m along the path) + engine._meeting[m]["b"]
	  - for a wid on the B-side (including m): dist_b = engine._dist["b"][wid],
	    dist_a = (hops from wid to m along the path) + engine._meeting[m]["a"]
	  - m itself: dist_a = engine._meeting[m]["a"], dist_b = engine._meeting[m]["b"]

	edge cases:
	  - a seed that is also a meeting node (dist 0/0, no _pred entries): yield a
	    path of length 1, [seed]. dists[seed] = (0, 0).
	  - the A-segment and B-segment share an interior node other than m: a loop
	    would form. detect via a visited set while walking _pred on both sides.
	    if the two segments share a wid besides m, truncate the B-segment to break
	    the loop (keep only the part from the shared wid to m).
	  - engine.want_paths is False: raise ValueError("engine.want_paths is False").

	raises ValueError if engine.want_paths is False.
	"""
	if not engine.want_paths:
		raise ValueError("engine.want_paths is False")

	if not engine._meeting:
		return {}

	result = {}

	for m in engine._meeting:
		# walk A-side backward from m to seed
		a_segment = []
		a_visited = set()
		current = m
		while current in engine._pred["a"]:
			a_segment.append(current)
			a_visited.add(current)
			current = engine._pred["a"][current]
		a_segment.append(current)  # seed has no _pred entry
		a_visited.add(current)
		a_segment.reverse()  # now [seed_a, ..., m]

		# walk B-side backward from m to seed, checking for loop
		b_segment = []
		current = m
		shared = None
		while current in engine._pred["b"]:
			if current in a_visited and current != m:
				# found shared node (other than m)
				shared = current
				break
			b_segment.append(current)
			current = engine._pred["b"][current]
		if current in a_visited and current != m:
			shared = current
		b_segment.append(current)  # seed has no _pred entry
		# now b_segment is [m, ..., seed_b] in backward order
		b_segment.reverse()  # now [seed_b, ..., m] in forward order

		# if shared node found, truncate B-segment to start from shared
		if shared is not None:
			shared_idx = b_segment.index(shared)
			b_segment = b_segment[shared_idx:]

		# splice: A-segment (seed_a->...->m) + B-segment reversed and trimmed
		b_segment_extended = b_segment[::-1][1:]  # reverse to [m, ..., seed_b], skip m
		path = a_segment + b_segment_extended

		# compute distances for each wid on path
		dists = {}

		# A-side nodes
		for i, w in enumerate(a_segment):
			hop_to_m = len(a_segment) - 1 - i
			dist_a = engine._dist["a"][w]
			dist_b = hop_to_m + engine._meeting[m]["b"]
			dists[w] = (dist_a, dist_b)

		# m itself
		dists[m] = (engine._meeting[m]["a"], engine._meeting[m]["b"])

		# B-side nodes
		for i, w in enumerate(b_segment_extended):
			hop_to_m = i + 1
			dist_b = engine._dist["b"][w]
			dist_a = hop_to_m + engine._meeting[m]["a"]
			dists[w] = (dist_a, dist_b)

		result[m] = (path, dists)

	return result


def relax(D, paths_by_pair, canon) -> None:
	"""propagate distance bounds through paths via group-wise aggregation (mutates D).

	D: {wid: {orig_group_idx: int}} — hop-distance to each original group.
	paths_by_pair: {(i, j): reconstruct_paths-output for current pair}.
	canon: list[set[str]] of current round's group seed sets.

	build D_group: for each current group index gi, aggregate the BEST known
	distances to seeds in canon[gi]. specifically:
	  D_group[gi] = {k: min over seeds s in canon[gi] of (D[s][k] if s in D)}
	              = {k: min({D[s][k] | s in canon[gi] AND s in D})}

	for each (i, j) pair and each path paper p with (da, db) distances:
	  - for each orig group k in D_group[i]: add to D[p][k] the minimum of its
	    current value and (da + D_group[i][k]). use inf as default for missing.
	  - likewise for k in D_group[j], using db and D_group[j][k].

	D is monotone non-increasing: documents D as "best hop-count found by
	budget-pruned search — an UPPER BOUND on true hop-distance from p to any
	original group, not a guaranteed minimum. used as a heuristic for coverage
	and candidate ranking."

	raises: none. mutates D in place.
	"""
	inf = float("inf")

	# build D_group: for each current group index gi
	D_group = {}
	for gi in range(len(canon)):
		D_group[gi] = {}
		for s in canon[gi]:
			if s in D:
				for k in D[s]:
					D_group[gi][k] = min(D_group[gi].get(k, inf), D[s][k])

	# relax D via path distances
	for (i, j), pp in paths_by_pair.items():
		for m, (path, dists) in pp.items():
			for p in path:
				da, db = dists[p]
				dp = D.setdefault(p, {})
				# relax via group i
				for k in D_group[i]:
					dp[k] = min(dp.get(k, inf), da + D_group[i][k])
				# relax via group j
				for k in D_group[j]:
					dp[k] = min(dp.get(k, inf), db + D_group[j][k])


def new_group(pair_paths, D, orig_seed_emb, path_emb, k, N, exclude) -> list[str]:
	"""select top-k candidate papers via semantic relevance to under-covered groups.

	input:
	  pair_paths: reconstruct_paths-output for one (i, j) pair.
	  D: {wid: {orig_group_idx: dist}} — coverage and distance bounds.
	  orig_seed_emb: {orig_group_idx: {wid: vec}} for original groups.
	  path_emb: {wid: vec} embeddings of path papers (titleless wids absent).
	  k: result size (top_k).
	  N: total number of original groups.
	  exclude: set of wids to skip (current-round seed endpoints).

	candidate pool: union of all wids across all path lists in pair_paths, minus
	`exclude`, minus any p with len(D.get(p, {})) >= N (already spans all original
	groups; not useful for forming a new group).

	for each candidate p:
	  missing = list of orig group indices NOT in D.get(p, {}).
	  score(p) = min over missing groups g of sim_to_set(path_emb[p], vec_array_g).
	    - vec_array_g = vstack of orig_seed_emb[g].values() (all seed embeddings).
	    - if orig_seed_emb[g] is empty, sim_to_set handles (0, dim) array -> 0.
	    - min aggregation: the worst missing group scores the candidate.
	  - a candidate p not in path_emb (titleless) gets a neutral score = median of
	    all embedded candidates' scores (or 0.0 if none have an embedding).

	rank candidates by score descending. return the top-k wids (or fewer if fewer
	candidates exist).

	raises: none. returns list of wids.
	"""
	# collect all wids from all paths in pair_paths
	union_wids = set()
	for m, (path, dists) in pair_paths.items():
		union_wids.update(path)

	# filter candidates
	candidates = [w for w in union_wids if w not in exclude and len(D.get(w, {})) < N]

	if not candidates:
		return []

	# pass 1: real scores for embedded candidates
	real = {}
	for p in candidates:
		if p in path_emb:
			missing = [gi for gi in range(N) if gi not in D.get(p, {})]
			scores_missing = []
			for gi in missing:
				if orig_seed_emb[gi]:
					target = np.vstack(list(orig_seed_emb[gi].values()))
				else:
					target = np.empty((0, path_emb[p].shape[0]))
				s_gi = sim_to_set(path_emb[p], target, agg="max")
				scores_missing.append(s_gi)
			if scores_missing:
				real[p] = min(scores_missing)

	# pass 2: assign score to every candidate
	neutral = float(np.median(list(real.values()))) if real else 0.0
	scored = {p: real.get(p, neutral) for p in candidates}

	# rank by score descending, tie-break by id for determinism
	ranked = sorted(candidates, key=lambda p: (-scored[p], p))

	return ranked[:k]


def find_bridges_refine(
	oracle,
	groups: list[list[str]],
	*,
	depth: int = 2,
	iters: int = 3,
	top_k: int = 25,
	min_groups: int | None = None,
	cap: int = 60,
	embedder: TitleEmbedder | None = None,
	n_jobs: int = 8,
	verbose: bool = False,
) -> list[dict]:
	"""bounded greedy refinement heuristic: iterate walk over evolving groups.

	refine greedily pools candidates from all pairwise walk searches, assigns each
	a semantic relevance score (worst missing group), and uses top candidates as
	seeds for the next round. no convergence claim: this is a heuristic designed
	to find multi-hop bridges via semantic bootstrapping.

	control flow:
	  1. validate inputs (non-empty groups, N >= 3, depth >= 1, iters >= 1, etc).
	  2. canonicalize seed ids; init D with distances 0 for original seeds.
	  3. embedder setup; precompute orig_seed_emb for all original groups.
	  4. init current = canon (list of sets), owned by the caller for iteration.
	  5. iteration loop (up to iters rounds):
	     - if fewer than 2 current groups remain, break.
	     - run _run_pairs on current groups (want_paths=True, deterministic, 1 run).
	     - extract paths via reconstruct_paths per engine.
	     - relax D with path distances.
	     - check for full N-group coverage; if found, emit and return.
	     - pool all path-paper embeddings; score via new_group for each pair.
	     - if fewer than 2 new candidate groups emerge, break.
	     - else set current = next_groups and continue.
	  6. no full coverage hit: emit best-effort with candidates spanning >= min_groups.
	  7. finally: call oracle.client.close().

	args:
	  oracle: CiteGraph with oracle.client (works, key, close) and oracle.grow.
	  groups: list of lists of paper ids. each non-empty; len(groups) >= 3.
	  depth: max hop depth per search. default 2. must be >= 1.
	  iters: max iteration count. default 3. must be >= 1.
	  top_k: result size (number of bridges to return). default 25. must be >= 1.
	  min_groups: min groups a candidate must bridge. default N (len(groups)).
	  cap: frontier cap per pairwise engine per round. default 60. must be > 0.
	  embedder: TitleEmbedder instance or None (lazy-build one).
	  n_jobs: i/o thread budget. default 8. must be > 0.
	  verbose: if True, joblib prints per-fetch progress.

	returns:
	  list of dicts (sorted by -groups_hit, score, -recurrence, id), trimmed to
	  top_k. each dict has: id, score, groups_hit, dists, recurrence, title, year,
	  doi, cited_by_count (as per _score output).

	raises ValueError if: groups empty, any group empty, N < 3, depth < 1,
	                       iters < 1, top_k < 1, min_groups < 1, cap < 1.
	"""
	# validation
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
	if min_groups is None:
		min_groups = N
	if min_groups < 1:
		raise ValueError("min_groups must be >= 1")

	# setup
	if N >= 4:
		log.warning("refine: %d groups, C(%d,2)=%d pairs; practical mainly for N=3", N, N, len(list(itertools.combinations(range(N), 2))))

	# canonicalize: {oracle.client.key(s) for s in g} for each group
	canon = [{oracle.client.key(s) for s in g} for g in groups]
	all_seeds = set().union(*canon)

	# build embedder if None
	if embedder is None:
		embedder = TitleEmbedder()

	# precompute original-group seed embeddings
	orig_seed_emb = _seed_emb(oracle, canon, embedder)

	# init distance map
	D = {}
	for gi, g in enumerate(canon):
		for w in g:
			D[w] = {gi: 0}

	# current list of sets for iteration
	current = [set(g) for g in canon]

	log.info("refine: %d group(s), %d iter(s), depth=%d", N, iters, depth)

	try:
		for r in range(iters):
			if len(current) < 2:
				break

			# per-round setup
			cur_seed_emb = _seed_emb(oracle, current, embedder)
			engines = _run_pairs(
				oracle, current, cur_seed_emb,
				depth=depth,
				frontier_cap=None,
				want_paths=True,
				runs=1,
				deterministic=True,
				cap=cap,
				p=0.15,
				T=0.1,
				embedder=embedder,
				n_jobs=n_jobs,
				verbose=verbose,
				seed=None
			)

			# extract pairs and paths
			pairs = sorted(set((i, j) for (_, i, j) in engines.keys()))
			paths_by_pair = {(i, j): reconstruct_paths(engines[(0, i, j)]) for (i, j) in pairs}
			log.info("refine: round %d, %d pair(s), %d paths total", r, len(pairs), sum(len(p) for p in paths_by_pair.values()))

			# distance relaxation
			relax(D, paths_by_pair, current)

			# check for full N-group coverage
			full = [w for w in D if w not in all_seeds and len(D[w]) == N]
			if full:
				return _emit(full, D, oracle, embedder, orig_seed_emb, canon, top_k, depth, min_groups)

			# pool and embed path papers
			pool = sorted(set().union(*(paths_by_pair[(i, j)][m][0] for (i, j) in pairs for m in paths_by_pair[(i, j)]))) if pairs else []
			meta = oracle.client.works(pool) if pool else {}
			path_titles = {w: meta.get(w, {}).get("title", "") for w in pool}
			path_emb = embedder.embed(path_titles)
			cur_seeds = set().union(*current)

			# generate next-round candidate groups
			next_groups = []
			for (i, j) in pairs:
				g = new_group(paths_by_pair[(i, j)], D, orig_seed_emb, path_emb, top_k, N, cur_seeds)
				if g:
					next_groups.append(set(g))

			if len(next_groups) < 2:
				break
			current = next_groups
			log.info("refine: round %d, %d new group(s) formed", r, len(next_groups))

		# best-effort emission (loop finished, no full-coverage hit). stays
		# inside the try so the oracle client is still open for _emit's fetch.
		cands = [w for w in D if w not in all_seeds and len(D[w]) >= min_groups]
		records = _emit(cands, D, oracle, embedder, orig_seed_emb, canon, top_k, depth, min_groups)
		log.info("refine: done, no full coverage, %d bridge(s)", len(records))
		return records

	finally:
		oracle.client.close()


def _emit(cands, D, oracle, embedder, orig_seed_emb, canon, top_k, depth, min_groups):
	"""score and rank candidates for return."""
	if not cands:
		return []
	meta = oracle.client.works(cands)
	cand_titles = {c: meta.get(c, {}).get("title", "") for c in cands}
	cand_emb = embedder.embed(cand_titles)
	# min_dist is D (reused); recur is {} (no recurrence tracking in refine)
	records = _score(cands, D, cand_emb, orig_seed_emb, canon, {}, meta, 0.7, depth, min_groups)
	records.sort(key=lambda r: (-r["groups_hit"], r["score"], -r["recurrence"], r["id"]))
	return records[:top_k]
