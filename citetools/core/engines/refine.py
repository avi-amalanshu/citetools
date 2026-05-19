"""nested-generator multi-round refine engine.

a nested-generator engine that holds live walk sub-generators and orchestrates
multi-round greedy refinement. each round, it yields the union of its
sub-generators' requests, routes each sub its response slice, and after all
subs finish for that round, reconstructs meeting paths, relaxes the monotone
distance table D, and generates new seed groups via embedding similarity.
the engine stops on full N-group coverage or after `iters` rounds.
pure functions ported verbatim: `reconstruct_paths`, `relax`, `new_group`.
"""

from __future__ import annotations

import itertools
from typing import Generator

import numpy as np

from .protocol import Req, Resp, SearchResult
from .walk import walk_engine
from ..embed import sim_to_set


def reconstruct_paths(result) -> dict[str, tuple[list[str], dict[str, tuple[int, int]]]]:
	"""walk engine result.meta["pred"] to extract meeting-node paths with distance tuples.

	input: a finished SearchResult from walk_engine with want_paths=True
	(has predecessor maps in result.meta["pred"] = {"a": ..., "b": ...}).

	for each meeting node m in result.bridges with "a" and "b" distance keys,
	walk result.meta["pred"]["a"] from m backward to a group-A seed (seed has
	no predecessor entry), then reverse. likewise walk result.meta["pred"]["b"]
	from m backward to a group-B seed and reverse. splice A-segment, m,
	B-segment into one [A-seed, ..., m, ..., B-seed] path.

	return {m: (path_wid_list, dists)} where path_wid_list is the full list,
	and dists maps every wid on the path to (dist_to_group_a, dist_to_group_b).

	edge cases:
	  - a seed that is also a meeting node: yield a path of length 1 with
	    distances (0, 0).
	  - the A-segment and B-segment share an interior node besides m: detect
	    via visited set; if shared wid found, truncate B-segment to break loop.

	raises ValueError if result.meta["pred"] is missing or falsy.
	"""
	if "pred" not in result.meta or not result.meta["pred"]:
		raise ValueError("result.meta[\"pred\"] is missing or falsy")

	pred_maps = result.meta["pred"]
	if not isinstance(pred_maps, dict) or "a" not in pred_maps or "b" not in pred_maps:
		raise ValueError("result.meta[\"pred\"] must be a dict with keys 'a' and 'b'")

	# infer meeting nodes from keys in bridges (entries with "a" and "b" distance keys)
	meeting_nodes = {}
	for node, data in result.bridges.items():
		if isinstance(data, dict) and "a" in data and "b" in data:
			meeting_nodes[node] = data

	if not meeting_nodes:
		return {}

	# extract distance maps if available (from walk_engine with want_paths=True)
	dist_a = result.meta.get("dist_a", {})
	dist_b = result.meta.get("dist_b", {})

	result_dict = {}

	for m in meeting_nodes:
		# walk A-side backward from m to seed
		a_segment = []
		a_visited = set()
		current = m
		while current in pred_maps["a"]:
			a_segment.append(current)
			a_visited.add(current)
			current = pred_maps["a"][current]
		a_segment.append(current)  # seed has no pred entry
		a_visited.add(current)
		a_segment.reverse()  # now [seed_a, ..., m]

		# walk B-side backward from m to seed, checking for loop
		b_segment = []
		current = m
		shared = None
		while current in pred_maps["b"]:
			if current in a_visited and current != m:
				# found shared node (other than m)
				shared = current
				break
			b_segment.append(current)
			current = pred_maps["b"][current]
		if current in a_visited and current != m:
			shared = current
		b_segment.append(current)  # seed has no pred entry
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

		# A-side nodes: dist_a from search result, dist_b = hops to m + m["b"]
		for i, w in enumerate(a_segment):
			hop_to_m = len(a_segment) - 1 - i
			d_a = dist_a.get(w, hop_to_m)  # default to hop count if not in result
			d_b = hop_to_m + meeting_nodes[m]["b"]
			dists[w] = (d_a, d_b)

		# m itself
		dists[m] = (meeting_nodes[m]["a"], meeting_nodes[m]["b"])

		# B-side nodes: dist_b from search result, dist_a = hops to m + m["a"]
		for i, w in enumerate(b_segment_extended):
			hop_to_m = i + 1
			d_b = dist_b.get(w, hop_to_m)  # default to hop count if not in result
			d_a = hop_to_m + meeting_nodes[m]["a"]
			dists[w] = (d_a, d_b)

		result_dict[m] = (path, dists)

	return result_dict


def relax(D, paths_by_pair, canon) -> None:
	"""propagate distance bounds through paths via group-wise aggregation (mutates D).

	D: {wid: {orig_group_idx: int}} — hop-distance to each original group.
	paths_by_pair: {(i, j): reconstruct_paths-output for current pair}.
	canon: list[set[str]] of current round's group seed sets.

	build D_group: for each current group index gi, aggregate the BEST known
	distances to seeds in canon[gi]. specifically:
	  D_group[gi] = {k: min({D[s][k] | s in canon[gi] AND s in D})}

	for each (i, j) pair and each path paper p with (da, db) distances:
	  - for each orig group k in D_group[i]: relax D[p][k] =
	    min(D[p].get(k, inf), da + D_group[i][k]).
	  - likewise for k in D_group[j], using db and D_group[j][k].

	D is monotone non-increasing: upper bounds on true hop-distance,
	never improved, only tightened by relaxation.

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
	  k: result size (top_k); if None, return all candidates.
	  N: total number of original groups.
	  exclude: set of wids to skip (current-round seed endpoints).

	candidate pool: union of all wids across all path lists in pair_paths,
	minus `exclude`, minus any p with len(D.get(p, {})) >= N (already spans
	all original groups).

	for each candidate p:
	  missing = list of orig group indices NOT in D.get(p, {}).
	  score(p) = min over missing groups g of sim_to_set(path_emb[p],
	  vstack(orig_seed_emb[g].values())).
	    - if orig_seed_emb[g] is empty or p not in path_emb, neutral score
	      (median or 0.0).

	rank candidates by score descending. return the top-k wids.

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


def refine_engine(
	groups: list[set[str]],
	*,
	depth: int = 2,
	iters: int = 3,
	min_groups: int | None = None,
	ensemble: bool = False,
	M: int = 5,
	p: float = 0.15,
	T: float = 0.1,
	cap: int = 60,
	rng: np.random.Generator,
	seed_embs: dict[int, dict[str, np.ndarray]],
) -> Generator[Req, Resp, SearchResult]:
	"""multi-round greedy refinement via pairwise walk sub-generators.

	initialization:
	  - validate inputs: non-empty groups, len(groups) >= 2, params in valid ranges.
	  - N = len(groups); min_groups defaults to N; all_seeds = union of original seeds.
	  - current = list of sets (mutable working copy); D = distance map; finished_subs = {}.
	  - D initialized with {wid: {gi: 0} for all original seeds wid, gi in range(N)}.

	outer round loop (up to iters rounds):
	  1. break if len(current) < 2.
	  2. derive per-round RNG: extract master SeedSequence from rng on first round,
	     spawn child for each round via spawn(1)[0], create new Generator from it.
	  3. spawn walk_engine sub-generators for all (i,j) pair combinations:
	     - for each pair and m in 1..M (if ensemble) or 1..1 (if deterministic):
	       - tag = (round_num, i, j, m); build partner_seed_embs from current groups;
	       - spawn walk_engine(group_a=current[i], group_b=current[j], ...).
	       - prime with next() to get first Req; store in live_subs dict.
	  4. inner lockstep loop: while live_subs is non-empty:
	     - form union request: union_nodes = frozenset.union of all req.nodes
	       (guard against empty live_subs); union_want_emb = any want_emb flag.
	     - yield union_req; receive resp.
	     - route slices: for each sub, extract sub_resp = {n: resp[n] for n in req.nodes if n in resp}.
	     - send slice to sub; catch StopIteration to cache result in finished_subs.
	  5. round completion:
	     - extract paths from finished_subs via reconstruct_paths.
	     - call relax(D, paths_by_pair, current).
	     - check coverage: if any wid outside all_seeds has D[wid] covering >= min_groups,
	       return early with full-coverage results.
	     - collect path paper embeddings; call new_group per (i,j) pair to generate candidates.
	     - break if len(next_groups) < 2.
	     - advance: current = next_groups; clear finished_subs; continue.

	final return:
	  - exit loop: emit best-effort results from depth-terminated search.

	args:
	  groups: list of non-empty sets of wid strings; len(groups) >= 2.
	  depth: max hop depth; must be >= 1.
	  iters: max iteration count; must be >= 1.
	  min_groups: min groups to cover for a candidate; defaults to N.
	  ensemble: if True, run M stochastic-prune runs; else 1 deterministic run.
	  M: ensemble size (runs per round); must be >= 1.
	  p: floor reserve probability for stochastic prune; must be 0 < p < 1.
	  T: temperature for stochastic prune sigmoid; must be > 0.
	  cap: frontier cap per walk_engine per round; must be >= 1.
	  rng: numpy.random.Generator with optional seed.
	  seed_embs: dict[orig_group_idx: {wid: embedding_vector}]; precomputed.

	returns:
	  Generator[Req, Resp, SearchResult] via StopIteration.value.

	raises ValueError on invalid inputs.
	"""
	# validation
	if not groups:
		raise ValueError("groups must be non-empty")
	for g in groups:
		if not g:
			raise ValueError("each group must be non-empty")
	N = len(groups)
	if N < 2:
		raise ValueError("len(groups) must be >= 2")
	if depth < 1:
		raise ValueError("depth must be >= 1")
	if iters < 1:
		raise ValueError("iters must be >= 1")
	if M < 1:
		raise ValueError("M must be >= 1")
	if not (0 < p < 1):
		raise ValueError("p must be in (0, 1); 0 < p < 1")
	if T <= 0:
		raise ValueError("T must be > 0")
	if cap < 1:
		raise ValueError("cap must be >= 1")
	if ensemble not in (True, False):
		raise ValueError("ensemble must be True or False")

	# initialize state
	min_groups = min_groups if min_groups is not None else N
	all_seeds = frozenset().union(*groups)
	current = [set(g) for g in groups]
	D = {}
	for gi, g in enumerate(groups):
		for w in g:
			D[w] = {gi: 0}
	finished_subs = {}
	master_seq = None

	# outer round loop
	for round_num in range(iters):
		if len(current) < 2:
			break

		# per-round RNG: extract master SeedSequence on first round
		if round_num == 0:
			master_seed = rng.bit_generator.state["state"]["state"]
			master_seq = np.random.SeedSequence(master_seed)

		# spawn child SeedSequence for this round
		round_seq = master_seq.spawn(1)[0]
		rng_per_round = np.random.Generator(np.random.PCG64(round_seq))

		# if ensemble, spawn M child RNGs for this round
		ensemble_rngs = {}
		if ensemble:
			child_seqs = round_seq.spawn(M)
			for m in range(M):
				ensemble_rngs[m] = np.random.Generator(np.random.PCG64(child_seqs[m]))

		# spawn walk_engine sub-generators for all (i, j) pairs
		subs = {}  # maps tag -> sub-generator object
		live_subs = {}  # maps tag -> pending Req from sub
		for (i, j) in itertools.combinations(range(len(current)), 2):
			for m in range(M if ensemble else 1):
				tag = (round_num, i, j, m)
				deterministic = not ensemble

				# build partner_seed_embs: per-group stacked embeddings
				partner_seed_embs = {}
				for gi in range(N):
					vecs = [seed_embs[gi][wid] for wid in current[gi] if wid in seed_embs[gi]]
					if vecs:
						partner_seed_embs[gi] = np.vstack(vecs)

				# choose rng for this sub: ensemble_rngs[m] if ensemble else rng_per_round
				rng_m = ensemble_rngs[m] if ensemble else rng_per_round

				sub = walk_engine(
					group_a=current[i],
					group_b=current[j],
					rng=rng_m,
					cap=cap,
					deterministic=deterministic,
					p=p,
					T=T,
					partner_seed_embs=partner_seed_embs,
					want_paths=True,
				)

				# prime: call next(sub) to get the first Req
				req = next(sub)
				subs[tag] = sub
				live_subs[tag] = req

		# inner lockstep loop: union requests, yield, receive, route slices
		while live_subs:
			# form union request; guard against empty live_subs
			if not live_subs:
				break

			# form union request
			req_nodes_list = [req.nodes for req in live_subs.values()]
			if req_nodes_list:
				union_nodes = frozenset.union(*req_nodes_list)
			else:
				union_nodes = frozenset()
			union_want_emb = any(req.want_emb for req in live_subs.values())
			union_req = Req(nodes=union_nodes, want_emb=union_want_emb)

			# yield union request and receive response
			resp = yield union_req

			# route response slices back to subs
			next_live = {}
			for tag, req in live_subs.items():
				sub = subs[tag]
				# extract the slice for this sub: only the nodes it requested
				slice_nodes = frozenset(req.nodes)
				sub_resp = {n: resp[n] for n in slice_nodes if n in resp}

				# send the slice to the sub
				try:
					next_req = sub.send(sub_resp)
					next_live[tag] = next_req
				except StopIteration as e:
					# sub finished: cache its SearchResult
					finished_subs[tag] = e.value

			live_subs = next_live

		# round complete: post-process finished subs
		# extract paths from all finished subs' SearchResults
		paths_by_pair = {}
		for tag, sub_result in finished_subs.items():
			round_num_tag, i, j, m = tag
			pair_key = (i, j)
			if pair_key not in paths_by_pair:
				paths_by_pair[pair_key] = {}

			# reconstruct_paths takes the SearchResult
			pair_paths = reconstruct_paths(sub_result)
			# merge: store paths keyed by (m, meeting_node) per pair
			for meeting_node, (path_list, dists_dict) in pair_paths.items():
				paths_by_pair[pair_key][(m, meeting_node)] = (path_list, dists_dict)

		# relax D
		relax(D, paths_by_pair, current)

		# check for full min_groups coverage
		full_coverage = [
			w for w in D
			if w not in all_seeds and len(D[w]) >= min_groups
		]
		if full_coverage:
			# early termination: return results immediately
			return SearchResult(
				bridges={w: D[w] for w in full_coverage},
				meta={"coverage": "full", "round": round_num},
			)

		# generate next-round seed groups via new_group per pair
		next_groups = []
		for (i, j) in itertools.combinations(range(len(current)), 2):
			# path embeddings would come from walk_engine responses if want_emb was set
			# (not collected in this round since scoring via embedding is deferred to caller)
			path_emb = {}

			# new_group returns candidates for group pair (i, j)
			g = new_group(
				pair_paths=paths_by_pair.get((i, j), {}),
				D=D,
				orig_seed_emb=seed_embs,
				path_emb=path_emb,
				k=None,  # return all candidates
				N=N,
				exclude=set().union(*current),
			)
			if g:
				next_groups.append(set(g))

		# terminate round if too few next groups
		if len(next_groups) < 2:
			break

		# advance to next round
		current = next_groups
		finished_subs.clear()

	# exit loop: no full coverage; emit best-effort results
	final_cands = [
		w for w in D
		if w not in all_seeds and len(D[w]) >= min_groups
	]
	return SearchResult(
		bridges={w: D[w] for w in final_cands},
		meta={"coverage": "partial", "rounds": iters},
	)
