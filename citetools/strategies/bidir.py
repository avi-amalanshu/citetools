"""Bidirectional meet-in-the-middle bridge search: engine + N=2 and N-way aggregators."""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass

from joblib import Parallel

from ..parallel import fetch_all

log = logging.getLogger("citetools.strategies.bidir")

__all__ = ["find_bridges_bidir", "find_bridges_bidir_nway"]


@dataclass(frozen=True)
class StepIn:
	"""Fetched neighbours of the frontier the engine emitted last step.

	nbrs: {node_id: {"refs": [...], "citers": [...]} | Exception}.
	A missing node or an Exception value is treated as "no neighbours"
	(not re-raised); this keeps a failed fetch deterministic.
	The first advance() call is given StepIn(nbrs={}).
	"""
	nbrs: dict


@dataclass(frozen=True)
class BiDirState:
	"""Immutable snapshot of the search after a step (carried in StepOut).

	meeting: {node_id: {"a": dist_a, "b": dist_b}} for nodes reached from
	         BOTH sides. dist_a / dist_b are hop distances to the nearest
	         group-A / group-B seed.
	depth_a, depth_b: the hop depth currently reached on each side.
	"""
	meeting: dict
	depth_a: int
	depth_b: int


@dataclass(frozen=True)
class StepOut:
	"""Engine output of one advance() step.

	frontier: the node-ids the aggregator must fetch and return next
	          (empty when done).
	state:    a BiDirState snapshot.
	radii:    {"a": depth_a, "b": depth_b} — used by the N-way aggregator.
	done:     True iff the search has terminated.
	"""
	frontier: set
	state: BiDirState
	radii: dict
	done: bool


class BiDirEngine:
	"""Pure-CPU bidirectional meet-in-the-middle search engine."""

	def __init__(
		self,
		group_a: set[str],
		group_b: set[str],
		*,
		mode: str = "budget",
		max_depth: int = 6,
		frontier_cap: int | None = 50,
		want_paths: bool = False,
	) -> None:
		"""Initialize bidirectional search engine.

		Raises ValueError if group_a or group_b is empty.
		Stores mode, max_depth, frontier_cap, want_paths.
		frontier_cap=None disables the internal cap (driver prunes instead).
		Initializes distance maps and frontier for both sides.
		Records any shared seeds (group_a & group_b) as meetings with dist 0/0.
		Proc:
		  1. validate non-empty groups; raise ValueError if any is empty
		  2. store mode, max_depth, frontier_cap, want_paths
		  3. init _dist = {"a": {s: 0 for s in group_a}, "b": {s: 0 for s in group_b}}
		  4. init _frontier = {"a": set(group_a), "b": set(group_b)}
		  5. init _meeting = {}; record shared seeds as {"a": 0, "b": 0}
		  6. init _expanding = None (no frontier emitted yet)
		  7. init _done = False
		  8. if want_paths: init _pred = {"a": {}, "b": {}}; else _pred = None
		"""
		if not group_a or not group_b:
			raise ValueError("group must be non-empty")

		self.mode = mode
		self.max_depth = max_depth
		self.frontier_cap = frontier_cap
		self.want_paths = want_paths

		self._dist = {
			"a": {s: 0 for s in group_a},
			"b": {s: 0 for s in group_b},
		}
		self._frontier = {
			"a": set(group_a),
			"b": set(group_b),
		}
		self._meeting = {}
		for node in group_a & group_b:
			self._meeting[node] = {"a": 0, "b": 0}

		self._expanding = None
		self._done = False

		if want_paths:
			self._pred = {"a": {}, "b": {}}
		else:
			self._pred = None

	def _depth(self, side: str) -> int:
		"""Current hop depth of side: max of dist[side].values(), or 0 if only seeds."""
		if not self._dist[side]:
			return 0
		return max(self._dist[side].values())

	def advance(self, step_in: StepIn) -> StepOut:
		"""Execute one search step: ingest, terminate check, pick next side, emit frontier.

		Proc:
		  1. ingest (skip if _expanding is None — first call):
		     - side = _expanding; other = "b" if side == "a" else "a"
		     - for each node in _frontier[side] emitted last step:
		       - res = step_in.nbrs.get(node)
		       - neighbours = union of refs+citers if res is valid dict; else empty
		       - d = _dist[side][node] + 1
		       - for each m in neighbours not in _dist[side]:
		         - _dist[side][m] = d
		         - if want_paths: _pred[side][m] = node
		         - add m to discovered set
		         - if m in _dist[other] and m not in _meeting:
		           _meeting[m] = {"a": _dist["a"][m], "b": _dist["b"][m]}
		     - budget mode: cap discovered to frontier_cap (sorted by node id)
		     - _frontier[side] = discovered
		  2. termination checks:
		     - mode == "budget" and _meeting non-empty -> _done = True
		     - mode == "exact" and _meeting non-empty and
		       _depth("a") + _depth("b") >= min(d["a"] + d["b"] for d in _meeting.values()) -> _done = True
		     - both _frontier["a"] and _frontier["b"] empty -> _done = True
		     - max(_depth("a"), _depth("b")) >= max_depth -> _done = True
		  3. build state and radii
		  4. if _done: return StepOut(frontier=set(), state, radii, done=True)
		  5. pick next side to expand: smaller frontier; tie-break "a" before "b"
		  6. set _expanding to chosen side
		  7. return StepOut(frontier, state, radii, done=False)
		"""
		# ingest phase (skip on first call when _expanding is None)
		if self._expanding is not None:
			side = self._expanding
			other = "b" if side == "a" else "a"
			discovered = set()

			for node in self._frontier[side]:
				res = step_in.nbrs.get(node)
				if res is None or isinstance(res, Exception):
					neighbours = set()
				else:
					neighbours = set(res.get("refs", [])) | set(res.get("citers", []))

				d = self._dist[side][node] + 1
				for m in neighbours:
					if m not in self._dist[side]:
						self._dist[side][m] = d
						if self.want_paths:
							self._pred[side][m] = node
						discovered.add(m)
						if m in self._dist[other] and m not in self._meeting:
							self._meeting[m] = {
								"a": self._dist["a"][m],
								"b": self._dist["b"][m],
							}

			# budget mode: cap discovered set (None disables cap)
			if self.mode == "budget" and self.frontier_cap is not None and len(discovered) > self.frontier_cap:
				discovered = set(sorted(discovered)[: self.frontier_cap])

			self._frontier[side] = discovered

		# termination checks
		if self.mode == "budget" and self._meeting:
			self._done = True
		elif self.mode == "exact" and self._meeting:
			best_total = min(d["a"] + d["b"] for d in self._meeting.values())
			if self._depth("a") + self._depth("b") >= best_total:
				self._done = True
		elif not self._frontier["a"] and not self._frontier["b"]:
			self._done = True
		elif max(self._depth("a"), self._depth("b")) >= self.max_depth:
			self._done = True

		# build state and radii
		state = BiDirState(
			meeting=dict(self._meeting),
			depth_a=self._depth("a"),
			depth_b=self._depth("b"),
		)
		radii = {"a": self._depth("a"), "b": self._depth("b")}

		# return early if done
		if self._done:
			return StepOut(frontier=set(), state=state, radii=radii, done=True)

		# pick next side to expand: smaller frontier, tie-break "a" before "b"
		frontier_a = len(self._frontier["a"])
		frontier_b = len(self._frontier["b"])

		if frontier_a == 0 and frontier_b == 0:
			# both empty; should have been caught above, but safeguard
			self._done = True
			return StepOut(frontier=set(), state=state, radii=radii, done=True)
		elif frontier_a == 0:
			chosen = "b"
		elif frontier_b == 0:
			chosen = "a"
		elif frontier_a <= frontier_b:
			chosen = "a"
		else:
			chosen = "b"

		self._expanding = chosen
		return StepOut(
			frontier=set(self._frontier[chosen]),
			state=state,
			radii=radii,
			done=False,
		)


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
	"""Bidirectional meet-in-the-middle bridge search for exactly 2 groups.

	Proc:
	  1. raise ValueError if group_a or group_b is empty
	  2. canonicalize: canon_a = {oracle.client.key(s) for s in group_a}
	     canon_b = {oracle.client.key(s) for s in group_b}
	  3. engine = BiDirEngine(canon_a, canon_b, mode=..., max_depth=..., frontier_cap=...)
	  4. drive with Parallel(..., backend="threading", verbose=...):
	     - step = engine.advance(StepIn(nbrs={}))
	     - while not step.done:
	       - nbrs = fetch_all(oracle, sorted(step.frontier), parallel)
	       - step = engine.advance(StepIn(nbrs=nbrs))
	  5. meeting = step.state.meeting; seeds = canon_a | canon_b
	  6. backfill metadata: meta = oracle.client.works([n for n in meeting if n not in seeds])
	  7. build records: id, score (a+b dist), groups_hit=2, dists, title, year, doi, cited_by_count
	  8. filter out seed nodes
	  9. sort by (score asc, -(cited_by_count or 0), id asc); return first top_k
	"""
	if not group_a or not group_b:
		raise ValueError("group must be non-empty")

	canon_a = {oracle.client.key(s) for s in group_a}
	canon_b = {oracle.client.key(s) for s in group_b}
	log.info(
		"bidir: %d + %d seeds, mode=%s, max_depth=%d",
		len(canon_a), len(canon_b), mode, max_depth,
	)

	engine = BiDirEngine(
		canon_a, canon_b, mode=mode, max_depth=max_depth, frontier_cap=frontier_cap
	)

	with Parallel(
		n_jobs=n_jobs, backend="threading", verbose=10 if verbose else 0
	) as parallel:
		step = engine.advance(StepIn(nbrs={}))
		hop = 0
		while not step.done:
			hop += 1
			log.info("  hop %d: fetching neighbours of %d node(s)", hop, len(step.frontier))
			nbrs = fetch_all(oracle, sorted(step.frontier), parallel)
			step = engine.advance(StepIn(nbrs=nbrs))
			log.info("  hop %d: %d meeting node(s) so far", hop, len(step.state.meeting))

	meeting = step.state.meeting
	seeds = canon_a | canon_b

	# backfill metadata for non-seed meeting nodes
	to_fetch = [n for n in meeting if n not in seeds]
	meta = oracle.client.works(to_fetch) if to_fetch else {}

	# build records
	records = []
	for node, d in meeting.items():
		if node in seeds:
			continue  # exclude seed nodes
		w = meta.get(node, {})
		records.append({
			"id": node,
			"score": d["a"] + d["b"],
			"groups_hit": 2,
			"dists": {"a": d["a"], "b": d["b"]},
			"title": w.get("title"),
			"year": w.get("publication_year"),
			"doi": w.get("doi"),
			"cited_by_count": w.get("cited_by_count"),
		})

	# sort by score asc, cited_by_count desc, id asc
	records.sort(
		key=lambda r: (r["score"], -(r.get("cited_by_count") or 0), r["id"])
	)
	log.info("bidir: done, %d bridge(s)", len(records))
	return records[:top_k]


def find_bridges_bidir_nway(
	oracle,
	groups,
	*,
	min_groups: int | None = None,
	mode: str = "budget",
	max_depth: int = 6,
	frontier_cap: int = 50,
	top_k: int = 25,
	n_jobs: int = 8,
	verbose: bool = False,
) -> list[dict]:
	"""N-way bidirectional search via pairwise decomposition.

	Proc:
	  1. raise ValueError if groups is empty or any group is empty
	  2. N = len(groups)
	  3. if N == 2: delegate to find_bridges_bidir(oracle, groups[0], groups[1], ...)
	  4. min_groups = N if min_groups is None else min_groups
	  5. canonicalize each group: canon = [{oracle.client.key(s) for s in g} for g in groups]
	  6. build one engine per pair: engines = {(i,j): BiDirEngine(canon[i], canon[j], ...) ...}
	  7. round-robin drive all engines in lockstep:
	     - with Parallel(..., backend="threading", ...):
	       - prime every engine: live = {key: step for key, step if not step.done}
	       - while live:
	         - union = sorted(set().union(*(s.frontier for s in live.values())))
	         - nbrs = fetch_all(oracle, union, parallel)
	         - advance each live engine with its subset of nbrs
	         - keep only non-done engines in next_live
	  8. aggregate across all engines:
	     - per_group = {node: {group_idx: min_dist}}
	     - recurrence = {node: pair_count}
	     - for each (i,j), eng: for each node, d in eng._meeting:
	       - per_group[node][i] = min(per_group[node].get(i, inf), d["a"])
	       - per_group[node][j] = min(per_group[node].get(j, inf), d["b"])
	       - recurrence[node] += 1
	  9. candidates = nodes in per_group with (not in seeds and len(per_group[node]) >= min_groups)
	  10. backfill metadata: meta = oracle.client.works(candidates)
	  11. build records: id, score (sum of dists), groups_hit, dists, recurrence, metadata
	  12. sort by (-groups_hit, score asc, -(cited_by_count or 0), id asc); return first top_k
	"""
	if not groups:
		raise ValueError("groups must be non-empty")
	for g in groups:
		if not g:
			raise ValueError("group must be non-empty")

	N = len(groups)
	if N == 2:
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

	# canonicalize each group
	canon = [{oracle.client.key(s) for s in g} for g in groups]

	# build engines for each pair
	pairs = list(itertools.combinations(range(N), 2))
	engines = {
		(i, j): BiDirEngine(
			canon[i], canon[j], mode=mode, max_depth=max_depth, frontier_cap=frontier_cap
		)
		for i, j in pairs
	}
	log.info(
		"bidir n-way: %d groups, %d pairs, mode=%s, max_depth=%d",
		N, len(pairs), mode, max_depth,
	)

	# round-robin drive all engines
	with Parallel(
		n_jobs=n_jobs, backend="threading", verbose=10 if verbose else 0
	) as parallel:
		# prime every engine
		live = {}  # (i, j) -> last StepOut
		for key, eng in engines.items():
			step = eng.advance(StepIn(nbrs={}))
			if not step.done:
				live[key] = step

		# rounds: drive all live engines in lockstep
		round_n = 0
		while live:
			round_n += 1
			union = sorted(set().union(*(s.frontier for s in live.values())))
			log.info(
				"  round %d: %d live engine(s), fetching %d node(s)",
				round_n, len(live), len(union),
			)
			nbrs = fetch_all(oracle, union, parallel)
			next_live = {}
			for key, step in live.items():
				# extract this engine's subset of fetched neighbours
				sub = {n: nbrs[n] for n in step.frontier if n in nbrs}
				nxt = engines[key].advance(StepIn(nbrs=sub))
				if not nxt.done:
					next_live[key] = nxt
			live = next_live

	# aggregate across all finished engines
	per_group = {}  # node_id -> {group_index: min hop distance}
	recurrence = {}  # node_id -> count of pairs that bridge it

	for (i, j), eng in engines.items():
		for node, d in eng._meeting.items():
			pg = per_group.setdefault(node, {})
			pg[i] = min(pg.get(i, float("inf")), d["a"])
			pg[j] = min(pg.get(j, float("inf")), d["b"])
			recurrence[node] = recurrence.get(node, 0) + 1

	# filter candidates: non-seed, meets min_groups threshold
	seeds = set().union(*canon)
	candidates = [
		node
		for node, pg in per_group.items()
		if node not in seeds and len(pg) >= min_groups
	]

	# backfill metadata
	meta = oracle.client.works(candidates) if candidates else {}

	# build records
	records = []
	for node in candidates:
		pg = per_group[node]
		w = meta.get(node, {})
		records.append({
			"id": node,
			"score": sum(pg.values()),
			"groups_hit": len(pg),
			"dists": pg,
			"recurrence": recurrence[node],
			"title": w.get("title"),
			"year": w.get("publication_year"),
			"doi": w.get("doi"),
			"cited_by_count": w.get("cited_by_count"),
		})

	# sort by (-groups_hit, score asc, -(cited_by_count or 0), id asc)
	records.sort(
		key=lambda r: (-r["groups_hit"], r["score"], -(r.get("cited_by_count") or 0), r["id"])
	)
	log.info("bidir n-way: done, %d candidate(s)", len(records))
	return records[:top_k]
