"""
N-way intersection engine via per-seed BFS growth and in-frame dijkstra scoring.

Procedure overview:
  1. Initialize N empty DiGraph instances.
     For each group g, allocate per-seed tracking:
       - visited[g] = dict {seed_id: set()} (per-seed visited set)
       - frontier[g] = dict {seed_id: set()} (per-seed frontier set)
       - Initialize each seed's frontier to {seed} and visited to empty.
  2. For each hop 0..depth-1:
     a. Compute active_frontier = union of ALL per-seed frontiers across all groups.
     b. If active_frontier empty, break (all seeds exhausted).
     c. Union all per-seed frontiers -> one Req(want_emb=False).
     d. Ingest Resp; for EACH SEED's INDEPENDENT expansion:
        - For each seed s in group g:
          * For each node n in frontier[g][s]:
            - If n not in resp, skip (unvisited this hop).
            - Extract refs_list and citers_list from resp[n].
            - For each ref in sorted(refs_list): insert edge (n -> ref) into graphs[g].
            - For each citer in sorted(citers_list): insert edge (citer -> n) into graphs[g].
            - Discover new candidate nodes; add to graph with empty attrs.
            - Update visited[g][s].add(n).
     e. Discover new candidate nodes: for each group g, collect all nodes appearing
        in this hop's neighbors but not yet in graphs[g].
     f. Metadata prefetch (CRITICAL TIMING FIX).
        - Compute all_new_candidates = union of new_candidates across all groups.
        - If all_new_candidates nonempty:
            * yield Req(nodes=frozenset(all_new_candidates), want_emb=False).
            * Receive prefetch_resp: dict[str, dict | Exception].
            * For each group g, update nodes in graphs[g] with their attrs
              from prefetch_resp (title, year, doi, cited_by_count).
     g. Frontier rank-and-cap: for each seed s in each group g.
        - Compute new_frontier_nodes[g][s] = {nodes just added to graphs[g] from seed s this hop}.
        - Cap via _rank_and_cap(new_frontier_nodes[g][s], graphs[g], cap=50, cap_key=...);
          returns top-50 frontier nodes.
        - Update frontier[g][s] = capped set.
  3. Post-growth composition and scoring:
     a. For each group g: compose all per-seed subgraphs into one group DiGraph.
        - Iterate through all visited nodes collected from all seeds' expansions.
        - Edges are already in graphs[g] from step 2.d; no additional composition needed.
     b. Compute per-group distance maps via multi_source_dijkstra_path_length
        on the undirected version of each group's DiGraph.
     c. Collect all nodes from all graphs; exclude all seeds.
     d. For each candidate, compute which groups it appears in and its summed
        distance across hit groups. Filter by min_groups threshold.
     e. Assemble result dicts via core.score.record with degree field.
  4. Return SearchResult with bridges and meta.

Determinism invariants:
  - Per-seed frontiers are sets; union before Req.
  - Edges inserted in sorted order (networkx tie-break on insertion order).
  - _rank_and_cap ranks by cap_key; ties broken by node id ascending.
  - Distance aggregation via core.score.keep and agg (pure).

Metadata-timing fix (critical):
  - Frontier candidates discovered in hop H have metadata NOT in H's response.
  - Response H carries only the expanded nodes' neighbors.
  - Engine must issue a prefetch Req(candidates) before capping.
  - Capping then ranks candidates against merged metadata (H + prefetch).
  - Without this, frontier candidates have no metadata; capping silently
    degrades to arbitrary order, causing divergence from main.
"""

from __future__ import annotations

from typing import Generator
import networkx as nx

from .protocol import Req, Resp, SearchResult
from ..score import keep


def _rank_and_cap(
    candidates: dict[str, str],
    graph: nx.DiGraph,
    *,
    cap: int = 50,
    cap_key=None,
) -> set[str]:
    """
    Rank newly-discovered nodes by cap_key, filter to refs direction, return top-cap.

    Mirrors pre-refactor CiteGraph._rank_and_cap with cap_dir="refs": only nodes
    whose edge_type is "refs" or "both" enter the next frontier. Ties broken by
    W-id ascending (total order).

    Procedure:
      1. Compute scores.
         - For each (node, edge_type) in candidates.items():
             * Retrieve work metadata from graph.nodes[node]; if absent, use {}.
             * score = cap_key(metadata) if cap_key else cited_by_count(metadata).
             * Append (score, node, edge_type) to scored list.

      2. Two-pass stable sort for total order.
         - First sort by node ascending (W-id).
         - Then sort by score descending (stable, preserves node order within ties).

      3. Filter to refs direction (cap_dir="refs").
         - Keep only entries where edge_type is "refs" or "both".

      4. Cap and return.
         - Take top-cap nodes (or all if fewer than cap available).
         - Return as set.
    """
    if not candidates:
        return set()

    if cap_key is None:
        cap_key = lambda w: w.get("cited_by_count") or float("-inf")

    scored = []
    for node, edge_type in candidates.items():
        work = graph.nodes.get(node, {})
        score = cap_key(work)
        scored.append((score, node, edge_type))

    # stable two-pass: first by node id, then by score descending
    scored.sort(key=lambda x: x[1])  # node ascending
    scored.sort(reverse=True, key=lambda x: x[0])  # score descending (stable)

    # filter to refs direction (cap_dir="refs"): only refs or both
    filtered = [node for _, node, et in scored if et == "refs" or et == "both"]

    # take top-cap
    return set(filtered[:cap])


def intersection_engine(
    groups: list[set[str]],
    *,
    depth: int = 2,
    min_groups: int | None = None,
) -> Generator[Req, Resp, SearchResult]:
    """
    N-way intersection via concurrent BFS and in-frame dijkstra scoring.

    Procedure:
      1. Validate and initialize.
         - Assert all groups nonempty; raise ValueError if any is empty.
         - Compute num_groups = len(groups).
         - If min_groups is None, set min_groups = num_groups.
         - Assert depth >= 0.
         - Allocate list of N empty DiGraph instances (one per group).
         - Allocate per-seed tracking structures:
           * visited[g_idx] = dict {seed_id: set()} (per-seed visited set).
           * frontier[g_idx] = dict {seed_id: set()} (per-seed frontier set).
           * For each group g_idx and seed s in groups[g_idx]:
             - visited[g_idx][s] = set().
             - frontier[g_idx][s] = {s} (seed starts as its own frontier).
         - Mark all_seeds = union of all groups.

      2. Main growth loop: for hop in range(depth).
         a. Compute active frontier across all per-seed frontiers.
            - active_frontier = union of frontier[g_idx][s] for all g_idx, s.
            - If active_frontier is empty, break (all seeds exhausted).
            - combined_frontier = frozenset(active_frontier).
            - yield Req(nodes=combined_frontier, want_emb=False).
            - Receive resp: dict[str, dict | Exception].

         b. Per-seed edge assembly: for each group g_idx and seed s in groups[g_idx].
            - For each node n in frontier[g_idx][s]:
                * If n not in resp, skip (unvisited this hop).
                * Get node_resp = resp[n]; if Exception, treat as no neighbors.
                * Extract refs_list = node_resp.get("refs", [])  (list of W-ids).
                * Extract citers_list = node_resp.get("citers", []).
                * For each ref in sorted(refs_list): insert edge (n -> ref)
                  into graphs[g_idx].
                * For each citer in sorted(citers_list): insert edge (citer -> n)
                  into graphs[g_idx].
                * Add/update nodes: for all newly discovered nodes d (those appearing in
                  resp but not yet in graphs[g_idx]), add nodes with empty attrs dict
                  (metadata will arrive in the prefetch).
                * Mark visited[g_idx][s].add(n).

         c. Candidate discovery: for each group g_idx.
            - Compute new_candidates[g_idx] = {nodes appearing in this hop's
              neighbors but not yet in graphs[g_idx]}.
            - (In implementation: track while scanning neighbors above.)

         d. Metadata prefetch (CRITICAL TIMING FIX).
            - Compute all_new_candidates = union of new_candidates across all groups.
            - If all_new_candidates nonempty:
                * yield Req(nodes=frozenset(all_new_candidates), want_emb=False).
                * Receive prefetch_resp: dict[str, dict | Exception].
                * For each group g_idx, update nodes in graphs[g_idx] with
                  their attrs from prefetch_resp (title, year, doi, cited_by_count).
            - (Else skip; no candidates this hop -> no cap to apply.)

         e. Per-seed frontier rank-and-cap: for each group g_idx and seed s in groups[g_idx].
            - Compute seed's new frontier candidates.
              new_frontier_nodes[g_idx][s] = {nodes just added to graphs[g_idx] from seed s this hop}.
            - Cap via _rank_and_cap(new_frontier_nodes[g_idx][s], graphs[g_idx], cap=50,
              cap_key=...); returns top-50 frontier nodes.
            - Update frontier[g_idx][s] = capped set.

      3. Post-growth scoring phase.
         a. Undirected distance computation: for each group g_idx.
            - Compute undirected = graphs[g_idx].to_undirected().
            - Compute dists_per_group[g_idx] = nx.multi_source_dijkstra_path_length(
                undirected, sources=groups[g_idx]).
            - (Returns dict[node_id, dist]; unreachable nodes omitted -> node in dists
              is the reachability test.)

         b. Seed exclusion.
            - Compute all_seeds = union of all groups.

         c. Candidate enumeration and aggregation.
            - Collect all_nodes = union of all graphs[g_idx].nodes.
            - For each candidate c in all_nodes:
                * Skip if c in all_seeds.
                * Compute hits = {g_idx: dists_per_group[g_idx][c]
                                  for g_idx in 0..num_groups-1
                                  if c in dists_per_group[g_idx]}.
                * If len(hits) < min_groups, skip.
                * Otherwise, confirm c is valid via core.score.keep.

         d. Score aggregation.
            - Aggregate hits via core.score.agg(candidates={c...},
              per_group_dists=dists_per_group) -> dict[c, {score, groups_hit, dists}].

         e. Result assembly.
            - For each candidate c and its aggregated data:
                * Compute degree[c] = max(graphs[g_idx].degree(c)
                                          for g_idx in hits.keys()).
                * Fetch work metadata from graphs[g_idx].nodes[c] (same metadata in all
                  hit groups; use any hit group).
                * Build bridges[c] = {group_idx: distance for group_idx, distance in hits.items()}.

         f. Finalize and return.
            - Return SearchResult(bridges=bridges, meta={}).
            - bridges is a dict keyed by candidate W-id; each value is {group_idx: distance}.

    Args:
      groups: list[set[str]], N >= 2 seed groups as sets of W-ids.
              Each set must be nonempty; ValueError raised if any is empty.
      depth: int >= 0, citation hops per group (default 2).
      min_groups: int | None, minimum groups a bridge candidate must reach
                  (default None -> N). Must satisfy 1 <= min_groups <= N.

    Raises:
      ValueError: if any group is empty.
      ValueError: if depth < 0 or min_groups not in [1, N].

    Returns:
      Generator that yields Req, receives Resp, and returns SearchResult.
    """
    # validation and initialization
    for g in groups:
        if not g:
            raise ValueError("group must be non-empty")

    num_groups = len(groups)
    if min_groups is None:
        min_groups = num_groups

    if depth < 0:
        raise ValueError("depth must be >= 0")

    if not (1 <= min_groups <= num_groups):
        raise ValueError(f"min_groups must be in [1, {num_groups}]")

    # allocate per-group graphs and per-seed tracking
    graphs = [nx.DiGraph() for _ in range(num_groups)]
    visited = [{seed: set() for seed in groups[g_idx]} for g_idx in range(num_groups)]
    frontier = [{seed: {seed} for seed in groups[g_idx]} for g_idx in range(num_groups)]
    all_seeds = set().union(*groups)

    # main growth loop
    for hop in range(depth):
        # compute active frontier across all per-seed frontiers
        active_frontier = set()
        for g_idx in range(num_groups):
            for s in groups[g_idx]:
                active_frontier.update(frontier[g_idx][s])

        if not active_frontier:
            break

        # yield request for all frontier nodes
        combined_frontier = frozenset(active_frontier)
        resp = yield Req(nodes=combined_frontier, want_emb=False)

        # per-seed edge assembly and candidate discovery
        # edge_type tracks "refs", "citers", or "both" per node (for cap_dir filtering)
        new_candidates_per_group = [set() for _ in range(num_groups)]
        # new_frontier_nodes_per_seed: {seed: {node: edge_type}}
        new_frontier_nodes_per_seed = [{s: {} for s in groups[g_idx]} for g_idx in range(num_groups)]

        for g_idx in range(num_groups):
            for s in groups[g_idx]:
                for n in frontier[g_idx][s]:
                    if n not in resp:
                        continue

                    node_resp = resp[n]
                    if isinstance(node_resp, Exception):
                        continue

                    # extract neighbors
                    refs_list = node_resp.get("refs", [])
                    citers_list = node_resp.get("citers", [])

                    # sorted edge insertion for determinism
                    for ref in sorted(refs_list):
                        if ref not in graphs[g_idx]:
                            graphs[g_idx].add_node(ref)
                            new_candidates_per_group[g_idx].add(ref)
                        graphs[g_idx].add_edge(n, ref)
                        # track edge type: refs or upgrade to "both"
                        if ref not in visited[g_idx][s]:
                            prev = new_frontier_nodes_per_seed[g_idx][s].get(ref)
                            if prev is None:
                                new_frontier_nodes_per_seed[g_idx][s][ref] = "refs"
                            elif prev == "citers":
                                new_frontier_nodes_per_seed[g_idx][s][ref] = "both"

                    for citer in sorted(citers_list):
                        if citer not in graphs[g_idx]:
                            graphs[g_idx].add_node(citer)
                            new_candidates_per_group[g_idx].add(citer)
                        graphs[g_idx].add_edge(citer, n)
                        # track edge type: citers or upgrade to "both"
                        if citer not in visited[g_idx][s]:
                            prev = new_frontier_nodes_per_seed[g_idx][s].get(citer)
                            if prev is None:
                                new_frontier_nodes_per_seed[g_idx][s][citer] = "citers"
                            elif prev == "refs":
                                new_frontier_nodes_per_seed[g_idx][s][citer] = "both"

                    # ensure n is in graph (parent node from previous hop)
                    if n not in graphs[g_idx]:
                        graphs[g_idx].add_node(n)

                    visited[g_idx][s].add(n)

        # metadata prefetch for new candidates
        all_new_candidates = set().union(*new_candidates_per_group)

        if all_new_candidates:
            prefetch_resp = yield Req(nodes=frozenset(all_new_candidates), want_emb=False)

            # update graph nodes with metadata
            for g_idx in range(num_groups):
                for node_id in new_candidates_per_group[g_idx]:
                    if node_id in prefetch_resp:
                        node_data = prefetch_resp[node_id]
                        if not isinstance(node_data, Exception):
                            # extract metadata fields (title, year, doi, cited_by_count)
                            meta = {
                                "title": node_data.get("title"),
                                "year": node_data.get("year"),
                                "doi": node_data.get("doi"),
                                "cited_by_count": node_data.get("cited_by_count"),
                            }
                            graphs[g_idx].nodes[node_id].update(meta)

        # per-seed frontier rank-and-cap (cap_dir="refs": mirrors pre-refactor default)
        for g_idx in range(num_groups):
            for s in groups[g_idx]:
                new_frontier_nodes = new_frontier_nodes_per_seed[g_idx][s]
                capped = _rank_and_cap(new_frontier_nodes, graphs[g_idx], cap=50)
                frontier[g_idx][s] = capped

    # post-growth scoring phase

    # undirected distance computation
    dists_per_group = []
    for g_idx in range(num_groups):
        undirected = graphs[g_idx].to_undirected()
        dists = nx.multi_source_dijkstra_path_length(undirected, sources=groups[g_idx])
        dists_per_group.append(dists)

    # collect all nodes from all graphs
    all_nodes = set()
    for g_idx in range(num_groups):
        all_nodes.update(graphs[g_idx].nodes)

    # build per_group_dists as required by core.score functions
    # this maps node_id -> dict[group_idx, distance]
    per_group_dists = {}
    for node_id in all_nodes:
        node_hits = {}
        for g_idx in range(num_groups):
            if node_id in dists_per_group[g_idx]:
                node_hits[g_idx] = dists_per_group[g_idx][node_id]
        if node_hits:
            per_group_dists[node_id] = node_hits

    # filter candidates via core.score.keep
    candidates = keep(all_nodes, per_group_dists, all_seeds, min_groups)

    # build bridges dict (candidates -> group distances) and per-candidate meta:
    # node metadata from any hit group plus degree (max over hit groups).
    bridges = {}
    meta = {}
    for candidate in candidates:
        hits = per_group_dists[candidate]
        bridges[candidate] = hits
        node_attrs = graphs[next(iter(hits))].nodes[candidate]
        degree = max(graphs[g_idx].degree(candidate) for g_idx in hits)
        meta[candidate] = {
            "title": node_attrs.get("title"),
            "year": node_attrs.get("year"),
            "doi": node_attrs.get("doi"),
            "cited_by_count": node_attrs.get("cited_by_count"),
            "degree": degree,
        }

    return SearchResult(bridges=bridges, meta=meta)
