import networkx as nx
from typing import Optional


def _grow_groups(
    oracle,
    groups: list[list[str]],
    depth: int
) -> tuple[list[nx.DiGraph], list[set[str]]]:
    """
    For each group of seed ids, expand every seed via oracle.grow(seed, depth).
    Precondition: read g.graph["seed"] from each per-seed graph BEFORE composing,
    because nx.compose merges .graph dicts and only the last would survive.
    Returns (group_graphs, seed_ids_per_group) where each group graph is a composed
    DiGraph and seed_ids_per_group is the canonical seed W-ids for each group.

    Pseudocode:
      group_graphs = []
      seed_ids_per_group = []
      for seeds in groups:
        gg = empty DiGraph
        canonical_seed_ids = set()
        for seed in seeds:
          sub_graph = oracle.grow(seed, depth)
          canonical_id = sub_graph.graph["seed"]
          canonical_seed_ids.add(canonical_id)
          gg = compose(gg, sub_graph)
        group_graphs.append(gg)
        seed_ids_per_group.append(canonical_seed_ids)
      return group_graphs, seed_ids_per_group
    """
    group_graphs = []
    seed_ids_per_group = []

    for seeds in groups:
        if not seeds:
            raise ValueError("group must be non-empty")

        gg = nx.DiGraph()
        canonical_seed_ids = set()

        for seed in seeds:
            sub_graph = oracle.grow(seed, depth)
            canonical_id = sub_graph.graph["seed"]
            canonical_seed_ids.add(canonical_id)
            gg = nx.compose(gg, sub_graph)

        group_graphs.append(gg)
        seed_ids_per_group.append(canonical_seed_ids)

    return group_graphs, seed_ids_per_group


def _score(
    group_graphs: list[nx.DiGraph],
    seed_ids_per_group: list[set[str]],
    min_groups: int,
    top_k: int
) -> list[dict]:
    """
    Score candidates by shortest paths within each group's own graph (never on a
    composed union, which would route paths through other groups). For each non-seed
    node c, compute the set of groups it appears in; if that set is >= min_groups,
    score it as the sum of shortest distances to seeds within each group it hits.
    Candidate selection folds into scoring: a node is scoreable iff it is reachable
    in its group's undirected graph.

    Pseudocode:
      # 1. compute each group's distance map ONCE -- this is the Stage-2 efficiency
      #    fix: exactly one multi-source BFS per group, never per-candidate, and on
      #    each group's OWN undirected graph (never a composed union).
      dist_per_group = []
      for gg, seeds in zip(group_graphs, seed_ids_per_group):
        udir = gg.to_undirected()
        dist_per_group.append(multi_source_dijkstra_path_length(udir, seeds))

      # 2. union of every group's seed ids -- a node that is ANY group's seed is
      #    excluded from candidacy entirely.
      all_seeds = set().union(*seed_ids_per_group)

      # 3. enumerate every node once, then score by dict lookups only.
      all_nodes = set()
      for gg in group_graphs:
        all_nodes.update(gg.nodes)

      candidates = []
      for c in all_nodes:
        if c in all_seeds:
          continue                       # a seed is never a bridge candidate
        hits = {g_idx: dist_per_group[g_idx][c]
                for g_idx in range(len(group_graphs))
                if c in dist_per_group[g_idx]}
        if len(hits) >= min_groups:
          score = sum(hits.values())
          groups_hit = len(hits)
          degree = max(group_graphs[g_idx].degree(c) for g_idx in hits)
          meta = group_graphs[next(iter(hits))].nodes[c]   # any hit group; same metadata
          candidates.append({"id": c, "score": score, "groups_hit": groups_hit,
                             "dists": hits, "title": meta.get("title"),
                             "year": meta.get("year"), "doi": meta.get("doi"),
                             "cited_by_count": meta.get("cited_by_count"),
                             "degree": degree})
      candidates.sort(key=lambda r: (-r["groups_hit"], r["score"],
                                     -(r.get("cited_by_count") or 0)))
      return candidates[:top_k]
    """
    # compute each group's distance map once
    dist_per_group = []
    for gg, seeds in zip(group_graphs, seed_ids_per_group):
        udir = gg.to_undirected()
        dist_per_group.append(nx.multi_source_dijkstra_path_length(udir, seeds))

    # union of all seed ids from all groups
    all_seeds = set().union(*seed_ids_per_group)

    # enumerate every node once
    all_nodes = set()
    for gg in group_graphs:
        all_nodes.update(gg.nodes)

    candidates = []
    for c in all_nodes:
        if c in all_seeds:
            continue  # a seed is never a bridge candidate

        # compute hits: groups this candidate is reachable from
        hits = {g_idx: dist_per_group[g_idx][c]
                for g_idx in range(len(group_graphs))
                if c in dist_per_group[g_idx]}

        if len(hits) >= min_groups:
            score = sum(hits.values())
            groups_hit = len(hits)
            degree = max(group_graphs[g_idx].degree(c) for g_idx in hits)
            meta = group_graphs[next(iter(hits))].nodes[c]

            candidates.append({
                "id": c,
                "score": score,
                "groups_hit": groups_hit,
                "dists": hits,
                "title": meta.get("title"),
                "year": meta.get("year"),
                "doi": meta.get("doi"),
                "cited_by_count": meta.get("cited_by_count"),
                "degree": degree
            })

    # sort by groups_hit desc, score asc, cited_by_count desc
    candidates.sort(key=lambda r: (-r["groups_hit"], r["score"],
                                   -(r.get("cited_by_count") or 0)))
    return candidates[:top_k]


def find_bridges(
    oracle,
    groups: list[list[str]],
    *,
    depth: int = 2,
    min_groups: Optional[int] = None,
    top_k: int = 25
) -> list[dict]:
    """
    N-way intersection strategy: find papers that bridge multiple research groups.

    Args:
        oracle: CiteGraph instance; exposes grow(seed, depth) -> DiGraph with
                g.graph["seed"] set and nodes carrying title, year, cited_by_count, doi.
        groups: list of lists; each inner list is the seed paper ids (W-ids, DOIs, or
                OpenAlex URLs) for one research group. Each group must be non-empty.
        depth: citation hops to expand each seed. Default 2.
        min_groups: require a candidate be reachable from at least this many groups.
                    Defaults to len(groups) (true N-way intersection). Relax to
                    len(groups) - 1 to include papers in the "almost intersection" tier.
        top_k: number of bridges to return. Default 25.

    Returns:
        list of dicts, sorted by (groups_hit desc, score asc, cited_by_count desc).
        Each dict has keys:
          id: paper W-id (canonical OpenAlex id).
          score: sum of shortest distances in groups this paper is reachable from.
          groups_hit: number of groups this paper bridges.
          dists: dict {group_index: distance_in_that_group}.
          title: paper title or None.
          year: publication year or None.
          doi: paper DOI or None.
          cited_by_count: citation count or None.
          degree: max node degree across hit groups (hub indicator).

    Raises:
        ValueError: if any group is empty.

    Procedure:
      1. validate all groups are non-empty; raise ValueError if any is.
      2. default min_groups to len(groups) if not provided.
      3. expand each group via _grow_groups.
      4. score and rank candidates via _score.
      5. return the ranked list (already sorted and trimmed to top_k).
    """
    # validate all groups are non-empty
    for g in groups:
        if not g:
            raise ValueError("group must be non-empty")

    # default min_groups to len(groups)
    if min_groups is None:
        min_groups = len(groups)

    # expand each group
    group_graphs, seed_ids_per_group = _grow_groups(oracle, groups, depth)

    # score and rank candidates
    return _score(group_graphs, seed_ids_per_group, min_groups, top_k)


__all__ = ["find_bridges"]
