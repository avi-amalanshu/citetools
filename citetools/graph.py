from __future__ import annotations

from typing import Callable, Literal
import networkx as nx


def _rank_cited_by_count(work: dict) -> float:
    """
    Default cap_key ranker: descending by cited_by_count, null-safe.

    Ranks cited_by_count descending. A work whose cited_by_count is missing or None
    sorts LAST (treated as -inf for descending order).

    Proc:
      - input: a full work dict (from client.work or client.works or client.citers)
      - extract cited_by_count; null-guard: absent/None -> return float('-inf')
      - otherwise return cited_by_count (negation is implicit in the sort context)

    Args:
      work: dict carrying cited_by_count field (may be missing/None)

    Returns:
      float, used as a sort key in descending order (higher=better rank)
    """
    cbc = work.get("cited_by_count")
    if cbc is None:
        return float("-inf")
    return cbc


class CiteGraph:
    """
    Lazy citation-graph oracle wrapping an OpenAlex client.

    Builds citation neighborhoods on demand via BFS with networkx. Caches both
    neighbor lists and work metadata. Ranks frontier nodes by a pluggable key function
    and applies directional capping.
    """

    def __init__(
        self,
        client,
        *,
        cap: int = 50,
        cap_dir: Literal["refs", "citers", "both"] = "refs",
        cap_key: Callable[[dict], float] | None = None,
    ) -> None:
        """
        Lazy citation-graph oracle wrapping an OpenAlex client.

        Proc:
          - store client
          - store cap (int > 0), cap_dir (one of "refs" "citers" "both"), cap_key
          - if cap_key is None, set it to _rank_cited_by_count
          - init two caches: _nbrs_cache (dict of node id -> {refs, citers}),
            _meta_cache (dict of node id -> work metadata dict)

        Args:
          client: OpenAlex instance (has methods key, work, works, citers, clear)
          cap: int, top-K size for expansion frontier. default 50.
          cap_dir: which edge direction(s) the cap applies to.
            "refs": cap reference expansion only (default, citers already capped upstream)
            "citers": cap citers only
            "both": cap both (seldom needed, citers are 1-page)
          cap_key: Callable[[dict], comparable] or None.
            signature: (work_metadata: dict) -> comparable (typically float).
            used to rank newly-seen nodes for inclusion in the next frontier.
            default: _rank_cited_by_count (descending by cited_by_count, null-safe).

            Known bias of the default ranker: ranking by cited_by_count deprioritises
            modestly-cited and recent papers, potentially filtering cross-domain bridge
            papers from the frontier. this is acceptable for v1 because excluded nodes
            remain in the graph as scoreable candidates and cap/K is tunable.

        Raises:
          None (client is assumed valid; validation is the caller's concern)
        """
        self.client = client
        self.cap = cap
        self.cap_dir = cap_dir
        self.cap_key = cap_key if cap_key is not None else _rank_cited_by_count
        self._nbrs_cache: dict[str, dict] = {}
        self._meta_cache: dict[str, dict] = {}

    def nbrs(self, node: str) -> dict[str, list[str]]:
        """
        Lazy + cached neighbor lists for a node.

        Proc:
          - if node in _nbrs_cache, return cached value immediately
          - fetch the work via client.work(node); if not found (raises or returns None),
            return {"refs": [], "citers": []}
          - extract refs: canonicalize EACH entry of work.get("referenced_works") via
            client.key(). referenced_works holds full OpenAlex URLs
            ("https://openalex.org/W123"), but every graph node id MUST be a bare
            canonical W-id so it matches citers() output and works() keys -- otherwise
            the graph gets two distinct nodes per paper (URL endpoint vs bare metadata
            node). null-guard: absent/None -> []
          - extract citers: client.citers(node) (already returns bare canonical W-ids);
            if client returns [] on unknown node, that is correct
          - cache both in _nbrs_cache[node] = {refs, citers}
          - cache the work metadata in _meta_cache[node] = work
          - return {"refs": refs, "citers": citers}

        Handles unknown nodes gracefully: unknown node -> both empty lists, no cache entry.

        Args:
          node: canonical W-id (bare "Wxxxx")

        Returns:
          dict with keys "refs" and "citers", each a list of canonical W-ids.
          unknown node -> {"refs": [], "citers": []}
        """
        if node in self._nbrs_cache:
            return self._nbrs_cache[node]

        try:
            work = self.client.work(node)
        except Exception:
            # node not found or other error; treat as unknown
            return {"refs": [], "citers": []}

        if work is None:
            return {"refs": [], "citers": []}

        # referenced_works holds full OpenAlex URLs; canonicalize to bare W-ids so
        # node identity is consistent with citers() and works().
        refs = [self.client.key(u) for u in (work.get("referenced_works") or [])]
        citers = self.client.citers(node)

        result = {"refs": refs, "citers": citers}
        self._nbrs_cache[node] = result
        self._meta_cache[node] = work

        return result

    def grow(self, seed: str, depth: int) -> nx.DiGraph:
        """
        BFS from seed out to depth hops over nbrs.

        Proc:
          - canonicalize seed via client.key(seed) and client.work(seed) to get canonical W-id
          - init: g = nx.DiGraph()
          - init: visited = set() (to prevent re-expansion on cycles)
          - init: frontier = {canonical_seed}

          for hop in range(depth):
            - if frontier is empty or already subset of visited, break
            - new_frontier_sources = {} (dict mapping newly-seen node id to "refs"|"citers"|"both")

            for each node in frontier - visited:
              - add node to visited
              - nbrs_dict = nbrs(node)
              - for each ref_id in nbrs_dict["refs"]:
                  * add edge (node -> ref_id) to g
                  * if ref_id not in visited:
                      - if ref_id not in new_frontier_sources: map ref_id -> "refs"
                      - else if current value is "citers": update to "both"
              - for each citer_id in nbrs_dict["citers"]:
                  * add edge (citer_id -> node) to g
                  * if citer_id not in visited:
                      - if citer_id not in new_frontier_sources: map citer_id -> "citers"
                      - else if current value is "refs": update to "both"

            - batch-fetch metadata for every newly-seen node:
              * if new_frontier_sources is nonempty:
                  meta_batch = client.works(list(new_frontier_sources.keys()))
                  for each (wid, work) in meta_batch.items():
                      - add/update node wid in g with attrs {title, year, cited_by_count, doi}
                      - cache work in _meta_cache[wid]

            - rank and cap the next frontier via _rank_and_cap(new_frontier_sources, visited)
            - frontier = result

          - after the loop, backfill metadata:
            * bare = [n for n, d in g.nodes(data=True) if "title" not in d]
            * if bare is nonempty:
                meta_batch = client.works(bare)
                for each (wid, work) in meta_batch.items():
                    update node wid with attrs {title, year, cited_by_count, doi}
                    cache work in _meta_cache[wid]

          - ensure seed node exists in graph (may be isolated if depth=0)
          - add seed node with metadata (backfilled or cached)
          - set g.graph["seed"] = canonical_seed
          - return g

        Edge cases:
          - depth == 0: seed-only graph (frontier never enters loop). return graph with
            just the seed node (metadata backfilled or from cache).
          - cycles in dirty data: visited prevents re-expansion; cycle edges are kept.
          - node as both ref and citer: both directed edges kept (one incoming, one outgoing).
          - unknown node in frontier: nbrs returns empty lists; no edges added, no
            metadata fetched for that node. it remains in the graph as a bare node from a
            previous edge.

        Args:
          seed: any id form (W-id, DOI, URL); client.key normalizes
          depth: int >= 0, number of hops

        Returns:
          nx.DiGraph with:
            - nodes: canonical W-ids
            - node attrs: title, year (publication_year), cited_by_count, doi
            - edges: (citing_id, cited_id), directed
            - graph attr: g.graph["seed"] = canonical seed W-id

        Raises:
          Any exception from client (BadId, OpenAlexError) propagates.
        """
        # canonicalize seed
        canonical_seed = self.client.key(seed)
        seed_work = self.client.work(canonical_seed)
        canonical_seed = seed_work["id"].rsplit("/", 1)[-1]  # extract bare W-id

        g = nx.DiGraph()
        visited: set[str] = set()
        frontier: set[str] = {canonical_seed}

        # BFS loop over hops
        for _ in range(depth):
            new_frontier_sources: dict[str, str] = {}  # {node_id: "refs"|"citers"|"both"}

            for node in frontier - visited:
                visited.add(node)
                nbrs_dict = self.nbrs(node)

                # refs (outgoing edges): node -> ref_id
                for ref_id in nbrs_dict["refs"]:
                    g.add_edge(node, ref_id)
                    if ref_id not in visited:
                        if ref_id not in new_frontier_sources:
                            new_frontier_sources[ref_id] = "refs"
                        elif new_frontier_sources[ref_id] == "citers":
                            new_frontier_sources[ref_id] = "both"

                # citers (incoming edges): citer_id -> node
                for citer_id in nbrs_dict["citers"]:
                    g.add_edge(citer_id, node)
                    if citer_id not in visited:
                        if citer_id not in new_frontier_sources:
                            new_frontier_sources[citer_id] = "citers"
                        elif new_frontier_sources[citer_id] == "refs":
                            new_frontier_sources[citer_id] = "both"

            # batch-fetch metadata for newly-seen nodes
            if new_frontier_sources:
                meta_batch = self.client.works(list(new_frontier_sources.keys()))
                for wid, work in meta_batch.items():
                    attrs = self._extract_attrs(work)
                    g.add_node(wid, **attrs)
                    self._meta_cache[wid] = work

            # rank and cap the frontier
            frontier = self._rank_and_cap(new_frontier_sources, visited)

        # backfill metadata for bare nodes
        bare = [n for n, d in g.nodes(data=True) if "title" not in d]
        if bare:
            meta_batch = self.client.works(bare)
            for wid, work in meta_batch.items():
                attrs = self._extract_attrs(work)
                g.nodes[wid].update(attrs)
                self._meta_cache[wid] = work

        # ensure seed is in graph with full metadata
        if canonical_seed not in g:
            g.add_node(canonical_seed)
        seed_work_cached = self._meta_cache.get(canonical_seed, seed_work)
        attrs = self._extract_attrs(seed_work_cached)
        g.nodes[canonical_seed].update(attrs)

        # mark seed for L3 read-before-compose contract
        g.graph["seed"] = canonical_seed

        return g

    def _extract_attrs(self, work: dict) -> dict:
        """
        Extract node metadata attributes from a work dict.

        Proc:
          - input: full work dict from client.work(s) or client.citers
          - extract: title (fallback to display_name, or empty string)
          - extract: publication_year as "year"
          - extract: cited_by_count (null-preserve)
          - extract: doi (null-preserve)
          - return dict with keys: title, year, cited_by_count, doi

        Args:
          work: full work dict from client.work(s) or client.citers

        Returns:
          dict with keys: title, year, cited_by_count, doi
          (null-guarding: missing/None fields are preserved as-is)
        """
        return {
            "title": work.get("title") or work.get("display_name") or "",
            "year": work.get("publication_year"),
            "cited_by_count": work.get("cited_by_count"),
            "doi": work.get("doi"),
        }

    def _rank_and_cap(
        self, candidates: dict[str, str], visited: set[str]
    ) -> set[str]:
        """
        Rank newly-seen nodes by cap_key and return top-cap, restricted to cap_dir.

        Proc:
          - for each (node, edge_type) in candidates.items():
              * retrieve metadata from _meta_cache; if not present, use empty dict
              * score = cap_key(metadata)
              * append (score, node, edge_type) to scored list
          - sort scored list descending by score (highest first)
          - filter by cap_dir:
              * if cap_dir == "both": keep all
              * else: keep only tuples where edge_type matches cap_dir or is "both"
          - take top-cap by position (or all if fewer than cap)
          - return as set of node ids

        Args:
          candidates: dict {node_id: edge_source_type}
            edge_source_type in ("refs", "citers", "both")
            (nodes whose new edges came from refs, citers, or both)
          visited: set of already-visited nodes (unused; kept for clarity)

        Returns:
          set of top-cap node ids (or fewer if fewer candidates available)
        """
        if not candidates:
            return set()

        scored = []
        for node, edge_type in candidates.items():
            work = self._meta_cache.get(node, {})
            score = self.cap_key(work)
            scored.append((score, node, edge_type))

        # sort descending by score
        scored.sort(reverse=True, key=lambda x: x[0])

        # filter by cap_dir
        if self.cap_dir == "both":
            filtered = scored
        else:
            filtered = [
                (score, node, edge_type)
                for score, node, edge_type in scored
                if self.cap_dir == edge_type or edge_type == "both"
            ]

        # take top-cap
        capped = [node for _, node, _ in filtered[: self.cap]]
        return set(capped)
