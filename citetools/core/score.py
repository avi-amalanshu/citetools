"""pure python module: distance aggregation, top-k filtering, result dict construction.

no i/o, no input mutation. all functions are composable pure helpers lifted from
four strategy implementations (intersection, bidir, walk, refine) to eliminate duplication.
"""


def attrs(work: dict) -> dict:
    """extract and None-safe metadata from a work dict.

    procedure: extract keys {title, year, doi, cited_by_count} from work,
    normalizing publication_year -> year, and returning None for missing keys.
    """
    if work is None:
        work = {}
    return {
        "title": work.get("title"),
        "year": work.get("publication_year") or work.get("year"),
        "doi": work.get("doi"),
        "cited_by_count": work.get("cited_by_count"),
    }


def record(
    id: str,
    dists: dict,
    groups_hit: int,
    score: float,
    work: dict,
    *,
    recurrence: int | None = None,
    degree: int | None = None
) -> dict:
    """build one result dict with all scoring context + optional recurrence/degree.

    procedure: extract metadata via attrs(work), populate base keys
    {id, score, groups_hit, dists, title, year, doi, cited_by_count},
    then conditionally add recurrence and degree if not None.
    """
    meta = attrs(work)
    result = {
        "id": id,
        "score": score,
        "groups_hit": groups_hit,
        "dists": dists,
        "title": meta["title"],
        "year": meta["year"],
        "doi": meta["doi"],
        "cited_by_count": meta["cited_by_count"],
    }
    if recurrence is not None:
        result["recurrence"] = recurrence
    if degree is not None:
        result["degree"] = degree
    return result


def keep(
    nodes: list[str] | set[str],
    per_group_dists: dict[str, dict],
    seeds: set[str],
    min_groups: int
) -> list[str]:
    """filter to non-seed candidates reachable from >= min_groups groups.

    procedure: iterate nodes, skip seeds and unreachable nodes,
    keep those appearing in per_group_dists with >= min_groups group hits.
    """
    result = []
    for node in nodes:
        if node in seeds:
            continue
        if node not in per_group_dists:
            continue
        dists = per_group_dists[node]
        if len(dists) >= min_groups:
            result.append(node)
    return result


def agg(
    candidates: list[str],
    per_group_dists: dict[str, dict]
) -> dict[str, dict]:
    """per candidate, build {score, groups_hit, dists} from distance dict.

    procedure: for each candidate, extract distances from per_group_dists,
    sum distances to score, count groups, build aggregated dict.
    """
    result = {}
    for candidate in candidates:
        dists = per_group_dists.get(candidate, {})
        score = sum(dists.values())
        groups_hit = len(dists)
        result[candidate] = {
            "score": score,
            "groups_hit": groups_hit,
            "dists": dists,
        }
    return result


def topk(
    records: list[dict],
    key: callable,
    k: int
) -> list[dict]:
    """sort by key, slice to k (graceful when fewer than k records).

    procedure: sort records in-place via list.sort(key=key), return [:k].
    """
    records.sort(key=key)
    return records[:k]
