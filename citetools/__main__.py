import argparse
import logging
import sys
import textwrap

from citetools import CiteGraph, OpenAlex, find_bridges, find_bridges_bidir, find_bridges_bidir_nway


def _setup_logging() -> None:
    """Attach a stderr handler to the citetools logger so progress is visible.

    coarse progress (group/seed/hop) is shown by default; --verbose additionally
    turns on joblib's live per-fetch ticking (handled inside find_bridges).
    """
    logger = logging.getLogger("citetools")
    if logger.handlers:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# demo seeds as canonical openalex w-ids (arxiv dois are not reliably indexed):
# "attention is all you need", the gcn paper, alphafold.
DEMO_GROUPS = [
    ["W2626778328"],
    ["W2519887557"],
    ["W3177828909"],
]


def main() -> None:
    """
    CLI entry point: parse arguments, construct pipeline, run, print results.

    Pipeline: argparse → validate groups → OpenAlex → CiteGraph → find_bridges.
    Results: print each record (score, groups_hit, year, cited_by_count, degree,
    title, doi) to stdout, one record per print block.

    With --resolve, instead prints the canonical W-id for each given id/link
    (W-id, DOI, or arxiv abs/pdf link) and exits without running the finder.

    With --search, instead prints candidate works (W-id, year, citations,
    title) for each title query and exits, to help find a W-id for --group.

    Known limitations & tuning:
    - depth=2 is the recommended default; depth=3+ adds exponential API cost.
    - --min-groups defaults to full intersection (all N groups); relax to N-1 or N-2
      if results are empty.
    - --top-k ranks by (groups_hit desc, score asc, cited_by_count desc); lower
      score indicates a more central bridge within reachable groups.
    - the "degree" field (node degree in its group graph) helps spot hub-dominated
      results; see --min-groups tuning.
    """
    _setup_logging()

    parser = argparse.ArgumentParser(prog="citetools")

    parser.add_argument(
        "--mailto",
        type=str,
        required=True,
        help="email for OpenAlex polite pool",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=2,
        help="BFS hops (default 2)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=25,
        help="return top K results (default 25)",
    )
    parser.add_argument(
        "--min-groups",
        type=int,
        default=None,
        help="minimum group count for a candidate to qualify (default: number of groups passed)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=8,
        help="worker threads for concurrent OpenAlex fetching (default 8; 1 = serial)",
    )
    parser.add_argument(
        "--group",
        type=str,
        action="append",
        help="comma-separated seed ids for one group (can be repeated)",
    )
    parser.add_argument(
        "--resolve",
        type=str,
        action="append",
        metavar="ID",
        help="resolve an id/link (W-id, DOI, or arxiv abs/pdf link) to its W-id and exit; repeatable",
    )
    parser.add_argument(
        "--search",
        type=str,
        action="append",
        metavar="QUERY",
        help="search works by title; print candidate W-ids and exit; repeatable",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="also stream live per-fetch progress (joblib) to stderr",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["intersection", "bidir"],
        default="intersection",
        help="bridging strategy: 'intersection' (default) or 'bidir' (bidirectional)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["budget", "exact"],
        default="budget",
        help="search mode for bidir strategy (default 'budget'); ignored for intersection",
    )
    parser.add_argument(
        "--frontier-cap",
        type=int,
        default=50,
        help="bidir budget-mode per-step frontier cap, per pairwise engine: the "
        "compute budget for each search step (default 50). ignored in exact mode "
        "and for the intersection strategy",
    )

    args = parser.parse_args()

    # search mode: list candidate works for each title query, then exit
    if args.search:
        client = OpenAlex(mailto=args.mailto)
        try:
            for q in args.search:
                print(f"search: {q}")
                hits = client.search(q)
                if not hits:
                    print("  (no matches)")
                for w in hits:
                    wid = w["id"].rsplit("/", 1)[-1].upper()
                    year = w.get("publication_year") or "????"
                    cites = w.get("cited_by_count") or 0
                    # collapse embedded whitespace so each hit stays one line
                    title = " ".join((w.get("title") or "[no title]").split())
                    print(f"  {wid}  {year}  cites={cites:>6}  {title[:75]}")
                print()
        finally:
            client.close()
        return

    # resolve mode: print canonical W-ids for the given ids/links, then exit
    if args.resolve:
        client = OpenAlex(mailto=args.mailto)
        try:
            for src in args.resolve:
                try:
                    wid = client.wid(src)
                    # collapse embedded whitespace so each line stays one line
                    title = " ".join((client.work(src).get("title") or "[no title]").split())
                    print(f"{wid}  {title[:80]}")
                except Exception as e:
                    print(f"resolve failed for {src}: {e}", file=sys.stderr)
        finally:
            client.close()
        return

    # validate depth
    if args.depth < 1:
        raise ValueError("depth must be >= 1")

    # validate top-k
    if args.top_k < 1:
        raise ValueError("top-k must be >= 1")

    # validate n-jobs
    if args.n_jobs < 1:
        raise ValueError("n-jobs must be >= 1")

    # validate frontier-cap
    if args.frontier_cap < 1:
        raise ValueError("frontier-cap must be >= 1")

    # validate strategy and mode interaction
    if args.strategy not in ["intersection", "bidir"]:
        raise ValueError(f"strategy must be 'intersection' or 'bidir', got {args.strategy}")

    if args.mode not in ["budget", "exact"]:
        raise ValueError(f"mode must be 'budget' or 'exact', got {args.mode}")

    if args.strategy == "intersection" and args.mode != "budget":
        # mode is only meaningful for bidir; silently allow for intersection
        pass

    # handle groups: parse or use demo
    if args.group is None:
        groups = DEMO_GROUPS
        print("Using demo groups (no --group passed).", file=sys.stderr)
    else:
        groups = []
        for group_str in args.group:
            # split on commas and strip whitespace
            ids = [id.strip() for id in group_str.split(",")]
            # check for empty groups
            if not ids or all(not id for id in ids):
                raise ValueError("empty group")
            # filter out empty strings
            ids = [id for id in ids if id]
            groups.append(ids)

    # strategy-specific group count validation
    if args.strategy == "bidir":
        if len(groups) == 1:
            raise ValueError("bidir strategy requires >= 2 groups; got 1")

    # finalize min_groups
    if args.min_groups is None:
        min_groups = len(groups)
    else:
        min_groups = args.min_groups

    # construct pipeline
    client = OpenAlex(mailto=args.mailto)
    oracle = CiteGraph(client)

    # dispatch on strategy
    if args.strategy == "intersection":
        results = find_bridges(
            oracle=oracle,
            groups=groups,
            depth=args.depth,
            min_groups=min_groups,
            top_k=args.top_k,
            n_jobs=args.n_jobs,
            verbose=args.verbose,
        )
    elif args.strategy == "bidir":
        if len(groups) == 2:
            # exactly 2 groups: use pairwise bidirectional
            results = find_bridges_bidir(
                oracle=oracle,
                group_a=groups[0],
                group_b=groups[1],
                mode=args.mode,
                max_depth=args.depth,
                frontier_cap=args.frontier_cap,
                top_k=args.top_k,
                n_jobs=args.n_jobs,
                verbose=args.verbose,
            )
        else:
            # >= 3 groups: use n-way bidirectional
            results = find_bridges_bidir_nway(
                oracle=oracle,
                groups=groups,
                min_groups=min_groups,
                mode=args.mode,
                max_depth=args.depth,
                frontier_cap=args.frontier_cap,
                top_k=args.top_k,
                n_jobs=args.n_jobs,
                verbose=args.verbose,
            )

    # print header
    strategy_info = f"Strategy: {args.strategy}"
    if args.strategy == "bidir":
        strategy_info += f", Mode: {args.mode}, Frontier-cap: {args.frontier_cap}"

    msg = textwrap.dedent(
        f"""\
        Citation bridge finder (citetools)
        ==================================
        Groups: {len(groups)}, Depth: {args.depth}, Min-groups: {min_groups}, {strategy_info}, N-jobs: {args.n_jobs}
        Top-K: {args.top_k}

        Note: depth-2 runs make many API calls; concurrency is rate-limited by OpenAlex.
        Ensure mailto is a real, monitored email address.

        Results (sorted by groups_hit desc, score asc, cited_by_count desc):
        """
    )
    print(msg, file=sys.stdout)

    # print each result
    for r in results:
        # safe extraction: handle missing/None fields
        score = r.get("score", "—")
        groups_hit = r.get("groups_hit", 0)
        year = r.get("year") or "—"
        cited_by = r.get("cited_by_count") or 0
        degree = r.get("degree") or 0
        title = r.get("title") or "[no title]"
        doi = r.get("doi") or "[no DOI]"

        # print in readable format
        print(
            f"{score:>3}  hit={groups_hit}  {year}  cites={cited_by:>5}  "
            f"degree={degree:>3}  {title[:80]}"
        )
        print(f"     {doi}")
        print()  # blank line between records


if __name__ == "__main__":
    try:
        main()
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)
