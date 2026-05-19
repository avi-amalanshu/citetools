"""CLI entry point for citation bridge finder (citetools).

three-layer refactor: argv -> Config (pure, declarative validation),
Config -> results (impure: OpenAlex, runtime, cleanup), results -> text
(pure formatting + printing).

exit modes: (1) resolve: resolve ids to W-ids; (2) search: list candidates
for title queries; (3) main: run a bridging strategy and output records.
all three modes use --mailto for the polite pool and manage the OpenAlex
client lifecycle in a try-finally.
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from typing import Optional

from citetools.io.openalex import OpenAlex
from citetools.strategies import (
    find_bridges,
    find_bridges_bidir,
    find_bridges_bidir_nway,
    find_bridges_walk,
    find_bridges_refine,
)


# demo seeds as canonical openalex w-ids (arxiv dois are not reliably indexed):
# "attention is all you need", the gcn paper, alphafold.
DEMO_GROUPS = [
    ["W2626778328"],
    ["W2519887557"],
    ["W3177828909"],
]


@dataclass(frozen=True)
class Config:
    """Immutable CLI configuration after argparse + validation.

    fields encode resolved, type-checked values:
    - mailto: polite-pool email (str, required, non-empty)
    - depth: BFS hops (int >= 1)
    - top_k: result count (int >= 1)
    - min_groups: group overlap threshold (int >= 1)
    - n_jobs: worker thread count (int >= 1)
    - groups: list[list[str]], canonical seed ids; empty for resolve/search
    - mode: "budget" | "exact"
    - frontier_cap: (int >= 1)
    - ensemble: bool
    - ensemble_runs: (int >= 1)
    - floor_prob: (float in (0, 1))
    - temperature: (float > 0)
    - alpha: (float in [0, 1])
    - seed: Optional[int]
    - cap: (int >= 1)
    - iters: (int >= 1, refine only)
    - strategy: "intersection" | "bidir" | "walk" | "refine"
    - verbose: bool
    - resolve_ids: Optional[list[str]], populated iff --resolve given
    - search_queries: Optional[list[str]], populated iff --search given

    invariant: exactly one of (resolve_ids, search_queries, groups)
    is non-empty; the other two are None or []. dispatch on which is
    non-empty.
    """
    mailto: str
    depth: int
    top_k: int
    min_groups: int
    n_jobs: int
    groups: list[list[str]]
    mode: str
    frontier_cap: int
    ensemble: bool
    ensemble_runs: int
    floor_prob: float
    temperature: float
    alpha: float
    seed: Optional[int]
    cap: int
    iters: int
    strategy: str
    verbose: bool
    resolve_ids: Optional[list[str]] = None
    search_queries: Optional[list[str]] = None


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


def _build_parser() -> argparse.ArgumentParser:
    """Construct and return the argument parser with all current flags intact.

    returns parser without calling parse_args().
    """
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
        choices=["intersection", "bidir", "walk", "refine"],
        default="intersection",
        help="bridging strategy: 'intersection' (default), 'bidir' (bidirectional), 'walk' (semantic-ensemble), or 'refine' (iterative refinement heuristic over walk)",
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
        help="bidir budget-mode per-step frontier cap, per pairwise engine: the compute budget for each search step (default 50). ignored in exact mode and for the intersection strategy",
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        default=False,
        help="walk/refine: run a randomized ensemble (default: single deterministic run)",
    )
    parser.add_argument(
        "-M",
        "--ensemble-runs",
        type=int,
        default=5,
        help="walk/refine: number of ensemble runs (default 5; used only with --ensemble)",
    )
    parser.add_argument(
        "-p",
        "--floor-prob",
        type=float,
        default=0.15,
        help="walk/refine: floor reserve probability for stochastic pruning (default 0.15)",
    )
    parser.add_argument(
        "-T",
        "--temperature",
        type=float,
        default=0.1,
        help="walk/refine: pruning sigmoid temperature (default 0.1)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="walk: structural-vs-embedding fusion weight (default 0.7; 1.0=pure structural)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="walk/refine: RNG master seed for reproducibility (default: unseeded)",
    )
    parser.add_argument(
        "--cap",
        type=int,
        default=60,
        help="walk: per-hop per-engine frontier cap; the embedding prune keeps the top --cap most-promising nodes -- raise it to widen the search (default 60)",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=3,
        help="refine: number of refinement rounds (each round runs the pairwise search; round 1 is the original groups) (default 3)",
    )

    return parser


def _validate_config(args: argparse.Namespace) -> Config:
    """Parse and validate args; return immutable Config.

    centralize all validation into a declarative loop. validation table:
    list of (condition, error_msg) tuples. iterate, check each condition;
    if any fail, raise ValueError(error_msg).

    handle groups parsing, strategy-specific group-count validation, and
    min_groups finalization before returning Config.
    """
    # common validations (all strategies)
    validations = [
        (args.mailto and args.mailto.strip(), "mailto is required and non-empty"),
        (args.depth >= 1, "depth must be >= 1"),
        (args.top_k >= 1, "top-k must be >= 1"),
        (args.n_jobs >= 1, "n-jobs must be >= 1"),
        (args.frontier_cap >= 1, "frontier-cap must be >= 1"),
        (args.mode in ["budget", "exact"], f"mode must be 'budget' or 'exact', got {args.mode}"),

        # walk-specific
        (args.strategy != "walk" or (0 < args.floor_prob < 1),
         f"floor-prob must be in (0, 1); got {args.floor_prob}"),
        (args.strategy != "walk" or args.temperature > 0,
         f"temperature must be > 0; got {args.temperature}"),
        (args.strategy != "walk" or args.ensemble_runs >= 1,
         f"ensemble-runs must be >= 1; got {args.ensemble_runs}"),
        (args.strategy != "walk" or (0 <= args.alpha <= 1),
         f"alpha must be in [0, 1]; got {args.alpha}"),

        # refine-specific
        (args.strategy != "refine" or args.iters >= 1, "iters must be >= 1"),
        (args.strategy != "refine" or (0 < args.floor_prob < 1),
         f"floor-prob must be in (0, 1); got {args.floor_prob}"),
        (args.strategy != "refine" or args.temperature > 0,
         f"temperature must be > 0; got {args.temperature}"),
        (args.strategy != "refine" or args.ensemble_runs >= 1,
         f"ensemble-runs must be >= 1; got {args.ensemble_runs}"),

        # cap (walk/refine)
        (args.strategy not in ["walk", "refine"] or args.cap >= 1, "cap must be >= 1"),
    ]

    for condition, msg in validations:
        if not condition:
            raise ValueError(msg)

    # handle resolve/search modes (groups are empty)
    if args.resolve:
        resolve_ids = args.resolve
        search_queries = None
        groups = []
    elif args.search:
        resolve_ids = None
        search_queries = args.search
        groups = []
    else:
        # main mode: parse groups or use demo
        resolve_ids = None
        search_queries = None

        if args.group is None:
            groups = DEMO_GROUPS
            print("Using demo groups (no --group passed).", file=sys.stderr)
        else:
            groups = []
            for group_str in args.group:
                # split on commas, strip whitespace
                ids = [id.strip() for id in group_str.split(",")]
                # check for empty groups
                if not ids or all(not id for id in ids):
                    raise ValueError("empty group")
                # filter empty strings
                ids = [id for id in ids if id]
                groups.append(ids)

        # strategy-specific group-count validation
        if args.strategy == "bidir":
            if len(groups) < 2:
                raise ValueError("bidir strategy requires >= 2 groups; got 1")
        if args.strategy == "refine":
            if len(groups) < 3:
                raise ValueError("refine strategy requires >= 3 groups")

    # finalize min_groups
    min_groups = args.min_groups if args.min_groups is not None else len(groups)

    return Config(
        mailto=args.mailto,
        depth=args.depth,
        top_k=args.top_k,
        min_groups=min_groups,
        n_jobs=args.n_jobs,
        groups=groups,
        mode=args.mode,
        frontier_cap=args.frontier_cap,
        ensemble=args.ensemble,
        ensemble_runs=args.ensemble_runs,
        floor_prob=args.floor_prob,
        temperature=args.temperature,
        alpha=args.alpha,
        seed=args.seed,
        cap=args.cap,
        iters=args.iters,
        strategy=args.strategy,
        verbose=args.verbose,
        resolve_ids=resolve_ids,
        search_queries=search_queries,
    )


def _strategy_intersection(cfg: Config, oracle: OpenAlex) -> list[dict]:
    """Run intersection strategy.

    call find_bridges(oracle, groups, depth, min_groups, top_k, n_jobs, verbose).
    """
    results = find_bridges(
        oracle=oracle,
        groups=cfg.groups,
        depth=cfg.depth,
        min_groups=cfg.min_groups,
        top_k=cfg.top_k,
        n_jobs=cfg.n_jobs,
        verbose=cfg.verbose,
    )
    return results


def _strategy_bidir(cfg: Config, oracle: OpenAlex) -> list[dict]:
    """Run bidir (2-way) or bidir_nway (N-way) strategy.

    if len(groups) == 2: use find_bridges_bidir.
    else (>= 3): use find_bridges_bidir_nway.
    """
    if len(cfg.groups) == 2:
        results = find_bridges_bidir(
            oracle=oracle,
            group_a=cfg.groups[0],
            group_b=cfg.groups[1],
            mode=cfg.mode,
            max_depth=cfg.depth,
            frontier_cap=cfg.frontier_cap,
            top_k=cfg.top_k,
            n_jobs=cfg.n_jobs,
            verbose=cfg.verbose,
        )
    else:
        results = find_bridges_bidir_nway(
            oracle=oracle,
            groups=cfg.groups,
            min_groups=cfg.min_groups,
            mode=cfg.mode,
            max_depth=cfg.depth,
            frontier_cap=cfg.frontier_cap,
            top_k=cfg.top_k,
            n_jobs=cfg.n_jobs,
            verbose=cfg.verbose,
        )
    return results


def _strategy_walk(cfg: Config, oracle: OpenAlex) -> list[dict]:
    """Run walk strategy.

    call find_bridges_walk with all ensemble/embedding params.
    """
    results = find_bridges_walk(
        oracle=oracle,
        groups=cfg.groups,
        depth=cfg.depth,
        min_groups=cfg.min_groups,
        top_k=cfg.top_k,
        cap=cfg.cap,
        ensemble=cfg.ensemble,
        M=cfg.ensemble_runs,
        p=cfg.floor_prob,
        T=cfg.temperature,
        alpha=cfg.alpha,
        seed=cfg.seed,
        n_jobs=cfg.n_jobs,
        verbose=cfg.verbose,
    )
    return results


def _strategy_refine(cfg: Config, oracle: OpenAlex) -> list[dict]:
    """Run refine strategy.

    call find_bridges_refine with all params including iters.
    """
    results = find_bridges_refine(
        oracle=oracle,
        groups=cfg.groups,
        depth=cfg.depth,
        iters=cfg.iters,
        top_k=cfg.top_k,
        min_groups=cfg.min_groups,
        cap=cfg.cap,
        ensemble=cfg.ensemble,
        M=cfg.ensemble_runs,
        p=cfg.floor_prob,
        T=cfg.temperature,
        seed=cfg.seed,
        embedder=None,
        n_jobs=cfg.n_jobs,
        verbose=cfg.verbose,
    )
    return results


def _run_strategy(cfg: Config, oracle: OpenAlex) -> list[dict]:
    """Dispatch to strategy via table.

    build a pure dict: {strategy_name: builder_fn}. each builder_fn takes
    (cfg, oracle) and returns list[dict] (records).
    """
    dispatch = {
        "intersection": _strategy_intersection,
        "bidir": _strategy_bidir,
        "walk": _strategy_walk,
        "refine": _strategy_refine,
    }

    builder = dispatch[cfg.strategy]
    return builder(cfg, oracle)


def _format_results(cfg: Config, results: list[dict]) -> str:
    """Format results into header + records string.

    build the full output string (header + records), pure. receives cfg and
    results; returns formatted text string. safe field extraction via .get()
    with defaults. result ordering: strategies return pre-sorted by groups_hit,
    score, cited_by_count; this function prints in that order (no re-sort).
    """
    # construct header
    strategy_info = f"Strategy: {cfg.strategy}"
    if cfg.strategy == "bidir":
        strategy_info += f", Mode: {cfg.mode}, Frontier-cap: {cfg.frontier_cap}"
    if cfg.strategy == "refine":
        strategy_info += f", Iters: {cfg.iters}"

    header = (
        "Citation bridge finder (citetools)\n"
        "==================================\n"
        f"Groups: {len(cfg.groups)}, Depth: {cfg.depth}, Min-groups: {cfg.min_groups}, {strategy_info}, N-jobs: {cfg.n_jobs}\n"
        f"Top-K: {cfg.top_k}\n"
        "\n"
        "Note: depth-2 runs make many API calls; concurrency is rate-limited by OpenAlex.\n"
        "Ensure mailto is a real, monitored email address.\n"
        "\n"
        "Results (sorted by groups_hit desc, score asc, cited_by_count desc):\n"
    )

    # format records
    lines = [header]
    for r in results:
        # safe extraction: handle missing/None fields
        score = r.get("score", "—")
        groups_hit = r.get("groups_hit", 0)
        year = r.get("year") or "—"
        cited_by = r.get("cited_by_count") or 0
        degree = r.get("degree") or 0
        title = r.get("title") or "[no title]"
        doi = r.get("doi") or "[no DOI]"

        lines.append(
            f"{score:>3}  hit={groups_hit}  {year}  cites={cited_by:>5}  "
            f"degree={degree:>3}  {title[:80]}"
        )
        lines.append(f"     {doi}")
        lines.append("")  # blank line between records

    return "\n".join(lines)


def _mode_resolve(cfg: Config) -> None:
    """Resolve IDs to W-ids and exit.

    open OpenAlex client, iterate --resolve ids, call wid() + work(),
    print result, close client in finally.
    """
    client = OpenAlex(mailto=cfg.mailto)
    try:
        for src in cfg.resolve_ids:
            try:
                wid = client.wid(src)
                # collapse whitespace in title
                title = " ".join((client.work(src).get("title") or "[no title]").split())
                print(f"{wid}  {title[:80]}")
            except Exception as e:
                print(f"resolve failed for {src}: {e}", file=sys.stderr)
    finally:
        client.close()


def _mode_search(cfg: Config) -> None:
    """Search works by title and exit.

    open OpenAlex client, iterate --search queries, call search(),
    print candidates, close client in finally.
    """
    client = OpenAlex(mailto=cfg.mailto)
    try:
        for q in cfg.search_queries:
            print(f"search: {q}")
            hits = client.search(q)
            if not hits:
                print("  (no matches)")
            for w in hits:
                wid = w["id"].rsplit("/", 1)[-1].upper()
                year = w.get("publication_year") or "????"
                cites = w.get("cited_by_count") or 0
                # collapse whitespace in title
                title = " ".join((w.get("title") or "[no title]").split())
                print(f"  {wid}  {year}  cites={cites:>6}  {title[:75]}")
            print()
    finally:
        client.close()


def _mode_main(cfg: Config) -> None:
    """Run bridging strategy and print results.

    open OpenAlex client, dispatch to strategy via table, format results,
    print, close client in finally. all exceptions propagate (let caller
    handle). strategy dispatch happens here.
    """
    client = OpenAlex(mailto=cfg.mailto)
    try:
        results = _run_strategy(cfg, client)
        text = _format_results(cfg, results)
        print(text, file=sys.stdout)
    finally:
        client.close()


def main() -> None:
    """CLI entry point: parse, validate, run, format, print.

    three-phase execution:
    1. setup logging (coarse progress to stderr).
    2. parse args -> validate -> Config.
    3. dispatch on resolve/search/main mode.
    4. for resolve/search: open client, run mode, close, return.
    5. for main: open client, run strategy, format, print, close.

    all three paths close the client in a finally block (fixes the
    current main-path leak).
    """
    # setup logging
    _setup_logging()

    # parse args
    parser = _build_parser()
    args = parser.parse_args()

    # validate
    cfg = _validate_config(args)

    # dispatch on mode (resolve/search/main)
    if cfg.resolve_ids:
        _mode_resolve(cfg)
        return

    if cfg.search_queries:
        _mode_search(cfg)
        return

    # main mode: run strategy
    _mode_main(cfg)


if __name__ == "__main__":
    try:
        main()
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)
