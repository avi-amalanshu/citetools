# citetools

a collection of thin lit review tools for specific scenarios i run into that aren't
covered by semantic scholar or whatever.

## Citation bridge finder

give it N groups of seed papers: each group standing for one research idea or area:
and it returns papers that sit in the citation neighbourhood of *every* group.

it runs on the OpenAlex API: free, no key, you just supply a `mailto` so they can
reach you if your script misbehaves (their "polite pool"). (Watch out for rate limits.)

treat it as an exploratory / hypothesis-generation tool: the output is candidates to
inspect, not ground truth. see [limitations](#limitations).

## Strategies

citetools ships two interchangeable *core* strategies — `intersection` and `bidir` —
plus two experimental **cousins**, `walk` and `refine`. All take the same input (N
groups of seed papers) and return a ranked list of bridge papers; they differ in
*how* they walk the citation graph, and the cousins also differ in *how they rank*.
Pick one with `--strategy` on the CLI, or call the matching function from the library.

`intersection` and `bidir` are interchangeable peers — same scoring, same
dependencies. `walk` and `refine` are *cousins*, not drop-in siblings: both build on
`bidir`'s search engine and add an optional local-embedding dependency, a semantic
ranking, and randomised ensembling. `walk` does this in a single pass; `refine` runs
several re-seeding rounds on top. They are more experimental tools, each with its own
philosophy (see [`walk`](#walk--semantic-ensemble) and
[`refine`](#refine--iterative-path-bootstrap) below).

### `intersection` — exhaustive directed search

Grows a citation neighbourhood of radius `--depth` around every group's seeds, then
scores every non-seed paper by its summed shortest-path distance to each group. Keeps
the papers reachable from enough groups and ranks them. It is a *true* N-way
intersection: within `--depth` hops, nothing is missed. Cost grows roughly
exponentially in `--depth`. Each result carries a `degree` field (the paper's degree
in its group graph, a hub indicator).

### `bidir` — bidirectional meet-in-the-middle

A lazy search: instead of pre-growing a fixed-radius ball, it expands a frontier out
from each group and stops where the frontiers meet. It never materialises more of the
graph than it needs. Two modes:

- **`budget`** (default): caps each frontier to `--frontier-cap` papers per step
  (default **50**; for N>=3 this cap applies *per pairwise engine*) and stops at the
  first meeting. A fast, bounded, approximate answer — the search is deliberately
  constrained to a sensible compute budget.
- **`exact`** (opt-in, `--mode exact`): no cap; runs until it can prove it has the
  genuinely shortest connecting bridges. Higher and unbounded cost.

For **N=2** this is a direct meet-in-the-middle search and the natural cheaper choice.
For **N>=3** it decomposes the problem into all C(N,2) pairwise searches and
aggregates the results: a heuristic that approximates N-way connectivity rather than
computing a true intersection (a paper can rank well by bridging several pairs without
being central to all groups at once). N>=3 results also carry a `recurrence` field:
how many group-pairs the paper bridged, a diagnostic.

### `walk` — semantic-ensemble

A **cousin of `bidir`**: `walk` reuses `bidir`'s meet-in-the-middle engine, but
replaces its *blunt* frontier cap with a *smart* one. At each step `bidir` keeps an
arbitrary slice of the frontier; `walk` keeps the `--cap` papers whose **titles** are
most semantically similar (local sentence-transformer embedding, cosine) to the group
it is steering toward. The compute budget is spent on papers heading *toward* a
bridge rather than on an arbitrary slice — so the search can be pushed deeper for the
same budget. On top of that, `walk`:

- **fuses** the structural distance with title-embedding similarity into the final
  `score` (weight `--alpha`; `1.0` = pure structural, `0.0` = pure semantic), so the
  ranking reflects *conceptual* proximity, not just citation-hop proximity. It also
  drops `cited_by_count` from the ranking — `walk` deliberately avoids the hub bias.
- runs an optional **ensemble** (`--ensemble`, `-M` stochastic runs): each run prunes
  the frontier randomly — biased by the embedding signal, with a floor keep-probability
  (`-p`) so every paper keeps a chance. Per-group distances are aggregated as the
  *minimum over runs*; a `recurrence` field counts how many runs found the bridge.

`walk` needs an optional dependency — a local embedding model (see
[Environment setup](#environment-setup)); it never sends paper data off-machine. It
is **experimental**: on closely-related groups it surfaces sensible bridges, but on
very distant groups a thin steered frontier can still miss a path (raise `--cap`, or
use `--ensemble`). Treat its output as a hypothesis, more so than the other two.

### `refine` — iterative path bootstrap

The most experimental strategy, and a **cousin of `walk`** (so, of `bidir`): where the
others find bridges in a single pass, `refine` runs several rounds and *re-seeds*
between them. It needs **N >= 3** groups.

Each round it runs `walk`-style embedding-steered searches between the current groups,
reconstructs the *paths* — the chains of papers — along which pairs met, and relaxes a
running table of best-known per-group distances. It then takes the papers lying on
those paths whose titles are most semantically similar to the groups still poorly
covered, and uses them as the refined seeds for the next round. So the search
bootstraps a connecting structure: each round walks a little further along the most
promising discovered paths, steered toward the areas not yet linked. It stops once
every group is reached, or after `--iters` rounds.

`refine` reuses `walk`'s engine and ensemble machinery, so it takes the same `--cap`,
`--ensemble` / `-M` / `-p` / `-T` and `--seed` knobs, plus `--iters` (round count).
Like `walk` it needs the optional embedding dependency and keeps all paper data
on-machine; results carry the usual fields plus `recurrence`.

Reach for `refine` when N >= 3 areas do not connect in a single pass and you want the
search to *iteratively* hunt a path rather than grow a fixed-radius ball — but it is
the least predictable of the four. Treat its output as a hypothesis first of all.

### which to use

| you want... | use |
|---|---|
| the complete, true N-way intersection within `--depth` | `intersection` |
| a fast N=2 connector | `bidir` (budget) |
| the provably-shortest bridge(s), cost no object | `bidir --mode exact` |
| N>=3 and completeness matters | `intersection` |
| N>=3 quick exploration | `bidir` (budget) |
| a deeper search steered toward a bridge, ranked by conceptual similarity | `walk` |
| robustness on a hard case — re-roll the steered search | `walk --ensemble` |
| N>=3 areas that don't connect in one pass — iteratively hunt a path | `refine` |

`intersection` is the default strategy. `walk` and `refine` require the optional
embedding dependency; `refine` needs N>=3 groups.

## Scoring

`intersection` and `bidir` score a candidate bridge the same way:

- **per-group distance**: the shortest undirected path, in citation hops, from the
  nearest seed of a group to the candidate. one citation link is one hop; edge
  direction is ignored when measuring distance.
- **`groups_hit`**: the number of groups the candidate is reachable from.
- **`score`**: the sum of per-group distances over the groups it reaches. lower score
  = closer to every group at once, i.e. a more central bridge.
- a candidate qualifies only if `groups_hit` is at least `--min-groups` (default: all
  groups). seed papers are never their own bridges and are excluded.

Results are ranked by `groups_hit` descending, then `score` ascending, then
`cited_by_count` descending. Each result is a dict:
`id, score, groups_hit, dists, title, year, doi, cited_by_count` — plus `degree` for
`intersection`, and `recurrence` for `bidir` N>=3.

Example, two groups: a paper one hop from each seed scores 2; a paper two hops from
each scores 4 and ranks below it; a paper reachable from only one group has
`groups_hit` 1 and is dropped when `--min-groups` is 2.

`walk` scores differently (see its [section](#walk--semantic-ensemble)): its `score`
*fuses* the structural distance with title-embedding similarity (`--alpha`), it adds a
`recurrence` field, and it drops `cited_by_count` from the ranking (still reported,
just not ranked on).

## Environment setup

```
pip install -r requirements.txt    # requests, networkx, joblib
```

Python 3.10+. That covers `intersection` and `bidir`.

`walk` additionally needs a local embedding model:

```
pip install -r requirements-walk.txt    # sentence-transformers (+ torch)
```

This is a heavier install (~600 MB-1 GB; it pulls in torch). It is kept in a separate
file so `intersection` / `bidir` users never pay for it — `walk` imports it lazily
and prints a clear install hint if it is missing. The model downloads once from
HuggingFace, then runs fully offline; no paper data ever leaves your machine.

## Usage

### CLI

```
# intersection (default strategy)
python -m citetools --mailto you@example.com \
    --group W2626778328 \
    --group W2519887557 \
    --depth 2 --top-k 15

# bidirectional, two groups
python -m citetools --mailto you@example.com --strategy bidir \
    --group W2626778328 --group W2519887557 --depth 2

# bidirectional, exact mode
python -m citetools --mailto you@example.com --strategy bidir --mode exact \
    --group W2626778328 --group W2519887557 --depth 2

# walk: deeper, embedding-steered search (needs requirements-walk.txt)
python -m citetools --mailto you@example.com --strategy walk \
    --group W2626778328 --group W2519887557 --depth 4 --cap 120

# walk with a randomised ensemble
python -m citetools --mailto you@example.com --strategy walk --ensemble -M 5 \
    --group W2626778328 --group W2519887557 --depth 4

# refine: iterative multi-round bootstrap (N>=3; needs requirements-walk.txt)
python -m citetools --mailto you@example.com --strategy refine \
    --group W2626778328 --group W2519887557 --group W3177828909 \
    --depth 2 --iters 3
```

`--group` is repeated once per research area; each takes comma-separated seed ids
(OpenAlex W-ids, DOIs, OpenAlex URLs, or arxiv abs/pdf links). with no `--group`, a
built-in demo set runs.

The first run bridges two ML areas: "Attention Is All You Need" (`W2626778328`) and
the graph-convolutional-networks paper (`W2519887557`), and prints, among others:

```
  2  hit=2  2016  cites=220033  degree=  0  Deep Residual Learning for Image Recognition
       https://doi.org/10.1109/cvpr.2016.90
```

| flag | meaning | default |
|---|---|---|
| `--mailto` | email for the OpenAlex polite pool (required) | : |
| `--strategy` | `intersection`, `bidir`, `walk`, or `refine` | `intersection` |
| `--mode` | `budget` or `exact`; applies to `bidir` only, ignored otherwise | `budget` |
| `--frontier-cap` | `bidir` budget-mode per-step frontier cap, *per pairwise engine* — a *blunt* compute-budget guard; ignored in exact mode and by other strategies | 50 |
| `--cap` | `walk`/`refine` per-step per-engine frontier cap — the *smart*, embedding-steered analogue of `--frontier-cap` (see [below](#--frontier-cap-vs---cap)); ignored by other strategies | 60 |
| `--iters` | `refine`: number of re-seeding rounds; ignored by other strategies | 3 |
| `--ensemble` | `walk`/`refine`: run a randomised ensemble instead of one deterministic pass | off |
| `-M`, `--ensemble-runs` | `walk`/`refine`: ensemble run count (used with `--ensemble`) | 5 |
| `-p`, `--floor-prob` | `walk`/`refine`: floor keep-probability in stochastic pruning | 0.15 |
| `-T`, `--temperature` | `walk`/`refine`: pruning-sigmoid temperature | 0.1 |
| `--alpha` | `walk`: structural-vs-embedding fusion weight (1.0 = pure structural) | 0.7 |
| `--seed` | `walk`/`refine`: RNG seed for a reproducible ensemble | unseeded |
| `--group` | one research area; comma-separated seed ids; repeat per area | demo set |
| `--depth` | citation hops expanded (per seed for `intersection`, per side for `bidir`/`walk`/`refine`) | 2 |
| `--top-k` | number of bridges to return | 25 |
| `--min-groups` | how many groups a candidate must reach to qualify | all groups |
| `--n-jobs` | worker threads for concurrent fetching; 1 = serial | 8 |
| `--resolve` | resolve an id/link to its W-id and exit; repeatable | : |
| `--search` | search works by title; print candidate W-ids and exit; repeatable | : |
| `-v`, `--verbose` | also stream live per-fetch progress to stderr | off |

A `--depth 2` run makes many API calls; concurrency speeds this up (bounded by the
~10 req/s polite pool rate). `bidir --mode budget` is typically faster than
`intersection` at the same depth because it stops at the first meeting instead of
fully growing every neighbourhood.

### `--frontier-cap` vs `--cap`

Both bound how many papers a search step expands — but they are *different knobs for
different strategies*, and they differ in *how* they choose what to keep:

- **`--frontier-cap`** (`bidir`) is a **blunt** cap: when a step discovers more papers
  than the cap, it keeps an *arbitrary* subset (by W-id order). Purely a compute-budget
  guard — it has no notion of which papers are promising.
- **`--cap`** (`walk`) is a **smart** cap: `walk` keeps the `--cap` papers whose titles
  are most semantically similar to the group it is steering toward. Same job — bound
  the step — but the kept subset is *chosen*, not arbitrary.

So raising `--frontier-cap` widens `bidir`'s *arbitrary* slice; raising `--cap` widens
`walk`'s *steered* slice. Each flag is ignored by the strategy it does not belong to.

### Progress logging

Each strategy logs coarse progress to stderr as the run proceeds, so a slow run is
easy to tell from a stuck one. The result output on stdout is unaffected, so piping
still works. Add `-v` / `--verbose` for joblib's live per-fetch counts on top
(`Done 16 out of 28 | elapsed: ...`), plus a `fetched <W-id>` line as each node's
neighbours arrive (`bidir` and `walk`).

**`intersection`** logs the group it is on, then each seed's breadth-first growth hop
by hop (neighbour fetches, then metadata fetches), then the scoring pass:

```
find_bridges: 2 group(s), depth=2, n_jobs=8, min_groups=2
group 1/2: expanding 1 seed(s)
grow W2626778328: depth=2
  W2626778328 hop 1/2: fetching neighbours of 1 node(s)
  W2626778328 hop 1/2: fetching metadata for 168 new node(s)
  W2626778328 hop 2/2: fetching neighbours of 50 node(s)
  W2626778328 hop 2/2: fetching metadata for 4631 new node(s)
  W2626778328: backfilling metadata for 27 node(s)
group 2/2: expanding 1 seed(s)
grow W2519887557: depth=2
  ...
scoring 8124 candidate node(s) across 2 group(s)
find_bridges: done, 25 bridge(s)
```

**`bidir`, N=2** logs the meet-in-the-middle search hop by hop, with the running
count of meeting (bridge) nodes:

```
bidir: 1 + 1 seeds, mode=budget, max_depth=2
  hop 1: fetching neighbours of 1 node(s)
  hop 1: 0 meeting node(s) so far
  hop 2: fetching neighbours of 1 node(s)
  hop 2: 0 meeting node(s) so far
  hop 3: fetching neighbours of 168 node(s)
  hop 3: 14 meeting node(s) so far
bidir: done, 5 bridge(s)
```

**`bidir`, N>=3** logs each round-robin round across the C(N,2) pairwise engines —
how many engines are still live and how many nodes that round fetches:

```
bidir n-way: 3 groups, 3 pairs, mode=budget, max_depth=2
  round 1: 3 live engine(s), fetching 3 node(s)
  round 2: 3 live engine(s), fetching 191 node(s)
  round 3: 2 live engine(s), fetching 437 node(s)
bidir n-way: done, 1 candidate(s)
```

**`walk`** logs its run plan, then each ensemble run's hops. With `--ensemble` the
runs execute concurrently, so their log lines interleave — the `run N` prefix tells
them apart:

```
walk: 2 group(s), 1 pairs, ensemble=True (M=3), depth=4
walk: driving 3 run(s), 2 i/o thread(s)/run
  run 0 hop 1: 1 live engine(s), 1 frontier node(s), 1 title(s) fetched
  run 2 hop 1: 1 live engine(s), 1 frontier node(s), 1 title(s) fetched
  run 1 hop 1: 1 live engine(s), 1 frontier node(s), 1 title(s) fetched
  run 0 hop 1: fetching 1 kept frontier node(s)
  run 2 hop 1: fetching 1 kept frontier node(s)
  run 0 hop 2: 1 live engine(s), 92 frontier node(s), 92 title(s) fetched
  ...
walk: done, 25 bridge(s)
```

Without `--ensemble` it is a single run (`run 0`) and the lines stay ordered. `walk`
logs one extra figure per hop — `title(s) fetched` — because it fetches and embeds
frontier titles *before* pruning, to steer the prune.

(Counts above are illustrative.) `degree` prints as `0` for `bidir` results — it is
an `intersection`-only field.

### getting a paper into `--group`

`--group` accepts the following ID forms:
- an OpenAlex W-id
- a DOI (`doi:10.xxxx/yyyy`)
- an OpenAlex URL
- an arxiv abs/pdf link

Most of the time you just paste the link:

```
python -m citetools --mailto you@example.com \
    --group https://arxiv.org/abs/2010.11929 \
    --group W2519887557
```

arxiv links resolve via the arxiv doi `10.48550/arxiv.<id>`. That works only when
OpenAlex has the paper registered under that doi: reliable for recent papers,
patchier for older arxiv-only ones. When a link can't be resolved the run fails
loudly with a clear error rather than guessing a wrong match.

If that happens (or if you only know a title), find the W-id by title with
`--search`, then pass the W-id you pick:

```
# step 1: find the paper
python -m citetools --mailto you@example.com \
    --search "relational inductive biases deep learning graph networks"
#   -> W2805516822  2018  cites=2401  Relational inductive biases, deep learning, ...

# step 2: run the finder with the W-id from step 1
python -m citetools --mailto you@example.com \
    --group W2805516822 --group W2519887557
```

`--search` prints the top title-match candidates (`W-id  year  cites  title`); you
pick the right one by eye. `--resolve` is the converse: it takes one id/link and
prints the single W-id it maps to, handy to confirm a link before a long run:

```
python -m citetools --mailto you@example.com --resolve https://arxiv.org/abs/2010.11929
```

Both `--search` and `--resolve` are repeatable and short-circuit the run: they print
and exit, never touching the finder (no `--group` needed). If both are passed,
`--search` runs.

### Library

```python
from citetools import (
    OpenAlex,
    find_bridges,             # intersection
    find_bridges_bidir,       # bidir, N=2
    find_bridges_bidir_nway,  # bidir, N>=3
    find_bridges_walk,        # walk
)

# the OpenAlex client is the oracle the strategies take directly
oracle = OpenAlex(mailto="you@example.com")

# intersection, any N
a = find_bridges(oracle, [["W2626778328"], ["W2519887557"]], depth=2)

# bidir, two groups
b = find_bridges_bidir(oracle, ["W2626778328"], ["W2519887557"], mode="budget")

# bidir, three or more groups
c = find_bridges_bidir_nway(
    oracle, [["W2626778328"], ["W2519887557"], ["W3177828909"]], mode="budget",
)

# walk, any N (needs requirements-walk.txt); optional ensemble
d = find_bridges_walk(
    oracle, [["W2626778328"], ["W2519887557"]], depth=4, cap=120,
    ensemble=True, M=5,
)

for x in b:
    print(x["score"], x["title"])
```

`find_bridges_bidir` takes the two groups as separate positional arguments;
`find_bridges`, `find_bridges_bidir_nway`, and `find_bridges_walk` take a list of
groups. All return a ranked `list[dict]` with the fields described under
[Scoring](#scoring) (`find_bridges_walk` adds a fused `score` and a `recurrence`
field).

## How it works

The codebase is split into a pure core and a single impure i/o shell: every search
engine is a deterministic pure generator, and one runtime is the only thing that
performs i/o.

- **`citetools/io/`** — the impure shell, the only place effects and mutable caches
  live:
  - `openalex.py` — the `OpenAlex` client: a cached wrapper over the OpenAlex REST
    API (id normalisation, single + batched work fetches, incoming-citation lookups,
    and `nbrs(node)` neighbour lists). Holds a shared rate limiter (keeps the
    aggregate request rate under the ~10 req/s polite-pool ceiling) and per-thread
    HTTP sessions, so it is safe under concurrent fetches.
  - `parallel.py` — the concurrent fetch helper (`fetch_all`): fetches many nodes'
    neighbour lists in parallel over a thread pool.
  - `models.py` — the optional local title-embedding model loader and cache, used
    only by `walk`.
  - `runtime.py` — the interpreter that drives the pure engines, performing all
    fetching and embedding: `run` for one engine, the lockstep-union `run_ensemble`
    for many.
- **`citetools/core/`** — the pure core: no i/o, no mutation of inputs.
  - `parse.py` — id canonicalisation and OpenAlex-json shaping.
  - `score.py` — distance aggregation, ranking, and result-record shaping, shared by
    every strategy.
  - `embed.py` — pure embedding math (cosine similarity).
  - `engines/` — the search strategies as pure Python generators over a shared
    `protocol.py` (`Req` / `Resp` / `SearchResult`). An engine `yield`s the batch of
    nodes it needs and receives their neighbours back; it performs no i/o and keeps
    no state outside its own frame. `bidir.py` is the meet-in-the-middle engine;
    `walk.py` and `refine.py` build on it; `intersection.py` is the exhaustive N-way
    engine.
- **`citetools/strategies.py`** — the public `find_bridges*` functions: thin impure
  wrappers that build the right engine(s), drive them through the runtime, and shape
  the result via `core.score`.
- **`citetools/cli/`** — the command-line interface.

This "pure modulo i/o" split makes the engines deterministic and offline-testable:
given the same seeds and the same sequence of fetched neighbours, an engine emits the
same requests and the same result.

Supporting files: `citetools/errors.py` (`OpenAlexError` / `BadId`),
`citetools/__init__.py` (the public API), `citetools/__main__.py` (the CLI shim).

## Concurrency

**I/O within a step (every strategy).** Each strategy parallelizes the I/O of a search
step: that step's frontier nodes have their neighbour lists fetched concurrently over
a thread pool (`--n-jobs`, default 8). The steps themselves stay sequential — each one
needs the previous step's results.

**Ensemble runs, lockstep-batched (`walk`).** `walk --ensemble` runs `M` independent
stochastic runs. The runtime drives them in lockstep: each round it unions every live
run's frontier into one concurrent fetch over the full `--n-jobs` pool, then hands
each run its slice. So all `M` runs share one thread pool (no `n_jobs // M` split),
one OpenAlex cache (a paper fetched for one run is free for the rest), and one rate
limiter — and a node needed by several runs in the same round is fetched once. A
single `walk` run (no `--ensemble`) is the same machinery driving one run.

**Shared rate limiter.** One rate limiter, shared across every thread — and, for
`walk`, every run — holds the aggregate request rate within the OpenAlex polite pool's
ceiling (~10 req/s), regardless of thread count. When OpenAlex rate-limits a request
(HTTP 429) it applies a brief *fleet-wide* cooldown — all threads back off together —
then retries, so a busy server degrades gracefully rather than stalling silently.
Concurrency hides per-request latency, not request volume: it reclaims the rate budget
a serial run leaves idle while stalling on latency, and cannot exceed the polite-pool
cap.

**Determinism.** `intersection`, `bidir`, and `walk` without `--ensemble` are
deterministic and independent of `--n-jobs`: a `--n-jobs 8` result equals a
`--n-jobs 1` result, thanks to fixed total-order tie-breaks in ranking and a pure-CPU
search core whose decisions never depend on fetch timing. A `walk --ensemble` run is
deterministic only with `--seed`: each run draws its own RNG spawned from that seed,
so the outcome is fixed however the run threads interleave; without `--seed` the
ensemble is genuinely random.

## Limitations

- Citation-graph proximity is only a proxy for conceptual proximity!!!
- Highly-cited hub papers (surveys, foundational works) tend to dominate. For
  `intersection`, every result carries a `degree` field so you can spot them; genuine
  bridges usually have low degree.
- Scoring is undirected and additive (sum of per-group distances), so a paper close
  to one area and far from another can tie a balanced bridge: the per-group `dists`
  are returned so you can tell them apart.
- Depth bounds reach. A bridge is discoverable only if it sits within `--depth`
  citation hops of every group's seeds; raising `--depth` reaches further at cost
  that grows roughly exponentially in depth.
- `bidir --mode budget` (the default) is a *bounded approximation*: it caps each
  frontier (`--frontier-cap`, default 50, per pairwise engine) and stops at the first
  meeting, so it can miss a better bridge that a fuller search would find. Raise
  `--frontier-cap` for a wider search, or use `--mode exact` / the `intersection`
  strategy, when completeness matters.
- `bidir` for N>=3 is a pairwise decomposition, not a true intersection: it can
  surface a paper that strongly bridges some group-pairs without being central to all
  groups. The `recurrence` field and the per-group `dists` let you judge this; for a
  guaranteed N-way intersection use `intersection`.
- Depth-2 over genuinely distant fields can return an empty result; relax
  `--min-groups`.
- `walk` is experimental. Its embedding steer is a heuristic, not ground truth: a thin
  steered frontier can still miss a path between very distant groups (raise `--cap` or
  use `--ensemble`), and title-embedding similarity is itself only another proxy for
  conceptual proximity.
- Concurrency gains are modest and rate-capped: `--n-jobs` only reclaims the rate
  budget a serial run wastes stalling on per-request latency; it cannot exceed the
  ~10 req/s polite-pool ceiling.
