# citetools

a collection of thin lit review tools for specific scenarios i run into that aren't
covered by semantic scholar or whatever.

## Citation bridge finder

give it N groups of seed papers: each group standing for one research idea or area:
and it returns papers that sit in the citation neighbourhood of *every* group. those
are candidate "bridge" papers: works that connect the areas you handed it.

- N=2 is a pairwise connector (an analogue of Inciteful's Literature Connector).
- N>=3 connects three or more areas at once.

it runs on the OpenAlex API: free, no key, you just supply a `mailto` so they can
reach you if your script misbehaves (their "polite pool").

treat it as an exploratory / hypothesis-generation tool: the output is candidates to
inspect, not ground truth. see [limitations](#limitations).

## Two strategies

citetools ships two interchangeable bridge-finding strategies. They take the same
input (N groups of seed papers) and return the same kind of ranked list of bridge
papers; they differ in *how* they walk the citation graph. Pick one with
`--strategy` on the CLI, or call the matching function from the library.

### `intersection` — exhaustive

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

- **`budget`** (default): caps each frontier and stops at the first meeting. A fast,
  bounded, approximate answer: the search is deliberately constrained to a sensible
  compute budget.
- **`exact`** (opt-in, `--mode exact`): no cap; runs until it can prove it has the
  genuinely shortest connecting bridges. Higher and unbounded cost.

For **N=2** this is a direct meet-in-the-middle search and the natural cheaper choice.
For **N>=3** it decomposes the problem into all C(N,2) pairwise searches and
aggregates the results: a heuristic that approximates N-way connectivity rather than
computing a true intersection (a paper can rank well by bridging several pairs without
being central to all groups at once). N>=3 results also carry a `recurrence` field:
how many group-pairs the paper bridged, a diagnostic.

### which to use

| you want... | use |
|---|---|
| the complete, true N-way intersection within `--depth` | `intersection` |
| a fast N=2 connector | `bidir` (budget) |
| the provably-shortest bridge(s), cost no object | `bidir --mode exact` |
| N>=3 and completeness matters | `intersection` |
| N>=3 quick exploration | `bidir` (budget) |

`intersection` is the default strategy.

## Scoring

Both strategies score a candidate bridge the same way:

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

## Environment setup

Same for both strategies:

```
pip install -r requirements.txt    # requests, networkx, joblib
```

Python 3.10+.

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
| `--strategy` | `intersection` or `bidir` | `intersection` |
| `--mode` | `budget` or `exact`; applies to `bidir` only, ignored otherwise | `budget` |
| `--group` | one research area; comma-separated seed ids; repeat per area | demo set |
| `--depth` | citation hops expanded (per seed for `intersection`, per side for `bidir`) | 2 |
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

### Progress logging

Both strategies log coarse progress to stderr as the run proceeds, so a slow run is
easy to tell from a stuck one. The result output on stdout is unaffected, so piping
still works. Add `-v` / `--verbose` for joblib's live per-fetch counts on top
(`Done 16 out of 28 | elapsed: ...`).

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
    OpenAlex, CiteGraph,
    find_bridges,             # intersection
    find_bridges_bidir,       # bidir, N=2
    find_bridges_bidir_nway,  # bidir, N>=3
)

client = OpenAlex(mailto="you@example.com")
oracle = CiteGraph(client)

# intersection, any N
a = find_bridges(oracle, [["W2626778328"], ["W2519887557"]], depth=2)

# bidir, two groups
b = find_bridges_bidir(oracle, ["W2626778328"], ["W2519887557"], mode="budget")

# bidir, three or more groups
c = find_bridges_bidir_nway(
    oracle, [["W2626778328"], ["W2519887557"], ["W3177828909"]], mode="budget",
)

for x in b:
    print(x["score"], x["title"])
```

`find_bridges_bidir` takes the two groups as separate positional arguments;
`find_bridges` and `find_bridges_bidir_nway` take a list of groups. All three return a
ranked `list[dict]` with the fields described under [Scoring](#scoring).

## How it works

Layered modules:

- **`citetools/openalex.py`** — L1, the `OpenAlex` client. A thin, cached wrapper over
  the OpenAlex REST API: id normalisation, single + batched work fetches, and
  incoming-citation lookups. Holds a shared rate limiter (keeps the aggregate request
  rate under the ~10 req/s polite-pool ceiling) and a per-thread HTTP session, so it
  is safe under concurrent fetches.
- **`citetools/parallel.py`** — L1.5, the concurrent fetch helper (`fetch_all`):
  fetches many nodes' neighbour lists in parallel over a thread pool.
- **`citetools/graph.py`** — L2, the `CiteGraph` lazy oracle. Exposes `nbrs(node)`
  (lazy, cached neighbour lists) and `grow(seed, depth)` (a breadth-first citation
  neighbourhood as a `networkx` graph). The shared substrate for every strategy.
- **`citetools/strategies/`** — L3, the pluggable strategies:
  - `intersection.py` — `find_bridges`: grows a neighbourhood per group, scores every
    non-seed node by summed shortest-path distance (one multi-source BFS per group),
    keeps the nodes reaching enough groups, ranks them.
  - `bidir.py` — `find_bridges_bidir` / `find_bridges_bidir_nway`: a pure-CPU
    bidirectional search engine plus two aggregators. The engine is a step state
    machine — it asks the aggregator for a frontier's neighbours and advances; the
    aggregator owns all I/O. N>=3 round-robins one engine per group pair.

Supporting files: `citetools/errors.py` (`OpenAlexError` / `BadId`),
`citetools/strategies/__init__.py` and `citetools/__init__.py` (the public API),
`citetools/__main__.py` (the CLI).

## Concurrency

Both strategies parallelize I/O *within* a search step: all of a step's frontier
nodes' neighbour lists are fetched concurrently over a thread pool (`--n-jobs`,
default 8). The steps themselves remain sequential.

A shared rate limiter keeps the aggregate request rate within the OpenAlex polite
pool's ceiling (~10 req/s) regardless of thread count. Concurrency hides per-request
latency, not request volume: it reclaims the rate budget that serial execution leaves
idle while stalling on latency, and cannot exceed the polite-pool cap.

Output is deterministic and independent of `--n-jobs`: results with `--n-jobs 8` are
identical to `--n-jobs 1`, thanks to fixed total-order tie-breaks in ranking and a
pure-CPU search core whose decisions never depend on fetch timing.

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
  frontier and stops at the first meeting, so it can miss a better bridge that a
  fuller search would find. Use `--mode exact`, or the `intersection` strategy, when
  completeness matters.
- `bidir` for N>=3 is a pairwise decomposition, not a true intersection: it can
  surface a paper that strongly bridges some group-pairs without being central to all
  groups. The `recurrence` field and the per-group `dists` let you judge this; for a
  guaranteed N-way intersection use `intersection`.
- Depth-2 over genuinely distant fields can return an empty result; relax
  `--min-groups`.
- Concurrency gains are modest and rate-capped: `--n-jobs` only reclaims the rate
  budget a serial run wastes stalling on per-request latency; it cannot exceed the
  ~10 req/s polite-pool ceiling.
