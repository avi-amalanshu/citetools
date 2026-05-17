# citetools

a collection of thin lit review tools for specific scenarios i run into that aren't
covered by semantic scholar or whatever. 

## Citation bridge finder (v1)

give it N groups of seed papers: each group standing for one research idea or area:
and it returns papers that sit in the citation neighbourhood of *every* group. those
are candidate "bridge" papers: works that connect the areas you handed it.

- N=2 is a pairwise connector (an analogue of Inciteful's Literature Connector).
- N>=3 is a true multi-way intersection.

each result is scored by the sum of citation-graph hop distances to the nearest seed
in each group: lower score = more central bridge. results are ranked by (number of
groups reached, score, citation count).

it runs on the OpenAlex API: free, no key, you just supply a `mailto` so they can
reach you if your script misbehaves (their "polite pool").

treat it as an exploratory / hypothesis-generation tool: the output is candidates to
inspect, not ground truth. see [limitations](#limitations).

## Scoring

for each group, the bridge finder builds a citation-neighbourhood graph around that
group's seeds. every non-seed paper `c` that appears in a group's graph is scored:

- **per-group distance** is the shortest undirected path, in citation hops, from the
  nearest seed of that group to `c`, measured within that group's own graph. one
  citation link is one hop; edge direction is ignored when measuring distance.
- **`groups_hit`** is the number of groups `c` is reachable from.
- **`score`** is the sum of `c`'s per-group distances over the groups it reaches. a
  lower score means `c` is close to every group at once, i.e. a more central bridge.
- `c` is kept as a result only if `groups_hit` is at least `--min-groups` (default:
  all groups, a true intersection). seed papers are never their own bridges and are
  excluded.

results are ranked by `groups_hit` descending, then `score` ascending, then
`cited_by_count` descending. each result also reports `dists` (the per-group distance
breakdown) and `degree` (the paper's degree in its group graph, a hub indicator).

example with two groups: a paper one hop from each seed scores 2; a paper two hops
from each scores 4 and ranks below it; a paper reachable from only one group has
`groups_hit` 1 and is dropped when `--min-groups` is 2.

## Environment setup

```
pip install -r requirements.txt    # requests, networkx, joblib
```

Python 3.10+.

## Usage

### CLI

```
python -m citetools --mailto you@example.com \
    --group W2626778328 \
    --group W2519887557 \
    --depth 2 --top-k 15
```

`--group` is repeated once per research area; each takes comma-separated seed ids
(OpenAlex W-ids, DOIs, or OpenAlex URLs). with no `--group`, a built-in demo set runs.

the run above bridges two ML areas: "Attention Is All You Need" (`W2626778328`) and
the graph-convolutional-networks paper (`W2519887557`): and prints, among others:

```
  3  hit=2  2020  cites=21512  degree=  2  An Image is Worth 16x16 Words: Transformers ...
       https://doi.org/10.48550/arxiv.2010.11929
  3  hit=2  2019  cites= 2816  degree=  7  Heterogeneous Graph Attention Network
       https://doi.org/10.1145/3308558.3313562
  3  hit=2  2018  cites= 2401  degree=  4  Relational inductive biases, deep learning, and graph networks
       https://doi.org/10.48550/arxiv.1806.01261
```

each a genuine attention <-> graph-networks bridge.

| flag | meaning | default |
|---|---|---|
| `--mailto` | email for the OpenAlex polite pool (required) |: |
| `--group` | one research area; comma-separated seed ids; repeat per area | demo set |
| `--depth` | citation hops expanded around each seed | 2 |
| `--top-k` | number of bridges to return | 25 |
| `--min-groups` | how many groups a candidate must reach to qualify | all groups |
| `--n-jobs` | worker threads for concurrent fetching; 1 = serial | 8 |

a `--depth 2` run makes many API calls and takes a few minutes; concurrency speeds this up (bounded by the ~10 req/s polite pool rate).

### Library

```python
from citetools import OpenAlex, CiteGraph, find_bridges

client = OpenAlex(mailto="you@example.com")
oracle = CiteGraph(client)
bridges = find_bridges(oracle, [["W2626778328"], ["W2519887557"]], depth=2)
for b in bridges:
    print(b["score"], b["title"])
```

each bridge is a dict: `id, score, groups_hit, dists, title, year, doi,
cited_by_count, degree`.

## How it works

three layers, each its own module:

- **`citetools/openalex.py`**: L1, the `OpenAlex` client. a thin, cached wrapper
  over the OpenAlex REST API: id normalisation, single + batched work fetches, and
  incoming-citation lookups. the one stateful object (an HTTP session + a response
  cache); every method is a memoised fetch.
- **`citetools/parallel.py`**: L1.5, the concurrent fetch layer (`fetch_all`). uses
  a thread pool and shared rate limiter to fetch multiple nodes' neighbour lists in
  parallel during each BFS hop, without exceeding the OpenAlex polite pool rate limit
  (~10 req/s).
- **`citetools/graph.py`**: L2, the `CiteGraph` lazy oracle. expands a citation
  neighbourhood around a seed by breadth-first search. each hop fetches the frontier's
  neighbours concurrently (via `parallel.py`), only when asked, and caps expansion so
  cost stays bounded. produces a `networkx` directed graph.
- **`citetools/strategies/intersection.py`**: L3, `find_bridges`. grows a
  neighbourhood per group, scores every non-seed node by its summed shortest-path
  distance to each group (computed per group, one multi-source BFS each), keeps the
  nodes reaching enough groups, and ranks them.

supporting files: `citetools/errors.py` (the `OpenAlexError` / `BadId` exceptions),
`citetools/strategies/__init__.py` (the `strategies` subpackage: future search
strategies plug in here behind the same `(oracle, groups, ...)` signature),
`citetools/__init__.py` (the public API), `citetools/__main__.py` (the CLI).

## Concurrency

the bridge finder parallelizes I/O *within* each BFS hop: all frontier nodes' neighbour lists are fetched concurrently over a thread pool (controlled by `--n-jobs`, default 8). BFS hops themselves remain sequential.

a shared rate limiter keeps the aggregate request rate within the OpenAlex polite pool's ceiling (~10 req/s), regardless of thread count. concurrency hides per-request latency, not request volume: it can only reclaim the rate budget that serial execution leaves idle while stalling on latency, never exceed the polite-pool cap. see [limitations](#limitations) for the measured speedup.

output is deterministic and independent of `--n-jobs`: results with `--n-jobs 8` are identical to `--n-jobs 1`, thanks to a fixed total-order tie-break in ranking.

## Limitations

- Citation-graph proximity is only a proxy for conceptual proximity!!!
- highly-cited hub papers (surveys, foundational works) tend to dominate. every
  result carries a `degree` field so you can spot them; genuine bridges usually have
  low degree.
- scoring is undirected and additive (sum of per-group distances), so a paper close
  to one area and far from another can tie a balanced bridge: the per-group `dists`
  are returned so you can tell them apart.
- depth bounds reach. a bridge is discoverable only if it sits within `--depth`
  citation hops of *every* group's seeds: at depth 2, each side of the seed-to-seed
  path through a discoverable bridge is at most two hops (four total). the limit is
  per-group, not a total: a paper three hops from one group is missed even if it is
  one hop from another. raising `--depth` reaches further, at cost that grows roughly
  exponentially in depth. the frontier cap can also drop an in-range bridge that is
  reachable only through a modestly-cited intermediate paper, so depth is necessary
  but not sufficient for discovery.
- depth-2 over genuinely distant fields can return an empty intersection; relax
  `--min-groups`.
- concurrency gains are modest and rate-capped. `--n-jobs` only reclaims the rate
  budget a serial run wastes stalling on per-request latency; it cannot exceed the
  ~10 req/s polite-pool ceiling. a depth-2 demo measured ~1.45x (35s -> 24s): the
  concurrent run saturates the polite pool, the serial run left it ~30% idle. the
  metadata batch fetches and the per-seed loop stay sequential and bound the gain
  further, so more than a handful of threads does not help.
