"""lazy embedding model loader and embedding cache (by W-id).

exports:
  Embedder: frozen dataclass with model and cache fields.
  load(model_name, device) -> model: thread-safe, idempotent singleton loader.
  make_embedder(model_name, device) -> Embedder: construct an Embedder pair.
  embed_titles(model, cache, titles): partition hits/misses, batch-encode misses
    via core.embed.encode, cache, return L2-normalized vectors.

control flow determinism fix: encode(model, texts) is not bit-pure across
batch composition. the cache must sort miss texts by W-id before encoding to
ensure order-stable batch encoding with a pinned batch_size.

module state:
  _model: singleton SentenceTransformer, loaded once.
  _lock: threading.Lock, guards load and cache writes.
"""

import threading
from dataclasses import dataclass

import numpy as np

from citetools.core import embed as core_embed


_model: object | None = None
_lock: threading.Lock = threading.Lock()


@dataclass(frozen=True)
class Embedder:
	"""embedder object carrying a loaded model and its embedding cache.

	fields:
	  model: sentence_transformers.SentenceTransformer instance, loaded via load().
	  cache: dict[str, np.ndarray], mapping W-id -> (dim,) L2-normalized float32 vectors.
	"""
	model: object
	cache: dict[str, np.ndarray]


def load(
	model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
	device: str = "cpu"
) -> object:
	"""lazily load and cache a SentenceTransformer model (thread-safe, idempotent).

	imports sentence_transformers on first call; subsequent calls return the cached
	singleton. all concurrent calls are serialized by an internal lock, so the model
	is safe to share across threads (e.g., ensemble runs in walk.py).

	args:
	  model_name: HuggingFace model identifier
	    (default: sentence-transformers/all-MiniLM-L6-v2, 384-dim).
	  device: cpu, cuda, or mps (passed to SentenceTransformer).

	returns:
	  sentence_transformers.SentenceTransformer instance.

	raises:
	  ImportError: if sentence_transformers is not installed.

	note:
	  if load is called a second time with a different device, the first device
	  is retained (the model is a singleton). call load once per process.
	"""
	global _model

	with _lock:
		if _model is None:
			try:
				import sentence_transformers
			except ImportError as e:
				raise ImportError(
					"sentence_transformers not found. install dependencies:\n"
					"  pip install -r requirements-walk.txt"
				) from e

			_model = sentence_transformers.SentenceTransformer(
				model_name_or_path=model_name,
				device=device
			)

		return _model


def make_embedder(
	model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
	device: str = "cpu"
) -> Embedder:
	"""create a new Embedder (model + cache) pair.

	calls load(model_name, device) to obtain the lazy-loaded transformer singleton,
	and initializes a fresh empty cache dict. the returned Embedder bundles both
	for convenient passing to runtime/strategies functions.

	args:
	  model_name: HuggingFace model identifier
	    (default: sentence-transformers/all-MiniLM-L6-v2, 384-dim).
	  device: cpu, cuda, or mps (passed to load).

	returns:
	  Embedder(model=..., cache={}), ready for use in embed_titles calls.
	"""
	model = load(model_name, device)
	cache = {}
	return Embedder(model=model, cache=cache)


def embed_titles(
	model: object,
	cache: dict[str, np.ndarray],
	titles: dict[str, str]
) -> dict[str, np.ndarray]:
	"""embed a batch of paper titles, partitioning cache hits and misses.

	partition titles into cache hits (served from the supplied cache dict)
	and misses (batch-encoded via core.embed.encode). newly encoded vectors
	are cached in-place. titleless (missing, empty, or non-str) W-ids are
	skipped and absent from the result.

	args:
	  model: sentence_transformers.SentenceTransformer instance (or compatible).
	  cache: dict[str, np.ndarray], keyed by W-id, initially empty or populated.
	    mutated in-place with newly encoded vectors (shape (dim,), dtype float32,
	    L2-norm == 1).
	  titles: dict[str, str], mapping W-id -> title text. title may be None, empty,
	    or non-string; such W-ids are skipped.

	returns:
	  dict[str, np.ndarray], mapping W-id -> (dim,) L2-normalized float32 vector.
	  only W-ids with usable titles (and a cached vector) are present.

	control flow:
	  1. partition titles into cache hits and misses.
	  2. filter misses to only usable (non-empty str) titles.
	  3. sort usable W-ids and batch-encode texts in that order via
	     core.embed.encode, ensuring order-stable encoding with pinned batch_size.
	  4. cache newly encoded vectors in-place.
	  5. return all cached vectors for W-ids in titles (hits + newly cached).

	determinism note:
	  core.embed.encode is not bit-pure across batch composition (transformer
	  batch-size and padding sensitivity). to ensure reproducible results, this
	  function sorts misses by W-id before encoding and relies on a pinned
	  batch_size in the runtime (io/runtime.py). all inputs to encode are
	  canonical (sorted, deduplicated).

	concurrency note:
	  the cache dict is mutated in-place. if multiple threads call this function
	  with overlapping titles, both may encode the same miss (wasteful but safe,
	  producing identical vectors). the runtime (io/runtime.py) does not call
	  this concurrently; thread-safety is the caller's responsibility if needed.
	"""
	# step 1: return early if titles is empty
	if not titles:
		return {}

	# step 2: partition into cache hits and misses
	hits = {wid: cache[wid] for wid in titles if wid in cache}
	misses_raw = {wid: titles[wid] for wid in titles if wid not in cache}

	# step 3: return hits if no misses
	if not misses_raw:
		return hits

	# step 4: filter misses to only those with usable (non-empty str) titles
	usable_wids = sorted(
		wid for wid in misses_raw
		if isinstance(misses_raw[wid], str) and misses_raw[wid].strip()
	)

	# step 5: encode usable misses and cache them
	if usable_wids:
		texts = [misses_raw[wid] for wid in usable_wids]
		vectors = core_embed.encode(model, texts)
		for wid, vec in zip(usable_wids, vectors):
			cache[wid] = vec

	# step 6: build result: all cached vectors for W-ids in titles
	return {wid: cache[wid] for wid in titles if wid in cache}
