"""lazy offline embedding of paper titles with batched inference and cache."""

import threading

import numpy


class TitleEmbedder:
    """
    lazy sentence-transformer embedder with per-W-id caching.

    control flow:
      __init__ stores config, delays model load
      _load() imports sentence_transformers and instantiates model (once)
      embed() partitions items into cache hits/misses, batch-encodes misses,
             stores results, returns all from cache
      dim property triggers load and returns embedding dimension

    thread-safety: embed() and dim hold an internal lock, so one embedder is
    safe to share across the concurrent ensemble-run threads in walk.py.
    """

    def __init__(
        self, model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder=None, device="cpu"
    ) -> None:
        """
        Initialize a lazy sentence-transformer embedder with optional disk caching.
        The model is NOT loaded here; it loads on first embed() or dim access.

        Args:
          model_name: model identifier on HuggingFace (default: all-MiniLM-L6-v2, 384-dim).
          cache_folder: optional path to cache downloaded model weights (passed to
            SentenceTransformer). if None, use HuggingFace's default.
          device: "cpu", "cuda", or "mps" (passed to SentenceTransformer).

        Embeddings are stored in self._cache, keyed by W-id, as (dim,) float32
        L2-normalized numpy arrays.
        """
        self.model_name = model_name
        self.cache_folder = cache_folder
        self.device = device
        self._cache = {}
        self._model = None
        self._lock = threading.Lock()

    def _load(self) -> None:
        """
        Lazily import sentence_transformers and load the model. Called once, on first
        embed() or dim access. Raises ImportError if sentence_transformers is missing.
        """
        try:
            import sentence_transformers
        except ImportError as e:
            raise ImportError(
                "sentence_transformers not found. Install dependencies:\n"
                "  pip install -r requirements-walk.txt"
            ) from e

        if self._model is None:
            self._model = sentence_transformers.SentenceTransformer(
                model_name_or_path=self.model_name,
                cache_folder=self.cache_folder,
                device=self.device,
            )

    def embed(self, items: dict[str, str]) -> dict[str, numpy.ndarray]:
        """
        Embed a batch of paper titles, serving cache hits and batch-encoding misses.

        Args:
          items: dict mapping W-id (str) -> title text (str). Title may be empty.

        Returns:
          dict mapping W-id -> (dim,) float32 L2-normalized vector. a wid whose
          title is missing / empty / non-str is OMITTED (see Edge cases).

        Control flow:
          1. If items is empty, return {}.
          2. Partition items into cache hits and misses.
          3. For misses with a usable (non-empty str) title, batch-encode in one
             self._model.encode() call and cache each (dim,) L2-normalized vector.
          4. Return the cached vectors for all input wids that have one.

        Edge cases:
          a missing, empty, or non-str title carries no semantic signal, so that
          wid is skipped -- never encoded, absent from the returned dict. callers
          detect this via `wid not in result` and apply a neutral fallback.
          (OpenAlex works can carry an explicit "title": null.)
        """
        if not items:
            return {}

        # lock guards the lazy _load() and the encode/cache-write path so the
        # embedder is safe under concurrent calls from walk.py's run threads.
        with self._lock:
            self._load()

            # partition into hits and misses
            hits = {wid: self._cache[wid] for wid in items if wid in self._cache}
            misses = {wid: items[wid] for wid in items if wid not in self._cache}

            if not misses:
                return hits

            # encode only misses with a usable (non-empty str) title; a missing,
            # empty, or non-str title carries no semantic signal -- skip it.
            sorted_wids = sorted(
                wid for wid in misses
                if isinstance(misses[wid], str) and misses[wid].strip()
            )
            if sorted_wids:
                vectors = self._model.encode(
                    [misses[wid] for wid in sorted_wids],
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                for wid, vec in zip(sorted_wids, vectors):
                    self._cache[wid] = vec

            # return cached vectors; titleless wids are absent by design
            return {wid: self._cache[wid] for wid in items if wid in self._cache}

    @property
    def dim(self) -> int:
        """
        Embedding dimension (int). Queries the model via get_sentence_embedding_dimension().
        Triggers model load if not yet loaded.
        """
        with self._lock:
            self._load()
            return self._model.get_sentence_embedding_dimension()


def sim_to_set(vec: numpy.ndarray, targets: numpy.ndarray, agg: str = "max") -> float:
    """
    Aggregate cosine similarities from one vector to a set of targets.

    All vectors are pre-normalized (L2-norm == 1), so dot product == cosine.

    Args:
      vec: shape (dim,), dtype float32, L2-norm == 1 (query vector).
      targets: shape (k, dim), dtype float32, all L2-norm == 1 (target vectors).
                k >= 0; if k == 0 (empty), return 0.0.
      agg: aggregation method in {"max", "mean", "min"}.

    Returns:
      Python float, aggregated similarity. clamped to [0.0, 1.0] for safety
      (normalized vectors have dot products in [-1, 1], but due to rounding
      and ensemble effects, clip to [0, 1]).

    Computation:
      1. compute sims = targets @ vec, shape (k,).
      2. if k == 0: return 0.0.
      3. apply agg ("max" -> sims.max(), "mean" -> sims.mean(), "min" -> sims.min()).
      4. clamp to [0.0, 1.0] and return as Python float.
    """
    if targets.shape[0] == 0:
        return 0.0

    sims = targets @ vec  # shape (k,), dot product over L2-normalized vectors
    if agg == "max":
        result = float(numpy.max(sims))
    elif agg == "mean":
        result = float(numpy.mean(sims))
    elif agg == "min":
        result = float(numpy.min(sims))
    else:
        raise ValueError(f"agg must be in {{'max', 'mean', 'min'}}, got {agg!r}")

    return float(numpy.clip(result, 0.0, 1.0))
