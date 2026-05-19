"""pure embedding math: encode text via injected model, aggregate cosine sims."""

import numpy as np


def encode(model, texts: list[str]) -> np.ndarray:
    """
    L2-normalize and batch-encode texts via injected sentence_transformers model.

    the model is a frozen, pre-loaded sentence_transformers.SentenceTransformer
    (injected by the impure io/models.load()). given a fixed model and fixed
    batch_size, this function is deterministic: same input texts -> same output
    vectors (up to determinism contract below).

    Args:
      model: loaded sentence_transformers.SentenceTransformer instance with
        encode(texts, ...) method returning ndarray of shape (len(texts), dim).
      texts: list of strings to encode; may be empty.

    Returns:
      ndarray of shape (len(texts), dim), dtype float32, each row L2-normalized
      to unit norm. if texts is empty, return shape (0, dim) array (dim inferred
      from model.get_sentence_embedding_dimension()).

    determinism contract:
      encode is deterministic IF the caller sorts texts before passing (or uses
      a deterministic order). model.encode() is sensitive to batch-size and
      padding, which can introduce small floating point deltas across batch
      compositions. the caller (io/models.embed_titles) is responsible for:
        1. sorting the texts before passing to encode (e.g. by W-id),
        2. pinning batch_size in the call below.

    procedure:
      1. if texts is empty, return a (0, dim) shaped float32 array where dim is
         queried via model.get_sentence_embedding_dimension().
      2. call model.encode(texts, batch_size=32, normalize_embeddings=True,
         convert_to_numpy=True, show_progress_bar=False).
      3. explicitly cast the result to float32 via np.asarray(..., dtype=np.float32).
      4. return the normalized ndarray (shape (len(texts), dim), dtype float32).
    """
    if not texts:
        dim = model.get_sentence_embedding_dimension()
        return np.zeros((0, dim), dtype=np.float32)

    # batch_size=32 is pinned for determinism; caller must sort texts before passing
    vecs = model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    # explicit cast to float32 in case model returns float64 or another dtype
    return np.asarray(vecs, dtype=np.float32)


def sim_to_set(vec: np.ndarray, targets: np.ndarray, agg: str = "max") -> float:
    """
    cosine similarity of one vector to a set of targets, aggregated.

    all vectors are pre-normalized (L2-norm == 1), so dot product == cosine.
    this function is pure: no side effects, deterministic, no randomness.

    Args:
      vec: shape (dim,), dtype float32, L2-norm == 1 (query vector).
      targets: shape (k, dim), dtype float32, all rows L2-norm == 1 (set of targets).
        k >= 0; if k == 0, return 0.0.
      agg: str in {"max", "mean", "min"}; default "max".
        - "max": return the largest similarity (best match in the set).
        - "mean": return the average similarity across the set.
        - "min": return the smallest similarity (worst match).

    Returns:
      Python float, clamped to [0.0, 1.0].

    procedure:
      1. if targets.shape[0] == 0 (empty set): return 0.0 immediately.
      2. compute sims = targets @ vec, shape (k,). each element is the dot product
         of targets[i] with vec, which is cosine(theta_i) for L2-normalized vectors.
      3. apply aggregation based on agg:
           - if agg == "max": result = float(np.max(sims)).
           - elif agg == "mean": result = float(np.mean(sims)).
           - elif agg == "min": result = float(np.min(sims)).
           - else: raise ValueError.
      4. clamp result to [0.0, 1.0] via np.clip(). due to numerical rounding and
         ensemble effects, dot products of normalized vectors can marginally exceed
         \\pm 1.
      5. return the clamped result as a Python float.
    """
    if targets.shape[0] == 0:
        return 0.0

    sims = targets @ vec  # (k,): dot product over L2-normalized vectors
    if agg == "max":
        result = float(np.max(sims))
    elif agg == "mean":
        result = float(np.mean(sims))
    elif agg == "min":
        result = float(np.min(sims))
    else:
        raise ValueError(f"agg must be in {{'max', 'mean', 'min'}}, got {agg!r}")

    return float(np.clip(result, 0.0, 1.0))
