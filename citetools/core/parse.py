"""citetools.core.parse

pure: id canonicalization (key), arxiv doi extraction, OpenAlex json shaping,
HTTP header parsing. lifted from io/openalex.py and graph.py; serves the pure
core and i/o layer's impure shells (runtime, models).

DETERMINISM CONTRACT: all functions are pure; side-effect-free; idempotent
(key(key(x)) == key(x) for all input forms). no mutation of arguments.
functions are offline-testable.

API surface:
  key(pid: str) -> str
      idempotent canonicalizer: any pid form -> canonical lookup key.
      raises BadId on malformed input.
  arxiv_doi(pid: str) -> str | None
      arxiv URL or id -> arxiv doi, or None.
  retry_after(resp) -> float | None
      HTTP response Retry-After header -> seconds (float), or None.
  refs(work: dict) -> list[str]
      work dict -> list of canonical W-ids from referenced_works.
  meta(work: dict) -> dict
      work dict -> {"title", "year", "doi", "cited_by_count"} (None-safe).
"""

import re
from ..errors import BadId


_ARXIV_URL = re.compile(
    r"^(?:https?://)?(?:www\.)?arxiv\.org/(?:abs|pdf)/(?P<id>.+?)(?:\.pdf)?/?$",
    re.IGNORECASE,
)
_ARXIV_ID = re.compile(r"^\d{4}\.\d{4,5}(?:v\d+)?$")


def _arxiv_doi(pid: str) -> str | None:
    """(private helper) map an arxiv abs/pdf link or bare new-style arxiv id to its
    arxiv doi.

    -> string '10.48550/arxiv.<id>' (lowercase), or None if pid is not arxiv.

    used internally by key() to detect arxiv inputs. exposed here for clarity
    (separates arxiv parsing from the main key() logic).
    """
    m = _ARXIV_URL.match(pid)
    if m:
        aid = m.group("id")
    elif _ARXIV_ID.match(pid):
        aid = pid
    else:
        return None
    # strip version suffix (vN)
    aid = re.sub(r"v\d+$", "", aid)
    return f"10.48550/arxiv.{aid.lower()}"


def key(pid: str) -> str:
    """normalize pid (DOI, bare W-id, OpenAlex URL, or arxiv link) to canonical
    lookup key. idempotent: key(key(x)) == key(x) for every output form.

    -> 'Wxxxxx' (uppercase bare W-id), 'doi:10.xxxx/yyyy' (lowercase doi), or
    raises BadId.

    accepts:
      - 'W123', 'W12345' (bare W-id, any case)
      - 'https://openalex.org/W123' (OpenAlex full URL)
      - '10.1038/nature12373' (bare doi)
      - 'https://doi.org/10.1038/nature12373' (doi URL)
      - 'https://arxiv.org/abs/2010.11929' (arxiv abs URL, http(s) tolerant)
      - 'https://arxiv.org/pdf/2010.11929v2' (arxiv pdf URL, .pdf and vN stripped)
      - '2010.11929', '2010.11929v2' (bare new-style arxiv id)
      - 'http://arxiv.org/...' (http variant)
      - 'https://www.arxiv.org/...' (www variant)

    arxiv links resolve to arxiv doi (10.48550/arxiv.<id>, vN suffix stripped).

    example trace:
      key('W12345') -> 'W12345' (already canonical W-id)
      key('doi:10.1038/nature12373') -> 'doi:10.1038/nature12373' (idempotent)
      key('10.1038/nature12373') -> 'doi:10.1038/nature12373'
      key('https://doi.org/10.1038/nature12373') -> 'doi:10.1038/nature12373'
      key('https://arxiv.org/abs/2010.11929v2') -> 'doi:10.48550/arxiv.2010.11929'
      key('2010.11929') -> 'doi:10.48550/arxiv.2010.11929'
      key('') -> raises BadId("empty id")
      key('bad@id') -> raises BadId("malformed id: bad@id")
    """
    # 1. empty check
    if not pid:
        raise BadId("empty id")

    # 2. idempotent doi prefix (MUST be early, before "/" check)
    if pid.startswith("doi:"):
        return "doi:" + pid[4:].lower()

    # 3. OpenAlex full URL
    if pid.startswith("https://openalex.org/"):
        canonical = pid.rsplit("/", 1)[-1]
        if not re.match(r"^W\d+$", canonical):
            raise BadId(f"malformed OpenAlex URL: {pid}")
        return canonical.upper()

    # 4. arxiv doi (checked before "/" branch: arxiv URLs contain "/")
    adoi = _arxiv_doi(pid)
    if adoi is not None:
        return f"doi:{adoi}"

    # 5. doi with / (bare or https://doi.org/...)
    if "/" in pid:
        if pid.startswith("https://doi.org/"):
            doi_part = pid.split("https://doi.org/")[-1]
        else:
            doi_part = pid
        return f"doi:{doi_part.lower()}"

    # 6. bare W-id
    if re.match(r"^W\d+$", pid):
        return pid.upper()

    # 7. fallthrough
    raise BadId(f"malformed id: {pid}")


def arxiv_doi(pid: str) -> str | None:
    """map an arxiv abs/pdf link or bare new-style arxiv id to its arxiv doi.

    -> string '10.48550/arxiv.<id>' (lowercase), or None if pid is not arxiv.

    accepts:
      - 'https://arxiv.org/abs/2010.11929v2' -> '10.48550/arxiv.2010.11929'
        (version suffix vN stripped, URL case insensitive)
      - 'https://arxiv.org/pdf/2010.11929' -> '10.48550/arxiv.2010.11929'
      - 'http://arxiv.org/abs/2010.11929' (http variant)
      - 'https://www.arxiv.org/abs/2010.11929' (www variant)
      - '2010.11929' -> '10.48550/arxiv.2010.11929' (bare new-style id)
      - '2010.11929v2' -> '10.48550/arxiv.2010.11929' (vN suffix stripped)
      - old-style id 'cs/0701001' -> '10.48550/arxiv.cs/0701001' (keeps internal /)
      - anything not arxiv-shaped -> None

    note: old-style arxiv ids (category/id format) keep their internal '/',
    matching arxiv's published doi format.
    """
    return _arxiv_doi(pid)


def retry_after(resp) -> float | None:
    """parse a Retry-After HTTP response header to seconds.

    -> float (seconds) if header present and is a positive integer-seconds form,
    -> None if absent, malformed, or <= 0.

    input: resp is a requests.Response object (or any object with
      .headers dict-like access).

    note: HTTP-date form (e.g., "Mon, 18 May 2026 10:00:00 GMT") is not
    supported; such headers are treated as malformed and return None. the
    caller falls back to exponential backoff in that case.

    example:
      resp with header "Retry-After: 60" -> 60.0
      resp with header "Retry-After: 0" -> None (non-positive)
      resp with no header -> None
      resp with header "Retry-After: Mon, ..." -> None (http-date unsupported)
    """
    raw = resp.headers.get("Retry-After")
    if raw is None:
        return None
    try:
        secs = float(raw)
    except ValueError:
        return None
    return secs if secs > 0 else None


def refs(work: dict) -> list[str]:
    """extract canonical W-ids from a work's referenced_works.

    -> list of canonical W-ids (e.g., ['W123', 'W456']), or [] if no refs.

    input: work is a full work dict from OpenAlex (client.work, client.works,
      client.citers, or client.search). work["referenced_works"] is a list of
      full OpenAlex URLs or None/absent.

    procedure:
      - extract work.get("referenced_works") (defaults to None)
      - if None or absent -> return []
      - for each url in referenced_works:
          * call key(url) to canonicalize to bare W-id
          * collect results in a list (preserving order)
      - return list of canonical W-ids

    edge case: a reference URL that fails key() validation raises BadId
    propagates up (don't swallow; malformed refs are a data-quality issue).

    example:
      work = {
        "referenced_works": [
          "https://openalex.org/W123",
          "https://openalex.org/W456"
        ]
      }
      refs(work) -> ['W123', 'W456']

      work = {"referenced_works": None}
      refs(work) -> []

      work = {}
      refs(work) -> []
    """
    ref_urls = work.get("referenced_works") or []
    return [key(url) for url in ref_urls]


def meta(work: dict) -> dict:
    """shape OpenAlex work dict into trimmed metadata record.

    -> dict with keys {"title", "year", "doi", "cited_by_count"}.
    all values are None-safe: missing/None fields preserved as-is.

    input: work is a full work dict from OpenAlex (client.work, client.works,
      client.citers, or client.search).

    field mapping (None-safe):
      "title" <- work.get("title") or work.get("display_name") or ""
          (fallback chain: title -> display_name -> empty string)
      "year" <- work.get("publication_year")
          (missing/None preserved)
      "doi" <- work.get("doi")
          (missing/None preserved; may be None or url-form 'https://doi.org/...')
      "cited_by_count" <- work.get("cited_by_count")
          (missing/None preserved; typically int >= 0)

    example:
      work = {
        "title": "Example Paper",
        "publication_year": 2020,
        "cited_by_count": 42,
        "doi": "10.1038/nature12373"
      }
      meta(work) -> {
        "title": "Example Paper",
        "year": 2020,
        "doi": "10.1038/nature12373",
        "cited_by_count": 42
      }

      work = {
        "title": None,
        "display_name": "Fallback Title",
        "publication_year": None,
        "cited_by_count": None,
        "doi": None
      }
      meta(work) -> {
        "title": "Fallback Title",
        "year": None,
        "doi": None,
        "cited_by_count": None
      }

      work = {}  # all missing
      meta(work) -> {
        "title": "",
        "year": None,
        "doi": None,
        "cited_by_count": None
      }
    """
    return {
        "title": work.get("title") or work.get("display_name") or "",
        "year": work.get("publication_year"),
        "doi": work.get("doi"),
        "cited_by_count": work.get("cited_by_count"),
    }
