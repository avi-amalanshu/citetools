"""OpenAlex HTTP client with polite-pool retry and dual caching."""

import logging
import re
import threading
import time
from collections.abc import Iterable
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .errors import BadId, OpenAlexError

OPENALEX_BASE = "https://api.openalex.org"

log = logging.getLogger("citetools.openalex")

# bounded GET retries on a 429/5xx before giving up
_MAX_ATTEMPTS = 5
# ceiling (seconds) on any single fleet-wide cooldown, incl. a server Retry-After
_COOLDOWN_CAP = 60.0

# arxiv abs/pdf url; captures the id, tolerates http(s)://, www., a .pdf suffix
# and a trailing slash. the id keeps any internal '/' for old-style arxiv ids.
_ARXIV_URL = re.compile(
	r"^(?:https?://)?(?:www\.)?arxiv\.org/(?:abs|pdf)/(?P<id>.+?)(?:\.pdf)?/?$",
	re.IGNORECASE,
)
# bare new-style arxiv id, e.g. 2010.11929 or 2010.11929v2
_ARXIV_ID = re.compile(r"^\d{4}\.\d{4,5}(?:v\d+)?$")


def _arxiv_doi(pid: str) -> str | None:
	"""Map an arxiv abs/pdf link or bare new-style arxiv id to its arxiv doi.

	'https://arxiv.org/abs/2010.11929v2' -> '10.48550/arxiv.2010.11929'
	'2010.11929'                         -> '10.48550/arxiv.2010.11929'
	anything not recognisably arxiv      -> None.
	the version suffix (vN) and a trailing '.pdf' are stripped; an old-style
	id keeps its internal '/' (e.g. cs/0701001).
	"""
	m = _ARXIV_URL.match(pid)
	if m:
		aid = m.group("id")
	elif _ARXIV_ID.match(pid):
		aid = pid
	else:
		return None
	aid = re.sub(r"v\d+$", "", aid)  # strip version suffix
	return f"10.48550/arxiv.{aid.lower()}"


def _retry_after(response: requests.Response) -> float | None:
	"""Parse a Retry-After header to seconds.

	integer-seconds form -> float; absent or http-date form -> None (the
	caller falls back to exponential backoff). a non-positive value is
	treated as absent.
	"""
	raw = response.headers.get("Retry-After")
	if raw is None:
		return None
	try:
		secs = float(raw)
	except ValueError:
		return None
	return secs if secs > 0 else None


class _RateLimiter:
	"""Shared rate-limiter for concurrent threads: caps aggregate request rate at 1/interval req/s.

	acquire() reserves a thread's slot under a lock (O(1) per-thread), then sleeps outside the
	lock. N threads thus reserve staggered slots and reach throughput N threads concurrently
	requesting, with aggregate rate exactly 1/interval req/s, without serializing to 1 req/s.

	penalize() registers a server throttle by pushing the shared slot forward, so every thread
	cools down together; relax() clears the escalation after a success.
	"""

	def __init__(self, interval: float) -> None:
		"""Store interval and initialize lock and next-slot timestamp.

		interval: seconds between requests (e.g. 0.1 -> 10 req/s aggregate).
		"""
		self.interval = interval
		self._lock = threading.Lock()
		self._next = 0.0
		self._strikes = 0  # consecutive server-throttle events

	def acquire(self) -> None:
		"""Reserve this thread's slot and sleep appropriately.

		Under lock: read current time, compute the slot time max(now, self._next),
		reserve it by setting self._next = slot + interval, then release lock.
		Outside lock: sleep for (slot - now) if positive. This allows N threads to
		reserve staggered slots concurrently without holding the lock during sleep.
		"""
		with self._lock:
			now = time.monotonic()
			start = max(now, self._next)
			self._next = start + self.interval

		wait = start - now
		if wait > 0:
			time.sleep(wait)

	def penalize(self, retry_after: float | None = None) -> float:
		"""Register a server throttle (429/5xx): push the shared slot forward.

		Every thread's next acquire() then blocks until the cooldown elapses,
		so the whole client backs off together. cooldown = the server's
		Retry-After when given, else exponential in the consecutive-strike
		count; capped at _COOLDOWN_CAP. returns the cooldown applied (seconds).
		"""
		with self._lock:
			self._strikes += 1
			if retry_after is not None:
				cooldown = min(retry_after, _COOLDOWN_CAP)
			else:
				cooldown = min(2.0 ** self._strikes, _COOLDOWN_CAP)
			self._next = max(self._next, time.monotonic() + cooldown)
			return cooldown

	def relax(self) -> None:
		"""A request succeeded: clear the consecutive-strike counter."""
		with self._lock:
			self._strikes = 0


class OpenAlex:
	"""HTTP client for OpenAlex API with thread-safe concurrent requests and rate limiting.

	Each thread gets its own requests.Session via threading.local(), initialized lazily
	on first _session() call and registered for cleanup. A shared _RateLimiter caps
	aggregate request rate at 1/sleep req/s across all threads without serializing.

	Caches full works in _cache (keyed by canonical W-id); trimmed metadata in _meta.
	Both caches are shared plain dicts, safe under concurrent fetches because callers
	only ever write to distinct keys (CPython per-key dict writes are GIL-atomic).

	session param allows test mocking; if provided, used as single shared session.
	Call close() to release all sessions. clear() is not safe to call concurrently
	with work/works/citers.
	"""

	def __init__(
		self,
		mailto: str,
		*,
		sleep: float = 0.1,
		timeout: float = 30.0,
		citer_k: int = 50,
		session: requests.Session | None = None,
	) -> None:
		"""Initialize OpenAlex client with polite-pool settings and thread-safe concurrency.

		mailto: email address for User-Agent and polite-pool qualification.
		sleep: delay after each successful HTTP response (seconds); also caps aggregate request rate.
		timeout: request timeout (seconds).
		citer_k: default page size for citers() calls.
		session: optional requests.Session for test injection; if None, uses thread-local sessions.
		"""
		self.mailto = mailto
		self.sleep = sleep
		self.timeout = timeout
		self.citer_k = citer_k

		# rate limiter shared across all threads
		self._limiter = _RateLimiter(self.sleep)

		# session management
		if session is not None:
			# test path: single injected session
			self._injected_session = session
			session.headers["User-Agent"] = f"citetools/1.0 ({mailto})"
		else:
			# production path: thread-local sessions
			self._injected_session = None
			self._local = threading.local()
			self._sessions: list = []
			self._sessions_lock = threading.Lock()

		# dual caches: both empty dicts
		self._cache: dict[str, dict[str, Any]] = {}
		self._meta: dict[str, dict[str, Any]] = {}

		self.params = {"mailto": mailto}

	def _session(self) -> requests.Session:
		"""Return the session for the current thread.

		Test path: if session was injected, return it directly (shared across threads).
		Production path: lazily build per-thread session from self._local, mount a
		Retry adapter for transport-level errors only (total=2, no status_forcelist
		-- 429/5xx are handled by _get's fleet-wide cooldown), set User-Agent header,
		and register it in self._sessions for cleanup on close().
		"""
		if self._injected_session is not None:
			return self._injected_session

		# thread-local production path
		if not hasattr(self._local, "session"):
			s = requests.Session()
			s.headers["User-Agent"] = f"citetools/1.0 ({self.mailto})"

			# retry transport-level errors only; 429/5xx go through _get's
			# fleet-wide cooldown so all threads back off together
			retry = Retry(total=2, status_forcelist=[], respect_retry_after_header=False)
			adapter = HTTPAdapter(max_retries=retry)
			s.mount("http://", adapter)
			s.mount("https://", adapter)

			self._local.session = s

			# register in global list for close()
			with self._sessions_lock:
				self._sessions.append(s)

		return self._local.session

	def _get(self, url: str, params: dict, *, desc: str) -> dict:
		"""Rate-limited GET with fleet-wide cooldown retry on 429 / 5xx.

		desc: short human label for the resource, used in logs and errors.
		Proc: for up to _MAX_ATTEMPTS attempts:
		  - _limiter.acquire() -- blocks for any pending fleet-wide cooldown
		  - GET; a transport-level error -> raise OpenAlexError immediately
		  - status 429 or >= 500 -> _limiter.penalize(Retry-After), log a
		    warning, retry
		  - else -> raise_for_status, _limiter.relax(), return parsed json
		Exhausting all attempts -> raise OpenAlexError naming desc.
		raises OpenAlexError (transport failure, non-retryable status, or
		retries exhausted).
		"""
		for attempt in range(1, _MAX_ATTEMPTS + 1):
			self._limiter.acquire()
			try:
				response = self._session().get(url, params=params, timeout=self.timeout)
			except requests.RequestException as e:
				raise OpenAlexError(f"failed to fetch {desc}: {e}") from e

			if response.status_code == 429 or response.status_code >= 500:
				cooldown = self._limiter.penalize(_retry_after(response))
				log.warning(
					"openalex %d on %s; cooling down %.1fs (attempt %d/%d)",
					response.status_code, desc, cooldown, attempt, _MAX_ATTEMPTS,
				)
				continue

			try:
				response.raise_for_status()
				data = response.json()
			except requests.RequestException as e:
				raise OpenAlexError(f"failed to fetch {desc}: {e}") from e

			self._limiter.relax()
			return data

		raise OpenAlexError(
			f"failed to fetch {desc}: still rate-limited after {_MAX_ATTEMPTS} attempts"
		)

	@staticmethod
	def key(pid: str) -> str:
		"""Normalize pid (DOI, bare W-id, OpenAlex URL, or arxiv link) to canonical lookup key.

		-> 'W12345' (uppercase), 'doi:10.xxxx/yyyy' (lowercase doi), or raises BadId.
		Accepts: 'W123', 'https://openalex.org/W123', '10.1038/nature12373',
		  'https://doi.org/10.1038/nature12373', 'https://arxiv.org/abs/2010.11929',
		  'https://arxiv.org/pdf/2010.11929v2', and bare new-style arxiv ids.
		arxiv links resolve via the arxiv doi 10.48550/arxiv.<id>.
		"""
		# reject empty
		if not pid:
			raise BadId("empty id")

		# already a canonical doi key -> key() must be idempotent
		if pid.startswith("doi:"):
			return "doi:" + pid[4:].lower()

		# openalex full url
		if pid.startswith("https://openalex.org/"):
			canonical = pid.rsplit("/", 1)[-1]
			if not re.match(r"^W\d+$", canonical):
				raise BadId(f"malformed OpenAlex URL: {pid}")
			return canonical.upper()

		# arxiv abs/pdf link or bare new-style arxiv id -> arxiv doi
		# (checked before the doi branch: arxiv urls also contain '/')
		adoi = _arxiv_doi(pid)
		if adoi is not None:
			return f"doi:{adoi}"

		# doi: has / but not url scheme (or is https://doi.org/...)
		if "/" in pid:
			if pid.startswith("https://doi.org/"):
				doi_part = pid.split("https://doi.org/")[-1]
			else:
				doi_part = pid
			return f"doi:{doi_part.lower()}"

		# bare w-id
		if not re.match(r"^W\d+$", pid):
			raise BadId(f"malformed id: {pid}")
		return pid.upper()

	def work(self, pid: str) -> dict[str, Any]:
		"""Fetch one full work by any id form; cache by canonical W-id.

		pid -> key (normalized) -> HTTP GET /works/{key} -> extract canonical_wid
		  from response["id"] -> cache full object in _cache[canonical_wid] and
		  _cache[key]; also in _meta. rate-limit, return data.
		raises BadId (from key), OpenAlexError (HTTP failures).
		"""
		lookup_key = self.key(pid)

		# check _cache
		if lookup_key in self._cache:
			return self._cache[lookup_key]

		# fetch via the rate-limited, cooldown-aware GET helper
		url = f"{OPENALEX_BASE}/works/{lookup_key}"
		data = self._get(url, self.params, desc=f"work {lookup_key}")

		# extract canonical wid
		canonical_wid = data["id"].rsplit("/", 1)[-1].upper()

		# cache in both slots
		self._cache[canonical_wid] = data
		self._cache[lookup_key] = data
		self._meta[canonical_wid] = data

		return data

	def works(self, pids: Iterable[str]) -> dict[str, dict[str, Any]]:
		"""Batch-fetch up to 50 full works per HTTP call via filter=openalex:W1|W2|...

		Skips ids already in _cache; returns dict {canonical_wid: work_dict} for found ids.
		empty input -> {}, no HTTP call. missing ids -> absent from result.
		Only bare W-ids batched; DOIs passed through but not fetched (would require
		separate work() calls). caches full objects in _cache and _meta by canonical_wid.
		raises BadId (from key), OpenAlexError (HTTP failures).
		"""
		pids_list = list(pids)

		# early exit
		if not pids_list:
			return {}

		# collect to_fetch and populate out from cache
		to_fetch: list[str] = []
		out: dict[str, dict[str, Any]] = {}

		for pid in pids_list:
			lookup_key = self.key(pid)
			if lookup_key in self._cache:
				# extract canonical from cached data
				cached_data = self._cache[lookup_key]
				canonical_wid = cached_data["id"].rsplit("/", 1)[-1].upper()
				out[canonical_wid] = cached_data
			else:
				to_fetch.append(lookup_key)

		# filter: only bare w-ids, skip dois
		w_ids_to_fetch = [k for k in to_fetch if not k.startswith("doi:")]

		# batch by 50
		for i in range(0, len(w_ids_to_fetch), 50):
			chunk = w_ids_to_fetch[i : i + 50]
			filter_str = "openalex:" + "|".join(chunk)

			batch = self._get(
				f"{OPENALEX_BASE}/works",
				{**self.params, "filter": filter_str, "per-page": 50},
				desc=f"batch {filter_str}",
			)
			batch_data = batch["results"]

			# cache each work
			for w in batch_data:
				canonical_wid = w["id"].rsplit("/", 1)[-1].upper()
				self._cache[canonical_wid] = w
				self._meta[canonical_wid] = w
				out[canonical_wid] = w

		return out

	def citers(self, pid: str, k: int | None = None) -> list[str]:
		"""Fetch ONE page of works that cite pid (sorted by cited_by_count desc).

		pid -> canonical_wid (via work()) -> GET /works?filter=cites:{wid}&...&select=...
		-> extract canonical W-ids from trimmed response. cache trimmed objects in _meta
		ONLY (not _cache, to prevent partial-cache poisoning). returns list of canonical
		W-ids; zero citers -> []. ONE page only, no pagination. k defaults to citer_k.
		raises BadId (from key), OpenAlexError (HTTP failures).
		"""
		if k is None:
			k = self.citer_k

		# fetch seed work to get canonical wid
		seed_work = self.work(pid)
		canonical_wid = seed_work["id"].rsplit("/", 1)[-1].upper()

		# fetch citers via the rate-limited, cooldown-aware GET helper
		params = {
			"mailto": self.mailto,
			"filter": f"cites:{canonical_wid}",
			"sort": "cited_by_count:desc",
			"per-page": k,
			"select": "id,title,publication_year,cited_by_count,doi",
		}
		page_data = self._get(
			f"{OPENALEX_BASE}/works", params, desc=f"citers of {canonical_wid}"
		)

		# extract and cache in _meta only
		results = page_data.get("results", [])
		out: list[str] = []

		for w in results:
			w_id = w["id"].rsplit("/", 1)[-1].upper()
			self._meta[w_id] = w
			out.append(w_id)

		return out

	def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
		"""Search works by title; return up to k trimmed candidate records.

		query -> GET /works?filter=title.search:{query}&per-page=k&select=...
		-> list of trimmed work dicts (id, title, publication_year,
		cited_by_count, doi), ranked by OpenAlex relevance. caches the trimmed
		objects in _meta only. blank query -> []. ONE page, no pagination.
		raises OpenAlexError (HTTP failures).
		"""
		if not query.strip():
			return []

		params = {
			"mailto": self.mailto,
			"filter": f"title.search:{query}",
			"per-page": k,
			"select": "id,title,publication_year,cited_by_count,doi",
		}
		page = self._get(
			f"{OPENALEX_BASE}/works", params, desc=f"works matching {query!r}"
		)
		results = page.get("results", [])

		# cache trimmed objects in _meta only
		for w in results:
			w_id = w["id"].rsplit("/", 1)[-1].upper()
			self._meta[w_id] = w

		return results

	def wid(self, pid: str) -> str:
		"""Resolve any id form to its bare canonical OpenAlex W-id.

		pid (W-id, DOI, OpenAlex URL, or arxiv abs/pdf link) -> work() ->
		bare 'Wxxxxx'. raises BadId (from key), OpenAlexError (HTTP / not found).
		"""
		return self.work(pid)["id"].rsplit("/", 1)[-1].upper()

	def clear(self) -> None:
		"""Drop all cached full works and metadata.

		Useful for long-running or memory-constrained scripts.
		next fetch will hit the API.
		"""
		self._cache.clear()
		self._meta.clear()

	def close(self) -> None:
		"""Close all managed sessions.

		Test path: if session was injected, close it. Production path: close all
		thread-local sessions registered in self._sessions. Idempotent (safe to
		call multiple times).
		"""
		if self._injected_session is not None:
			self._injected_session.close()
		else:
			with self._sessions_lock:
				for s in self._sessions:
					s.close()
