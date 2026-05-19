"""OpenAlex HTTP client with polite-pool retry and dual caching.

Stateful impure module: manages thread-local sessions, rate limiting, and
shared caches. No behavior change from the current implementation; functions
for id normalization and header parsing are moved to core.parse (and imported
back where needed).

Public surface: OpenAlex class with methods work, works, citers, search, wid,
nbrs, clear, close.
"""

import json
import logging
import os
import threading
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..core import parse
from ..errors import BadId, OpenAlexError

OPENALEX_BASE = "https://api.openalex.org"

log = logging.getLogger("citetools.openalex")

# bounded GET retries on a 429/5xx before giving up
_MAX_ATTEMPTS = 5
# ceiling (seconds) on the exponential backoff used when a 429/5xx carries no
# Retry-After; a server-supplied Retry-After is honored in full, uncapped
_FALLBACK_COOLDOWN = 60.0


class _RateLimiter:
	"""Shared rate-limiter for concurrent threads.

	Caps aggregate request rate at 1/interval req/s without serializing.
	acquire() reserves thread slot under lock, sleeps outside lock.
	penalize() pushes the shared slot forward (fleet-wide cooldown).
	relax() clears consecutive-strike counter on success.
	"""

	def __init__(self, interval: float) -> None:
		"""Store interval and initialize lock and next-slot timestamp.

		interval: seconds between requests (e.g. 0.1 -> 10 req/s aggregate).
		"""
		self.interval = interval
		self._lock = threading.Lock()
		self._next = 0.0
		self._strikes = 0

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
		Retry-After honored IN FULL when given (a fixed-window reset -- the
		server 429s every earlier request, so a shorter wait is wasted), else
		exponential in the consecutive-strike count capped at _FALLBACK_COOLDOWN.
		returns the cooldown applied (seconds).
		"""
		with self._lock:
			self._strikes += 1
			if retry_after is not None:
				cooldown = retry_after
			else:
				cooldown = min(2.0 ** self._strikes, _FALLBACK_COOLDOWN)
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
		cache_path: str | None = None,
		cache_flush_every: int = 200,
		session: requests.Session | None = None,
	) -> None:
		"""Initialize OpenAlex client with polite-pool settings and thread-safe concurrency.

		Proc:
		  - store mailto (email for User-Agent and polite-pool qualification)
		  - store sleep (interval seconds, caps aggregate request rate)
		  - store timeout (request timeout seconds)
		  - store citer_k (default page size for citers() calls)
		  - build shared _RateLimiter(_RateLimiter(sleep))
		  - if session is injected (not None):
		      set User-Agent header to "citetools/1.0 (mailto)"
		      store in self._injected_session
		      set self._local to None (test path)
		  - else (production path):
		      store _injected_session = None
		      init self._local = threading.local() (thread-local storage)
		      init self._sessions = [] (list of managed sessions)
		      init self._sessions_lock = threading.Lock()
		  - init self._cache = {} (canonical_wid -> full work dict)
		  - init self._meta = {} (canonical_wid -> trimmed work dict)
		  - build self.params = {"mailto": mailto} (passed to every HTTP call)

		Args:
		  mailto: email address for User-Agent header and polite-pool qualification
		  sleep: delay (seconds) after each successful HTTP response; also caps
		    aggregate request rate across threads
		  timeout: HTTP request timeout (seconds)
		  citer_k: default page size for citers() calls (overridable per call)
		  session: optional requests.Session for test injection; if None, uses
		    thread-local sessions (production)

		Raises:
		  None (validation deferred to the caller)
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

		# optional on-disk cache: persists _cache/_meta across runs so repeated
		# fetches never re-hit openalex (cuts the 429 rate). disabled when
		# cache_path is None -- then behavior is unchanged.
		self._cache_path = Path(cache_path) if cache_path else None
		self._flush_every = max(1, cache_flush_every)
		self._cache_lock = threading.Lock()
		self._gets_ok = 0
		if self._cache_path is not None and self._cache_path.exists():
			self._load_cache()

		self.params = {"mailto": mailto}

	def _load_cache(self) -> None:
		"""Populate _cache/_meta from the on-disk cache file.

		A malformed or unreadable file is ignored (start empty) with a warning;
		a stale cache must never abort a run.
		"""
		try:
			data = json.loads(self._cache_path.read_text())
			self._cache = data.get("cache", {})
			self._meta = data.get("meta", {})
			log.info("loaded %d cached work(s) from %s", len(self._cache),
			         self._cache_path)
		except Exception as e:
			log.warning("ignoring unreadable cache %s: %s", self._cache_path, e)

	def _flush(self) -> None:
		"""Atomically persist _cache/_meta to the on-disk cache file.

		No-op when on-disk caching is off. Snapshots both dicts (a C-level
		atomic copy, safe under concurrent fetches), writes a sibling temp file,
		then os.replace() -- so a kill mid-write leaves the prior file intact,
		never a corrupt one.
		"""
		if self._cache_path is None:
			return
		snapshot = {"cache": dict(self._cache), "meta": dict(self._meta)}
		self._cache_path.parent.mkdir(parents=True, exist_ok=True)
		tmp = self._cache_path.with_suffix(self._cache_path.suffix + ".tmp")
		tmp.write_text(json.dumps(snapshot))
		os.replace(tmp, self._cache_path)

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

		desc: human label for the resource (used in logs/errors)

		Proc: up to _MAX_ATTEMPTS attempts:
		  - _limiter.acquire() blocks for fleet-wide cooldown
		  - GET request; transport error -> raise OpenAlexError
		  - status 429 or >= 500 -> parse Retry-After header via parse.retry_after,
		    _limiter.penalize(retry_after), log warning, retry
		  - else -> raise_for_status, _limiter.relax(), return parsed json
		  - exhaust attempts -> raise OpenAlexError

		Raises:
		  OpenAlexError (transport failure, non-retryable status, or exhausted retries)
		"""
		for attempt in range(1, _MAX_ATTEMPTS + 1):
			self._limiter.acquire()
			try:
				response = self._session().get(url, params=params, timeout=self.timeout)
			except requests.RequestException as e:
				raise OpenAlexError(f"failed to fetch {desc}: {e}") from e

			if response.status_code == 429 or response.status_code >= 500:
				cooldown = self._limiter.penalize(parse.retry_after(response))
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

			# periodic on-disk cache flush so a killed run still persists most
			# of its fetches; no-op when on-disk caching is disabled
			if self._cache_path is not None:
				with self._cache_lock:
					self._gets_ok += 1
					due = self._gets_ok % self._flush_every == 0
				if due:
					self._flush()

			return data

		raise OpenAlexError(
			f"failed to fetch {desc}: still rate-limited after {_MAX_ATTEMPTS} attempts"
		)

	def work(self, pid: str) -> dict[str, Any]:
		"""Fetch one full work by any id form; cache by canonical W-id.

		Proc:
		  - normalize pid via parse.key(pid) -> lookup_key
		  - check _cache[lookup_key]; if present, return cached data
		  - GET /works/{lookup_key}; parse JSON response
		  - extract canonical W-id: data["id"].rsplit("/", 1)[-1].upper()
		  - cache full data in _cache[canonical_wid] and _cache[lookup_key]
		  - cache full data in _meta[canonical_wid]
		  - return data

		Raises:
		  BadId (from parse.key())
		  OpenAlexError (HTTP failures via _get)
		"""
		lookup_key = parse.key(pid)

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

		Proc:
		  - convert pids to list
		  - if empty, return {}
		  - init to_fetch = [], out = {}
		  - for each pid in pids_list:
		      * normalize via parse.key(pid) -> lookup_key
		      * check _cache[lookup_key]:
		          - if hit: extract canonical_wid from cached data; add to out
		          - if miss: append lookup_key to to_fetch
		  - filter to_fetch to bare W-ids (skip "doi:" prefixed, as they require
		    separate work() calls)
		  - batch by 50:
		      * build filter_str = "openalex:" + "|".join(chunk)
		      * GET /works with filter and per-page=50
		      * cache each returned work in _cache[canonical_wid], _meta[canonical_wid]
		      * add to out
		  - return out

		Raises:
		  BadId (from parse.key())
		  OpenAlexError (HTTP failures)

		Edge cases:
		  - empty input -> return {} (no HTTP call)
		  - missing ids -> absent from result
		  - DOIs -> not fetched via batch (cached only, not in to_fetch)
		"""
		pids_list = list(pids)

		# early exit
		if not pids_list:
			return {}

		# collect to_fetch and populate out from cache
		to_fetch: list[str] = []
		out: dict[str, dict[str, Any]] = {}

		for pid in pids_list:
			lookup_key = parse.key(pid)
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
		"""Fetch one page of citing works for a work, ranked by cited_by_count desc.

		Proc:
		  - if k is None, set k = self.citer_k
		  - fetch seed work via work(pid) -> canonical_wid via rsplit
		  - GET /works?filter=cites:{canonical_wid}&sort=cited_by_count:desc&per-page=k
		    with select="id,title,publication_year,cited_by_count,doi"
		  - extract canonical W-ids from results; cache trimmed work dicts in _meta only
		    (NOT _cache, to avoid partial-cache poisoning)
		  - return list of canonical W-ids
		  - zero citers -> [] (graceful)

		Raises:
		  BadId (from work())
		  OpenAlexError (HTTP failures)

		Edge cases:
		  - unknown node -> work() may raise or return data with no citers
		  - missing k -> uses self.citer_k
		  - one page only, no pagination
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

		Proc:
		  - if query is blank, return []
		  - GET /works?filter=title.search:{query}&per-page=k
		    with select="id,title,publication_year,cited_by_count,doi"
		  - extract trimmed work dicts from results
		  - cache in _meta only
		  - return results

		Raises:
		  OpenAlexError (HTTP failures)

		Edge cases:
		  - blank query -> [] (no HTTP call)
		  - one page only, no pagination
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
		"""Resolve any id form to its canonical OpenAlex W-id.

		Proc:
		  - fetch work via work(pid)
		  - extract and return canonical W-id: data["id"].rsplit("/", 1)[-1].upper()

		Raises:
		  BadId (from work())
		  OpenAlexError (HTTP failures)
		"""
		return self.work(pid)["id"].rsplit("/", 1)[-1].upper()

	def nbrs(self, node: str) -> dict[str, Any]:
		"""Fetch neighborhood (refs + citers) and metadata for a node.

		Oracle method for core.engines.intersection and runtime.fetch_all.
		Composes work(node) (for referenced_works -> refs) and citers(node);
		shapes the result using core.parse helpers; caches in _meta.

		Proc:
		  - fetch work via work(node) -- handles canonicalization, caching, errors
		  - if work fetch raises or returns None: return {"refs": [], "citers": [],
		    "title": None, "year": None, "doi": None, "cited_by_count": None}
		  - extract refs:
		      * get work["referenced_works"] (list of full OpenAlex URLs)
		      * for each url: parse.key(url) to canonicalize to bare W-id
		      * null-guard: absent/None -> []
		  - extract citers: citers(node) -> list of bare W-ids (already canonical)
		  - extract metadata: meta_dict = parse.meta(work) -> dict with keys
		    {title, year, doi, cited_by_count}; None-safe
		  - merge and return: {
		      "refs": refs,
		      "citers": citers,
		      "title": meta_dict["title"],
		      "year": meta_dict["year"],
		      "doi": meta_dict["doi"],
		      "cited_by_count": meta_dict["cited_by_count"]
		    }

		Args:
		  node: canonical W-id (bare "Wxxxx") or any form accepted by work()

		Returns:
		  dict with keys: refs, citers, title, year, doi, cited_by_count.
		  - refs: list of canonical W-ids referenced by node
		  - citers: list of canonical W-ids citing node
		  - title, year, doi, cited_by_count: metadata (may be None)
		  - unknown or missing node -> refs=[], citers=[], metadata=None

		Raises:
		  BadId (from work() if normalization fails)
		  OpenAlexError (HTTP failures)

		Edge cases:
		  - node not found: work() may raise or return sparse data; return empty
		    neighbors + None metadata gracefully
		  - referenced_works absent or None: refs = []
		  - node has no citers: citers = [] (graceful)
		  - no metadata: parse.meta returns all-None dict
		"""
		try:
			work_data = self.work(node)
		except Exception:
			# unknown node or fetch error; return empty neighborhood
			return {
				"refs": [],
				"citers": [],
				"title": None,
				"year": None,
				"doi": None,
				"cited_by_count": None,
			}

		if work_data is None:
			return {
				"refs": [],
				"citers": [],
				"title": None,
				"year": None,
				"doi": None,
				"cited_by_count": None,
			}

		# extract refs: canonicalize each referenced_works URL to bare W-id
		refs_raw = work_data.get("referenced_works") or []
		refs = [parse.key(url) for url in refs_raw]

		# extract citers: already returns canonical W-ids
		citers_list = self.citers(node)

		# extract metadata using parse.meta (None-safe)
		meta_dict = parse.meta(work_data)

		return {
			"refs": refs,
			"citers": citers_list,
			"title": meta_dict["title"],
			"year": meta_dict["year"],
			"doi": meta_dict["doi"],
			"cited_by_count": meta_dict["cited_by_count"],
		}

	def clear(self) -> None:
		"""Drop all cached full works and metadata.

		Useful for long-running or memory-constrained scripts.
		Next fetch will hit the API.
		"""
		self._cache.clear()
		self._meta.clear()

	def close(self) -> None:
		"""Persist the on-disk cache (if enabled), then close all managed sessions.

		Test path: if session was injected, close it. Production path: close all
		thread-local sessions registered in self._sessions. Idempotent.
		"""
		self._flush()
		if self._injected_session is not None:
			self._injected_session.close()
		else:
			with self._sessions_lock:
				for s in self._sessions:
					s.close()
