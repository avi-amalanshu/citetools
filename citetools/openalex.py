"""OpenAlex HTTP client with polite-pool retry and dual caching."""

import re
import time
from collections.abc import Iterable
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .errors import BadId, OpenAlexError

OPENALEX_BASE = "https://api.openalex.org"


class OpenAlex:
	"""HTTP client for OpenAlex API with polite-pool settings.

	Builds a requests.Session with urllib3 Retry (429 + 5xx, honours Retry-After).
	Caches full works in _cache (keyed by canonical W-id); trimmed metadata in _meta.
	session param allows test mocking; if None, creates fresh Session.
	Not thread-safe; caller serializes access.
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
		"""Initialize OpenAlex client with polite-pool settings.

		mailto: email address for User-Agent and polite-pool qualification.
		sleep: delay after each successful HTTP response (seconds).
		timeout: request timeout (seconds).
		citer_k: default page size for citers() calls.
		session: optional requests.Session for test injection; if None, creates fresh one.
		"""
		self.mailto = mailto
		self.sleep = sleep
		self.timeout = timeout
		self.citer_k = citer_k

		if session is None:
			self.session = requests.Session()
		else:
			self.session = session

		self.session.headers["User-Agent"] = f"citetools/1.0 ({mailto})"

		# configure retry on both schemes
		retry = Retry(
			total=5,
			backoff_factor=1.0,
			status_forcelist=[429, 500, 502, 503, 504],
			respect_retry_after_header=True,
		)
		adapter = HTTPAdapter(max_retries=retry)
		self.session.mount("http://", adapter)
		self.session.mount("https://", adapter)

		# dual caches: both empty dicts
		self._cache: dict[str, dict[str, Any]] = {}
		self._meta: dict[str, dict[str, Any]] = {}

		self.params = {"mailto": mailto}

	@staticmethod
	def key(pid: str) -> str:
		"""Normalize pid (DOI, bare W-id, or OpenAlex URL) to canonical lookup key.

		-> 'W12345' (uppercase), 'doi:10.xxxx/yyyy' (lowercase doi), or raises BadId.
		Accepts: 'W123', 'https://openalex.org/W123', '10.1038/nature12373',
		  'https://doi.org/10.1038/nature12373'.
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
		  _cache[key]; also in _meta. sleep, return data.
		raises BadId (from key), OpenAlexError (HTTP failures).
		"""
		lookup_key = self.key(pid)

		# check _cache
		if lookup_key in self._cache:
			return self._cache[lookup_key]

		# fetch
		url = f"{OPENALEX_BASE}/works/{lookup_key}"
		try:
			response = self.session.get(url, params=self.params, timeout=self.timeout)
			response.raise_for_status()
			data = response.json()
		except requests.RequestException as e:
			raise OpenAlexError(f"failed to fetch work {lookup_key}: {e}") from e

		# extract canonical wid
		canonical_wid = data["id"].rsplit("/", 1)[-1].upper()

		# cache in both slots
		self._cache[canonical_wid] = data
		self._cache[lookup_key] = data
		self._meta[canonical_wid] = data

		time.sleep(self.sleep)
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
				canonical_wid = lookup_key if lookup_key.startswith("W") else lookup_key
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

			try:
				response = self.session.get(
					f"{OPENALEX_BASE}/works",
					params={**self.params, "filter": filter_str, "per-page": 50},
					timeout=self.timeout,
				)
				response.raise_for_status()
				batch_data = response.json()["results"]
			except requests.RequestException as e:
				raise OpenAlexError(f"failed to fetch batch {filter_str}: {e}") from e

			# cache each work
			for w in batch_data:
				canonical_wid = w["id"].rsplit("/", 1)[-1].upper()
				self._cache[canonical_wid] = w
				self._meta[canonical_wid] = w
				out[canonical_wid] = w

			time.sleep(self.sleep)

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

		# fetch citers
		try:
			params = {
				"mailto": self.mailto,
				"filter": f"cites:{canonical_wid}",
				"sort": "cited_by_count:desc",
				"per-page": k,
				"select": "id,title,publication_year,cited_by_count,doi",
			}
			response = self.session.get(
				f"{OPENALEX_BASE}/works",
				params=params,
				timeout=self.timeout,
			)
			response.raise_for_status()
			page_data = response.json()
		except requests.RequestException as e:
			raise OpenAlexError(f"failed to fetch citers of {canonical_wid}: {e}") from e

		# extract and cache in _meta only
		results = page_data.get("results", [])
		out: list[str] = []

		for w in results:
			w_id = w["id"].rsplit("/", 1)[-1].upper()
			self._meta[w_id] = w
			out.append(w_id)

		time.sleep(self.sleep)
		return out

	def clear(self) -> None:
		"""Drop all cached full works and metadata.

		Useful for long-running or memory-constrained scripts.
		next fetch will hit the API.
		"""
		self._cache.clear()
		self._meta.clear()
