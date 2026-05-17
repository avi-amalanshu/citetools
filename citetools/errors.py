"""exception hierarchy for openalax client failures (transport, http, network)."""


class OpenAlexError(Exception):
	"""base exception for openalax client failures (transport, http, network)."""


class BadId(OpenAlexError, ValueError):
	"""raised by openalax.key() on a malformed paper id (doi, w-id, or url)."""
