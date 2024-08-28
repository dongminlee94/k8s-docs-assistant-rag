"""Requester."""

from requests.adapters import HTTPAdapter, Retry
from requests.sessions import Session


def requests_retry_session(
    max_retries: int = 10,
    backoff_factor: int = 5,
    status_forcelist: list[int] = [400, 403, 429, 500, 502, 503, 504],
) -> Session:
    """
    Create a requests session with retry logic.

    :param max_retries: Maximum number of retries for failed requests.
    :param backoff_factor: Factor for exponential backoff strategy.
    :param status_forcelist: List of HTTP status codes to trigger a retry.
    :returns: A configured requests Session object.
    :note: If we do not receive the correct response when we send the request,
            retry as much as max_retries.
            We also use an "exponential backoff" strategy,
            where the delay is "backoff_factor * (2 ^ (retry count - 1))" after each failed retry.
    """
    # Configure retry strategy
    retry = Retry(
        total=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )

    # Create an HTTP adapter with the retry strategy
    adapter = HTTPAdapter(max_retries=retry)

    # Create and configure the session
    session = Session()
    session.mount(prefix="http://", adapter=adapter)

    return session
