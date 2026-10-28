"""Small, mockable helpers for resilient Lightkurve/MAST retrieval."""

_TRANSIENT_MARKERS = (
    "request failed", "remotedisconnected", "connection aborted",
    "connection reset", "max retries exceeded", "timeout", "timed out",
    "ssleoferror", "unexpected_eof", "temporarily unavailable",
    "service unavailable", " 429", " 500", " 502", " 503", " 504",
)


def looks_transient(message):
    """Return whether a MAST/HTTP error is plausibly temporary."""
    text = str(message).lower()
    return any(marker in text for marker in _TRANSIENT_MARKERS)


class ProductDownloadError(RuntimeError):
    """Raised when none of a target's individual light-curve products download."""


def download_search_result(search, lightkurve, download_dir, quality_bitmask):
    """Try Lightkurve's batch download, then retry products individually.

    A single stale or unavailable sector product can make download_all fail
    for an otherwise useful target. Individual retrieval preserves any
    sectors that are available. The batch error is retained if all fail.
    """
    batch_error = None
    try:
        collection = search.download_all(
            download_dir=download_dir,
            quality_bitmask=quality_bitmask,
        )
        if collection is not None:
            return collection
        batch_error = RuntimeError("download_all returned no collection")
    except Exception as exc:
        batch_error = exc

    curves = []
    failures = []
    for index in range(len(search)):
        try:
            curve = search[index].download(
                download_dir=download_dir,
                quality_bitmask=quality_bitmask,
            )
            if curve is not None:
                curves.append(curve)
        except Exception as exc:
            failures.append(f"product {index}: {type(exc).__name__}: {exc}")

    if curves:
        return lightkurve.LightCurveCollection(curves)

    detail = "; ".join(failures) if failures else "no individual products returned data"
    raise ProductDownloadError(
        f"batch download failed ({type(batch_error).__name__}: {batch_error}); "
        f"individual downloads also failed: {detail}"
    ) from batch_error
