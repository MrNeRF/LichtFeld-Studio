# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared HTTP helpers with distrusted-CA hardening and a CA-bundle fallback for bundled Python runtimes."""

from __future__ import annotations
from _lfs_path_io import native_path

import logging
import os
from functools import lru_cache
from typing import Optional
from .portal_security import redact

_log = logging.getLogger(__name__)

# eMudhra's compromised intermediate CAs. Windows still lists the roots behind them, and
# OpenSSL ignores Windows' own distrust of them, so the bundled Python refuses them itself.
_DISTRUSTED_CAFILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "distrusted_cas.pem")


def __getattr__(name):
    """Keep the historical lazy ``http.urllib`` patch surface for callers."""
    if name == "urllib":
        import urllib
        import urllib.error
        import urllib.request

        return urllib
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _load_ca_bundle_from_certifi() -> Optional[str]:
    for module_name in ("certifi", "pip._vendor.certifi"):
        try:
            module = __import__(module_name, fromlist=["where"])
        except ImportError:
            continue

        where = getattr(module, "where", None)
        if not callable(where):
            continue

        try:
            cafile = where()
        except Exception as exc:
            _log.debug("Failed to resolve CA bundle from %s: %s", module_name, redact(exc))
            continue

        if cafile and os.path.exists(native_path(cafile)):
            return cafile
    return None


def _context_with_distrusted_cas_first():
    """Create a client context whose store holds the distrusted CAs before any other CA.

    OpenSSL keeps the first copy of a certificate added to a store. A distrusted
    intermediate that arrives later as an ordinary CA, as Windows' "CA" store may
    supply it, would otherwise stay trusted.
    """
    import ssl

    return ssl.create_default_context(cafile=_DISTRUSTED_CAFILE)


@lru_cache(maxsize=1)
def _system_ssl_context():
    context = _context_with_distrusted_cas_first()
    context.load_default_certs()
    return context


@lru_cache(maxsize=1)
def _fallback_ssl_context() -> Optional[object]:
    cafile = _load_ca_bundle_from_certifi()
    if not cafile:
        return None

    try:
        context = _context_with_distrusted_cas_first()
        context.load_verify_locations(cafile=cafile)
        return context
    except Exception as exc:
        _log.warning("Failed to create fallback SSL context from '%s': %s", cafile, redact(exc))
        return None


def _is_cert_verify_error(exc: BaseException) -> bool:
    import ssl
    import urllib.error

    if isinstance(exc, ssl.SSLCertVerificationError):
        return True

    if isinstance(exc, urllib.error.URLError):
        reason = exc.reason
        if isinstance(reason, BaseException):
            return _is_cert_verify_error(reason)
        return "CERTIFICATE_VERIFY_FAILED" in str(reason)

    if isinstance(exc, ssl.SSLError):
        return "CERTIFICATE_VERIFY_FAILED" in str(exc)

    return False


def urlopen(url, *, timeout: float, no_redirect: bool = False, **kwargs):
    """Open a URL and retry certificate failures with a certifi CA bundle if available."""
    import urllib.request

    def open_request(**options):
        if not no_redirect:
            return urllib.request.urlopen(url, timeout=timeout, **options)

        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                return None

        context = options.pop("context", None)
        opener = urllib.request.build_opener(NoRedirect(), urllib.request.HTTPSHandler(context=context))
        return opener.open(url, timeout=timeout, **options)

    try:
        return open_request(context=_system_ssl_context(), **kwargs)
    except Exception as exc:
        if not _is_cert_verify_error(exc):
            raise
        first_error = exc

    context = _fallback_ssl_context()
    if context is None:
        raise first_error

    _log.info("Retrying HTTPS request with fallback CA bundle")
    return open_request(context=context, **kwargs)
