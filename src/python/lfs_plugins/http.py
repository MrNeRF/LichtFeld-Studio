# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared HTTP helpers with a CA-bundle fallback for bundled Python runtimes."""

from __future__ import annotations

import logging
import os
import ssl
import urllib.error
import urllib.request
from functools import lru_cache
from typing import Optional

_log = logging.getLogger(__name__)


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
            _log.debug("Failed to resolve CA bundle from %s: %s", module_name, exc)
            continue

        if cafile and os.path.exists(cafile):
            return cafile
    return None


@lru_cache(maxsize=1)
def _fallback_ssl_context() -> Optional[ssl.SSLContext]:
    cafile = _load_ca_bundle_from_certifi()
    if not cafile:
        return None

    try:
        return ssl.create_default_context(cafile=cafile)
    except Exception as exc:
        _log.warning("Failed to create fallback SSL context from '%s': %s", cafile, exc)
        return None


def _is_cert_verify_error(exc: BaseException) -> bool:
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


def urlopen(url, *, timeout: float, **kwargs):
    """Open a URL and retry certificate failures with a certifi CA bundle if available."""

    try:
        return urllib.request.urlopen(url, timeout=timeout, **kwargs)
    except Exception as exc:
        if not _is_cert_verify_error(exc):
            raise
        first_error = exc

    context = _fallback_ssl_context()
    if context is None:
        raise first_error

    _log.info("Retrying HTTPS request with fallback CA bundle: %s", getattr(url, "full_url", url))
    return urllib.request.urlopen(url, timeout=timeout, context=context, **kwargs)
