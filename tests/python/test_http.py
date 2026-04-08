# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for HTTP certificate fallback handling."""

import ssl
import urllib.error

import pytest


class _ResponseStub:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return b"ok"


def test_urlopen_retries_with_fallback_context(monkeypatch):
    from lfs_plugins import http

    fallback_context = object()
    calls = []

    monkeypatch.setattr(http, "_fallback_ssl_context", lambda: fallback_context)

    cert_error = urllib.error.URLError(
        ssl.SSLCertVerificationError(1, "CERTIFICATE_VERIFY_FAILED")
    )

    def fake_urlopen(url, timeout=None, **kwargs):
        calls.append(kwargs.get("context"))
        if len(calls) == 1:
            raise cert_error
        return _ResponseStub()

    monkeypatch.setattr(http.urllib.request, "urlopen", fake_urlopen)

    resp = http.urlopen("https://example.com", timeout=5)

    assert isinstance(resp, _ResponseStub)
    assert calls == [None, fallback_context]


def test_urlopen_does_not_retry_non_certificate_errors(monkeypatch):
    from lfs_plugins import http

    calls = []

    monkeypatch.setattr(http, "_fallback_ssl_context", lambda: object())

    def fake_urlopen(url, timeout=None, **kwargs):
        calls.append(kwargs.get("context"))
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(http.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(urllib.error.URLError, match="connection refused"):
        http.urlopen("https://example.com", timeout=5)

    assert calls == [None]


def test_urlopen_propagates_cert_error_without_fallback_context(monkeypatch):
    from lfs_plugins import http

    calls = []
    cert_error = urllib.error.URLError(
        ssl.SSLCertVerificationError(1, "CERTIFICATE_VERIFY_FAILED")
    )

    monkeypatch.setattr(http, "_fallback_ssl_context", lambda: None)

    def fake_urlopen(url, timeout=None, **kwargs):
        calls.append(kwargs.get("context"))
        raise cert_error

    monkeypatch.setattr(http.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(urllib.error.URLError):
        http.urlopen("https://example.com", timeout=5)

    assert calls == [None]
