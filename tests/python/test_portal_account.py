# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Contract tests for the LichtFeld portal account client."""

from __future__ import annotations

import json
import logging
import os
import stat
import time
import urllib.error
import urllib.parse
from collections import deque
from pathlib import Path

import pytest

from lfs_plugins import portal_account


class FakeResponse:
    def __init__(self, status: int, payload=None, headers=None):
        self.status = status
        self.headers = headers or {}
        if payload is None:
            self._raw = b""
        elif isinstance(payload, bytes):
            self._raw = payload
        else:
            self._raw = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def getcode(self):
        return self.status

    def read(self):
        return self._raw


class StubUrlopen:
    def __init__(self, *responses):
        self.responses = deque(responses)
        self.requests = []

    def __call__(self, request, *, timeout, **kwargs):
        del timeout, kwargs
        self.requests.append(request)
        if not self.responses:
            raise AssertionError(f"Unexpected request: {request.full_url}")
        response = self.responses.popleft()
        if callable(response):
            response = response(request)
        if isinstance(response, BaseException):
            raise response
        if isinstance(response, tuple):
            return FakeResponse(*response)
        return FakeResponse(200, response)


def token_pair(access="access-new", refresh="refresh-new"):
    return {
        "access_token": access,
        "expires_in": 3600,
        "refresh_token": refresh,
        "refresh_expires_in": 90 * 24 * 60 * 60,
        "token_type": "Bearer",
    }


def profile(name="Ada Lovelace", tier="Professional"):
    return {
        "display_name": name,
        "email": "ada@example.com",
        "customer_status": "active",
        "customer_tier": tier,
        "has_download_access": True,
        "member_since": "2025-01-02T03:04:05Z",
        "session": {
            "device_name": "LichtFeld Studio",
            "created_at": "2026-02-03T04:05:06Z",
        },
    }


def start_response(interval=0, expires_in=600, device_code="device-secret"):
    return {
        "device_code": device_code,
        "user_code": "KXQ4-7RTM",
        "verification_uri": "https://portal.lichtfeld.io/link/",
        "verification_uri_complete": "https://portal.lichtfeld.io/link/?code=KXQ4-7RTM",
        "expires_in": expires_in,
        "interval": interval,
    }


def write_credentials(
    path: Path,
    *,
    origin=portal_account.DEFAULT_PORTAL_URL,
    access="access-old",
    refresh="refresh-old",
    access_expires_at=None,
    refresh_expires_at=None,
    display_name="Ada Lovelace",
    tier="Professional",
):
    now = time.time()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "version": portal_account.CREDENTIALS_VERSION,
                "portal_origin": origin,
                "access_token": access,
                "access_expires_at": access_expires_at if access_expires_at is not None else now + 3600,
                "refresh_token": refresh,
                "refresh_expires_at": refresh_expires_at if refresh_expires_at is not None else now + 86400,
                "display_name": display_name,
                "email": "ada@example.com",
                "customer_tier": tier,
                "member_since": "2025-01-02T03:04:05Z",
                "connected_since": "2026-02-03T04:05:06Z",
            }
        ),
        encoding="utf-8",
    )
    os.chmod(path, 0o600)


def request_path(request):
    return urllib.parse.urlsplit(request.full_url).path


def request_json(request):
    return json.loads(request.data.decode("utf-8")) if request.data else None


def make_service(tmp_path, *, waiter=None, base_url=None):
    return portal_account.PortalAccountService(
        base_url=base_url,
        credentials_path=tmp_path / "account" / "credentials.json",
        client_version="1.2.3",
        platform="TestOS",
        waiter=waiter,
    )


def test_device_flow_state_machine_polls_and_caches_profile(tmp_path, monkeypatch):
    service_holder = {}
    poll_count = 0

    def pending_then_tokens(request):
        nonlocal poll_count
        poll_count += 1
        snapshot = service_holder["service"].snapshot()
        assert snapshot.linking is True
        assert snapshot.user_code == "KXQ4-7RTM"
        assert 0 < snapshot.countdown_seconds <= 600
        if poll_count == 1:
            return (400, {"error": "authorization_pending"})
        return (200, token_pair())

    stub = StubUrlopen(start_response(), pending_then_tokens, pending_then_tokens, profile())
    monkeypatch.setattr(portal_account, "urlopen", stub)
    service = make_service(tmp_path, waiter=lambda _seconds: False)
    service_holder["service"] = service

    assert service.start_device_flow() is True
    assert service.start_device_flow() is False
    service.wait_for_idle()

    snapshot = service.snapshot()
    assert snapshot.signed_in is True
    assert snapshot.linking is False
    assert snapshot.display_name == "Ada Lovelace"
    assert snapshot.tier == "Professional"
    assert snapshot.label == "AL"
    assert [request_path(request) for request in stub.requests] == [
        portal_account.DEVICE_START_PATH,
        portal_account.DEVICE_TOKEN_PATH,
        portal_account.DEVICE_TOKEN_PATH,
        portal_account.ME_PATH,
    ]
    assert request_json(stub.requests[0]) == {
        "client_name": "LichtFeld Studio",
        "client_version": "1.2.3",
        "platform": "TestOS",
    }
    assert request_json(stub.requests[1]) == {"device_code": "device-secret"}
    stored = json.loads(service.credentials_file.read_text(encoding="utf-8"))
    assert stored["portal_origin"] == portal_account.DEFAULT_PORTAL_URL
    assert stored["display_name"] == "Ada Lovelace"
    assert stored["customer_tier"] == "Professional"


def test_device_start_failure_publishes_signed_out_error(tmp_path, monkeypatch):
    stub = StubUrlopen(OSError("offline"))
    monkeypatch.setattr(portal_account, "urlopen", stub)
    service = make_service(tmp_path)

    assert service.start_device_flow() is True
    service.wait_for_idle()

    snapshot = service.snapshot()
    assert snapshot.signed_in is False
    assert snapshot.linking is False
    assert snapshot.error == "sign_in_unavailable"
    assert [request_path(request) for request in stub.requests] == [portal_account.DEVICE_START_PATH]


@pytest.mark.parametrize("terminal_error", ["access_denied", "expired_token", "invalid_grant"])
def test_terminal_device_errors_stop_polling(tmp_path, monkeypatch, terminal_error):
    stub = StubUrlopen(start_response(), (400, {"error": terminal_error}))
    monkeypatch.setattr(portal_account, "urlopen", stub)
    service = make_service(tmp_path, waiter=lambda _seconds: False)

    service.start_device_flow()
    service.wait_for_idle()

    assert service.snapshot().signed_in is False
    assert service.snapshot().linking is False
    assert service.snapshot().error == terminal_error
    assert len(stub.requests) == 2
    assert not service.credentials_file.exists()


def test_slow_down_adds_five_seconds_to_poll_interval(tmp_path, monkeypatch):
    waits = []
    stub = StubUrlopen(
        start_response(interval=2),
        (400, {"error": "slow_down"}),
        token_pair(),
        profile(),
    )
    monkeypatch.setattr(portal_account, "urlopen", stub)
    service = make_service(tmp_path, waiter=lambda seconds: waits.append(seconds) or False)

    service.start_device_flow()
    service.wait_for_idle()

    assert waits == [2.0, 7.0]
    assert service.snapshot().signed_in is True


def test_poll_wait_is_capped_by_device_code_ttl(tmp_path, monkeypatch):
    now = [1000.0]
    waits = []

    def wait(seconds):
        waits.append(seconds)
        now[0] += seconds
        return False

    stub = StubUrlopen(start_response(interval=10, expires_in=3))
    monkeypatch.setattr(portal_account, "urlopen", stub)
    monkeypatch.setattr(portal_account.time, "time", lambda: now[0])
    service = make_service(tmp_path, waiter=wait)

    service.start_device_flow()
    service.wait_for_idle()

    assert waits == [3.0]
    assert service.snapshot().error == "expired_token"
    assert len(stub.requests) == 1


def test_rate_limit_honors_retry_after_and_keeps_polling(tmp_path, monkeypatch):
    waits = []
    stub = StubUrlopen(
        start_response(interval=2),
        (429, {"error": "slow_down"}, {"Retry-After": "11"}),
        token_pair(),
        profile(),
    )
    monkeypatch.setattr(portal_account, "urlopen", stub)
    service = make_service(tmp_path, waiter=lambda seconds: waits.append(seconds) or False)

    service.start_device_flow()
    service.wait_for_idle()

    assert waits == [2.0, 11.0]
    assert service.snapshot().signed_in is True


@pytest.mark.parametrize(
    "transient_response",
    [
        OSError("offline"),
        (503, {"error": "service_unavailable"}),
        (200, b"not-json"),
    ],
    ids=["network", "http", "protocol"],
)
def test_transient_poll_failure_retries_and_recovers(tmp_path, monkeypatch, transient_response):
    stub = StubUrlopen(start_response(), transient_response, token_pair(), profile())
    monkeypatch.setattr(portal_account, "urlopen", stub)
    service = make_service(tmp_path, waiter=lambda _seconds: False)

    service.start_device_flow()
    service.wait_for_idle()

    assert service.snapshot().signed_in is True
    assert [request_path(request) for request in stub.requests] == [
        portal_account.DEVICE_START_PATH,
        portal_account.DEVICE_TOKEN_PATH,
        portal_account.DEVICE_TOKEN_PATH,
        portal_account.ME_PATH,
    ]


def test_five_consecutive_poll_failures_end_the_flow(tmp_path, monkeypatch):
    failures = [(503, {"error": "service_unavailable"})] * 5
    stub = StubUrlopen(start_response(), *failures)
    monkeypatch.setattr(portal_account, "urlopen", stub)
    service = make_service(tmp_path, waiter=lambda _seconds: False)

    service.start_device_flow()
    service.wait_for_idle()

    snapshot = service.snapshot()
    assert snapshot.signed_in is False
    assert snapshot.linking is False
    assert snapshot.error == "sign_in_unavailable"
    assert len(stub.requests) == 1 + portal_account._MAX_CONSECUTIVE_POLL_FAILURES


def test_membership_required_during_device_poll_is_terminal(tmp_path, monkeypatch):
    stub = StubUrlopen(start_response(), (403, {"error": "membership_required"}))
    monkeypatch.setattr(portal_account, "urlopen", stub)
    service = make_service(tmp_path, waiter=lambda _seconds: False)

    service.start_device_flow()
    service.wait_for_idle()

    snapshot = service.snapshot()
    assert snapshot.signed_in is False
    assert snapshot.linking is False
    assert snapshot.error == "membership_required"
    assert len(stub.requests) == 2


def test_credentials_use_atomic_replace_and_mode_0600(tmp_path, monkeypatch):
    service = make_service(tmp_path)
    real_replace = portal_account.os.replace
    replaced = []

    def checked_replace(source, destination):
        replaced.append((Path(source), Path(destination)))
        assert stat.S_IMODE(Path(source).stat().st_mode) == 0o600
        real_replace(source, destination)

    monkeypatch.setattr(portal_account.os, "replace", checked_replace)
    credentials = service._credentials_from_token_pair(token_pair("access-atomic", "refresh-atomic"))

    service._save_credentials(credentials)

    assert len(replaced) == 1
    assert replaced[0][1] == service.credentials_file
    assert stat.S_IMODE(service.credentials_file.stat().st_mode) == 0o600
    stored = json.loads(service.credentials_file.read_text(encoding="utf-8"))
    assert stored["version"] == portal_account.CREDENTIALS_VERSION
    assert stored["access_token"] == "access-atomic"
    assert stored["refresh_token"] == "refresh-atomic"


def test_401_refreshes_once_and_retries_with_rotated_access_token(tmp_path, monkeypatch):
    credentials_path = tmp_path / "account" / "credentials.json"
    write_credentials(credentials_path)
    stub = StubUrlopen(
        (401, {"error": "invalid_token"}),
        token_pair("access-rotated", "refresh-rotated"),
        profile(),
    )
    monkeypatch.setattr(portal_account, "urlopen", stub)
    service = make_service(tmp_path)

    assert service.sync_profile() is True

    assert [request_path(request) for request in stub.requests] == [
        portal_account.ME_PATH,
        portal_account.REFRESH_PATH,
        portal_account.ME_PATH,
    ]
    assert stub.requests[0].get_header("Authorization") == "Bearer access-old"
    assert request_json(stub.requests[1]) == {"refresh_token": "refresh-old"}
    assert stub.requests[2].get_header("Authorization") == "Bearer access-rotated"
    stored = json.loads(credentials_path.read_text(encoding="utf-8"))
    assert stored["refresh_token"] == "refresh-rotated"


def test_second_401_does_not_refresh_twice_and_signs_out(tmp_path, monkeypatch):
    credentials_path = tmp_path / "account" / "credentials.json"
    write_credentials(credentials_path)
    stub = StubUrlopen(
        (401, {"error": "invalid_token"}),
        token_pair("access-rotated", "refresh-rotated"),
        (401, {"error": "invalid_token"}),
    )
    monkeypatch.setattr(portal_account, "urlopen", stub)
    service = make_service(tmp_path)

    assert service.sync_profile() is False

    assert sum(request_path(request) == portal_account.REFRESH_PATH for request in stub.requests) == 1
    assert not credentials_path.exists()
    assert service.snapshot().signed_in is False


def test_definitive_refresh_401_clears_credentials_and_signs_out(tmp_path, monkeypatch):
    credentials_path = tmp_path / "account" / "credentials.json"
    write_credentials(credentials_path)
    stub = StubUrlopen(
        (401, {"error": "invalid_token"}),
        (401, {"error": "invalid_token"}),
    )
    monkeypatch.setattr(portal_account, "urlopen", stub)
    service = make_service(tmp_path)

    assert service.sync_profile() is False

    assert [request_path(request) for request in stub.requests] == [
        portal_account.ME_PATH,
        portal_account.REFRESH_PATH,
    ]
    assert not credentials_path.exists()
    assert service.snapshot().signed_in is False


@pytest.mark.parametrize(
    "refresh_failure",
    [
        OSError("offline"),
        (503, {"error": "service_unavailable"}),
        (200, b"not-json"),
    ],
    ids=["network", "http", "protocol"],
)
def test_transient_refresh_failure_keeps_credentials_and_cached_state(
    tmp_path,
    monkeypatch,
    refresh_failure,
):
    credentials_path = tmp_path / "account" / "credentials.json"
    write_credentials(credentials_path, display_name="Grace Hopper", tier="Supporter")
    stub = StubUrlopen(
        (401, {"error": "invalid_token"}),
        refresh_failure,
    )
    monkeypatch.setattr(portal_account, "urlopen", stub)
    service = make_service(tmp_path)
    cached_snapshot = service.snapshot()

    assert service.sync_profile() is False

    assert credentials_path.exists()
    assert service.snapshot() == cached_snapshot
    stored = json.loads(credentials_path.read_text(encoding="utf-8"))
    assert stored["access_token"] == "access-old"
    assert stored["refresh_token"] == "refresh-old"


def test_membership_required_keeps_cached_identity_and_does_not_refresh(tmp_path, monkeypatch):
    credentials_path = tmp_path / "account" / "credentials.json"
    write_credentials(credentials_path, display_name="Grace Hopper", tier="Supporter")
    stub = StubUrlopen((403, {"error": "membership_required"}))
    monkeypatch.setattr(portal_account, "urlopen", stub)
    service = make_service(tmp_path)

    assert service.sync_profile() is False

    snapshot = service.snapshot()
    assert snapshot.signed_in is True
    assert snapshot.membership_required is True
    assert snapshot.display_name == "Grace Hopper"
    assert snapshot.tier == "Supporter"
    assert snapshot.label == "GH"
    assert credentials_path.exists()
    assert [request_path(request) for request in stub.requests] == [portal_account.ME_PATH]


def test_membership_required_during_refresh_keeps_credentials(tmp_path, monkeypatch):
    credentials_path = tmp_path / "account" / "credentials.json"
    write_credentials(credentials_path)
    stub = StubUrlopen(
        (401, {"error": "invalid_token"}),
        (403, {"error": "membership_required"}),
    )
    monkeypatch.setattr(portal_account, "urlopen", stub)
    service = make_service(tmp_path)

    assert service.sync_profile() is False

    assert service.snapshot().membership_required is True
    assert credentials_path.exists()
    stored = json.loads(credentials_path.read_text(encoding="utf-8"))
    assert stored["refresh_token"] == "refresh-old"


def test_origin_binding_never_transmits_tokens_to_different_base_url(tmp_path, monkeypatch):
    credentials_path = tmp_path / "account" / "credentials.json"
    write_credentials(credentials_path)
    stub = StubUrlopen()
    monkeypatch.setattr(portal_account, "urlopen", stub)
    service = make_service(tmp_path, base_url="https://staging.example.org")

    service.initialize_async()
    service.wait_for_idle()

    assert service.snapshot().signed_in is False
    assert service.snapshot().error == "portal_origin_mismatch"
    assert service.sync_profile() is False
    assert stub.requests == []
    assert credentials_path.exists()
    assert "staging.example.org" in service.snapshot().tooltip


@pytest.mark.parametrize(
    "url",
    [
        "http://portal.lichtfeld.io",
        "http://localhost:8000",
        "http://127.0.0.1.example.com:8000",
    ],
)
def test_http_rejected_except_exact_loopback_host(tmp_path, url):
    with pytest.raises(ValueError, match="127.0.0.1"):
        make_service(tmp_path, base_url=url)


def test_http_loopback_origin_is_allowed_and_normalized(tmp_path):
    service = make_service(tmp_path, base_url="http://127.0.0.1:8000/")

    assert service.base_url == "http://127.0.0.1:8000"
    assert service.portal_url == "http://127.0.0.1:8000/"


def test_refresh_failure_rereads_credentials_before_signing_out(tmp_path, monkeypatch):
    credentials_path = tmp_path / "account" / "credentials.json"
    write_credentials(credentials_path)

    def failed_refresh_after_other_instance_rotates(request):
        assert request_json(request) == {"refresh_token": "refresh-old"}
        write_credentials(
            credentials_path,
            access="access-from-other-instance",
            refresh="refresh-from-other-instance",
        )
        return (401, {"error": "invalid_token"})

    def profile_with_fresh_access(request):
        assert request.get_header("Authorization") == "Bearer access-from-other-instance"
        return (200, profile())

    stub = StubUrlopen(
        (401, {"error": "invalid_token"}),
        failed_refresh_after_other_instance_rotates,
        profile_with_fresh_access,
    )
    monkeypatch.setattr(portal_account, "urlopen", stub)
    service = make_service(tmp_path)

    assert service.sync_profile() is True

    assert credentials_path.exists()
    assert service.snapshot().signed_in is True
    assert sum(request_path(request) == portal_account.REFRESH_PATH for request in stub.requests) == 1


def test_sign_out_deletes_local_credentials_when_offline(tmp_path, monkeypatch):
    credentials_path = tmp_path / "account" / "credentials.json"
    write_credentials(credentials_path)
    stub = StubUrlopen(urllib.error.URLError("offline"))
    monkeypatch.setattr(portal_account, "urlopen", stub)
    service = make_service(tmp_path)

    service.sign_out()

    assert [request_path(request) for request in stub.requests] == [portal_account.REVOKE_PATH]
    assert not credentials_path.exists()
    assert service.snapshot().signed_in is False


def test_sign_out_refreshes_expired_access_before_revoke(tmp_path, monkeypatch):
    credentials_path = tmp_path / "account" / "credentials.json"
    write_credentials(credentials_path, access_expires_at=time.time() - 1)
    stub = StubUrlopen(token_pair("access-for-revoke", "refresh-for-revoke"), (204, None))
    monkeypatch.setattr(portal_account, "urlopen", stub)
    service = make_service(tmp_path)

    service.sign_out()

    assert [request_path(request) for request in stub.requests] == [
        portal_account.REFRESH_PATH,
        portal_account.REVOKE_PATH,
    ]
    assert stub.requests[1].get_header("Authorization") == "Bearer access-for-revoke"
    assert not credentials_path.exists()


def test_token_material_never_appears_in_logs_or_urls(tmp_path, monkeypatch, caplog):
    access_token = "ACCESS-MATERIAL-MUST-NOT-LOG"
    refresh_token = "REFRESH-MATERIAL-MUST-NOT-LOG"
    device_code = "DEVICE-MATERIAL-MUST-NOT-LOG"
    stub = StubUrlopen(
        start_response(device_code=device_code),
        token_pair(access_token, refresh_token),
        profile(),
    )
    monkeypatch.setattr(portal_account, "urlopen", stub)
    service = make_service(tmp_path, waiter=lambda _seconds: False)

    with caplog.at_level(logging.DEBUG, logger=portal_account.__name__):
        service.start_device_flow()
        service.wait_for_idle()

    log_output = caplog.text
    for token in (access_token, refresh_token, device_code):
        assert token not in log_output
        assert all(token not in request.full_url for request in stub.requests)
