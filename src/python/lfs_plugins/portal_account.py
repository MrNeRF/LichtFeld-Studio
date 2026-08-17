# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Portal account device authorization and credential management."""

from __future__ import annotations

import json
import logging
import math
import os
import platform as platform_module
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Callable, Iterator, Mapping, Optional

from .http import urlopen

_log = logging.getLogger(__name__)

DEFAULT_PORTAL_URL = "https://portal.lichtfeld.io"
PORTAL_URL_ENV = "LFS_PORTAL_URL"
CREDENTIALS_VERSION = 1
HTTP_TIMEOUT_SEC = 10.0

DEVICE_START_PATH = "/api/v1/auth/device/start/"
DEVICE_TOKEN_PATH = "/api/v1/auth/device/token/"
REFRESH_PATH = "/api/v1/auth/refresh/"
REVOKE_PATH = "/api/v1/auth/revoke/"
ME_PATH = "/api/v1/me/"

_TERMINAL_DEVICE_ERRORS = {"access_denied", "expired_token", "invalid_grant", "membership_required"}
_MAX_CONSECUTIVE_POLL_FAILURES = 5


class PortalAccountError(RuntimeError):
    """Base error for portal account operations."""


class PortalProtocolError(PortalAccountError):
    """Raised when the portal returns a response outside the pinned contract."""


class PortalHTTPError(PortalAccountError):
    """A contract-shaped HTTP error without response or token material."""

    def __init__(
        self,
        status: int,
        error: str,
        retry_after: Optional[float] = None,
        detail: Optional[Mapping[str, object]] = None,
    ) -> None:
        self.status = status
        self.error = error
        self.retry_after = retry_after
        self.detail = dict(detail) if detail is not None else None
        super().__init__(f"Portal request failed with HTTP {status}: {error or 'unknown_error'}")


class PortalOriginMismatchError(PortalAccountError):
    """Raised before credentials could be sent to a different portal origin."""


@dataclass(frozen=True)
class AccountSnapshot:
    """Token-free account state consumed by the account panel."""

    signed_in: bool = False
    linking: bool = False
    membership_required: bool = False
    label: str = "Sign in"
    tier: str = ""
    tooltip: str = ""
    display_name: str = ""
    email: str = ""
    connected_since: str = ""
    user_code: str = ""
    verification_uri: str = ""
    verification_uri_complete: str = ""
    expires_at: float = 0.0
    countdown_seconds: int = 0
    poll_interval: float = 0.0
    error: str = ""
    portal_host: str = ""
    custom_portal: bool = False


@dataclass(frozen=True)
class _Credentials:
    portal_origin: str
    access_token: str
    access_expires_at: float
    refresh_token: str
    refresh_expires_at: float
    display_name: str = ""
    email: str = ""
    customer_tier: str = ""
    member_since: str = ""
    connected_since: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "version": CREDENTIALS_VERSION,
            "portal_origin": self.portal_origin,
            "access_token": self.access_token,
            "access_expires_at": self.access_expires_at,
            "refresh_token": self.refresh_token,
            "refresh_expires_at": self.refresh_expires_at,
            "display_name": self.display_name,
            "email": self.email,
            "customer_tier": self.customer_tier,
            "member_since": self.member_since,
            "connected_since": self.connected_since,
        }

    @classmethod
    def from_dict(cls, value: object) -> Optional[_Credentials]:
        if not isinstance(value, dict) or value.get("version") != CREDENTIALS_VERSION:
            return None

        portal_origin = value.get("portal_origin")
        access_token = value.get("access_token")
        refresh_token = value.get("refresh_token")
        if not all(isinstance(item, str) and item for item in (portal_origin, access_token, refresh_token)):
            return None

        try:
            access_expires_at = float(value["access_expires_at"])
            refresh_expires_at = float(value["refresh_expires_at"])
        except (KeyError, TypeError, ValueError):
            return None

        def text(key: str) -> str:
            item = value.get(key, "")
            return item if isinstance(item, str) else ""

        return cls(
            portal_origin=portal_origin,
            access_token=access_token,
            access_expires_at=access_expires_at,
            refresh_token=refresh_token,
            refresh_expires_at=refresh_expires_at,
            display_name=text("display_name"),
            email=text("email"),
            customer_tier=text("customer_tier"),
            member_since=text("member_since"),
            connected_since=text("connected_since"),
        )


def _canonical_portal_origin(raw_url: str) -> str:
    value = raw_url.strip()
    parsed = urllib.parse.urlsplit(value)
    scheme = parsed.scheme.lower()
    hostname = parsed.hostname

    if scheme not in ("http", "https") or not hostname:
        raise ValueError("LFS_PORTAL_URL must be an absolute HTTPS origin")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("LFS_PORTAL_URL must not contain user information")
    if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        raise ValueError("LFS_PORTAL_URL must contain only a portal origin")
    if scheme == "http" and hostname != "127.0.0.1":
        raise ValueError("Plain HTTP is only allowed for 127.0.0.1 portal testing")

    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("LFS_PORTAL_URL contains an invalid port") from exc

    host = hostname.lower()
    if ":" in host:
        host = f"[{host}]"
    netloc = f"{host}:{port}" if port is not None else host
    return f"{scheme}://{netloc}"


def _default_client_version() -> str:
    try:
        import lichtfeld as lf

        return str(lf.build_info.version)
    except Exception:
        return "unknown"


def _initials(display_name: str, email: str) -> str:
    words = [part for part in display_name.strip().split() if part]
    if len(words) >= 2:
        return f"{words[0][0]}{words[-1][0]}".upper()
    if words:
        return words[0][:2].upper()
    if email:
        return email[0].upper()
    return "LF"


def _tier_name(customer_tier: str) -> str:
    return {
        "standard": "Standard",
        "supporter": "Supporter",
    }.get(customer_tier, customer_tier)


def _error_response(raw: bytes) -> tuple[str, Optional[dict[str, object]]]:
    if not raw:
        return "", None
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return "", None
    if not isinstance(payload, dict):
        return "", None
    error = payload.get("error")
    detail = payload.get("detail")
    return (
        error if isinstance(error, str) else "",
        dict(detail) if isinstance(detail, dict) else None,
    )


def _retry_after_seconds(headers: object) -> Optional[float]:
    if headers is None:
        return None
    try:
        value = headers.get("Retry-After")
    except AttributeError:
        return None
    if value is None:
        value = headers.get("retry-after")
    if value is None:
        return None

    try:
        seconds = float(value)
    except (TypeError, ValueError):
        pass
    else:
        return max(0.0, seconds) if math.isfinite(seconds) else None

    try:
        retry_at = parsedate_to_datetime(str(value))
    except (TypeError, ValueError, OverflowError):
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=timezone.utc)
    return max(0.0, retry_at.timestamp() - time.time())


@contextmanager
def _locked_sidecar(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        if os.name == "nt":
            import msvcrt

            if os.fstat(fd).st_size == 0:
                os.write(fd, b"\0")
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX)

        try:
            yield
        finally:
            if os.name == "nt":
                import msvcrt

                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


class PortalAccountService:
    """Owns portal device linking, refresh, profile state, and sign-out."""

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        credentials_path: Optional[os.PathLike[str] | str] = None,
        client_name: str = "LichtFeld Studio",
        client_version: Optional[str] = None,
        platform: Optional[str] = None,
        timeout: float = HTTP_TIMEOUT_SEC,
        waiter: Optional[Callable[[float], bool]] = None,
    ) -> None:
        configured_url = base_url if base_url is not None else os.environ.get(PORTAL_URL_ENV, DEFAULT_PORTAL_URL)
        self.base_url = _canonical_portal_origin(configured_url)
        self.portal_host = urllib.parse.urlsplit(self.base_url).netloc
        self.is_custom_portal = self.base_url != DEFAULT_PORTAL_URL
        self.portal_url = f"{self.base_url}/"
        self.billing_url = f"{self.base_url}/billing/"

        if self.is_custom_portal:
            _log.warning("Using non-default LichtFeld portal host: %s", self.portal_host)

        resolved_data = os.environ.get("LFS_RESOLVED_DATA_DIR")
        default_credentials = (
            Path(resolved_data) / "account" / "credentials.json"
            if resolved_data
            else Path.home() / ".lichtfeld" / "account" / "credentials.json"
        )
        self.credentials_path = (
            Path(credentials_path).expanduser()
            if credentials_path
            else default_credentials
        )
        self._lock_path = self.credentials_path.with_suffix(self.credentials_path.suffix + ".lock")
        self._client_name = client_name
        self._client_version = client_version if client_version is not None else _default_client_version()
        self._platform = platform if platform is not None else platform_module.system()
        self._timeout = timeout
        self._test_waiter = waiter

        self._lock = threading.Lock()
        self._refresh_lock = threading.Lock()
        self._cancel_event = threading.Event()
        self._credentials: Optional[_Credentials] = None
        self._flow_thread: Optional[threading.Thread] = None
        self._sync_thread: Optional[threading.Thread] = None
        self._sign_out_thread: Optional[threading.Thread] = None
        self._initialized = False
        self._snapshot = AccountSnapshot(
            portal_host=self.portal_host,
            custom_portal=self.is_custom_portal,
            tooltip=self._with_portal_host(""),
        )

        stored = self._read_credentials_file()
        if stored is not None and stored.portal_origin == self.base_url:
            self._credentials = stored
            self._apply_credentials_state(stored)
        elif stored is not None:
            self._set_signed_out("portal_origin_mismatch")
        else:
            self._publish_account_state()

    def snapshot(self) -> AccountSnapshot:
        with self._lock:
            return self._snapshot

    @property
    def credentials_file(self) -> Path:
        return self.credentials_path

    def request_json_authenticated(
        self,
        method: str,
        path: str,
        body: Optional[Mapping[str, object]] = None,
        timeout: float = 30,
    ) -> dict[str, object]:
        """Make one bearer request with the shared single-refresh ladder."""
        return self._authenticated_request(method, path, body, timeout=timeout)

    def _redaction_tokens(self) -> tuple[str, ...]:
        credentials = self._current_credentials()
        if credentials is None:
            return ()
        return tuple(
            token
            for token in (credentials.access_token, credentials.refresh_token)
            if token
        )

    def initialize_async(self) -> None:
        """Validate a stored session once, without blocking panel registration."""
        with self._lock:
            if self._initialized:
                return
            self._initialized = True
            has_credentials = self._credentials is not None
        if has_credentials:
            self.sync_profile_async()

    def sync_profile_async(self) -> None:
        with self._lock:
            if self._sync_thread is not None and self._sync_thread.is_alive():
                return
            thread = threading.Thread(target=self.sync_profile, daemon=True, name="lfs-portal-profile")
            self._sync_thread = thread
        thread.start()

    def sync_profile(self) -> bool:
        """Fetch ``/me/``, refreshing exactly once after a bearer 401."""
        try:
            payload = self._authenticated_request("GET", ME_PATH)
        except PortalHTTPError as exc:
            if not (exc.status == 403 and exc.error == "membership_required"):
                _log.debug("Portal profile request failed with HTTP %d", exc.status)
            return False
        except (OSError, PortalProtocolError):
            _log.debug("Portal profile request was unavailable")
            return False

        credentials = self._current_credentials()
        if credentials is None:
            return False
        updated = self._credentials_with_profile(credentials, payload)
        if updated is None:
            _log.debug("Portal profile response did not match the pinned contract")
            return False
        self._merge_and_save_profile(updated)
        return True

    def start_device_flow(self) -> bool:
        """Start the device flow on a single daemon worker."""
        with self._lock:
            if self._flow_thread is not None and self._flow_thread.is_alive():
                return False
            self._cancel_event.clear()
            self._snapshot = AccountSnapshot(
                linking=True,
                label="",
                tooltip=self._with_portal_host(""),
                portal_host=self.portal_host,
                custom_portal=self.is_custom_portal,
            )
            thread = threading.Thread(target=self._device_flow_worker, daemon=True, name="lfs-portal-device")
            self._flow_thread = thread
        self._publish_account_state()
        thread.start()
        return True

    start_linking = start_device_flow

    def cancel_device_flow(self) -> None:
        self._cancel_event.set()
        self._set_signed_out("")

    cancel_linking = cancel_device_flow

    def sign_out_async(self) -> None:
        with self._lock:
            if self._sign_out_thread is not None and self._sign_out_thread.is_alive():
                return
            thread = threading.Thread(target=self.sign_out, daemon=True, name="lfs-portal-sign-out")
            self._sign_out_thread = thread
        thread.start()

    def sign_out(self) -> None:
        """Best-effort server revocation followed by unconditional local removal."""
        self._cancel_event.set()
        credentials = self._current_credentials()
        if credentials is not None and credentials.access_expires_at <= time.time():
            if credentials.refresh_expires_at > time.time():
                self._refresh_tokens(credentials.access_token)

        with self._refresh_lock:
            with _locked_sidecar(self._lock_path):
                disk = self._read_credentials_file()
                if disk is not None and disk.portal_origin == self.base_url:
                    credentials = disk
                elif credentials is not None and credentials.portal_origin != self.base_url:
                    credentials = None

                if credentials is not None and credentials.access_expires_at > time.time():
                    try:
                        self._request_with_bearer("POST", REVOKE_PATH, credentials, {})
                    except (OSError, PortalAccountError):
                        pass

                if disk is None or disk.portal_origin == self.base_url:
                    try:
                        self.credentials_path.unlink()
                    except FileNotFoundError:
                        pass
                    except OSError:
                        _log.warning("Could not remove local portal credentials")

        self._clear_current_credentials()
        self._set_signed_out("")

    def wait_for_idle(self, timeout: float = 5.0) -> None:
        """Join current workers; intended for deterministic shutdown and tests."""
        deadline = time.monotonic() + timeout
        for thread in (self._flow_thread, self._sync_thread, self._sign_out_thread):
            if thread is None or thread is threading.current_thread():
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            thread.join(remaining)

    def _device_flow_worker(self) -> None:
        try:
            start = self._request_json(
                "POST",
                DEVICE_START_PATH,
                {
                    "client_name": self._client_name,
                    "client_version": self._client_version,
                    "platform": self._platform,
                },
            )
            device_code = self._required_text(start, "device_code")
            user_code = self._required_text(start, "user_code")
            verification_uri = self._required_text(start, "verification_uri")
            verification_uri_complete = self._required_text(start, "verification_uri_complete")
            expires_in = self._required_number(start, "expires_in")
            interval = self._required_number(start, "interval")
        except PortalHTTPError as exc:
            self._set_signed_out(exc.error or "sign_in_failed")
            return
        except (OSError, PortalProtocolError):
            self._set_signed_out("sign_in_unavailable")
            return

        if self._cancel_event.is_set():
            return

        expires_at = time.time() + max(0.0, expires_in)
        interval = max(0.0, interval)
        self._set_linking(
            user_code=user_code,
            verification_uri=verification_uri,
            verification_uri_complete=verification_uri_complete,
            expires_at=expires_at,
            interval=interval,
        )
        consecutive_failures = 0

        while not self._cancel_event.is_set():
            if time.time() >= expires_at:
                self._set_signed_out("expired_token")
                return
            remaining_lifetime = max(0.0, expires_at - time.time())
            if not self._wait_for_poll(min(interval, remaining_lifetime)):
                return
            if time.time() >= expires_at:
                self._set_signed_out("expired_token")
                return

            try:
                token_pair = self._request_json("POST", DEVICE_TOKEN_PATH, {"device_code": device_code})
            except PortalHTTPError as exc:
                if exc.status == 400 and exc.error == "authorization_pending":
                    consecutive_failures = 0
                    continue
                if (exc.status == 400 and exc.error == "slow_down") or exc.status == 429:
                    interval += 5.0
                    if exc.retry_after is not None:
                        interval = max(interval, exc.retry_after)
                    consecutive_failures = 0
                    self._set_linking(
                        user_code=user_code,
                        verification_uri=verification_uri,
                        verification_uri_complete=verification_uri_complete,
                        expires_at=expires_at,
                        interval=interval,
                    )
                    continue
                if exc.error in _TERMINAL_DEVICE_ERRORS:
                    self._set_signed_out(exc.error)
                    return
                consecutive_failures += 1
                if consecutive_failures >= _MAX_CONSECUTIVE_POLL_FAILURES:
                    self._set_signed_out("sign_in_unavailable")
                    return
                continue
            except (OSError, PortalProtocolError):
                consecutive_failures += 1
                if consecutive_failures >= _MAX_CONSECUTIVE_POLL_FAILURES:
                    self._set_signed_out("sign_in_unavailable")
                    return
                continue

            if self._cancel_event.is_set():
                return

            try:
                credentials = self._credentials_from_token_pair(token_pair)
            except PortalProtocolError:
                self._set_signed_out("sign_in_failed")
                return
            self._save_credentials(credentials)
            self._apply_credentials_state(credentials)
            self.sync_profile()
            return

    def _wait_for_poll(self, interval: float) -> bool:
        if self._test_waiter is not None:
            return not self._test_waiter(interval) and not self._cancel_event.is_set()

        deadline = time.monotonic() + interval
        while not self._cancel_event.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return True
            if self._cancel_event.wait(min(1.0, remaining)):
                return False
            self._update_countdown()
        return False

    def _refresh_tokens(
        self,
        failed_access_token: str,
        *,
        timeout: Optional[float] = None,
    ) -> str:
        """Return ``ok``, ``membership_required``, ``invalid``, or ``unavailable``."""
        with self._refresh_lock:
            with _locked_sidecar(self._lock_path):
                credentials = self._read_credentials_file()
                if credentials is None or credentials.portal_origin != self.base_url:
                    return "unavailable"
                if credentials.access_token != failed_access_token:
                    self._set_current_credentials(credentials)
                    self._apply_credentials_state(credentials)
                    return "ok"

                self._assert_active_origin(credentials)
                attempted_refresh_token = credentials.refresh_token
                try:
                    token_pair = self._request_json(
                        "POST",
                        REFRESH_PATH,
                        {"refresh_token": attempted_refresh_token},
                        timeout=timeout,
                    )
                except PortalHTTPError as exc:
                    if exc.status == 403 and exc.error == "membership_required":
                        self._set_current_credentials(credentials)
                        self._set_membership_required(credentials)
                        return "membership_required"
                    if exc.status == 401 and exc.error == "invalid_token":
                        fresh = self._read_credentials_file()
                        if (
                            fresh is not None
                            and fresh.portal_origin == self.base_url
                            and fresh.refresh_token != attempted_refresh_token
                        ):
                            self._set_current_credentials(fresh)
                            self._apply_credentials_state(fresh)
                            return "ok"
                        return "invalid"
                    return "unavailable"
                except (OSError, PortalProtocolError):
                    return "unavailable"

                try:
                    rotated = self._credentials_from_token_pair(token_pair, credentials)
                except PortalProtocolError:
                    return "unavailable"
                self._write_credentials_file(rotated)
                self._set_current_credentials(rotated)
                self._apply_credentials_state(rotated)
                return "ok"

    def _authenticated_request(
        self,
        method: str,
        path: str,
        body: Optional[Mapping[str, object]] = None,
        *,
        timeout: Optional[float] = None,
    ) -> dict[str, object]:
        credentials = self._current_credentials()
        if credentials is None:
            raise PortalHTTPError(401, "invalid_token")

        failed_access_token = credentials.access_token
        try:
            return self._request_with_bearer(
                method,
                path,
                credentials,
                body,
                timeout=timeout,
            )
        except PortalHTTPError as exc:
            if exc.status == 403 and exc.error == "membership_required":
                self._set_membership_required(credentials)
                raise
            if exc.status != 401 or exc.error != "invalid_token":
                raise

        refresh_result = self._refresh_tokens(failed_access_token, timeout=timeout)
        if refresh_result == "membership_required":
            raise PortalHTTPError(403, "membership_required")
        if refresh_result == "invalid":
            self._clear_local_credentials()
            raise PortalHTTPError(401, "invalid_token")
        if refresh_result == "unavailable":
            raise PortalProtocolError("Portal token refresh was unavailable")

        credentials = self._current_credentials()
        if credentials is None:
            self._set_signed_out("invalid_token")
            raise PortalHTTPError(401, "invalid_token")
        try:
            return self._request_with_bearer(
                method,
                path,
                credentials,
                body,
                timeout=timeout,
            )
        except PortalHTTPError as exc:
            if exc.status == 403 and exc.error == "membership_required":
                self._set_membership_required(credentials)
            elif exc.status == 401 and exc.error == "invalid_token":
                self._clear_local_credentials()
            raise

    def _request_with_bearer(
        self,
        method: str,
        path: str,
        credentials: _Credentials,
        body: Optional[Mapping[str, object]] = None,
        *,
        timeout: Optional[float] = None,
    ) -> dict[str, object]:
        self._assert_active_origin(credentials)
        return self._request_json(
            method,
            path,
            body,
            {"Authorization": f"Bearer {credentials.access_token}"},
            timeout=timeout,
        )

    def _request_json(
        self,
        method: str,
        path: str,
        body: Optional[Mapping[str, object]] = None,
        headers: Optional[Mapping[str, str]] = None,
        *,
        timeout: Optional[float] = None,
    ) -> dict[str, object]:
        request_headers = {"Accept": "application/json"}
        data = None
        if body is not None:
            request_headers["Content-Type"] = "application/json"
            data = json.dumps(dict(body), separators=(",", ":")).encode("utf-8")
        if headers:
            request_headers.update(headers)

        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers=request_headers,
            method=method,
        )
        response_headers = None
        try:
            with urlopen(
                request,
                timeout=self._timeout if timeout is None else timeout,
            ) as response:
                response_status = getattr(response, "status", None)
                if response_status is None:
                    response_status = response.getcode()
                status = int(response_status)
                response_headers = getattr(response, "headers", None)
                raw = response.read()
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            retry_after = _retry_after_seconds(getattr(exc, "headers", None))
            error, detail = _error_response(raw)
            raise PortalHTTPError(int(exc.code), error, retry_after, detail) from None

        if status == 204:
            return {}
        if status < 200 or status >= 300:
            error, detail = _error_response(raw)
            raise PortalHTTPError(
                status,
                error,
                _retry_after_seconds(response_headers),
                detail,
            )
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PortalProtocolError("Portal response was not valid JSON") from exc
        if not isinstance(payload, dict):
            raise PortalProtocolError("Portal response was not a JSON object")
        return payload

    def _credentials_from_token_pair(
        self,
        payload: Mapping[str, object],
        cached: Optional[_Credentials] = None,
    ) -> _Credentials:
        access_token = self._required_text(payload, "access_token")
        refresh_token = self._required_text(payload, "refresh_token")
        token_type = self._required_text(payload, "token_type")
        if token_type != "Bearer":
            raise PortalProtocolError("Portal returned an unsupported token type")
        expires_in = self._required_number(payload, "expires_in")
        refresh_expires_in = self._required_number(payload, "refresh_expires_in")
        now = time.time()
        return _Credentials(
            portal_origin=self.base_url,
            access_token=access_token,
            access_expires_at=now + max(0.0, expires_in),
            refresh_token=refresh_token,
            refresh_expires_at=now + max(0.0, refresh_expires_in),
            display_name=cached.display_name if cached else "",
            email=cached.email if cached else "",
            customer_tier=cached.customer_tier if cached else "",
            member_since=cached.member_since if cached else "",
            connected_since=cached.connected_since if cached else "",
        )

    def _credentials_with_profile(
        self,
        credentials: _Credentials,
        payload: Mapping[str, object],
    ) -> Optional[_Credentials]:
        display_name = payload.get("display_name")
        email = payload.get("email")
        customer_tier = payload.get("customer_tier")
        if not all(isinstance(item, str) for item in (display_name, email, customer_tier)):
            return None
        session = payload.get("session")
        if not isinstance(session, dict):
            return None
        connected_since = session.get("created_at", "")
        if not isinstance(connected_since, str):
            return None
        member_since = payload.get("member_since", "")
        if member_since is None:
            member_since = ""
        if not isinstance(member_since, str):
            return None
        return replace(
            credentials,
            display_name=display_name,
            email=email,
            customer_tier=customer_tier,
            member_since=member_since,
            connected_since=connected_since,
        )

    def _merge_and_save_profile(self, profile: _Credentials) -> None:
        with _locked_sidecar(self._lock_path):
            disk = self._read_credentials_file()
            if disk is None or disk.portal_origin != self.base_url:
                self._clear_current_credentials()
                self._set_signed_out("")
                return
            merged = replace(
                disk,
                display_name=profile.display_name,
                email=profile.email,
                customer_tier=profile.customer_tier,
                member_since=profile.member_since,
                connected_since=profile.connected_since,
            )
            self._write_credentials_file(merged)
        self._set_current_credentials(merged)
        self._apply_credentials_state(merged)

    def _save_credentials(self, credentials: _Credentials) -> None:
        with _locked_sidecar(self._lock_path):
            self._write_credentials_file(credentials)
        self._set_current_credentials(credentials)

    def _write_credentials_file(self, credentials: _Credentials) -> None:
        directory = self.credentials_path.parent
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        temp_path = directory / (
            f".{self.credentials_path.name}.{os.getpid()}.{threading.get_ident()}."
            f"{secrets.token_hex(8)}.tmp"
        )
        encoded = (json.dumps(credentials.to_dict(), indent=2, sort_keys=True) + "\n").encode("utf-8")
        fd = -1
        try:
            fd = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as handle:
                fd = -1
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self.credentials_path)
        finally:
            if fd >= 0:
                os.close(fd)
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass

    def _read_credentials_file(self) -> Optional[_Credentials]:
        try:
            with self.credentials_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except FileNotFoundError:
            return None
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            _log.warning("Ignoring unreadable portal credentials")
            return None
        credentials = _Credentials.from_dict(payload)
        if credentials is None:
            _log.warning("Ignoring portal credentials with an unsupported format")
        return credentials

    def _clear_local_credentials(self) -> None:
        try:
            self.credentials_path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            _log.warning("Could not remove local portal credentials")
        self._clear_current_credentials()
        self._set_signed_out("")

    def _current_credentials(self) -> Optional[_Credentials]:
        with self._lock:
            return self._credentials

    def _set_current_credentials(self, credentials: _Credentials) -> None:
        with self._lock:
            self._credentials = credentials

    def _clear_current_credentials(self) -> None:
        with self._lock:
            self._credentials = None

    def _assert_active_origin(self, credentials: _Credentials) -> None:
        if credentials.portal_origin != self.base_url:
            raise PortalOriginMismatchError("Stored portal credentials belong to another origin")

    def _with_portal_host(self, tooltip: str) -> str:
        if self.is_custom_portal and tooltip:
            return f"{tooltip} — {self.portal_host}"
        if self.is_custom_portal:
            return self.portal_host
        return tooltip

    def _apply_credentials_state(self, credentials: _Credentials) -> None:
        name = credentials.display_name or credentials.email
        tooltip = name or "LichtFeld Portal account"
        with self._lock:
            self._snapshot = AccountSnapshot(
                signed_in=True,
                label=_initials(credentials.display_name, credentials.email),
                tier=_tier_name(credentials.customer_tier),
                tooltip=self._with_portal_host(tooltip),
                display_name=credentials.display_name,
                email=credentials.email,
                connected_since=credentials.connected_since,
                portal_host=self.portal_host,
                custom_portal=self.is_custom_portal,
            )
        self._publish_account_state()

    def _set_membership_required(self, credentials: _Credentials) -> None:
        name = credentials.display_name or credentials.email
        with self._lock:
            self._snapshot = AccountSnapshot(
                signed_in=True,
                membership_required=True,
                label=_initials(credentials.display_name, credentials.email),
                tier=_tier_name(credentials.customer_tier),
                tooltip=self._with_portal_host(name),
                display_name=credentials.display_name,
                email=credentials.email,
                connected_since=credentials.connected_since,
                error="membership_required",
                portal_host=self.portal_host,
                custom_portal=self.is_custom_portal,
            )
        self._publish_account_state()

    def _set_linking(
        self,
        *,
        user_code: str,
        verification_uri: str,
        verification_uri_complete: str,
        expires_at: float,
        interval: float,
    ) -> None:
        remaining = max(0, int(expires_at - time.time() + 0.999))
        with self._lock:
            self._snapshot = AccountSnapshot(
                linking=True,
                label="",
                tooltip=self._with_portal_host(f"{remaining // 60}:{remaining % 60:02d}"),
                user_code=user_code,
                verification_uri=verification_uri,
                verification_uri_complete=verification_uri_complete,
                expires_at=expires_at,
                countdown_seconds=remaining,
                poll_interval=interval,
                portal_host=self.portal_host,
                custom_portal=self.is_custom_portal,
            )
        self._publish_account_state()

    def _update_countdown(self) -> None:
        with self._lock:
            current = self._snapshot
            if not current.linking:
                return
            remaining = max(0, int(current.expires_at - time.time() + 0.999))
            if remaining == current.countdown_seconds:
                return
            self._snapshot = replace(
                current,
                countdown_seconds=remaining,
                tooltip=self._with_portal_host(f"{remaining // 60}:{remaining % 60:02d}"),
            )
        self._publish_account_state()

    def _set_signed_out(self, error: str) -> None:
        with self._lock:
            self._snapshot = AccountSnapshot(
                label="",
                tooltip=self._with_portal_host(""),
                error=error,
                portal_host=self.portal_host,
                custom_portal=self.is_custom_portal,
            )
        self._publish_account_state()

    def _publish_account_state(self) -> None:
        snapshot = self.snapshot()
        try:
            from .ui.store import RuntimeState

            RuntimeState.account_state.value = {
                "signed_in": snapshot.signed_in,
                "linking": snapshot.linking,
                "membership_required": snapshot.membership_required,
                "label": snapshot.label,
                "tier": snapshot.tier,
                "tooltip": snapshot.tooltip,
            }
        except Exception:
            _log.debug("Portal account state bridge is not available")

    @staticmethod
    def _required_text(payload: Mapping[str, object], key: str) -> str:
        value = payload.get(key)
        if not isinstance(value, str) or not value:
            raise PortalProtocolError(f"Portal response is missing '{key}'")
        return value

    @staticmethod
    def _required_number(payload: Mapping[str, object], key: str) -> float:
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PortalProtocolError(f"Portal response is missing '{key}'")
        return float(value)


_service_lock = threading.Lock()
_service: Optional[PortalAccountService] = None


def get_portal_account_service() -> PortalAccountService:
    """Return the process-wide account service."""
    global _service
    with _service_lock:
        if _service is None:
            _service = PortalAccountService()
        return _service


def initialize_portal_account() -> PortalAccountService:
    service = get_portal_account_service()
    service.initialize_async()
    return service
