# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Diagnostic collection and portal submission for in-app bug reports."""

from __future__ import annotations

import getpass
import hashlib
import logging
import os
import re
import socket
import threading
import urllib.parse
from collections.abc import Mapping
from pathlib import Path
from typing import Optional

import lichtfeld as lf

from .portal_account import (
    PortalHTTPError,
    PortalProtocolError,
    get_portal_account_service,
)
from .ui.store import RuntimeState, new_bug_report_state

_log = logging.getLogger(__name__)

BUG_REPORT_PATH = "/api/v1/bugs/"
CLIENT_NAME = "LichtFeld Studio"
LOG_MAX_BYTES = 1_048_576
CATEGORY_DETAIL_MAX_LENGTH = 120

CATEGORIES = ("crash", "training", "ui", "performance", "export", "other")
FREQUENCIES = ("always", "sometimes", "once")
STAGES = ("startup", "loading", "training", "export", "viewer", "other")

_DIAGNOSTIC_KEYS = frozenset(
    {
        "app_version",
        "app_commit",
        "build_type",
        "os",
        "os_build",
        "cpu",
        "ram_mb",
        "gpu",
        "gpu_driver",
        "cuda_runtime",
        "vram_mb",
        "vram_used_mb",
        "strategy",
        "iteration",
        "max_iterations",
        "num_gaussians",
        "sh_degree",
        "resize_factor",
        "max_width",
        "image_count",
        "dataset_fingerprint",
        "render_mode",
    }
)

_EMAIL_RE = re.compile(
    r"(?i)(?<![\w.+-])[A-Z0-9.!#$%&'*+/=?^_`{|}~-]+@"
    r"[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?(?:\."
    r"[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?)+"
)
_WINDOWS_USER_RE = re.compile(r"(?i)\b[A-Z]:[\\/]+Users[\\/]+[^\\/\s]+")
_UNC_PREFIX_RE = re.compile(r"(?<!\\)\\\\[^\\/\s]+[\\/]+[^\\/\s]+")
_AUTHORIZATION_RE = re.compile(r"(?i)Authorization\s*:\s*Bearer\s+\S+")

_submit_lock = threading.Lock()
_submit_thread: Optional[threading.Thread] = None


def _property_values(config: object) -> dict[str, object]:
    try:
        if config is None or not config.has_params():
            return {}
        properties = config.properties()
    except Exception:
        return {}

    values: dict[str, object] = {}
    for item in properties:
        try:
            prop_id = str(item["id"])
            values[prop_id] = item["value"]
        except (KeyError, TypeError):
            continue
    return values


def _dataset_properties() -> dict[str, object]:
    try:
        return _property_values(lf.dataset_params())
    except Exception:
        return {}


def _optimization_properties() -> dict[str, object]:
    try:
        return _property_values(lf.optimization_params())
    except Exception:
        return {}


def _render_mode_name(value: object) -> str:
    return str(getattr(value, "name", value))


def collect_diagnostics() -> dict[str, object]:
    """Collect only fields allowed by the frozen desktop bug-report contract."""
    diagnostics: dict[str, object] = {
        "app_version": str(getattr(lf.build_info, "version", "unknown")),
        "app_commit": str(getattr(lf.build_info, "commit", "unknown")),
        "build_type": str(getattr(lf.build_info, "build_type", "unknown")),
    }

    try:
        native = lf.diagnostics.collect()
        if isinstance(native, Mapping):
            diagnostics.update(
                (str(key), value)
                for key, value in native.items()
                if str(key) in _DIAGNOSTIC_KEYS
            )
    except Exception:
        _log.debug("Native diagnostic collection was unavailable")

    if RuntimeState.training_running.value:
        optimization = _optimization_properties()
        if optimization:
            for key in ("strategy", "sh_degree"):
                if key in optimization:
                    diagnostics[key] = optimization[key]
            runtime_fields = {
                "iteration": "trainer_current_iteration",
                "max_iterations": "trainer_total_iterations",
                "num_gaussians": "get_num_gaussians",
            }
            for key, function_name in runtime_fields.items():
                try:
                    diagnostics[key] = getattr(lf, function_name)()
                except Exception:
                    pass
            try:
                diagnostics["render_mode"] = _render_mode_name(lf.get_render_mode())
            except Exception:
                pass

        dataset = _dataset_properties()
        if dataset:
            for key in ("resize_factor", "max_width"):
                if key in dataset:
                    diagnostics[key] = dataset[key]
            data_path = dataset.get("data_path")
            if isinstance(data_path, (str, os.PathLike)) and os.fspath(data_path):
                canonical_path = os.path.realpath(os.fspath(data_path))
                diagnostics["dataset_fingerprint"] = hashlib.sha256(
                    os.fsencode(canonical_path)
                ).hexdigest()[:12]

    return {
        key: value
        for key, value in diagnostics.items()
        if key in _DIAGNOSTIC_KEYS
    }


def _replace_path(text: str, raw_path: object, replacement: str) -> str:
    if not isinstance(raw_path, (str, os.PathLike)):
        return text
    path = os.fspath(raw_path)
    if not path:
        return text
    variants = {path, os.path.realpath(path)}
    variants.update(value.rstrip("/\\") for value in tuple(variants))
    for value in sorted((value for value in variants if value), key=len, reverse=True):
        boundary = "" if value.endswith(("/", "\\")) else r"(?![A-Za-z0-9._-])"
        text = re.sub(
            re.escape(value) + boundary,
            lambda _match, token=replacement: token,
            text,
            flags=re.IGNORECASE if os.name == "nt" else 0,
        )
    return text


def redact(text: str) -> str:
    """Remove known path, identity, host, email, and bearer-token material.

    Paths from earlier datasets on non-home mounts are not tracked; the panel's
    editable log remains the review point for that residual case.
    """
    result = str(text)

    dataset = _dataset_properties()
    result = _replace_path(result, dataset.get("data_path"), "<dataset>")
    result = _replace_path(result, dataset.get("output_path"), "<output>")
    result = _replace_path(result, os.path.expanduser("~"), "~")

    result = _EMAIL_RE.sub("<email>", result)
    hostname = socket.gethostname()
    if hostname:
        result = re.sub(
            rf"(?<![\w.-]){re.escape(hostname)}(?![\w.-])",
            "<host>",
            result,
            flags=re.IGNORECASE,
        )
    result = _WINDOWS_USER_RE.sub("~", result)
    result = _UNC_PREFIX_RE.sub("<network>", result)

    username = getpass.getuser()
    if len(username) >= 3:
        result = re.sub(
            rf"(?<!\w){re.escape(username)}(?!\w)",
            "<user>",
            result,
            flags=re.IGNORECASE if os.name == "nt" else 0,
        )

    try:
        tokens = get_portal_account_service()._redaction_tokens()
    except Exception:
        tokens = ()
    for token in sorted((token for token in tokens if token), key=len, reverse=True):
        result = result.replace(token, "<token>")
    return _AUTHORIZATION_RE.sub("Authorization: Bearer <token>", result)


def truncate_utf8_tail(text: str, max_bytes: int = LOG_MAX_BYTES) -> str:
    if max_bytes < 0:
        raise ValueError("max_bytes must be non-negative")
    encoded = str(text).encode("utf-8")
    if len(encoded) <= max_bytes:
        return str(text)
    if max_bytes == 0:
        return ""
    tail = encoded[-max_bytes:]
    newline = tail.find(b"\n")
    if newline < 0:
        return tail.decode("utf-8", errors="replace")
    return tail[newline + 1 :].decode("utf-8", errors="replace")


def current_session_log() -> str:
    try:
        text = lf.log.buffered_text(LOG_MAX_BYTES)
    except Exception:
        return ""
    return truncate_utf8_tail(redact(str(text)))


def previous_session_log_available() -> bool:
    try:
        previous = lf.log.previous_session()
    except Exception:
        return False
    if not isinstance(previous, Mapping) or previous.get("clean_exit") is not False:
        return False
    path = previous.get("log_path")
    return isinstance(path, str) and bool(path) and Path(path).is_file()


def _read_file_tail(path: Path, max_bytes: int) -> str:
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        truncated = size > max_bytes
        handle.seek(max(0, size - max_bytes), os.SEEK_SET)
        tail = handle.read(max_bytes)
    if truncated:
        newline = tail.find(b"\n")
        if newline >= 0:
            tail = tail[newline + 1 :]
    return tail.decode("utf-8", errors="replace")


def previous_session_log() -> str:
    if not previous_session_log_available():
        return ""
    previous = lf.log.previous_session()
    try:
        text = _read_file_tail(Path(str(previous["log_path"])), LOG_MAX_BYTES)
    except (OSError, KeyError, TypeError):
        return ""
    return truncate_utf8_tail(redact(text))


def stage_infer() -> str:
    if RuntimeState.training_running.value:
        return "training"
    if RuntimeState.trainer_loaded.value:
        return "viewer"
    return "startup"


def build_payload(
    form: Mapping[str, object],
    *,
    consent_to_logs: bool,
    current_log_text: str = "",
    include_previous_log: bool = False,
    previous_log_text: Optional[str] = None,
) -> dict[str, object]:
    category = str(form.get("category", ""))
    if category not in CATEGORIES:
        raise ValueError("Unsupported bug-report category")

    diagnostics = collect_diagnostics()
    payload: dict[str, object] = {
        "category": category,
        "title": redact(str(form.get("title", ""))),
        "expected": redact(str(form.get("expected", ""))),
        "actual": redact(str(form.get("actual", ""))),
        "app_version": str(diagnostics.get("app_version", "unknown")),
        "client_name": CLIENT_NAME,
        "diagnostics": diagnostics,
    }

    steps = [
        redact(str(form.get(f"step_{index}", ""))).strip()
        for index in range(1, 4)
    ]
    steps = [step for step in steps if step]
    if steps:
        payload["steps"] = steps

    frequency = str(form.get("frequency", ""))
    if frequency in FREQUENCIES:
        payload["frequency"] = frequency
    stage = str(form.get("stage", ""))
    if stage in STAGES:
        payload["stage"] = stage

    if category == "other":
        category_detail = redact(str(form.get("category_detail", ""))).strip()
        category_detail = category_detail[:CATEGORY_DETAIL_MAX_LENGTH]
        if category_detail:
            payload["category_detail"] = category_detail

    if consent_to_logs:
        current_text = truncate_utf8_tail(redact(current_log_text))
        if current_text:
            payload["log"] = {
                "file_name": "lichtfeld-session.log",
                "text": current_text,
            }
        if include_previous_log:
            source = previous_session_log() if previous_log_text is None else previous_log_text
            prior_text = truncate_utf8_tail(redact(source))
            if prior_text:
                payload["previous_log"] = {
                    "file_name": "lichtfeld-previous.log",
                    "text": prior_text,
                }

    return payload


def is_safe_portal_url(url: str, portal_origin: str) -> bool:
    try:
        candidate = urllib.parse.urlsplit(url)
        origin = urllib.parse.urlsplit(portal_origin)
    except (TypeError, ValueError):
        return False
    return (
        candidate.scheme.lower() in {"http", "https"}
        and bool(candidate.hostname)
        and bool(origin.hostname)
        and candidate.hostname.lower() == origin.hostname.lower()
    )


def open_portal_url(url: str) -> bool:
    service = get_portal_account_service()
    if not is_safe_portal_url(url, service.base_url):
        return False
    lf.ui.open_url(url)
    return True


def _publish_state(state: Mapping[str, object]) -> dict[str, object]:
    published = new_bug_report_state()
    published.update(dict(state))
    RuntimeState.bug_report_state.value = published
    return published


def _redacted_detail(detail: object) -> dict[str, object]:
    if not isinstance(detail, Mapping):
        return {}
    result: dict[str, object] = {}
    for key, value in detail.items():
        if isinstance(value, list):
            result[str(key)] = [redact(str(item)) for item in value]
        else:
            result[str(key)] = redact(str(value))
    return result


def submit_report(
    payload: Mapping[str, object],
    *,
    service: object | None = None,
) -> dict[str, object]:
    """Submit a report synchronously and publish its terminal state.

    When called by :func:`submit_async`, the daemon worker can be lost if the
    process exits before the request completes; callers accept that bounded
    loss window in exchange for keeping shutdown non-blocking.
    """
    account = service if service is not None else get_portal_account_service()
    try:
        response = account.request_json_authenticated("POST", BUG_REPORT_PATH, payload)
    except PortalHTTPError as exc:
        if exc.status == 404:
            error = "portal_outdated"
        elif exc.status == 429:
            error = "rate_limited"
        elif exc.status == 400 and exc.error == "report_invalid":
            error = "report_invalid"
        elif exc.error in {"invalid_token", "membership_required"}:
            error = exc.error
        else:
            error = "generic"
        return _publish_state(
            {
                "error": error,
                "retry_after": exc.retry_after,
                "detail": _redacted_detail(exc.detail),
            }
        )
    except (OSError, PortalProtocolError):
        return _publish_state({"error": "generic"})
    except Exception:
        _log.exception("Unexpected in-app bug-report failure")
        return _publish_state({"error": "generic"})

    url = response.get("url")
    status = response.get("status")
    problems = response.get("completeness_problems")
    return _publish_state(
        {
            "success": True,
            "url": url if isinstance(url, str) else "",
            "status": status if isinstance(status, str) else "",
            "completeness_problems": (
                [redact(str(item)) for item in problems]
                if isinstance(problems, list)
                else []
            ),
        }
    )


def submit_async(payload: Mapping[str, object]) -> bool:
    """Start a daemon report submission, returning ``False`` if one is active.

    The daemon worker is intentionally not joined during process shutdown, so
    an exit while the request is in flight may lose that report.  A completed
    worker publishes a non-submitting terminal state.
    """
    global _submit_thread
    with _submit_lock:
        if _submit_thread is not None and _submit_thread.is_alive():
            return False
        safe_payload = dict(payload)
        _publish_state({"submitting": True})

        def worker() -> None:
            submit_report(safe_payload)

        _submit_thread = threading.Thread(
            target=worker,
            daemon=True,
            name="lfs-portal-bugreport",
        )
        _submit_thread.start()
        return True
