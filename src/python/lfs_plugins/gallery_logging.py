# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Structured, secret-safe diagnostics for gallery transfers."""
from __future__ import annotations

import logging
import re
import traceback
from urllib.parse import urlsplit, urlunsplit

from .portal_security import redact


_EMAIL = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")


def safe_text(value: object) -> str:
    """Keep diagnostics useful without copying account or URL credentials."""
    text = redact(value)
    text = _EMAIL.sub("[REDACTED_EMAIL]", text)
    return re.sub(r"https?://[^\s]+", lambda match: safe_url(match.group(0)), text)


def safe_url(value: object) -> str:
    """Return an URL location without its query or fragment."""
    try:
        parsed = urlsplit(str(value))
        path_parts = []
        for part in (parsed.path or "/").split("/"):
            path_parts.append("[REDACTED]" if part.lower() in {
                "access", "authorization", "credential", "private", "secret", "token"
            } else part)
        return urlunsplit((parsed.scheme, parsed.netloc, "/".join(path_parts) or "/", "", ""))
    except (TypeError, ValueError):
        return "[invalid URL]"


def _fields(values: dict[str, object]) -> str:
    return " ".join(f"{key}={safe_text(values[key])}" for key in sorted(values))


def stage(stage_name: str, **values: object) -> None:
    """Write one machine-searchable INFO line for a transfer stage."""
    logging.getLogger("lfs_plugins.portal_gallery").info("gallery stage=%s %s", stage_name, _fields(values))


def failure(stage_name: str, exc: BaseException, **values: object) -> None:
    """Write a failure with the active traceback and any safe HTTP body."""
    details = {
        "exception_class": type(exc).__name__,
        "message": safe_text(exc),
        **values,
    }
    response_body = getattr(exc, "response_body", None)
    if response_body:
        details["last_http_response_body"] = safe_text(response_body)
    traceback_text = traceback.format_exc()
    if traceback_text == "NoneType: None\n":
        traceback_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    logging.getLogger("lfs_plugins.portal_gallery").error(
        "gallery failure stage=%s %s\ntraceback=%s",
        stage_name,
        _fields(details),
        safe_text(traceback_text),
    )
