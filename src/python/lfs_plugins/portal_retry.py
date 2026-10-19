# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""One bounded transient retry policy, opt-in for idempotent operations only."""
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import timezone
from email.utils import parsedate_to_datetime
from http.client import RemoteDisconnected
import math
import errno
import random
import socket
import time
import urllib.error

_context = ContextVar('portal_retry_context', default=(None, None))


def retry_after(headers):
    value = headers.get('Retry-After', headers.get('retry-after')) if headers is not None else None
    if value is None:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        try:
            at = parsedate_to_datetime(str(value))
            seconds = (at if at.tzinfo else at.replace(tzinfo=timezone.utc)).timestamp() - time.time()
        except (TypeError, ValueError, OverflowError):
            return None
    return max(0., seconds) if math.isfinite(seconds) else None


@contextmanager
def transfer_attempts(callback, cancel):
    token = _context.set((callback, cancel))
    try:
        yield
    finally:
        _context.reset(token)


def is_transient(exc):
    status = getattr(exc, 'status', getattr(exc, 'code', None))
    reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
    # A clean HTTP close is not a socket reset, despite its Python base class.
    if isinstance(reason, RemoteDisconnected):
        return False
    return (status == 429 or (isinstance(status, int) and 500 <= status <= 599)
            or isinstance(reason, (TimeoutError, socket.timeout, ConnectionError))
            or isinstance(reason, socket.gaierror) and reason.errno == socket.EAI_AGAIN
            or isinstance(exc, urllib.error.URLError) and isinstance(reason, OSError)
            and reason.errno in (errno.ENETUNREACH, errno.EHOSTUNREACH, errno.ETIMEDOUT))


def retry_call(operation, *, idempotent, attempts=4, sleep=None):
    callback, cancel = _context.get()
    for attempt in range(attempts):
        if cancel is not None and cancel.is_set():
            from .portal_gallery import GalleryTransferCanceled
            raise GalleryTransferCanceled('Transfer paused')
        if callback is not None:
            callback()
        try:
            return operation()
        except Exception as exc:
            if not idempotent or not is_transient(exc) or attempt + 1 >= attempts:
                raise
            after = getattr(exc, 'retry_after', None)
            if after is None:
                after = retry_after(getattr(exc, 'headers', None))
            delay = min(30., max(after or 0., min(30., 2. ** attempt) * random.uniform(.5, 1.5)))
            if isinstance(exc, urllib.error.HTTPError):
                exc.close()
            if sleep is not None:
                sleep(delay)
            elif cancel is not None:
                if cancel.wait(delay):
                    from .portal_gallery import GalleryTransferCanceled
                    raise GalleryTransferCanceled('Transfer paused') from None
            else:
                time.sleep(delay)
