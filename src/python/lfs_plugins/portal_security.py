# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared secret scrubbing for portal diagnostics."""
import re
import threading
from collections import deque

_lock = threading.Lock()
_secrets = deque(maxlen=128)


def remember_secrets(*values):
    with _lock:
        for value in values:
            if isinstance(value, str) and value and value not in _secrets:
                _secrets.append(value)


def redact(value):
    text = str(value)
    with _lock:
        secrets = sorted(_secrets, key=len, reverse=True)
    for secret in secrets:
        text = text.replace(secret, '[REDACTED]')
    text = re.sub(r'(?i)\bBearer\s+[^\s\'"<>,}]+', 'Bearer [REDACTED]', text)
    text = re.sub(r'''(?ix)(\b(?:access_token|refresh_token|user_code|device_code|authorization|token|signature|x-amz-signature)\b["']?\s*[:=]\s*)(?:"[^"\r\n]*"|'[^'\r\n]*'|[^\s&,}\]]+)''', r'\1[REDACTED]', text)
    return text


def safe_filename(title):
    stem = re.sub(r'[^\w -]', '', str(title)).strip(' .')[:114].rstrip(' .') or 'Gallery'
    if stem.upper() in {'CON', 'PRN', 'AUX', 'NUL', 'CLOCK$', 'CONIN$', 'CONOUT$',
                        *(f'{kind}{i}' for kind in ('COM', 'LPT') for i in '123456789¹²³')}:
        stem = '_' + stem
    return stem[:114].rstrip(' .') + '.licht'
