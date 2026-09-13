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


def portal_url(base_url, value, allowed_hosts=()):
    """Rebuild a URL only after pinning its origin. Never trust listing URLs."""
    from urllib.parse import urlsplit, urlunsplit, urljoin
    try:
        if not isinstance(value, str) or not value or any(ord(c) <= 32 or ord(c) == 127 for c in value) or "\\" in value:
            raise ValueError
        base = urlsplit(base_url)
        parsed = urlsplit(urljoin(base_url + '/', value))
        def origin(parts):
            if (parts.scheme not in ('https', 'http') or not parts.hostname
                    or parts.username is not None or parts.password is not None or parts.fragment
                    or (parts.scheme == 'http' and parts.hostname != '127.0.0.1')):
                raise ValueError
            return parts.scheme, parts.hostname.lower(), parts.port if parts.port is not None else (443 if parts.scheme == 'https' else 80)
        expected, actual = origin(base), origin(parsed)
        # /me may explicitly advertise portal-owned HTTPS hosts (host[:port]).
        permitted = {expected}
        if isinstance(allowed_hosts, (list, tuple)):
            for host in allowed_hosts:
                if isinstance(host, str) and host and not any(c in host for c in '/?#@\\'):
                    permitted.add(origin(urlsplit('https://' + host)))
        if actual not in permitted:
            raise ValueError
        authority = parsed.hostname.lower()
        if ':' in authority:
            authority = '[' + authority + ']'
        if parsed.port is not None:
            authority += ':' + str(parsed.port)
        return urlunsplit((parsed.scheme, authority, parsed.path or '/', parsed.query, ''))
    except (ValueError, TypeError, AttributeError):
        raise ValueError('Unsafe portal URL') from None


def checked_portal_url(account, value):
    """UI boundary: refuse navigation/copy with a localized notice."""
    try:
        return portal_url(account.base_url, value)
    except ValueError:
        import lichtfeld as lf
        raise ValueError(lf.ui.tr('asset_manager.gallery.error.unsafe_url')) from None


def storage_url(base_url, value, allowed_hosts=None):
    """Validate credential-free transfers; an absent allowlist permits any HTTPS host.

    HTTP is only available on the configured local portal's own origin. Explicit
    host[:port] allowlists are exact matches, and empty/malformed lists deny all.
    Navigation must continue to use portal_url instead.
    """
    from urllib.parse import urljoin, urlsplit
    try:
        if not isinstance(value, str) or not value or any(ord(c) <= 32 or ord(c) == 127 for c in value) or '\\' in value:
            raise ValueError
        parsed = urlsplit(urljoin(base_url + '/', value))
        # Reuse the strict URL syntax checks, allowing only this HTTPS authority
        # in addition to the portal origin. This never changes navigation policy.
        result = portal_url(base_url, value, [parsed.netloc])
        if allowed_hosts is not None:
            if not isinstance(allowed_hosts, (list, tuple)):
                raise ValueError
            authority = (parsed.hostname.lower(), parsed.port if parsed.port is not None else (443 if parsed.scheme == 'https' else 80))
            permitted = set()
            for host in allowed_hosts:
                if not isinstance(host, str) or not host or any(c in host for c in '/?#@\\'):
                    raise ValueError
                candidate = urlsplit(portal_url('https://' + host, 'https://' + host))
                permitted.add((candidate.hostname, candidate.port if candidate.port is not None else 443))
            if authority not in permitted:
                raise ValueError
        return result
    except (ValueError, TypeError, AttributeError):
        raise ValueError('Unsafe portal URL') from None
