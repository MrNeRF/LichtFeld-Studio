# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Bridge every Python logging record into LichtFeld's durable logger."""
from __future__ import annotations

import logging
import re


_EMAIL = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")


def install() -> bool:
    try:
        import lichtfeld as lf
    except Exception:
        return False

    class _LfLogHandler(logging.Handler):
        _lichtfeld_bridge = True

        def emit(self, record):
            try:
                message = self.format(record)
                message = _EMAIL.sub("[REDACTED_EMAIL]", message)
                if record.levelno >= logging.ERROR:
                    lf.log.error(message)
                elif record.levelno >= logging.WARNING:
                    lf.log.warn(message)
                elif record.levelno >= logging.INFO:
                    lf.log.info(message)
                else:
                    lf.log.debug(message)
            except Exception:
                # A diagnostic path must never alter the operation being logged.
                pass

    root = logging.getLogger()
    if not any(getattr(handler, "_lichtfeld_bridge", False) for handler in root.handlers):
        root.addHandler(_LfLogHandler())
    root.setLevel(logging.DEBUG)
    return True
