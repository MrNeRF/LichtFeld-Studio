# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Bridge every Python logging record into LichtFeld's durable logger."""
from __future__ import annotations

import logging
import sys

from .gallery_logging import safe_text



def install() -> bool:
    try:
        import lichtfeld as lf
    except ImportError as exc:
        sys.stderr.write(f"Python log bridge could not load the native logger: {safe_text(exc)}\n")
        return False

    class _LfLogHandler(logging.Handler):
        _lichtfeld_bridge = True

        def emit(self, record):
            try:
                message = self.format(record)
                message = safe_text(message)
                if record.levelno >= logging.ERROR:
                    lf.log.error(message)
                elif record.levelno >= logging.WARNING:
                    lf.log.warn(message)
                elif record.levelno >= logging.INFO:
                    lf.log.info(message)
                else:
                    lf.log.debug(message)
            except Exception as exc:
                sys.stderr.write(f"Native log write failed: {safe_text(exc)}; {safe_text(record.getMessage())}\n")

    root = logging.getLogger()
    if not any(getattr(handler, "_lichtfeld_bridge", False) for handler in root.handlers):
        root.addHandler(_LfLogHandler())
    root.setLevel(logging.DEBUG)
    return True
