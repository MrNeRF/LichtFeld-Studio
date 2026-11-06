# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Public local training protocol; importing it requires neither host nor NumPy.

Readers use :mod:`transport` and :mod:`geometry`. Producers capture at a native
model read boundary with :mod:`capture`; :mod:`publisher` supplies an optional
ScopedHandler adapter for ordinary LichtFeld trainers.
"""

import importlib
import sys


def ensure_numpy(lf, should_cancel=None):
    """Prepare numerical dependencies on a worker, never in a capture callback."""
    try:
        import numpy
    except ImportError:
        options = {"cancel": should_cancel} if should_cancel is not None else {}
        lf.packages.init(**options)
        lf.packages.install("numpy>=2,<3", **options)
        site = lf.packages.site_packages_dir()
        if site not in sys.path:
            sys.path.insert(0, site)
        importlib.invalidate_caches()
        import numpy
    return numpy
