# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Public local training protocol; importing it requires neither host nor NumPy.

Readers use :mod:`transport` and :mod:`geometry`. Producers capture at a native
model read boundary with :mod:`capture`; :mod:`publisher` supplies an optional
ScopedHandler adapter for ordinary LichtFeld trainers.
"""
