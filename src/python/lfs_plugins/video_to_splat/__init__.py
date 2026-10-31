# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Video to splat pipeline: frames -> COLMAP poses -> project -> training.

The planning modules (probe, sfm, toolchain, pipeline) import nothing from the
app so they stay testable without it; register()/unregister() arrive in Task 8.
"""
