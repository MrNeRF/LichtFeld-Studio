# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Video to splat pipeline: frames -> COLMAP poses -> project -> training.

The planning modules (probe, sfm, toolchain, pipeline) import nothing from the
app so they stay testable without it. register()/unregister() import lichtfeld
lazily for the same reason.
"""

__lfs_operator_classes__ = ["NewProjectFromVideoOperator"]


def register() -> None:
    import lichtfeld as lf

    from .operators import NewProjectFromVideoOperator

    lf.register_class(NewProjectFromVideoOperator)


def unregister() -> None:
    import lichtfeld as lf

    from .operators import NewProjectFromVideoOperator

    lf.unregister_class(NewProjectFromVideoOperator)
