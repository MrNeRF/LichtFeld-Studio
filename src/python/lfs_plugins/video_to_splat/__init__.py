# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Video to splat pipeline: frames -> COLMAP poses -> project -> training.

The planning modules (probe, sfm, toolchain, pipeline) import nothing from the
app so they stay testable without it. register()/unregister() import lichtfeld
lazily for the same reason.
"""

__lfs_operator_classes__ = ["NewProjectFromVideoOperator", "NewProjectFromPhotosOperator"]


def register() -> None:
    import lichtfeld as lf

    from .operators import NewProjectFromPhotosOperator, NewProjectFromVideoOperator

    lf.register_class(NewProjectFromVideoOperator)
    lf.register_class(NewProjectFromPhotosOperator)
    from .mcp_tools import register_mcp_tools

    register_mcp_tools()


def unregister() -> None:
    import lichtfeld as lf

    from .operators import NewProjectFromPhotosOperator, NewProjectFromVideoOperator

    from .mcp_tools import unregister_mcp_tools

    unregister_mcp_tools()
    lf.unregister_class(NewProjectFromPhotosOperator)
    lf.unregister_class(NewProjectFromVideoOperator)
