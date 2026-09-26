# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Saved-project content evidence for deciding PATCH versus replace."""

from lichtfeld import io


def saved_content_stamp(path):
    return io.project_content_stamp(path)
