# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Failures during GUI callback setup must be reported to the caller."""

import builtins
from types import SimpleNamespace

from test_lightfield_render_runtime import _verify_in_gui


def test_missing_numpy_is_reported_immediately(monkeypatch, capsys):
    original_import = builtins.__import__

    def missing_numpy(name, *args, **kwargs):
        if name == "numpy":
            raise ImportError("numpy unavailable in test GUI")
        if name == "lichtfeld":
            return SimpleNamespace()
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_numpy)
    _verify_in_gui()
    output = capsys.readouterr().out
    assert "LIGHTFIELD_RENDER_FAIL" in output
    assert "ImportError: numpy unavailable in test GUI" in output
