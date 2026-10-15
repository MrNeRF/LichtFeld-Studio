# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

import threading

import pytest


@pytest.mark.parametrize("outcome", ["project", "missing", "error"])
def test_library_service_verify_uses_worker(outcome):
    from lfs_plugins.asset_index import LibraryService

    expected = object() if outcome == "project" else None
    threads = []

    class Index:
        def verify_asset(self, asset_id):
            assert asset_id == "project-id"
            threads.append(threading.current_thread())
            if outcome == "error":
                raise ValueError("verification failed")
            return expected

    with LibraryService(Index()) as service:
        if outcome == "error":
            with pytest.raises(ValueError, match="verification failed"):
                service.verify("project-id")
        else:
            assert service.verify("project-id") is expected
        assert threads == [service._worker]


