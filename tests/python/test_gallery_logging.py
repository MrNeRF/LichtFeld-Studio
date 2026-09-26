# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for durable gallery transfer diagnostics."""
from __future__ import annotations

import io
import logging
import sys
import urllib.error
from types import SimpleNamespace
from pathlib import Path

import pytest

from lfs_plugins import portal_gallery


def test_python_logging_bridge_delivers_records_to_native_logger(monkeypatch):
    delivered = []
    fake = SimpleNamespace(log=SimpleNamespace(
        debug=lambda message: delivered.append(("debug", message)),
        info=lambda message: delivered.append(("info", message)),
        warn=lambda message: delivered.append(("warn", message)),
        error=lambda message: delivered.append(("error", message)),
    ))
    monkeypatch.setitem(sys.modules, "lichtfeld", fake)
    from lfs_plugins import logging_bridge

    root = logging.getLogger()
    previous = [handler for handler in root.handlers if getattr(handler, "_lichtfeld_bridge", False)]
    for handler in previous:
        root.removeHandler(handler)
    assert logging_bridge.install()
    bridge = next(handler for handler in root.handlers if getattr(handler, "_lichtfeld_bridge", False))
    logger = logging.getLogger("lfs_plugins.bridge_test")
    try:
        logger.info("bridge marker")
    finally:
        root.removeHandler(bridge)
        for handler in previous:
            root.addHandler(handler)
    assert any(level == "info" and "bridge marker" in message for level, message in delivered)


@pytest.mark.parametrize("status", [403, 503])
def test_failing_part_put_has_stage_and_failure_lines(tmp_path, monkeypatch, caplog, status):
    upload_id = "11111111-1111-4111-8111-111111111111"
    calls = []

    def request(method, path, body=None, **_kwargs):
        calls.append((method, path))
        if path.endswith("/me"):
            return {"id": "owner", "gallerySyncVersion": 1, "revisionDomains": 1,
                    "sourceFormats": ["licht"], "storageHosts": ["portal.example"]}
        if path.endswith("/uploads"):
            return {"id": upload_id, "status": "created", "partSize": 4, "uploadedParts": []}
        if path.endswith("/part-upload-urls"):
            return {"urls": [{"partNumber": 1,
                              "url": "https://portal.example/part/1?signature=secret"}]}
        raise AssertionError(path)

    account = SimpleNamespace(base_url="https://portal.example", _client_version="test",
                              request_json_authenticated=request)
    source = tmp_path / "scene.licht"
    source.write_bytes(b"data")

    put_calls = []
    def failed_put(_request, **kwargs):
        put_calls.append(kwargs["timeout"])
        raise urllib.error.HTTPError(
            "https://portal.example/part/1?signature=secret", status, "bad part", {}, io.BytesIO(b"bad"))

    monkeypatch.setattr(portal_gallery, "urlopen", failed_put)
    with caplog.at_level(logging.DEBUG, logger="lfs_plugins.portal_gallery"):
        with pytest.raises(urllib.error.HTTPError):
            portal_gallery.PortalGalleryClient(account).upload(source, {"title": "Scene"})

    assert "gallery stage=upload_create" in caplog.text
    assert "gallery failure stage=part_put" in caplog.text
    assert "exception_class=HTTPError" in caplog.text
    assert "signature=secret" not in caplog.text
    assert calls[-1][1].endswith("/part-upload-urls")
    assert put_calls == [120] * (2 if status == 403 else 4)


def test_preparation_exception_is_logged_and_journaled(tmp_path, monkeypatch, caplog):
    from test_gallery_preparation import native_service, staging, finish
    from lfs_plugins import gallery_preparation

    service = native_service(tmp_path, monkeypatch)
    directory = staging(tmp_path)
    monkeypatch.setattr(gallery_preparation, "read_staging",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(
                            RuntimeError("preparation marker")))
    with caplog.at_level(logging.DEBUG, logger="lfs_plugins.portal_gallery"):
        job_id = service.queue_prepared_upload(directory, {"title": "Scene"}, "project")
        finish(service)

    job = next(job for job in service.snapshot()["jobs"] if job["id"] == job_id)
    assert job["status"] == "error"
    assert "preparation marker" in job["failureReason"]
    assert "gallery failure stage=transfer" in caplog.text
    assert "RuntimeError" in caplog.text
    assert not directory.exists() and not Path(job["path"]).exists()
    assert job["requiresPreparation"]


def test_malformed_url_does_not_recurse_or_expose_credentials():
    from lfs_plugins.gallery_logging import safe_text
    assert safe_text('request https://[bad?token=secret failed') == 'request [invalid URL] failed'


def test_log_bridge_fallback_keeps_failure_visible(monkeypatch, capsys):
    from lfs_plugins import logging_bridge
    def fail(_message):
        raise OSError('sink unavailable')
    monkeypatch.setitem(sys.modules, 'lichtfeld', SimpleNamespace(log=SimpleNamespace(error=fail)))
    root = logging.getLogger()
    previous = list(root.handlers)
    for handler in previous:
        root.removeHandler(handler)
    try:
        assert logging_bridge.install()
        logging.getLogger(__name__).error('failed path=项目.licht https://host/file?token=secret')
        output = capsys.readouterr().err
        assert 'sink unavailable' in output and '项目.licht' in output
        assert 'token=secret' not in output
    finally:
        root.handlers[:] = previous
