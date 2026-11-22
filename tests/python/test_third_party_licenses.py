# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""License URLs must not be opened as local or UNC filesystem locations."""

import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "generate_third_party_licenses",
    Path(__file__).resolve().parents[2] / "tools/generate_third_party_licenses.py",
)
notices = importlib.util.module_from_spec(spec)
spec.loader.exec_module(notices)


@pytest.mark.parametrize(
    "reference",
    [
        "https://raw.githubusercontent.com/fmtlib/fmt/master/LICENSE",
        "//license-server/share/LICENSE",
        "/licenses/LICENSE",
        "nested/LICENSE",
    ],
)
def test_license_reference_uses_local_sources_without_opening_remote_paths(
    tmp_path, monkeypatch, reference
):
    triplet = tmp_path / "installed"
    metadata = triplet / "share" / "example"
    metadata.mkdir(parents=True)
    (metadata / "vcpkg.spdx.json").write_text(
        json.dumps({"packages": [{"SPDXID": "SPDXRef-port", "name": "example", "licenseConcluded": "MIT"}]}),
        encoding="utf-8",
    )
    (metadata / "copyright").write_text(f"See {reference} for license terms.", encoding="utf-8")
    vcpkg = tmp_path / "vcpkg"
    source = vcpkg / "buildtrees" / "example" / "src" / "unpacked"
    source.mkdir(parents=True)
    (source / "LICENSE").write_text("Source license text", encoding="utf-8")
    if reference == "nested/LICENSE":
        (metadata / "nested").mkdir()
        (metadata / reference).write_text("Installed relative license text", encoding="utf-8")

    read_text = notices.read_text

    def read_local(path: Path):
        # Fail before open(): the regression must never wait for a network share.
        assert path.is_relative_to(tmp_path), f"Unexpected external filesystem read: {path}"
        return read_text(path)

    monkeypatch.setattr(notices, "read_text", read_local)
    ports = notices.installed_ports(triplet, vcpkg)
    assert len(ports) == 1
    assert "Source license text" in ports[0]["text"]
    if reference == "nested/LICENSE":
        assert "Installed relative license text" in ports[0]["text"]
