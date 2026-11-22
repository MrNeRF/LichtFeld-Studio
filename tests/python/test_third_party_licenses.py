# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Preserve license lookup and contents without repeated source I/O."""

import importlib.util
import json
import os
import sys
from collections import Counter
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


def write_port(triplet, name, text, version="1.0", license_id="MIT"):
    metadata = triplet / "share" / name
    metadata.mkdir(parents=True)
    (metadata / "vcpkg.spdx.json").write_text(json.dumps({"packages": [{
        "SPDXID": "SPDXRef-port", "name": name,
        "versionInfo": version, "licenseConcluded": license_id,
    }]}), encoding="utf-8")
    (metadata / "copyright").write_text(text, encoding="utf-8")
    return metadata


@pytest.mark.skipif(os.name != "nt", reason="Windows UNC share-name semantics")
def test_url_without_unc_filename_does_not_read_source_directories(tmp_path, monkeypatch):
    triplet, vcpkg = tmp_path / "installed", tmp_path / "vcpkg"
    text = "See https://example.invalid/LICENSE"
    metadata = write_port(triplet, "example", text)
    (vcpkg / "buildtrees/example/src/nested").mkdir(parents=True)
    read_text = notices.read_text

    def read_copyright(path):
        assert path == metadata / "copyright", f"Unexpected directory read: {path}"
        return read_text(path)

    monkeypatch.setattr(notices, "read_text", read_copyright)
    assert notices.installed_ports(triplet, vcpkg)[0]["text"] == text


def test_source_lookup_reads_each_path_once_and_preserves_all_versions(tmp_path, monkeypatch):
    triplet, vcpkg = tmp_path / "installed", tmp_path / "vcpkg"
    metadata = write_port(
        triplet, "example",
        "See LICENSE; nested/LICENSE; COPYING; https://example.invalid/LICENSE; LICENSE",
    )
    (metadata / "nested").mkdir()
    (metadata / "nested/LICENSE").write_text("Installed nested terms", encoding="utf-8")
    source = vcpkg / "buildtrees/example/src"
    files = {
        "old/LICENSE": "Old version terms",
        "new/deep/LICENSE": "New version terms",
        "new/deep/COPYING": "Additional terms",
        "mixed/License": "Mixed case terms",
        "unicode/COPYİNG": "Unicode case terms",
        "new/irrelevant.cpp": "Not a license",
    }
    for relative, text in files.items():
        file = source / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(text, encoding="utf-8")
    # The original recursive glob is an independent oracle for platform and
    # Python-version-specific filename matching (notably Python 3.10 vs 3.12).
    mixed_case = source / "mixed/License" in set(source.glob("**/LICENSE"))
    unicode_case = source / "unicode/COPYİNG" in set(source.glob("**/COPYING"))
    reads = Counter()
    read_text = notices.read_text

    def read_local(path):
        assert path.is_relative_to(tmp_path), f"External read: {path}"
        reads[path] += 1
        return read_text(path)

    monkeypatch.setattr(notices, "read_text", read_local)
    for old_terms in ("Old version terms", "Updated old terms"):
        reads.clear()
        (source / "old/LICENSE").write_text(old_terms, encoding="utf-8")
        text = notices.installed_ports(triplet, vcpkg)[0]["text"]
        for terms in (old_terms, "New version terms", "Additional terms", "Installed nested terms"):
            assert terms in text
        assert ("Mixed case terms" in text) == mixed_case
        assert ("Unicode case terms" in text) == unicode_case
        assert "Not a license" not in text
        assert max(reads.values()) == 1
        assert reads[source / "new/irrelevant.cpp"] == 0


def test_cli_reuses_inventory_for_document_ffmpeg_details_and_count(tmp_path, monkeypatch, capsys):
    triplet, vcpkg = tmp_path / "installed", tmp_path / "vcpkg"
    write_port(triplet, "ffmpeg", "FFmpeg license terms", "8.0.1")
    write_port(triplet, "x264", "x264 license terms", "0.164", "GPL-2.0-only")
    metadata = write_port(triplet, "example", "Example terms", "2.0", "BSD-2-Clause")
    project, payload = tmp_path / "project", tmp_path / "payload"
    project.mkdir()
    payload.mkdir()
    output = tmp_path / "notices.md"
    read_text = Path.read_text
    inventory_reads = []

    def read(path, *args, **kwargs):
        if path == metadata / "vcpkg.spdx.json":
            inventory_reads.append(path)
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(sys, "argv", ["notices", "--source", str(project),
                                     "--triplet-dir", str(triplet), "--install-prefix", str(payload),
                                     "--output", str(output), "--vcpkg-root", str(vcpkg), "--check"])
    assert notices.main() == 0
    text = output.read_text(encoding="utf-8")
    assert "FFmpeg license terms" in text and "x264 license terms" in text
    assert "- Version: 8.0.1" in text and "- x264 version/source: 0.164;" in text
    assert "Example terms" in text
    assert "(3 vcpkg ports)" in capsys.readouterr().out
    # One inventory read, plus the independent --check coverage pass.
    assert len(inventory_reads) == 2
