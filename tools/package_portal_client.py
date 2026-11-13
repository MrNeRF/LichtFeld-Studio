#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Package the canonical Portal owners for consumers outside LichtFeld Studio."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from normalize_shared_wheel import normalize

MODULES = ("portal_account", "portal_gallery", "credential_storage", "private_directory",
           "http", "portal_security", "portal_retry", "gallery_logging", "gallery_storage",
           "portable_project", "gallery_validation", "project_identity")


def build(output, version):
    source = Path(__file__).resolve().parents[1]
    output = Path(output).resolve()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        package = root / "lichtfeld_portal"
        package.mkdir()
        (package / "__init__.py").write_text('''# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Portal account and gallery client; host presentation helpers are not SDK APIs."""
from .portal_account import PortalAccountService
from .portal_gallery import PortalGalleryClient
__all__ = ("PortalAccountService", "PortalGalleryClient")
''', encoding="utf-8", newline="\n")
        for module in MODULES:
            text = (source / "src/python/lfs_plugins" / (module + ".py")).read_text(encoding="utf-8")
            # Namespace adaptation only. No second implementation or host path injection.
            text = text.replace("from _lfs_path_io import ", "from .path_io import ")
            (package / (module + ".py")).write_text(text, encoding="utf-8", newline="\n")
        (package / "path_io.py").write_text(
            (source / "src/python/_lfs_path_io.py").read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
        (package / "distrusted_cas.pem").write_text(
            (source / "src/python/lfs_plugins/distrusted_cas.pem").read_text(encoding="utf-8"),
            encoding="utf-8", newline="\n")
        (root / "LICENSE").write_text((source / "LICENSE").read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
        (root / "pyproject.toml").write_text('''[build-system]
requires = ["setuptools>=77"]
build-backend = "setuptools.build_meta"
[project]
name = "lichtfeld-portal-client"
version = "PORTAL_CLIENT_VERSION"
requires-python = ">=3.11"
license = "GPL-3.0-or-later"
license-files = ["LICENSE"]
authors = [{name = "LichtFeld Studio Authors"}]
[tool.setuptools]
packages = ["lichtfeld_portal"]
[tool.setuptools.package-data]
lichtfeld_portal = ["distrusted_cas.pem"]
'''.replace("PORTAL_CLIENT_VERSION", version), encoding="utf-8", newline="\n")
        dist = root / "dist"
        subprocess.run([sys.executable, "-m", "build", "--wheel", "--no-isolation",
                        "--outdir", str(dist)], cwd=root, check=True,
                       env={**os.environ, "SOURCE_DATE_EPOCH": "315532800"})
        wheel = dist / f"lichtfeld_portal_client-{version}-py3-none-any.whl"
        normalize(wheel)
        output.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=wheel.name + '-', suffix='.tmp', dir=output)
        try:
            with os.fdopen(descriptor, 'wb') as destination, wheel.open('rb') as source_file:
                shutil.copyfileobj(source_file, destination)
            os.replace(temporary, output / wheel.name)
        finally:
            Path(temporary).unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    build(args.output, args.version)
