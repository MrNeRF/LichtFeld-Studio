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

MODULES = ("portal_account", "portal_gallery", "credential_storage", "private_directory",
           "http", "portal_security", "portal_retry", "gallery_logging", "gallery_storage",
           "portable_project", "gallery_validation", "project_identity")


def build(output):
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
''', encoding="utf-8")
        for module in MODULES:
            text = (source / "src/python/lfs_plugins" / (module + ".py")).read_text(encoding="utf-8")
            # Namespace adaptation only. No second implementation or host path injection.
            text = text.replace("from _lfs_path_io import ", "from .path_io import ")
            (package / (module + ".py")).write_text(text, encoding="utf-8")
        shutil.copyfile(source / "src/python/_lfs_path_io.py", package / "path_io.py")
        shutil.copyfile(source / "src/python/lfs_plugins/distrusted_cas.pem", package / "distrusted_cas.pem")
        shutil.copyfile(source / "LICENSE", root / "LICENSE")
        (root / "pyproject.toml").write_text('''[build-system]
requires = ["setuptools>=77"]
build-backend = "setuptools.build_meta"
[project]
name = "lichtfeld-portal-client"
version = "0.1.2"
requires-python = ">=3.11"
license = "GPL-3.0-or-later"
license-files = ["LICENSE"]
authors = [{name = "LichtFeld Studio Authors"}]
[tool.setuptools]
packages = ["lichtfeld_portal"]
[tool.setuptools.package-data]
lichtfeld_portal = ["distrusted_cas.pem"]
''', encoding="utf-8")
        subprocess.run([sys.executable, "-m", "build", "--wheel", "--no-isolation",
                        "--outdir", str(output)], cwd=root, check=True,
                       env={**os.environ, "SOURCE_DATE_EPOCH": "315532800"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
