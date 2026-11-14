# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Keeps the BCR-format registry in bazel/registry consistent.

For every modules/<name>/<version>/source.json it
  * recomputes the SRI integrity of all files below overlay/ and patches/,
  * downloads the source archive and fills in "integrity" when it is missing
    (or when --refresh-archives is given),
and it rewrites modules/<name>/metadata.json so that "versions" lists all
version directories.

Usage (from the LichtFeld-Studio directory):
    python bazel/tools/registry_integrity.py                 # all modules
    python bazel/tools/registry_integrity.py rmlui           # one module
    python bazel/tools/registry_integrity.py rmlui/6.3       # one version
"""

import argparse
import base64
import hashlib
import json
import pathlib
import sys
import urllib.request

REGISTRY = pathlib.Path(__file__).resolve().parent.parent / "registry"


def sri(data: bytes) -> str:
    return "sha256-" + base64.b64encode(hashlib.sha256(data).digest()).decode()


def file_sri(path: pathlib.Path) -> str:
    return sri(path.read_bytes())


def download_sri(url: str) -> str:
    print(f"  downloading {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "lfs-registry-integrity"})
    with urllib.request.urlopen(request) as response:
        return sri(response.read())


def version_key(version: str):
    parts = []
    for part in version.replace("-", ".").split("."):
        parts.append((0, int(part), "") if part.isdigit() else (1, 0, part))
    return parts


def update_version_dir(version_dir: pathlib.Path, refresh_archives: bool) -> None:
    source_json = version_dir / "source.json"
    if not source_json.exists():
        return
    source = json.loads(source_json.read_text(encoding="utf-8"))

    if "url" in source and (refresh_archives or not source.get("integrity")):
        source["integrity"] = download_sri(source["url"])

    overlay_dir = version_dir / "overlay"
    if overlay_dir.is_dir():
        source["overlay"] = {
            path.relative_to(overlay_dir).as_posix(): file_sri(path)
            for path in sorted(overlay_dir.rglob("*"))
            if path.is_file()
        }
    else:
        source.pop("overlay", None)

    patches_dir = version_dir / "patches"
    if patches_dir.is_dir():
        source["patches"] = {
            path.name: file_sri(path) for path in sorted(patches_dir.iterdir()) if path.is_file()
        }
        source.setdefault("patch_strip", 1)
    else:
        source.pop("patches", None)
        source.pop("patch_strip", None)

    ordered = {}
    for key in ("url", "mirror_urls", "integrity", "strip_prefix", "archive_type", "overlay", "patches", "patch_strip"):
        if key in source:
            ordered[key] = source.pop(key)
    ordered.update(source)

    source_json.write_text(json.dumps(ordered, indent=4) + "\n", encoding="utf-8", newline="\n")
    print(f"  updated {source_json.relative_to(REGISTRY)}")


def update_module(module_dir: pathlib.Path, only_version: str | None, refresh_archives: bool) -> None:
    versions = sorted(
        (path.name for path in module_dir.iterdir() if path.is_dir()),
        key=version_key,
    )
    for version in versions:
        if only_version is None or version == only_version:
            update_version_dir(module_dir / version, refresh_archives)

    metadata_json = module_dir / "metadata.json"
    metadata = json.loads(metadata_json.read_text(encoding="utf-8")) if metadata_json.exists() else {}
    metadata.setdefault("homepage", "")
    metadata.setdefault("maintainers", [])
    metadata.setdefault("repository", [])
    metadata["versions"] = versions
    metadata.setdefault("yanked_versions", {})
    metadata_json.write_text(json.dumps(metadata, indent=4) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("modules", nargs="*", help="<module> or <module>/<version>; default: all")
    parser.add_argument("--refresh-archives", action="store_true", help="re-download source archives")
    args = parser.parse_args()

    modules_root = REGISTRY / "modules"
    selections = args.modules or [path.name for path in sorted(modules_root.iterdir()) if path.is_dir()]
    for selection in selections:
        name, _, version = selection.partition("/")
        module_dir = modules_root / name
        if not module_dir.is_dir():
            print(f"unknown module: {name}", file=sys.stderr)
            return 1
        print(f"{name}{'/' + version if version else ''}")
        update_module(module_dir, version or None, args.refresh_archives)
    return 0


if __name__ == "__main__":
    sys.exit(main())
