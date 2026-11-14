# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Assembles an application directory from a JSON manifest (//bazel:bundle.bzl).

  assemble_bundle.py <manifest.json>

The manifest contains:
  out:   the output directory (a Bazel tree artifact),
  files: [source file, destination] pairs,
  trees: [source directory, destination directory] pairs (contents are merged),
  text:  {destination: content} for generated text files.
Destinations are relative to `out`.
"""

import json
import os
import shutil
import sys


def main(argv):
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    with open(argv[0], encoding="utf-8") as f:
        manifest = json.load(f)

    out = manifest["out"]
    os.makedirs(out, exist_ok=True)

    def destination(relative):
        path = os.path.normpath(os.path.join(out, relative))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    for source, relative in manifest["trees"]:
        shutil.copytree(source, os.path.normpath(os.path.join(out, relative)), dirs_exist_ok=True)
    for source, relative in manifest["files"]:
        shutil.copyfile(source, destination(relative))
    for relative, content in manifest["text"].items():
        with open(destination(relative), "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
