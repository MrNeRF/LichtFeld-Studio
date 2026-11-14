# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Copies files into a directory, e.g. a Bazel tree artifact.

  copy_to_directory.py <directory> <file> <destination> [<file> <destination>...]
  copy_to_directory.py @<params file with one argument per line>

Destinations are relative to <directory>; missing directories are created.
"""

import os
import shutil
import sys


def main(argv):
    if len(argv) == 1 and argv[0].startswith("@"):
        with open(argv[0][1:], encoding="utf-8") as params:
            argv = params.read().splitlines()
    if not argv or len(argv) % 2 != 1:
        print(__doc__, file=sys.stderr)
        return 2
    directory, *pairs = argv
    os.makedirs(directory, exist_ok=True)
    for source, destination in zip(pairs[0::2], pairs[1::2]):
        target = os.path.join(directory, destination)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copyfile(source, target)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
