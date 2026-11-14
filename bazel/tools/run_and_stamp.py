# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Runs a program and writes an empty stamp file if it succeeds.

Bazel validation actions (e.g. spirv-val on SPIR-V) must produce an output.

  run_and_stamp.py <stamp> <program> [<argument>...]
"""

import os
import subprocess
import sys


def main(argv):
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    stamp, program, *arguments = argv
    try:
        returncode = subprocess.run([os.path.normpath(program), *arguments]).returncode
    except OSError as e:
        print(f"run_and_stamp: failed to run {program}: {e}", file=sys.stderr)
        return 1
    if returncode != 0:
        return returncode
    with open(stamp, "wb"):
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
