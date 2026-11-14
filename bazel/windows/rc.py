# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Runs the Windows resource compiler (rc.exe).

rc.exe is part of the Windows SDK. It is looked up on the PATH of the action,
which is the environment of the MSVC C++ toolchain (it contains the SDK's bin
directory), with the default SDK location as fallback.

Arguments are passed to rc.exe unchanged, except that forward slashes in
non-option arguments (Bazel paths) are converted to backslashes.
"""

import glob
import os
import shutil
import subprocess
import sys


def find_rc():
    rc = shutil.which("rc.exe", path=os.environ.get("PATH", ""))
    if rc:
        return rc
    sdk_bin = os.path.join(
        os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
        "Windows Kits",
        "10",
        "bin",
    )
    candidates = sorted(glob.glob(os.path.join(sdk_bin, "10.*", "x64", "rc.exe")), reverse=True)
    return candidates[0] if candidates else None


def main():
    rc = find_rc()
    if not rc:
        sys.exit("rc.exe (Windows SDK resource compiler) not found")
    args = [arg if arg.startswith("/") else arg.replace("/", "\\") for arg in sys.argv[1:]]
    sys.exit(subprocess.call([rc] + args))


if __name__ == "__main__":
    main()
