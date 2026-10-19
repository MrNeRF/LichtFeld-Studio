# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Retry only download failures in CMake's manifest-mode vcpkg install."""

import subprocess
import sys
import time


def run_command(command):
    output = []
    # CMake writes UTF-8. A Windows console code page cannot show every character,
    # so replace those instead of failing the configure step.
    console = sys.stdout.encoding or "utf-8"
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          encoding="utf-8", errors="replace") as process:
        for line in process.stdout:
            print(line.encode(console, errors="replace").decode(console), end="", flush=True)
            output.append(line)
        return process.wait(), "".join(output)


def retry_configure(command, run=run_command, sleep=time.sleep):
    for attempt in range(3):
        code, output = run(command)
        if code == 0:
            return 0
        download_failed = any(marker in output.lower() for marker in (
            "download failed, halting portfile", "curl operation failed", "failed to download",
        ))
        if attempt == 2 or not download_failed or "vcpkg install failed" not in output.lower():
            return code
        delay = 15 * (attempt + 1)
        print(f"vcpkg download failed; retrying configure in {delay}s ({attempt + 2}/3).", flush=True)
        sleep(delay)
    raise AssertionError("unreachable")


if __name__ == "__main__":
    command = sys.argv[1:]
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        sys.exit("Usage: ci_retry_configure.py -- cmake <configure arguments>")
    sys.exit(retry_configure(command))
