# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Bound retry time and keep compiler/configuration failures visible."""

import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock


spec = importlib.util.spec_from_file_location(
    "ci_retry_configure", Path(__file__).resolve().parents[2] / "tools/ci_retry_configure.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
DOWNLOAD_ERROR = "error: curl operation failed with response code 503.\nvcpkg install failed"


class RetryConfigureTest(unittest.TestCase):
    def run_case(self, results):
        run = Mock(side_effect=results)
        sleep = Mock()
        code = module.retry_configure(["cmake", "-B", "build"], run=run, sleep=sleep)
        return code, run, sleep

    def test_success_runs_once(self):
        code, run, sleep = self.run_case([(0, "Configured")])
        self.assertEqual(code, 0)
        self.assertEqual(run.call_count, 1)
        sleep.assert_not_called()

    def test_download_failure_recovers(self):
        code, run, sleep = self.run_case([(1, DOWNLOAD_ERROR), (0, "Configured")])
        self.assertEqual(code, 0)
        self.assertEqual(run.call_count, 2)
        sleep.assert_called_once_with(15)

    def test_persistent_failure_stops_after_three_attempts(self):
        code, run, sleep = self.run_case([(7, DOWNLOAD_ERROR)] * 3)
        self.assertEqual(code, 7)
        self.assertEqual(run.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [15, 30])

    def test_compile_or_configure_failure_is_not_retried(self):
        for output in ["CXX compiler not found", "vcpkg install failed: compiler error",
                       "curl operation failed in unrelated application check"]:
            with self.subTest(output=output):
                code, run, sleep = self.run_case([(2, output)])
                self.assertEqual(code, 2)
                self.assertEqual(run.call_count, 1)
                sleep.assert_not_called()

    def test_non_download_failure_after_retry_stops_immediately(self):
        code, run, sleep = self.run_case([(1, DOWNLOAD_ERROR), (3, "CMake syntax error")])
        self.assertEqual(code, 3)
        self.assertEqual(run.call_count, 2)
        sleep.assert_called_once_with(15)

    def test_output_the_console_cannot_encode_is_replaced(self):
        # CMake status banners use box-drawing characters; cp1252 has no mapping for them.
        console = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
        script = "import sys; sys.stdout.buffer.write('\\u2550 tcp\\n'.encode('utf-8'))"
        with contextlib.redirect_stdout(console):
            code, output = module.run_command([sys.executable, "-c", script])
        self.assertEqual(code, 0)
        self.assertEqual(output, "\u2550 tcp\n")


if __name__ == "__main__":
    unittest.main()
