# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Self-tests for the Phase 0 lexical error-debt census."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import error_debt_census as census


class ErrorDebtCensusTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name) / "src"
        self.root.mkdir()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write(self, relative_path: str, contents: str) -> Path:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
        return path

    def rule_hits(self, rule_id: str) -> list[census.Hit]:
        return sorted(
            (hit for hit in census.scan(self.root) if hit.rule == rule_id),
            key=lambda hit: (hit.file, hit.line, hit.module),
        )

    def assert_rule_hits(
        self, rule_id: str, expected: list[tuple[str, int, str]]
    ) -> None:
        actual = [
            (hit.file, hit.line, hit.module) for hit in self.rule_hits(rule_id)
        ]
        self.assertEqual(expected, actual)

    def test_expected_string_detects_exact_last_top_level_argument(self) -> None:
        self.write(
            "io/result.hpp",
            "#include <expected>\n"
            "using Result = std::expected<std::vector<std::pair<int, int>>, std::string>;\n",
        )
        self.assert_rule_hits(
            "expected-string", [("src/io/result.hpp", 2, "io")]
        )

    def test_expected_string_ignores_other_and_malformed_error_types(self) -> None:
        self.write(
            "io/result.hpp",
            "using A = std::expected<int, std::string_view>;\n"
            "using B = std::expected<int, wrapper<std::string>>;\n"
            "using C = std::expected<int, std::string;\n",
        )
        self.assert_rule_hits("expected-string", [])

    def test_discarded_cuda_detects_only_bare_runtime_statement(self) -> None:
        self.write(
            "training/cuda.cpp",
            "void allocate(void* ptr, size_t size) {\n"
            "  cudaGetLastError();\n"
            "  cudaMalloc(&ptr, size);\n"
            "  LFS_CUDA_CHECK(cudaMalloc(&ptr, size));\n"
            "}\n",
        )
        self.assert_rule_hits(
            "discarded-status-macro-free-cuda",
            [("src/training/cuda.cpp", 3, "training")],
        )

    def test_discarded_cuda_ignores_consumed_and_expression_calls(self) -> None:
        self.write(
            "training/cuda.cpp",
            "cudaError_t allocate(void* ptr, size_t size) {\n"
            "  auto status = cudaMalloc(&ptr, size);\n"
            "  if (cudaMalloc(&ptr, size) != cudaSuccess) {}\n"
            "  consume(cudaMalloc(&ptr, size));\n"
            "  cudaPeekAtLastError();\n"
            "  return cudaMalloc(&ptr, size);\n"
            "}\n",
        )
        self.assert_rule_hits("discarded-status-macro-free-cuda", [])

    def test_raw_vk_check_detects_calls_but_not_definition(self) -> None:
        self.write(
            "rendering/vulkan.cpp",
            "#define _THROW_ERROR(result) handle(result)\n"
            "void submit() {\n"
            "  _THROW_ERROR(vkQueueSubmit());\n"
            "}\n",
        )
        self.assert_rule_hits(
            "raw-vk-check", [("src/rendering/vulkan.cpp", 3, "rendering")]
        )

    def test_raw_vk_check_ignores_own_definitions(self) -> None:
        self.write(
            "rendering/vulkan.hpp",
            "#define _THROW_ERROR(result) handle(result)\n"
            "#define LFS_VK_CHECK_MSG(result, message) handle(result)\n",
        )
        self.assert_rule_hits("raw-vk-check", [])

    def test_raw_vk_check_masks_comment_and_string_mentions(self) -> None:
        self.write(
            "rendering/diagnostic.cpp",
            "// _THROW_ERROR(vkFailure);\n"
            "const char* message = \"_THROW_ERROR(vkFailure)\";\n"
            "/* LFS_VK_CHECK_MSG(vkFailure, message); */\n",
        )
        self.assert_rule_hits("raw-vk-check", [])

    def test_empty_catch_detects_empty_handler(self) -> None:
        self.write(
            "app/worker.cpp",
            "void run() {\n"
            "  try { work(); } catch (...) {\n"
            "  }\n"
            "}\n",
        )
        self.assert_rule_hits("empty-catch", [("src/app/worker.cpp", 2, "app")])

    def test_empty_catch_accepts_reviewed_handler(self) -> None:
        self.write(
            "app/worker.cpp",
            "void run() {\n"
            "  try { work(); } catch (const std::exception& error) {\n"
            "    LOG_ERROR(\"{}\", error.what());\n"
            "  }\n"
            "}\n",
        )
        self.assert_rule_hits("empty-catch", [])

    def test_empty_catch_does_not_accept_log_text_inside_literal(self) -> None:
        self.write(
            "app/worker.cpp",
            "void run() {\n"
            "  try { work(); } catch (...) {\n"
            "    \"LOG_ERROR fallback\";\n"
            "  }\n"
            "}\n",
        )
        self.assert_rule_hits("empty-catch", [("src/app/worker.cpp", 2, "app")])

    def test_empty_catch_accepts_typed_error_return(self) -> None:
        self.write(
            "app/worker.cpp",
            "lfs::Result<int> run() {\n"
            "  try { return work(); } catch (const std::exception& e) {\n"
            "    return lfs::make_error(lfs::ErrorInit{.detail = e.what()});\n"
            "  }\n"
            "}\n",
        )
        self.assert_rule_hits("empty-catch", [])

    def test_empty_catch_accepts_python_callback_containment(self) -> None:
        self.write(
            "python/lfs/py_ui.cpp",
            "void draw() {\n"
            "  try { invoke_cb(); } catch (const std::exception& e) {\n"
            "    (void)contain_cxx_callback(e.what(), PyCallbackPolicy::WarnAndContinue);\n"
            "  }\n"
            "}\n",
        )
        self.assert_rule_hits("empty-catch", [])

    def test_empty_catch_accepts_normalize_current_exception(self) -> None:
        self.write(
            "tcp/tcp_responder.cpp",
            "void run() {\n"
            "  try { respond(); } catch (...) {\n"
            "    const lfs::Error error = detail::normalize_current_exception(ctx);\n"
            "  }\n"
            "}\n",
        )
        self.assert_rule_hits("empty-catch", [])

    def test_empty_catch_accepts_legacy_unexpected_return(self) -> None:
        self.write(
            "app/worker.cpp",
            "std::expected<int, std::string> run() {\n"
            "  try { return work(); } catch (const std::exception& e) {\n"
            "    return std::unexpected(std::string(e.what()));\n"
            "  }\n"
            "}\n",
        )
        self.assert_rule_hits("empty-catch", [])

    def test_empty_catch_does_not_accept_make_error_inside_literal(self) -> None:
        self.write(
            "app/worker.cpp",
            "void run() {\n"
            "  try { work(); } catch (...) {\n"
            "    \"make_error fallback\";\n"
            "  }\n"
            "}\n",
        )
        self.assert_rule_hits("empty-catch", [("src/app/worker.cpp", 2, "app")])

    def test_local_check_macro_detects_non_core_definition(self) -> None:
        self.write(
            "rendering/config.h",
            "#if DEBUG_BUILD\n"
            "#  define LFS_VK_DEBUG_ASSERT(value) assert(value)\n"
            "#endif\n",
        )
        self.assert_rule_hits(
            "local-check-macro", [("src/rendering/config.h", 2, "rendering")]
        )

    def test_local_check_macro_ignores_core_and_nonmatching_names(self) -> None:
        self.write(
            "core/include/core/assert.hpp",
            "#define LFS_CHECK(value) validate(value)\n",
        )
        self.write(
            "rendering/config.h", "#define LFS_VALIDATE(value) validate(value)\n"
        )
        self.assert_rule_hits("local-check-macro", [])

    def test_fatal_invariant_detects_call_but_not_definition(self) -> None:
        self.write(
            "core/allocator.cpp",
            "#define LFS_FATAL_INVARIANT(reason) terminate(reason)\n"
            "void fail() {\n"
            "  LFS_FATAL_INVARIANT(FatalReason::Corruption);\n"
            "}\n",
        )
        self.assert_rule_hits(
            "fatal-invariant-site", [("src/core/allocator.cpp", 3, "core")]
        )

    def test_fatal_invariant_ignores_own_definition(self) -> None:
        self.write(
            "core/fatal.hpp",
            "#define LFS_FATAL_INVARIANT(reason) terminate(reason)\n",
        )
        self.assert_rule_hits("fatal-invariant-site", [])

    def test_vk_infinite_wait_detects_forever_token_in_arguments(self) -> None:
        self.write(
            "rendering/swapchain.cpp",
            "void acquire() {\n"
            "  vkAcquireNextImageKHR(device, swapchain, kWaitForeverNs, semaphore, fence, &index);\n"
            "}\n",
        )
        self.assert_rule_hits(
            "vk-infinite-wait",
            [("src/rendering/swapchain.cpp", 2, "rendering")],
        )

    def test_vk_infinite_wait_ignores_finite_wait(self) -> None:
        self.write(
            "rendering/swapchain.cpp",
            "vkWaitForFences(device, 1, &fence, VK_TRUE, timeout_ns);\n",
        )
        self.assert_rule_hits("vk-infinite-wait", [])

    def test_vk_infinite_wait_masks_diagnostic_call_syntax_in_string(self) -> None:
        self.write(
            "rendering/diagnostic.cpp",
            "const char* message = \"vkWaitForFences(device, 1, &f, VK_TRUE, UINT64_MAX)\";\n",
        )
        self.assert_rule_hits("vk-infinite-wait", [])

    def test_unchecked_kernel_launch_detects_missing_nearby_check(self) -> None:
        self.write(
            "training/kernel.cu",
            "void launch() {\n"
            "  kernel<<<grid, block>>>(input, output);\n"
            "\n"
            "\n"
            "\n"
            "  LFS_CUDA_CHECK(cudaGetLastError());\n"
            "}\n",
        )
        self.assert_rule_hits(
            "unchecked-kernel-launch",
            [("src/training/kernel.cu", 2, "training")],
        )

    def test_unchecked_kernel_launch_accepts_check_within_three_lines(self) -> None:
        self.write(
            "training/kernel.cu",
            "void launch() {\n"
            "  kernel<<<grid, block>>>(input, output);\n"
            "\n"
            "\n"
            "  LFS_CUDA_CHECK(cudaGetLastError());\n"
            "}\n",
        )
        self.assert_rule_hits("unchecked-kernel-launch", [])

    def test_unchecked_kernel_launch_counts_after_digit_separator(self) -> None:
        """C++14 digit separators must not blank the rest of the TU via mask_source."""
        self.write(
            "training/digit_sep.cu",
            "int x = 100'000;\n"
            "void launch() {\n"
            "  kernel<<<grid, block>>>(input, output);\n"
            "}\n",
        )
        self.assert_rule_hits(
            "unchecked-kernel-launch",
            [("src/training/digit_sep.cu", 3, "training")],
        )

    def test_result_in_cu_detects_each_token_inside_tests_path(self) -> None:
        self.write(
            "tests/cuda/result_leak.cu",
            "using A = lfs::Result<int>;\n"
            "using B = std::expected<int, Error>;\n",
        )
        self.assert_rule_hits(
            "result-in-cu",
            [
                ("src/tests/cuda/result_leak.cu", 1, "tests"),
                ("src/tests/cuda/result_leak.cu", 2, "tests"),
            ],
        )

    def test_result_in_cu_ignores_non_cuda_and_masked_tokens(self) -> None:
        self.write(
            "training/result.cpp",
            "using A = lfs::Result<int>;\nusing B = std::expected<int, Error>;\n",
        )
        self.write(
            "training/result.cu",
            "// lfs::Result<int>\nconst char* text = \"std::expected\";\n",
        )
        self.assert_rule_hits("result-in-cu", [])

    def test_frozen_path_exclusions_apply_except_tests_for_result_rule(self) -> None:
        self.write("external/result.cu", "using A = lfs::Result<int>;\n")
        self.write("Vendor/result.cu", "using B = std::expected<int, Error>;\n")
        self.write("rendering/GeneratedShaders/result.cu", "using C = lfs::Result<int>;\n")
        self.write("rendering/foo_generated_kernel.cu", "using D = lfs::Result<int>;\n")
        self.write(
            "rendering/html_viewer_resources.hpp", "_THROW_ERROR(vkFailure);\n"
        )
        self.write("tests/raw.cpp", "_THROW_ERROR(vkFailure);\n")
        self.write("test/result.cuh", "using E = lfs::Result<int>;\n")

        self.assert_rule_hits("raw-vk-check", [])
        self.assert_rule_hits(
            "result-in-cu", [("src/test/result.cuh", 1, "test")]
        )

    def test_mask_source_preserves_length_and_newlines(self) -> None:
        source = (
            "// _THROW_ERROR(comment)\n"
            "/* block _THROW_ERROR(\ncomment */\n"
            "const char* escaped = \"prefix \\\" _THROW_ERROR( \\\\ suffix\";\n"
            "const char* raw = R\"tag(_THROW_ERROR(\nraw)tag\";\n"
            "char quote = '\\'';\n"
        )
        masked = census.mask_source(source)
        self.assertEqual(len(source), len(masked))
        self.assertEqual(source.count("\n"), masked.count("\n"))
        self.assertNotIn("_THROW_ERROR", masked)

    def test_mask_source_leaves_unterminated_raw_literal_unmasked(self) -> None:
        source = 'auto text = R"tag(_THROW_ERROR(\nstill raw\n'
        self.assertEqual(source, census.mask_source(source))

    def test_cli_write_and_gate_contract_reports_only_new_location(self) -> None:
        self.write(
            "rendering/vulkan.cpp",
            "void submit() {\n  _THROW_ERROR(vkQueueSubmit());\n}\n",
        )
        baseline_path = Path(self.temporary_directory.name) / "baseline.json"

        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            write_status = census.main(
                [
                    "--root",
                    str(self.root),
                    "--write-baseline",
                    str(baseline_path),
                ]
            )
        self.assertEqual(0, write_status)
        self.assertEqual("", stdout.getvalue())
        stored = json.loads(baseline_path.read_text(encoding="utf-8"))
        self.assertEqual(
            ["src/rendering/vulkan.cpp:2"],
            stored["raw-vk-check"]["rendering"],
        )

        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            clean_status = census.main(
                ["--root", str(self.root), "--baseline", str(baseline_path)]
            )
        self.assertEqual(0, clean_status)
        self.assertEqual("", stdout.getvalue())

        baseline_path.write_text("{}\n", encoding="utf-8")
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            drift_status = census.main(
                ["--root", str(self.root), "--baseline", str(baseline_path)]
            )
        self.assertEqual(1, drift_status)
        self.assertEqual(
            "raw-vk-check\trendering\tsrc/rendering/vulkan.cpp:2\n",
            stdout.getvalue(),
        )

    def test_gate_ignores_line_drift_of_preexisting_debt(self) -> None:
        self.write(
            "rendering/vulkan.cpp",
            "void submit() {\n  _THROW_ERROR(vkQueueSubmit());\n}\n",
        )
        baseline_path = Path(self.temporary_directory.name) / "baseline.json"
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(
                0,
                census.main(
                    ["--root", str(self.root), "--write-baseline", str(baseline_path)]
                ),
            )

        self.write(
            "rendering/vulkan.cpp",
            "void prelude() {}\n\nvoid submit() {\n  _THROW_ERROR(vkQueueSubmit());\n}\n",
        )
        stdout = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(io.StringIO()):
            shifted_status = census.main(
                ["--root", str(self.root), "--baseline", str(baseline_path)]
            )
        self.assertEqual(0, shifted_status)
        self.assertEqual("", stdout.getvalue())

        self.write(
            "rendering/vulkan.cpp",
            "void submit() {\n  _THROW_ERROR(vkQueueSubmit());\n"
            "  _THROW_ERROR(vkQueuePresent());\n}\n",
        )
        stdout = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(io.StringIO()):
            grown_status = census.main(
                ["--root", str(self.root), "--baseline", str(baseline_path)]
            )
        self.assertEqual(1, grown_status)
        self.assertEqual(
            "raw-vk-check\trendering\tsrc/rendering/vulkan.cpp:3\n",
            stdout.getvalue(),
        )

    def _write_baseline(self, baseline_path: Path) -> None:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(
                0,
                census.main(
                    ["--root", str(self.root), "--write-baseline", str(baseline_path)]
                ),
            )

    def test_gate_fails_on_count_decrease_without_regeneration(self) -> None:
        self.write(
            "rendering/vulkan.cpp",
            "void submit() {\n  _THROW_ERROR(vkQueueSubmit());\n"
            "  _THROW_ERROR(vkQueuePresent());\n}\n",
        )
        baseline_path = Path(self.temporary_directory.name) / "baseline.json"
        self._write_baseline(baseline_path)

        self.write(
            "rendering/vulkan.cpp",
            "void submit() {\n  _THROW_ERROR(vkQueueSubmit());\n}\n",
        )
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = census.main(
                ["--root", str(self.root), "--baseline", str(baseline_path)]
            )
        self.assertEqual(1, status)
        self.assertEqual(
            "stale-baseline\traw-vk-check\trendering\tsrc/rendering/vulkan.cpp\t2->1\n",
            stdout.getvalue(),
        )
        self.assertIn("--write-baseline", stderr.getvalue())

    def test_gate_fails_when_baselined_file_disappears(self) -> None:
        path = self.write(
            "rendering/vulkan.cpp",
            "void submit() {\n  _THROW_ERROR(vkQueueSubmit());\n}\n",
        )
        baseline_path = Path(self.temporary_directory.name) / "baseline.json"
        self._write_baseline(baseline_path)

        path.unlink()
        stdout = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(io.StringIO()):
            status = census.main(
                ["--root", str(self.root), "--baseline", str(baseline_path)]
            )
        self.assertEqual(1, status)
        self.assertEqual(
            "stale-baseline\traw-vk-check\trendering\tsrc/rendering/vulkan.cpp\t1->0\n",
            stdout.getvalue(),
        )

    def test_gate_passes_after_regeneration_acknowledges_decrease(self) -> None:
        self.write(
            "rendering/vulkan.cpp",
            "void submit() {\n  _THROW_ERROR(vkQueueSubmit());\n"
            "  _THROW_ERROR(vkQueuePresent());\n}\n",
        )
        baseline_path = Path(self.temporary_directory.name) / "baseline.json"
        self._write_baseline(baseline_path)

        self.write(
            "rendering/vulkan.cpp",
            "void submit() {\n  _THROW_ERROR(vkQueueSubmit());\n}\n",
        )
        self._write_baseline(baseline_path)

        stdout = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(io.StringIO()):
            status = census.main(
                ["--root", str(self.root), "--baseline", str(baseline_path)]
            )
        self.assertEqual(0, status)
        self.assertEqual("", stdout.getvalue())

    def test_gate_reports_increase_and_decrease_together(self) -> None:
        self.write("rendering/a.cpp", "void a() {\n  _THROW_ERROR(vkA());\n}\n")
        self.write(
            "rendering/b.cpp",
            "void b() {\n  _THROW_ERROR(vkB());\n  _THROW_ERROR(vkC());\n}\n",
        )
        baseline_path = Path(self.temporary_directory.name) / "baseline.json"
        self._write_baseline(baseline_path)

        self.write(
            "rendering/a.cpp",
            "void a() {\n  _THROW_ERROR(vkA());\n  _THROW_ERROR(vkA2());\n}\n",
        )
        self.write("rendering/b.cpp", "void b() {\n  _THROW_ERROR(vkB());\n}\n")
        stdout = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(io.StringIO()):
            status = census.main(
                ["--root", str(self.root), "--baseline", str(baseline_path)]
            )
        self.assertEqual(1, status)
        output = stdout.getvalue()
        self.assertIn("raw-vk-check\trendering\tsrc/rendering/a.cpp:3\n", output)
        self.assertIn(
            "stale-baseline\traw-vk-check\trendering\tsrc/rendering/b.cpp\t2->1\n",
            output,
        )

    def test_gate_catches_wrapper_refactor_that_hides_sites(self) -> None:
        self.write(
            "rendering/wait.cpp",
            "void wait_all(VkDevice device, VkFence fence) {\n"
            "  vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);\n"
            "  vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);\n"
            "}\n",
        )
        baseline_path = Path(self.temporary_directory.name) / "baseline.json"
        self._write_baseline(baseline_path)

        self.write(
            "rendering/wait.cpp",
            "void wait_all(VkDevice device, VkFence fence) {\n"
            "  waitBounded(fence);\n"
            "  waitBounded(fence);\n"
            "}\n",
        )
        stdout = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(io.StringIO()):
            status = census.main(
                ["--root", str(self.root), "--baseline", str(baseline_path)]
            )
        self.assertEqual(1, status)
        self.assertEqual(
            "stale-baseline\tvk-infinite-wait\trendering\tsrc/rendering/wait.cpp\t2->0\n",
            stdout.getvalue(),
        )

    def test_empty_catch_reviewed_annotation_is_exempt(self) -> None:
        self.write(
            "core/reviewed.cpp",
            "void f() {\n  try {\n    g();\n  } catch (...) {\n"
            "    // LFS-CENSUS-OK(empty-catch): best-effort diagnostics only.\n"
            "  }\n}\n",
        )
        self.write(
            "core/unreviewed.cpp",
            "void f() {\n  try {\n    g();\n  } catch (...) {\n  }\n}\n",
        )
        hits = census.scan(self.root)
        locations = [hit.location for hit in hits if hit.rule == "empty-catch"]
        self.assertEqual(["src/core/unreviewed.cpp:4"], locations)

    def test_allowlist_exempts_sanctioned_legacy_bridge_declaration(self) -> None:
        self.write(
            "core/include/core/error.hpp",
            "template <class T>\n"
            "Result<T> from_legacy_expected(std::expected<T, std::string>&& value);\n"
            "std::expected<int, std::string> not_sanctioned();\n",
        )
        hits = census.scan(self.root)
        locations = [hit.location for hit in hits if hit.rule == "expected-string"]
        self.assertEqual(["src/core/include/core/error.hpp:3"], locations)

    def test_cli_rejects_unknown_rule_and_missing_baseline(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            unknown_status = census.main(
                ["--root", str(self.root), "--list", "not-a-rule"]
            )
        self.assertEqual(2, unknown_status)
        self.assertIn("unknown rule id", stderr.getvalue())

        stderr = io.StringIO()
        with redirect_stdout(io.StringIO()), redirect_stderr(stderr):
            missing_status = census.main(
                [
                    "--root",
                    str(self.root),
                    "--baseline",
                    str(Path(self.temporary_directory.name) / "missing.json"),
                ]
            )
        self.assertEqual(2, missing_status)
        self.assertIn("cannot load baseline", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
