# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Self-tests for the Windows build preflight."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import subprocess
import unittest
from unittest import mock

try:
    import windows_build_preflight as preflight
except ImportError:
    from tools import windows_build_preflight as preflight


class WindowsBuildPreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        (self.root / ".git").mkdir()
        (self.root / "src" / "visualizer" / "gui").mkdir(parents=True)
        (self.root / "tests").mkdir()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write(self, relative_path: str, contents: str) -> Path:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
        return path

    def test_visualizer_test_function_requires_export(self) -> None:
        self.write(
            "src/visualizer/gui/widget.hpp",
            "#pragma once\n"
            "namespace lfs::vis::gui {\n"
            "[[nodiscard]] bool makeWidget(int size);\n"
            "}\n",
        )
        self.write(
            "tests/test_widget.cpp",
            '#include "gui/widget.hpp"\n'
            "using lfs::vis::gui::makeWidget;\n"
            "void test() { (void)makeWidget(3); }\n",
        )
        findings = preflight.check_visualizer_test_exports(self.root)
        self.assertEqual(["visualizer-test-dll-export"], [item.rule for item in findings])
        self.assertEqual(3, findings[0].line)

    def test_exported_and_constexpr_functions_are_accepted(self) -> None:
        self.write(
            "src/visualizer/gui/widget.hpp",
            "#pragma once\n"
            "namespace lfs::vis::gui {\n"
            "[[nodiscard]] LFS_VIS_API bool makeWidget(int size);\n"
            "constexpr int widgetLimit() { return 4; }\n"
            "}\n",
        )
        self.write(
            "tests/test_widget.cpp",
            '#include "gui/widget.hpp"\n'
            "using lfs::vis::gui::makeWidget;\n"
            "void test() { (void)makeWidget(widgetLimit()); }\n",
        )
        self.assertEqual([], preflight.check_visualizer_test_exports(self.root))

    def test_methods_do_not_require_function_level_export(self) -> None:
        self.write(
            "src/visualizer/gui/widget.hpp",
            "#pragma once\n"
            "namespace lfs::vis::gui {\n"
            "class LFS_VIS_API Widget {\n"
            "public:\n"
            "  bool valid() const;\n"
            "};\n"
            "}\n",
        )
        self.write(
            "tests/test_widget.cpp",
            '#include "gui/widget.hpp"\n'
            "void test(Widget& widget) { (void)widget.valid(); }\n",
        )
        self.assertEqual([], preflight.check_visualizer_test_exports(self.root))

    def test_unqualified_call_without_explicit_import_is_ignored(self) -> None:
        self.write(
            "src/visualizer/gui/widget.hpp",
            "namespace lfs::vis::gui { bool makeWidget(int size); }\n",
        )
        self.write(
            "tests/test_widget.cpp",
            '#include "gui/widget.hpp"\n'
            "using namespace lfs::vis::gui;\n"
            "void test() { (void)makeWidget(3); }\n",
        )
        self.assertEqual([], preflight.check_visualizer_test_exports(self.root))

    def test_fully_qualified_visualizer_call_requires_export(self) -> None:
        self.write(
            "src/visualizer/gui/widget.hpp",
            "namespace lfs::vis::gui {\n"
            "bool makeWidget(int size);\n"
            "}\n",
        )
        self.write(
            "tests/test_widget.cpp",
            '#include "gui/widget.hpp"\n'
            "void test() { (void)lfs::vis::gui::makeWidget(3); }\n",
        )
        findings = preflight.check_visualizer_test_exports(self.root)
        self.assertEqual(["makeWidget"], [item.message.split()[2] for item in findings])

    def test_header_body_calls_do_not_look_like_declarations(self) -> None:
        self.write(
            "src/visualizer/gui/widget.hpp",
            "namespace lfs::vis::gui {\n"
            "inline bool wrapper() { return Factory::makeWidget(); }\n"
            "}\n",
        )
        self.write(
            "tests/test_widget.cpp",
            '#include "gui/widget.hpp"\n'
            "using lfs::vis::gui::makeWidget;\n",
        )
        self.assertEqual([], preflight.check_visualizer_test_exports(self.root))

    def test_inline_definition_is_available_without_dll_export(self) -> None:
        self.write(
            "src/visualizer/gui/widget.hpp",
            "namespace lfs::vis::gui {\n"
            "inline bool makeWidget(int size) { return size > 0; }\n"
            "}\n",
        )
        self.write(
            "tests/test_widget.cpp",
            '#include "gui/widget.hpp"\n'
            "using lfs::vis::gui::makeWidget;\n"
            "void test() { (void)makeWidget(3); }\n",
        )
        self.assertEqual([], preflight.check_visualizer_test_exports(self.root))

    def test_compile_database_validation(self) -> None:
        database = self.write(
            "compile_commands.json",
            json.dumps(
                [
                    {
                        "directory": str(self.root),
                        "command": '"C:/VS/cl.exe" /c source.cpp',
                        "file": str(self.root / "source.cpp"),
                    }
                ]
            ),
        )
        commands = preflight.load_compile_commands(database)
        self.assertEqual(1, len(commands))
        self.assertEqual((self.root / "source.cpp").resolve(), commands[0].file)

    def test_compile_database_accepts_arguments_array(self) -> None:
        database = self.write(
            "compile_commands.json",
            json.dumps(
                [
                    {
                        "directory": str(self.root),
                        "arguments": ["C:/VS/cl.exe", "/c", "source.cpp"],
                        "file": "source.cpp",
                    }
                ]
            ),
        )
        commands = preflight.load_compile_commands(database)
        self.assertEqual((self.root / "source.cpp").resolve(), commands[0].file)
        self.assertTrue(preflight._is_msvc_command(commands[0]))

    def test_msvc_detection_rejects_nvcc_host_compilation(self) -> None:
        source = self.root / "source.cpp"
        self.assertTrue(
            preflight._is_msvc_command(
                preflight.CompileCommand(source, self.root, '"C:/VS/cl.exe" /c source.cpp')
            )
        )
        self.assertFalse(
            preflight._is_msvc_command(
                preflight.CompileCommand(
                    source, self.root, 'nvcc.exe -ccbin "C:/VS/cl.exe" source.cu'
                )
            )
        )

    def test_syntax_command_bypasses_sccache_and_show_includes(self) -> None:
        command = (
            'sccache "C:/Program Files/VS/cl.exe" /nologo /showIncludes '
            '/Iinclude /Foout.obj /c source.cpp'
        )
        syntax_command = preflight._msvc_syntax_command(command)
        self.assertTrue(syntax_command.startswith('"C:/Program Files/VS/cl.exe"'))
        self.assertNotIn("sccache", syntax_command.lower())
        self.assertNotIn("showincludes", syntax_command.lower())
        self.assertTrue(syntax_command.endswith("/Zs"))

    def test_syntax_command_removes_pch_options_without_changing_other_arguments(self):
        prefix = '"C:/Program Files/VS/cl.exe" /DNAME="hello world" /I"C:/my includes"'
        for pch in (
            '/Yc"pch header.hpp" /Fp"C:/build dir/cache.pch"',
            '/Yu "pch header.hpp" /Fp "C:/build dir/cache.pch"',
            '/Ycpch.hpp /Yupch.hpp /Fpcache.pch',
            '/Yc /Yu /Fp',
            '"/Yupch header.hpp" "/FpC:/build dir/cache.pch"',
        ):
            with self.subTest(pch=pch):
                command = f'{prefix} {pch} /Fo"C:/out dir/file.obj" /c "src/my file.cpp"'
                self.assertEqual(
                    f'{prefix} /Fo"C:/out dir/file.obj" /c "src/my file.cpp" /Zs',
                    preflight._msvc_syntax_command(command),
                )

    def test_syntax_command_preserves_escaped_quotes_and_trailing_backslashes(self):
        arguments = [
            "C:/Program Files/VS/cl.exe", r'-DNAME="C:\source\src\python"',
            '-DMESSAGE="hello /Yu /showIncludes world"',
            "-I" + "C:\\include with spaces\\", "-I" + "C:\\no-spaces\\",
            "/c", "src/my source.cpp",
        ]
        command = subprocess.list2cmdline(arguments)
        self.assertEqual(len(arguments), len(preflight._windows_command_tokens(command)))
        self.assertEqual(command + " /Zs", preflight._msvc_syntax_command(command))

    def test_header_change_selects_transitive_translation_unit(self) -> None:
        header = self.write("src/visualizer/detail.hpp", "#pragma once\n")
        self.write(
            "src/visualizer/wrapper.hpp",
            '#include "detail.hpp"\n',
        )
        source = self.write(
            "src/visualizer/widget.cpp",
            '#include "wrapper.hpp"\n',
        )
        unrelated = self.write("src/visualizer/other.cpp", "void other() {}\n")
        commands = [
            preflight.CompileCommand(source.resolve(), self.root, "cl.exe /c widget.cpp"),
            preflight.CompileCommand(
                unrelated.resolve(), self.root, "cl.exe /c other.cpp"
            ),
        ]
        selected = preflight.select_compile_commands(
            self.root, commands, {header.resolve()}, False
        )
        self.assertEqual([source.resolve()], [item.file for item in selected])

    def test_deleted_header_selects_its_consumer(self) -> None:
        deleted = self.root / "src" / "visualizer" / "deleted.hpp"
        source = self.write(
            "src/visualizer/widget.cpp",
            '#include "deleted.hpp"\n',
        )
        commands = [
            preflight.CompileCommand(source.resolve(), self.root, "cl.exe /c widget.cpp")
        ]
        selected = preflight.select_compile_commands(
            self.root, commands, {deleted.resolve()}, False
        )
        self.assertEqual([source.resolve()], [item.file for item in selected])

    def test_force_all_selects_project_cxx_but_not_cuda_or_external(self) -> None:
        source = self.write("src/visualizer/widget.cpp", "void widget() {}\n")
        cuda = self.write("src/visualizer/widget.cu", "__global__ void widget() {}\n")
        external = self.write("external/library.cpp", "void library() {}\n")
        commands = [
            preflight.CompileCommand(source.resolve(), self.root, "cl.exe /c widget.cpp"),
            preflight.CompileCommand(cuda.resolve(), self.root, "nvcc widget.cu"),
            preflight.CompileCommand(
                external.resolve(), self.root, "cl.exe /c library.cpp"
            ),
        ]
        selected = preflight.select_compile_commands(
            self.root, commands, set(), True
        )
        self.assertEqual([source.resolve()], [item.file for item in selected])

    def test_all_commands_excludes_generated_and_dependency_sources(self):
        paths = ["src/core/real.cpp", "tests/real.cpp", "external/vendor.cpp",
                 "build/vcpkg_installed/x64-windows/src/nanobind.cpp",
                 "build/_deps/dependency-src/vendor.cpp", "_deps/vendor.cpp",
                 "build-release/generated.cpp", "build/generated.cpp"]
        commands = [preflight.CompileCommand(self.write(path, "").resolve(),
                                             self.root, "cl.exe /c file.cpp")
                    for path in paths]
        selected = preflight.select_compile_commands(self.root, commands, set(), True)
        self.assertEqual(commands[:2], selected)

    def test_generated_config_does_not_alias_vulkan_config(self):
        build = self.root / "custom build"
        generated = self.write("custom build/include/config.h", "#pragma once\n")
        vulkan = self.write("src/rendering/vulkan/config.h", "#pragma once\n")
        actual = self.write("src/rendering/vulkan/renderer.cpp", '#include "config.h"\n')
        unrelated = self.write("src/visualizer/widget.cpp", '#include "config.h"\n')
        commands = [preflight.CompileCommand(path.resolve(), build, "cl.exe /c file.cpp")
                    for path in (actual, unrelated)]
        self.assertEqual([commands[0]], preflight.select_compile_commands(
            self.root, commands, {vulkan.resolve()}, False, build))
        self.assertEqual([commands[1]], preflight.select_compile_commands(
            self.root, commands, {generated.resolve()}, False, build))

    def test_generated_headers_in_build_outside_checkout(self):
        with tempfile.TemporaryDirectory() as temporary:
            build = Path(temporary).resolve()
            (build / "include").mkdir()
            (build / "include/config.h").write_text("", encoding="utf-8")
            vulkan = self.write("src/rendering/vulkan/config.h", "")
            source = self.write("src/visualizer/widget.cpp", '#include "config.h"\n')
            command = preflight.CompileCommand(source.resolve(), build, "cl.exe /c widget.cpp")
            self.assertEqual([], preflight.select_compile_commands(
                self.root, [command], {vulkan.resolve()}, False, build))

    def test_budget_skip_emits_github_warning_without_replay(self):
        source = self.write("src/core/sample.cpp", "int sample;\n")
        database = self.write("build/compile_commands.json", json.dumps([
            {"file": str(source), "directory": str(self.root), "command": "cl.exe /c sample.cpp"}
        ] * 2))
        output = StringIO()
        with mock.patch.dict(preflight.os.environ, {"GITHUB_ACTIONS": "true"}), \
                mock.patch.object(preflight, "run_msvc_syntax_checks") as replay, \
                redirect_stdout(output):
            result = preflight.main(["--root", str(self.root), "--skip-source-checks",
                                     "--compile-commands", str(database),
                                     "--all-commands", "--max-commands", "1"])
        self.assertEqual(0, result)
        self.assertIn("::warning::MSVC preflight: configured replay skipped", output.getvalue())
        replay.assert_not_called()

    def test_replay_uses_database_without_cmake_metadata_or_build_commands(self):
        source = self.write("src/core/sample.cpp", "int sample;\n")
        database = self.write("build/compile_commands.json", json.dumps([
            {"file": str(source), "directory": str(self.root), "command": "cl.exe /c sample.cpp"}
        ]))
        host = mock.Mock(wraps=preflight.os)
        host.name = "nt"
        with mock.patch.object(preflight, "os", host), \
                mock.patch.object(preflight.subprocess, "run") as native, \
                mock.patch.object(preflight, "run_msvc_syntax_checks", return_value=[]) as replay, \
                redirect_stdout(StringIO()):
            result = preflight.main(["--root", str(self.root), "--skip-source-checks",
                                     "--compile-commands", str(database), "--all-commands"])
        self.assertEqual(0, result)
        replay.assert_called_once()
        native.assert_not_called()

    def test_build_graph_only_change_does_not_replay_every_source(self) -> None:
        source = self.write("src/visualizer/widget.cpp", "void widget() {}\n")
        cmake = self.write("CMakeLists.txt", "project(sample)\n")
        commands = [
            preflight.CompileCommand(source.resolve(), self.root, "cl.exe /c widget.cpp")
        ]
        selected = preflight.select_compile_commands(
            self.root, commands, {cmake.resolve()}, False
        )
        self.assertEqual([], selected)

    def test_selected_command_report_is_relative_sorted_and_counts_duplicates(
        self,
    ) -> None:
        alpha = self.write("src/visualizer/alpha.cpp", "void alpha() {}\n")
        beta = self.write("src/visualizer/beta.cpp", "void beta() {}\n")
        commands = [
            preflight.CompileCommand(beta.resolve(), self.root, "cl.exe /c beta.cpp"),
            preflight.CompileCommand(alpha.resolve(), self.root, "cl.exe /c alpha.cpp"),
            preflight.CompileCommand(beta.resolve(), self.root, "cl.exe /c beta.cpp"),
        ]
        output = StringIO()

        with redirect_stdout(output):
            preflight._print_selected_compile_commands(self.root, commands)

        self.assertEqual(
            [
                "MSVC preflight: selected 2 source file(s) for 3 compile command(s):",
                "  - src/visualizer/alpha.cpp",
                "  - src/visualizer/beta.cpp (2 compile commands)",
            ],
            output.getvalue().splitlines(),
        )
        self.assertNotIn(self.root.as_posix(), output.getvalue())


class GitDiscoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        environment = mock.patch.dict(preflight.os.environ,
                                      {"GITHUB_ACTIONS": "false", "LFS_PREFLIGHT_BASE": ""})
        environment.start()
        self.addCleanup(environment.stop)
        self.git("init", "--initial-branch=master")
        self.header = self.write("src/core/include/core/internal/detail.hpp", "#pragma once\n")
        self.write("src/core/include/core/tensor.hpp", "#include <core/internal/detail.hpp>\n")
        self.source = self.write("src/app/sample.cpp", '#include <vector>\n#include "core/tensor.hpp"\n')
        self.commands = [preflight.CompileCommand(self.source, self.root, "cl.exe /c sample.cpp")]
        self.commit("baseline")

    def git(self, *arguments):
        result = subprocess.run([
            "git", "-c", "user.name=Preflight Test", "-c", "user.email=preflight@example.invalid",
            "-c", "commit.gpgsign=false", "-c", f"core.hooksPath={self.root / 'no-hooks'}",
            "-C", str(self.root), *arguments,
        ], capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        return result.stdout.strip()

    def write(self, relative, contents):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
        return path.resolve()

    def commit(self, message):
        self.git("add", ".")
        self.git("commit", "-m", message)

    def assert_header_change_reaches_consumer(self):
        changed, force_all = preflight.discover_changed_files(self.root, None)
        self.assertFalse(force_all)
        self.assertIn(self.header, changed)
        self.assertEqual(self.commands, preflight.select_compile_commands(
            self.root, self.commands, changed, force_all))
        return changed

    def test_staged_header_deletion(self):
        self.git("rm", self.header.relative_to(self.root).as_posix())
        self.assert_header_change_reaches_consumer()

    def test_unstaged_header_deletion(self):
        self.header.unlink()
        self.assert_header_change_reaches_consumer()

    def test_git_mv_reports_old_and_new_header(self):
        renamed = self.header.with_name("renamed.hpp")
        self.git("mv", str(self.header), str(renamed))
        self.assertIn(renamed, self.assert_header_change_reaches_consumer())

    def test_angle_include_transitive_graph_in_git_and_source_archive(self):
        self.header.write_text("// changed\n", encoding="utf-8")
        self.assert_header_change_reaches_consumer()
        # Both the accelerated git-grep and source-archive fallback must agree.
        with mock.patch.object(preflight, "_git", return_value=mock.Mock(returncode=128)):
            self.assertEqual(self.commands, preflight.select_compile_commands(
                self.root, self.commands, {self.header}, False))
        graph = preflight.build_reverse_include_graph(self.root)
        self.assertTrue(all(path.is_relative_to(self.root) for path in graph))
        self.assertFalse(any(path.name == "vector" for path in graph))

    def test_pr_merge_base_does_not_replay_new_master_sources(self):
        old_base = self.git("rev-parse", "HEAD")
        self.git("checkout", "-b", "feature")
        cmake = self.write("CMakeLists.txt", "# PR changes only build files\n")
        self.commit("feature build change")
        self.git("checkout", "master")
        self.source.write_text("int added_on_master;\n", encoding="utf-8")
        self.commit("new master source change")
        self.git("merge", "--no-ff", "feature", "-m", "simulated PR merge checkout")
        # The old event base incorrectly included the master's C++ changes.
        old_changed, _ = preflight.discover_changed_files(self.root, old_base)
        self.assertIn(self.source, old_changed)
        with mock.patch.dict(preflight.os.environ,
                             {"GITHUB_ACTIONS": "true", "LFS_PREFLIGHT_BASE": "HEAD^1"}):
            changed, forced = preflight.discover_changed_files(self.root, None)
        self.assertFalse(forced)
        self.assertEqual({cmake}, changed)
        self.assertEqual([], preflight.select_compile_commands(
            self.root, self.commands, changed, forced))


if __name__ == "__main__":
    unittest.main()
