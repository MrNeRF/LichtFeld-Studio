# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Self-tests for the Windows build preflight."""

from __future__ import annotations

from contextlib import ExitStack, redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
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


class GeneratedHeaderPreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name).resolve()
        (self.root / ".git").mkdir()
        self.build = self.root / "build with spaces"
        (self.build / "CMakeFiles").mkdir(parents=True)
        self.source = self.root / "src" / "core" / "abi.cpp"
        self.source.parent.mkdir(parents=True)
        self.source.write_text('#include "lfs_core_abi_stamp.h"\n', encoding="utf-8")
        self.database = self.build / "compile_commands.json"
        self.entries = [{
            "directory": str(self.build),
            "command": '"C:/VS/cl.exe" /c abi.cpp',
            "file": str(self.source),
        }]
        self.write_database()
        self.cmake = str(self.root / "CMake with spaces" / "cmake.exe")
        self.write_metadata()
        # Mock the module's OS facade, not os.name globally: changing the latter
        # makes pathlib choose WindowsPath even on the Linux source-check job.
        host = mock.Mock(wraps=preflight.os)
        host.name = "nt"
        patches = self.enterContext(ExitStack())
        patches.enter_context(mock.patch.object(preflight, "os", host))
        self.native = patches.enter_context(mock.patch.object(preflight.subprocess, "run"))
        self.native.return_value.returncode = 0
        self.syntax = patches.enter_context(
            mock.patch.object(preflight, "run_msvc_syntax_checks", return_value=[])
        )
        self.output = StringIO()

    def write_database(self) -> None:
        self.database.write_text(json.dumps(self.entries), encoding="utf-8")

    def write_metadata(
        self, *, target: bool = True, generator: str = "Ninja",
        build_type: str = "Release", configurations: str = "Debug;Release;RelWithDebInfo",
        source_root: Path | None = None,
    ) -> None:
        (self.build / "CMakeCache.txt").write_text(
            f"CMAKE_HOME_DIRECTORY:INTERNAL={source_root or self.root}\n"
            f"CMAKE_COMMAND:INTERNAL={self.cmake}\n"
            f"CMAKE_GENERATOR:INTERNAL={generator}\n"
            f"CMAKE_BUILD_TYPE:STRING={build_type}\n"
            f"CMAKE_CONFIGURATION_TYPES:STRING={configurations}\n",
            encoding="utf-8",
        )
        targets = ["lfs_git_version", "LichtFeld-Studio"]
        if target:
            targets.append("lfs_core_abi_stamp")
        (self.build / "CMakeFiles" / "TargetDirectories.txt").write_text(
            "".join(f"{self.build.as_posix()}/CMakeFiles/{name}.dir\n" for name in targets),
            encoding="utf-8",
        )

    def run_preflight(self, *extra: str, all_commands: bool = True) -> int:
        arguments = ["--root", str(self.root), "--skip-source-checks",
                     "--compile-commands", str(self.database)]
        if all_commands:
            arguments.append("--all-commands")
        with redirect_stdout(self.output), redirect_stderr(self.output):
            return preflight.main([*arguments, *extra])

    def test_clean_tree_generates_abi_header_before_msvc(self) -> None:
        header = self.build / "include" / "Release" / "lfs_core_abi_stamp.h"
        events = []

        def generate(command, **kwargs):
            self.assertEqual([
                self.cmake, "--build", str(self.build), "--target",
                "lfs_core_abi_stamp", "--config", "Release",
            ], command)
            header.parent.mkdir(parents=True)
            header.write_text("#define LFS_CORE_ABI_STAMP \"fixture\"\n", encoding="utf-8")
            events.append("generate")
            return mock.Mock(returncode=0)

        def check(commands, jobs):
            self.assertTrue(header.is_file())
            self.assertEqual(self.source, commands[0].file)
            events.append("syntax")
            return []

        self.native.side_effect = generate
        self.syntax.side_effect = check
        self.assertEqual(0, self.run_preflight())
        self.assertEqual(["generate", "syntax"], events)
        self.native.assert_called_once()

    def test_legacy_configure_time_header_needs_no_build_target(self) -> None:
        self.write_metadata(target=False)
        self.assertEqual(0, self.run_preflight())
        self.native.assert_not_called()
        self.syntax.assert_called_once()

    def test_ninja_uses_build_type_despite_cached_configuration_types(self) -> None:
        for configuration in ("Debug", "Release", "RelWithDebInfo", "MinSizeRel", ""):
            with self.subTest(configuration=configuration):
                self.write_metadata(build_type=configuration)
                commands = preflight.generated_header_build_commands(self.root, self.database)
                expected = [
                    self.cmake, "--build", str(self.build), "--target", "lfs_core_abi_stamp"
                ]
                if configuration:
                    expected.extend(["--config", configuration])
                self.assertEqual([expected], commands)

    def test_multi_config_prepares_each_configuration_once(self) -> None:
        self.write_metadata(
            generator="Ninja Multi-Config", configurations="Debug;Release;RelWithDebInfo;Debug"
        )
        self.assertEqual(0, self.run_preflight())
        self.assertEqual(
            ["Debug", "Release", "RelWithDebInfo"],
            [call.args[0][-1] for call in self.native.call_args_list],
        )
        self.syntax.assert_called_once()

    def test_empty_multi_config_is_an_actionable_error(self) -> None:
        self.write_metadata(generator="Ninja Multi-Config", configurations="")
        self.assertEqual(1, self.run_preflight())
        self.assertIn("no configurations", self.output.getvalue())
        self.native.assert_not_called()
        self.syntax.assert_not_called()

    def test_preparation_failure_prevents_msvc(self) -> None:
        self.native.return_value.returncode = 7
        self.assertEqual(1, self.run_preflight())
        self.assertIn("preparation failed (exit 7)", self.output.getvalue())
        self.syntax.assert_not_called()

    def test_missing_cmake_prevents_msvc_with_a_clear_error(self) -> None:
        self.native.side_effect = FileNotFoundError("cmake fixture missing")
        self.assertEqual(1, self.run_preflight())
        self.assertIn("cannot prepare generated headers", self.output.getvalue())
        self.syntax.assert_not_called()

    def test_missing_metadata_does_not_silently_skip_preparation(self) -> None:
        (self.build / "CMakeFiles" / "TargetDirectories.txt").unlink()
        self.assertEqual(1, self.run_preflight())
        self.assertIn("original build directory", self.output.getvalue())
        self.native.assert_not_called()
        self.syntax.assert_not_called()

    def test_another_source_tree_is_not_built(self) -> None:
        self.write_metadata(source_root=self.root / "other checkout")
        self.assertEqual(1, self.run_preflight())
        self.assertIn("different CMake source tree", self.output.getvalue())
        self.native.assert_not_called()
        self.syntax.assert_not_called()

    def test_dry_run_never_prepares_headers_or_invokes_msvc(self) -> None:
        self.assertEqual(0, self.run_preflight("--dry-run"))
        self.native.assert_not_called()
        self.syntax.assert_not_called()

    def test_source_only_never_prepares_headers_or_invokes_msvc(self) -> None:
        with redirect_stdout(self.output):
            self.assertEqual(0, preflight.main(["--root", str(self.root), "--source-only"]))
        self.native.assert_not_called()
        self.syntax.assert_not_called()

    def test_no_affected_commands_never_prepares_headers(self) -> None:
        with mock.patch.object(
            preflight, "discover_changed_files", return_value=(set(), False)
        ):
            self.assertEqual(0, self.run_preflight(all_commands=False))
        self.native.assert_not_called()
        self.syntax.assert_not_called()

    def test_over_budget_never_prepares_headers(self) -> None:
        self.entries *= 2
        self.write_database()
        self.assertEqual(0, self.run_preflight("--max-commands", "1"))
        self.native.assert_not_called()
        self.syntax.assert_not_called()

    def test_non_windows_host_never_prepares_headers(self) -> None:
        preflight.os.name = "posix"
        self.assertEqual(2, self.run_preflight())
        self.native.assert_not_called()
        self.syntax.assert_not_called()

    def test_no_msvc_commands_never_prepares_headers(self) -> None:
        self.entries[0]["command"] = "g++ -c abi.cpp"
        self.write_database()
        self.assertEqual(2, self.run_preflight())
        self.native.assert_not_called()
        self.syntax.assert_not_called()

    def test_regenerated_compile_commands_are_reloaded_before_msvc(self) -> None:
        def regenerate(command, **kwargs):
            self.entries[0]["command"] += " /DREFRESHED=1"
            self.write_database()
            return mock.Mock(returncode=0)

        self.native.side_effect = regenerate
        self.assertEqual(0, self.run_preflight())
        self.assertIn("/DREFRESHED=1", self.syntax.call_args.args[0][0].command)
        self.native.assert_called_once()

    def test_regeneration_rechecks_the_command_budget(self) -> None:
        def regenerate(command, **kwargs):
            self.entries *= 2
            self.write_database()
            return mock.Mock(returncode=0)

        self.native.side_effect = regenerate
        self.assertEqual(0, self.run_preflight("--max-commands", "1"))
        self.assertIn("exceed the 1-command budget", self.output.getvalue())
        self.native.assert_called_once()
        self.syntax.assert_not_called()


if __name__ == "__main__":
    unittest.main()
