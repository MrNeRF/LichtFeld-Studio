# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Self-tests for the Windows build preflight."""

from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()
