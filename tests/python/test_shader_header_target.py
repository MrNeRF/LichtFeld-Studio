# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Exercise the shader dependency graph using a compiler-free CMake fixture."""

import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
CMAKE = os.environ.get("CMAKE_COMMAND") or shutil.which("cmake")
NINJA = shutil.which("ninja")


@unittest.skipUnless(CMAKE and NINJA, "CMake and Ninja are required for the codegen fixture")
class ShaderHeaderTargetTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="lfs shader fixture ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.build = self.root / "b"
        self.outputs = [self.build / "generated/first.spv.h",
                        self.build / "generated/second.spv.h"]
        self.calls = self.root / "calls.txt"
        # An imported Python wrapper stands in for lfs_shader_compiler. No C++,
        # CUDA, or GLSL compiler is configured or invoked by this fixture.
        script = self.root / "fake_shader.py"
        script.write_text(
            "import argparse\nfrom pathlib import Path\n"
            "p = argparse.ArgumentParser()\n"
            "p.add_argument('--input'); p.add_argument('--output'); p.add_argument('--symbol')\n"
            "a = p.parse_args()\n"
            "source = Path(a.input)\n"
            "if source.read_text(encoding='utf-8') == 'FAIL':\n"
            "    raise SystemExit(42)\n"
            "output = Path(a.output)\noutput.parent.mkdir(parents=True, exist_ok=True)\n"
            "output.write_text(a.symbol + '\\n', encoding='utf-8')\n"
            f"with Path({str(self.calls)!r}).open('a', encoding='utf-8') as calls:\n"
            "    calls.write(a.symbol + '\\n')\n",
            encoding="utf-8",
        )
        if os.name == "nt":
            compiler = self.root / "fake shader.cmd"
            compiler.write_text(
                f'@echo off\n"{sys.executable}" -B "{script}" %*\nexit /b %errorlevel%\n',
                encoding="utf-8",
            )
        else:
            compiler = self.root / "fake shader"
            compiler.write_text(
                f"#!/bin/sh\nexec {shlex.quote(sys.executable)} -B "
                f'{shlex.quote(str(script))} "$@"\n',
                encoding="utf-8",
            )
            compiler.chmod(0o755)
        for name in ("first", "second"):
            (self.root / f"{name}.vert").write_text("fixture", encoding="utf-8")
        (self.root / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.20)\n"
            "project(ShaderHeaderFixture LANGUAGES NONE)\n"
            "add_executable(lfs_shader_compiler IMPORTED GLOBAL)\n"
            "set_target_properties(lfs_shader_compiler PROPERTIES "
            f'IMPORTED_LOCATION "{compiler.as_posix()}")\n'
            'add_custom_target(lfs_visualizer ALL COMMAND "${CMAKE_COMMAND}" -E touch '
            '"${CMAKE_BINARY_DIR}/consumer-ran")\n'
            f'include("{ROOT.as_posix()}/cmake/CompileShaders.cmake")\n'
            'compile_shader(lfs_visualizer first.vert "${CMAKE_BINARY_DIR}/generated/first.spv.h" First)\n'
            'compile_shader(lfs_visualizer second.vert "${CMAKE_BINARY_DIR}/generated/second.spv.h" Second)\n',
            encoding="utf-8",
        )
        configured = self.command(
            "-S", str(self.root), "-B", str(self.build), "-G", "Ninja",
            f"-DCMAKE_MAKE_PROGRAM={Path(NINJA).as_posix()}",
        )
        self.assertEqual(0, configured.returncode, configured.stdout + configured.stderr)

    def command(self, *arguments):
        return subprocess.run(
            [CMAKE, *arguments], capture_output=True, text=True,
            encoding="utf-8", errors="replace",
        )

    def generate(self):
        return self.command("--build", str(self.build), "--target", "lfs_shader_headers")

    def test_header_target_generates_missing_outputs_without_building_consumer(self):
        self.assertFalse(any(path.exists() for path in self.outputs))
        result = self.generate()
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        self.assertEqual(["First", "Second"], sorted(self.calls.read_text().splitlines()))
        self.assertTrue(all(path.is_file() for path in self.outputs))
        self.assertFalse((self.build / "consumer-ran").exists())
        # No-op builds reuse outputs; deleting one header regenerates only it.
        result = self.generate()
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        self.assertEqual(2, len(self.calls.read_text().splitlines()))
        self.outputs[0].unlink()
        result = self.generate()
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        self.assertEqual(["First", "First", "Second"], sorted(self.calls.read_text().splitlines()))
        self.assertFalse((self.build / "consumer-ran").exists())

    def test_failed_generation_fails_the_prerequisite_target(self):
        (self.root / "first.vert").write_text("FAIL", encoding="utf-8")
        result = self.generate()
        self.assertNotEqual(0, result.returncode)
        self.assertFalse(self.outputs[0].exists())
        self.assertFalse((self.build / "consumer-ran").exists())

    def configure_vulkan_validation(self, enabled, rejected_suffix=""):
        source = self.root / f"vulkan-{enabled}-{rejected_suffix}"
        source.mkdir()
        build = source / "build"
        shader = source / "slang/test.slang"
        shader.parent.mkdir()
        shader.write_text("fixture", encoding="utf-8")
        self.calls.write_text("", encoding="utf-8")
        (self.root / "fake_shader.py").write_text(
            "import sys\nfrom pathlib import Path\n"
            "if '-o' in sys.argv:\n"
            "    output = Path(sys.argv[sys.argv.index('-o') + 1])\n"
            "    output.write_text('compiled', encoding='utf-8')\n"
            "else:\n"
            "    output = Path(sys.argv[1])\n"
            "    assert output.read_text(encoding='utf-8') == 'compiled'\n"
            f"    with Path({str(self.calls)!r}).open('a', encoding='utf-8') as calls:\n"
            "        calls.write(output.name + '\\n')\n"
            f"    if {bool(rejected_suffix)!r} and output.name.endswith({rejected_suffix!r}):\n"
            "        raise SystemExit(42)\n",
            encoding="utf-8",
        )
        compiler = self.root / ("fake shader.cmd" if os.name == "nt" else "fake shader")
        cmake_source = (ROOT / "src/rendering/rasterizer/vulkan/CMakeLists.txt").read_text(encoding="utf-8")
        start = cmake_source.index("function(lfs_add_shader_ir)")
        end = cmake_source.index("function(_vulkan_add_glsl_shader ", start)
        (source / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.20)\n"
            "project(VulkanValidationFixture LANGUAGES NONE)\n"
            f'set(SLANGC "{compiler.as_posix()}")\n'
            f'set(_VULKAN_SPIRV_VAL "{compiler.as_posix() if enabled else ""}")\n'
            f'set(_VULKAN_SHADER_SRC "{source.as_posix()}")\n'
            'set(_SHARED_SLANG_SRC "${_VULKAN_SHADER_SRC}/shared")\n'
            f'set(_VULKAN_SLANG_ALL "{shader.as_posix()}")\n'
            'set(_VULKAN_GENERATED_OUT "${CMAKE_BINARY_DIR}/generated")\n'
            'set(_lfs_user_shader_src "${_VULKAN_SHADER_SRC}/slang")\n'
            'set(_lfs_user_shader_dir "${CMAKE_BINARY_DIR}/user")\n'
            'set(_lfs_user_shader_reflection "${CMAKE_BINARY_DIR}/reflection")\n'
            'set(_lfs_user_shader_package fixture_ir)\n'
            'file(MAKE_DIRECTORY "${_VULKAN_GENERATED_OUT}")\n'
            + cmake_source[start:end] + "\n"
            "_vulkan_add_slang_shader(projection_forward_survivors test.slang LFS_PROJECTION_SURVIVORS=1)\n"
            "_vulkan_add_slang_shader(projection_forward_3dgut test.slang LFS_3DGUT_PROJECTION=1)\n"
            "_vulkan_add_slang_shader(cull_splats test.slang)\n"
            "_vulkan_add_slang_shader(rasterize_forward_3dgut test.slang LFS_3DGUT_RASTERIZE=1)\n"
            "get_property(rows GLOBAL PROPERTY LFS_SHADER_VARIANTS)\n"
            'foreach(expected "projection_forward_survivors;Host;Survivors" '
            '"projection_forward_3dgut;Host;Gut" "cull_splats;Spirv;Survivors" '
            '"rasterize_forward_3dgut;Spirv;Gut")\n'
            '  list(GET expected 0 name)\n'
            '  list(GET expected 1 kind)\n'
            '  list(GET expected 2 chain)\n'
            '  if(NOT rows MATCHES "${name}[^;]*CustomKind::${kind}, CustomChain::${chain}")\n'
            '    message(FATAL_ERROR "Missing chain for ${name}")\n'
            '  endif()\n'
            "endforeach()\n"
            "add_custom_target(shaders DEPENDS ${VULKAN_RASTERIZER_SHADER_OUTPUTS})\n",
            encoding="utf-8",
        )
        configured = self.command("-S", str(source), "-B", str(build), "-G", "Ninja",
                                  f"-DCMAKE_MAKE_PROGRAM={Path(NINJA).as_posix()}")
        self.assertEqual(0, configured.returncode, configured.stdout + configured.stderr)
        return build

    def test_vulkan_reveal_variants_run_optional_validation_after_compilation(self):
        expected = {f"{name}{suffix}.spv"
                    for name in ("projection_forward_survivors", "projection_forward_3dgut")
                    for suffix in ("", "_banded", "_reveal", "_reveal_banded")}
        expected.update({"cull_splats.spv", "cull_splats_reveal.spv", "cull_splats_custom.spv",
                         "rasterize_forward_3dgut.spv", "rasterize_forward_3dgut_banded.spv",
                         "rasterize_forward_3dgut_reveal.spv", "rasterize_forward_3dgut_reveal_banded.spv",
                         "rasterize_forward_3dgut_custom.spv", "rasterize_forward_3dgut_custom_banded.spv"})
        for enabled in (False, True):
            with self.subTest(validation=enabled):
                build = self.configure_vulkan_validation(enabled)
                result = self.command("--build", str(build), "--target", "shaders", "--parallel", "1")
                self.assertEqual(0, result.returncode, result.stdout + result.stderr)
                self.assertEqual(expected, {path.name for path in (build / "generated").glob("*.spv")})
                calls = self.calls.read_text().splitlines()
                self.assertEqual(sorted(expected) if enabled else [], sorted(calls))

    def test_vulkan_reveal_validation_failure_fails_shader_target(self):
        for suffix in ("_reveal.spv", "_reveal_banded.spv", "_custom.spv", "_custom_banded.spv"):
            with self.subTest(suffix=suffix):
                build = self.configure_vulkan_validation(True, suffix)
                result = self.command("--build", str(build), "--target", "shaders", "--parallel", "1")
                self.assertNotEqual(0, result.returncode)
                self.assertTrue(any(name.endswith(suffix) for name in self.calls.read_text().splitlines()))


@unittest.skipUnless(CMAKE and NINJA, "CMake and Ninja are required for the codegen fixture")
class GpuProgramHeaderTargetTests(unittest.TestCase):
    def test_header_target_generates_gpu_program_sources(self):
        temporary = tempfile.TemporaryDirectory(prefix="lfs gpu program fixture ")
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name).resolve()
        build = root / "b"
        # The embed step runs from the fixture's source tree, as it does from the project's.
        (root / "tools").mkdir()
        shutil.copy(ROOT / "tools/embed_gpu_program.py", root / "tools")
        # A Python stand-in for slangc writes one artifact and its reflection for any target.
        script = root / "fake_slangc.py"
        script.write_text(
            "import json, sys\nfrom pathlib import Path\n"
            "a = sys.argv[1:]\n"
            "Path(a[a.index('-o') + 1]).write_bytes(b'\\x03\\x02\\x23\\x07')\n"
            "field = {'type': {'kind': 'pointer'}, 'binding': {'offset': 0}}\n"
            "layout = {'binding': {'size': 8}, 'type': {'fields': [field]}}\n"
            "entry = {'name': a[a.index('-entry') + 1], 'threadGroupSize': [64, 1, 1]}\n"
            "Path(a[a.index('-reflection-json') + 1]).write_text(json.dumps(\n"
            "    {'parameters': [{'type': {'elementVarLayout': layout}}], 'entryPoints': [entry]}))\n",
            encoding="utf-8",
        )
        if os.name == "nt":
            compiler = root / "fake slangc.cmd"
            compiler.write_text(
                f'@echo off\n"{sys.executable}" -B "{script}" %*\nexit /b %errorlevel%\n',
                encoding="utf-8",
            )
        else:
            compiler = root / "fake slangc"
            compiler.write_text(
                f"#!/bin/sh\nexec {shlex.quote(sys.executable)} -B "
                f'{shlex.quote(str(script))} "$@"\n',
                encoding="utf-8",
            )
            compiler.chmod(0o755)
        (root / "program.slang").write_text("fixture", encoding="utf-8")
        # No compiler is configured: the consumer only carries the generated sources.
        (root / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.20)\n"
            "project(GpuProgramFixture LANGUAGES NONE)\n"
            f'set(LFS_GPU_SLANGC "{compiler.as_posix()}" CACHE FILEPATH "")\n'
            "set(LFS_TENSOR_VULKAN ON)\n"
            "add_library(lfs_core OBJECT)\n"
            "set_target_properties(lfs_core PROPERTIES LINKER_LANGUAGE NONE)\n"
            f'include("{ROOT.as_posix()}/cmake/GpuPrograms.cmake")\n'
            "lfs_add_gpu_program(lfs_core fixture_program SOURCE program.slang COMPUTE main)\n"
            'set_source_files_properties("${CMAKE_CURRENT_BINARY_DIR}/gpu_programs/fixture_program/'
            'fixture_program.cpp" PROPERTIES HEADER_FILE_ONLY ON)\n',
            encoding="utf-8",
        )
        run = lambda *arguments: subprocess.run(
            [CMAKE, *arguments], capture_output=True, text=True, encoding="utf-8", errors="replace")
        configured = run("-S", str(root), "-B", str(build), "-G", "Ninja",
                         f"-DCMAKE_MAKE_PROGRAM={Path(NINJA).as_posix()}",
                         f"-DPython3_EXECUTABLE={Path(sys.executable).as_posix()}")
        self.assertEqual(0, configured.returncode, configured.stdout + configured.stderr)
        result = run("--build", str(build), "--target", "lfs_shader_headers")
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        directory = build / "gpu_programs/fixture_program"
        self.assertIn("fixture_program_entries()", (directory / "fixture_program.hpp").read_text(encoding="utf-8"))
        self.assertTrue((directory / "fixture_program.cpp").is_file())


if __name__ == "__main__":
    unittest.main()
