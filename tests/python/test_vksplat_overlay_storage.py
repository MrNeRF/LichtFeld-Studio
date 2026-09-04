# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Host/shader contracts for #2014; also runnable with stdlib unittest.

These checks read source and evaluate the writer's integer address expressions.
They do not compile Slang or replace the scripted Vulkan and real GPU tests.
"""

import ast
from pathlib import Path
import random
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
VULKAN = ROOT / "src/rendering/rasterizer/vulkan"
SHADERS = VULKAN / "shader/src/slang"


def read(path):
    return path.read_text(encoding="utf-8")


def body(source, signature):
    start = source.index("{", source.index(signature))
    depth = 1
    for end in range(start + 1, len(source)):
        depth += (source[end] == "{") - (source[end] == "}")
        if depth == 0:
            return source[start + 1:end]
    raise AssertionError(f"Unclosed function: {signature}")


def integer_expression(expression, values):
    """Interpret only the integer operations used by the actual shader writer."""
    expression = re.sub(r"\b(0x[0-9a-fA-F]+|[0-9]+)[uU]\b", r"\1", expression)
    tree = ast.parse(expression, mode="eval")
    allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name,
               ast.Load, ast.BitAnd, ast.BitOr, ast.LShift, ast.RShift,
               ast.Invert, ast.Add, ast.Sub, ast.Mult)
    if any(not isinstance(node, allowed) for node in ast.walk(tree)):
        raise AssertionError(f"Unexpected shader address expression: {expression}")
    return eval(compile(tree, "<shader integer expression>", "eval"), {"__builtins__": {}}, values)


class VkSplatOverlayStorageTests(unittest.TestCase):
    def test_every_shader_uses_byte_addressed_overlay_storage(self):
        vertex = read(SHADERS / "vertex_shader.slang")
        self.assertIn("layout(binding=13) RWByteAddressBuffer out_overlay_flags;", vertex)
        self.assertNotRegex(vertex, r"out_overlay_flags\s*\[")
        for name in ("alphablend_shader.slang", "tile_batch_shader.slang",
                     "macro_raster.slang", "macro_compose.slang", "utils.slang"):
            with self.subTest(shader=name):
                source = read(SHADERS / name)
                self.assertRegex(source, r"\bByteAddressBuffer\s+gaussian_overlay_flags\b")
                self.assertNotRegex(source, r"StructuredBuffer<[^>]+>\s+gaussian_overlay_flags")
                self.assertNotRegex(source, r"gaussian_overlay_flags\s*\[")
        for name in ("macro_raster.slang", "macro_compose.slang"):
            self.assertIn("read_byte(gaussian_overlay_flags, slot)", read(SHADERS / name))
        self.assertIn("read_byte(gaussian_overlay_flags, splat_id)", read(SHADERS / "utils.slang"))

    def test_shader_writer_guards_dummy_and_uses_atomic_or(self):
        writer = body(read(SHADERS / "utils.slang"), "void write_overlay_byte(")
        self.assertRegex(writer, r"lod_enabled\s*&\s*LFS_VK_OVERLAY_WRITE_BIT")
        self.assertLess(writer.index("return;"), writer.index(".InterlockedOr("))
        self.assertNotIn(".Store(", writer)
        self.assertNotIn(".Load(", writer)
        self.assertEqual(writer.count(".InterlockedOr("), 1)

    def test_actual_shader_address_expressions_preserve_neighbour_bytes(self):
        defines = {name: int(value) for name, value in re.findall(
            r"#define\s+(LFS_VK_OVERLAY_\w+)\s+(\d+)u", read(SHADERS / "overlay_flags.inc"))}
        self.assertEqual(defines["LFS_VK_OVERLAY_WORD_BYTES"], 4)
        writer = body(read(SHADERS / "utils.slang"), "void write_overlay_byte(")
        offset_expr = re.search(r"const uint word_offset\s*=\s*(.*?);", writer).group(1)
        shift_expr = re.search(r"const uint shift\s*=\s*(.*?);", writer).group(1)
        value_expr = re.search(r"\.InterlockedOr\(word_offset,\s*(.*?)\);", writer).group(1)
        rng = random.Random(2014)
        for count in (1, 2, 3, 4, 5, 31, 32, 33, 127, 128, 129, 513):
            # Survivor emission order may interleave the bytes of one word.
            # Exercise all currently valid flags, zero, and the full byte range.
            flags = [(i * 37 + count) & 0xff for i in range(count)]
            flags[:min(count, 8)] = list(range(min(count, 8)))
            indices = list(range(count))
            for _ in range(8):
                rng.shuffle(indices)
                storage = bytearray(((count + 3) // 4) * 4)
                for index in indices:
                    values = dict(defines, idx=index, flags=flags[index])
                    offset = integer_expression(offset_expr, values)
                    shift = integer_expression(shift_expr, values)
                    value = integer_expression(value_expr, dict(values, shift=shift))
                    self.assertEqual(offset % 4, 0)
                    self.assertLessEqual(offset + 4, len(storage))
                    word = int.from_bytes(storage[offset:offset + 4], "little") | value
                    storage[offset:offset + 4] = word.to_bytes(4, "little")
                self.assertEqual(list(storage[:count]), flags)
                self.assertFalse(any(storage[count:]))

    def test_host_clears_padded_storage_for_both_projection_paths(self):
        source = read(VULKAN / "src/gs_renderer.cpp")
        prepare = body(source, "VulkanGSRenderer::prepareOverlayFlags(")
        self.assertIn("LFS_VK_OVERLAY_WORD_BYTES", prepare)
        self.assertIn("clearDeviceBuffer(buffers.overlay_flags, bytes)", prepare)
        self.assertIn("resizeDeviceBuffer(buffers.overlay_flags, kWordBytes)", prepare)
        self.assertIn("if (!write_overlay_flags)", prepare)
        for signature in ("VulkanGSRenderer::executeProjectionForward(",
                          "VulkanGSRenderer::executeProjectionForwardSurvivors("):
            projection = body(source, signature)
            self.assertIn("prepareOverlayFlags(buffers,", projection)
            self.assertRegex(projection, r"overlay_flags,\s*write_overlay_flags\s*\?\s*BufferUse::ComputeReadWrite")
            self.assertIn("|= kLodEnabledWriteOverlayFlags", projection)
            self.assertIn("&= ~kLodEnabledWriteOverlayFlags", projection)
        self.assertIn("kLodEnabledWriteOverlayFlags = LFS_VK_OVERLAY_WRITE_BIT",
                      read(VULKAN / "src/gs_renderer.h"))
        self.assertIn("Buffer<uint8_t> overlay_flags", read(VULKAN / "src/buffer.h"))

    def test_arena_estimate_and_bind_use_one_byte_per_flag(self):
        source = read(ROOT / "src/visualizer/rendering/vksplat_viewport_renderer.cpp")
        estimate = body(source, "VksplatViewportRenderer::estimateSharedScratchBytes(")
        self.assertRegex(estimate, r"add_count\(per_visible,\s*sizeof\(std::uint8_t\)\);\s*// overlay_flags")
        bind = body(source, "VksplatViewportRenderer::bindSharedScratchBuffers(")
        self.assertIn("count * sizeof(Value)", bind)
        self.assertIn("bind_count(buffers_.overlay_flags, per_visible)", bind)
        # Changing only selection must retain the projection's classification.
        rerender = body(source, "VksplatViewportRenderer::rerenderSelectionOverlay(")
        self.assertNotIn("prepareOverlayFlags", rerender)
        self.assertNotIn("clearDeviceBuffer", rerender)

    def test_all_projection_reject_paths_receive_the_write_gate(self):
        source = read(SHADERS / "vertex_shader.slang")
        calls = re.findall(r"clear_forward_projection_output\(([^()]*)\)", source)
        self.assertGreaterEqual(len(calls), 6)
        for call in calls:
            self.assertEqual(len(call.split(",")), 3, call)
        for call in calls[1:]:
            self.assertTrue(call.strip().endswith("uniforms.lod_enabled"), call)
        clear = body(source, "void clear_forward_projection_output(")
        self.assertNotIn("uniforms.", clear)  # Uniforms are a main() parameter.
        self.assertIn("write_overlay_byte(out_overlay_flags, out_idx, overlay_flags, lod_enabled)", clear)

    def test_retired_radii_output_is_absent_from_forward_shader(self):
        self.assertNotIn("out_radii", read(SHADERS / "vertex_shader.slang"))
        self.assertIn("pipeline_projection_forward = _ComputePipeline(vksplatSkipBinding(24, 8))",
                      read(VULKAN / "src/gs_renderer.h"))


if __name__ == "__main__":
    unittest.main()
