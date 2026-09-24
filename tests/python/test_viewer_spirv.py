"""Inspect the compiled word streams shipped by the viewer, without a GPU."""
import os
import pathlib
import re
import struct
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
BUILD = pathlib.Path(os.environ.get('LFS_TEST_BUILD_DIR', ROOT / 'build'))


def capabilities(words):
    assert len(words) >= 5 and words[0] == 0x07230203
    result = set()
    i = 5
    while i < len(words):
        count, opcode = words[i] >> 16, words[i] & 0xffff
        assert count and i + count <= len(words)
        if opcode == 17:
            result.add(words[i + 1])
        if opcode == 22:
            assert words[i + 2] != 64, '64-bit floating type'
        i += count
    return result


class ViewerSpirv(unittest.TestCase):
    def test_compiled_modules(self):
        modules = {}
        table = BUILD / 'lfs_tensor_spv/lfs_tensor_shader_table.cpp'
        for name, body in re.findall(r'(kWords_\w+)\s*\{([^}]+)\}', table.read_text()):
            words = [int(w, 16) for w in re.findall(r'0x[0-9a-fA-F]+', body)]
            if words and words[0] == 0x07230203:
                modules['tensor/' + name] = words
        self.assertGreaterEqual(len(modules), 36)
        for directory, pattern in [('src/rendering/rasterizer/vulkan/shader', '*.spv'),
                                   ('src/visualizer/generated/shaders', '*.spv.h')]:
            paths = list((BUILD / directory).rglob(pattern))
            self.assertTrue(paths, directory)
            for path in paths:
                if path.suffix == '.spv':
                    blob = path.read_bytes()
                    words = struct.unpack('<%dI' % (len(blob) // 4), blob)
                else:
                    words = [int(w, 16) for w in re.findall(r'0x[0-9a-fA-F]+', path.read_text())]
                modules[str(path.relative_to(BUILD))] = words
        rml = ROOT / 'src/visualizer/gui/rmlui/vulkan/rmlui_shaders_spv.hpp'
        for name, body in re.findall(r'(shader_\w+)\[\]\s*=\s*\{([^}]+)\}', rml.read_text()):
            blob = bytes(int(w, 16) for w in re.findall(r'0x[0-9a-fA-F]+', body))
            modules['rmlui/' + name] = struct.unpack('<%dI' % (len(blob) // 4), blob)
        # Allowed capabilities map to explicit device requirements or tensor per-op gates.
        allowed = {1, 9, 11, 22, 49, 50, 61, 62, 63, 64, 65, 4434, 4466, 5347, 6033}
        for name, words in modules.items():
            with self.subTest(module=name):
                caps = capabilities(words)
                self.assertNotIn(10, caps, 'Float64 is not a viewer capability')
                self.assertFalse(caps - allowed, 'unreviewed optional capabilities: %s' % (caps - allowed))
