"""Ensure installation validation detects arbitrary missing nested resources."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class InventoryTest(unittest.TestCase):
    def test_new_nested_resources_require_no_allowlist_update(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, installed = root / 'source', root / 'installed'
            source.mkdir()
            installed.mkdir()
            script = root / 'verify.cmake'
            script.write_text(f'include("{ROOT}/cmake/VerifyInstalledTree.cmake")\n'
                              f'lfs_verify_installed_tree("{source}" "{installed}" "*" "[.]md$")\n')
            for relative in ['nested/future-image.png', 'other/new-helper.py', 'new/shader.spv', 'language/catalog.json']:
                with self.subTest(resource=relative):
                    src, dst = source / relative, installed / relative
                    src.parent.mkdir(parents=True, exist_ok=True)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    src.write_text('new resource')
                    result = subprocess.run(['cmake', '-P', str(script)], capture_output=True, text=True)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(str(dst), result.stderr)
                    dst.write_bytes(src.read_bytes())
                    subprocess.run(['cmake', '-P', str(script)], check=True, capture_output=True)
            (source / 'readme.md').write_text('excluded documentation')
            subprocess.run(['cmake', '-P', str(script)], check=True, capture_output=True)
            (installed / 'nested/future-image.png').unlink()
            result = subprocess.run(['cmake', '-P', str(script)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)

    def test_empty_inventory_is_an_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script = root / 'verify.cmake'
            script.write_text(f'include("{ROOT}/cmake/VerifyInstalledTree.cmake")\n'
                              f'lfs_verify_installed_tree("{root}" "{root}" "*.spv" "")\n')
            result = subprocess.run(['cmake', '-P', str(script)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Empty source inventory', result.stderr)


if __name__ == '__main__':
    unittest.main()
