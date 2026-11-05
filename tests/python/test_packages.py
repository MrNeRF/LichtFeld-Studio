# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for lichtfeld.packages module (UV package management)."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import pytest


class TestPackagesModule:
    """Basic packages module tests."""

    def test_packages_module_exists(self, lf):
        """Test that packages submodule exists."""
        assert hasattr(lf, "packages")

    def test_uv_available(self, lf):
        """Test is_uv_available function exists."""
        assert hasattr(lf.packages, "is_uv_available")
        result = lf.packages.is_uv_available()
        assert isinstance(result, bool)

    def test_site_packages_dir(self, lf):
        """Test site_packages_dir function exists and returns string."""
        assert hasattr(lf.packages, "site_packages_dir")
        result = lf.packages.site_packages_dir()
        assert isinstance(result, str)
        assert len(result) > 0


class TestPackageList:
    """Tests for package listing."""

    def test_list_exists(self, lf):
        """Test list function exists."""
        assert hasattr(lf.packages, "list")

    def test_list_returns_list(self, lf):
        """Test list returns a list."""
        packages = lf.packages.list()
        assert isinstance(packages, list)

    def test_list_package_info(self, lf):
        """Test packages have expected attributes if any are installed."""
        packages = lf.packages.list()
        if len(packages) > 0:
            pkg = packages[0]
            assert hasattr(pkg, "name")
            assert hasattr(pkg, "version")
            assert hasattr(pkg, "path")


class TestIsInstalled:
    """Tests for is_installed check."""

    def test_is_installed_exists(self, lf):
        """Test is_installed function exists."""
        assert hasattr(lf.packages, "is_installed")

    def test_is_installed_nonexistent(self, lf):
        """Test is_installed returns False for nonexistent package."""
        result = lf.packages.is_installed("nonexistent-package-xyz-12345")
        assert result is False


class TestPackageInstallation:
    """Tests for package installation (marked slow, may modify system)."""

    @pytest.mark.slow
    @pytest.mark.integration
    def test_install_uninstall_cycle(self, lf):
        """Test installing and uninstalling a small package."""
        if not lf.packages.is_uv_available():
            pytest.skip("UV not available")

        package = "six"  # Small, safe package

        # Ensure clean state
        if lf.packages.is_installed(package):
            lf.packages.uninstall(package)

        # Install - returns output string on success
        result = lf.packages.install(package)
        assert isinstance(result, str)
        assert lf.packages.is_installed(package)

        # Uninstall
        result = lf.packages.uninstall(package)
        assert isinstance(result, str)
        assert not lf.packages.is_installed(package)


class TestTorchInstallation:
    """Tests for PyTorch installation helpers."""

    def test_install_torch_function_exists(self, lf):
        """Test install_torch function exists."""
        assert hasattr(lf.packages, "install_torch")

    def test_install_torch_async_exists(self, lf):
        """Test install_torch_async function exists."""
        assert hasattr(lf.packages, "install_torch_async")


class TestVenvManagement:
    """Tests for virtual environment management."""

    def test_init_function_exists(self, lf):
        """Test init function exists for venv setup."""
        assert hasattr(lf.packages, "init")

    @pytest.mark.slow
    @pytest.mark.skipif(os.name != "nt", reason="Windows Unicode subprocess environment")
    def test_fresh_unicode_environment(self, tmp_path):
        build = Path(os.environ.get("LFS_TEST_BUILD_DIR", os.environ.get("LFS_BUILD_DIR", "build"))).resolve()
        if not (build / "bin" / "uv.exe").is_file():
            pytest.skip("A build with bundled uv is required")
        # The package manager discovers uv relative to the host executable.
        fd, name = tempfile.mkstemp(prefix="package-test-", suffix=".exe", dir=build)
        os.close(fd)
        executable = Path(name)
        runtime = tmp_path / "runtime-\u00e0-\u6e2c"
        code = f"""
import os, sys
from pathlib import Path
dll = os.add_dll_directory({str(build)!r})
sys.path.insert(0, {str(build / 'src' / 'python')!r})
import lichtfeld as lf
directory = Path(lf.packages.init())
assert directory.is_relative_to(Path(os.environ['LFS_HOME']))
assert (directory / 'Scripts' / 'python.exe').is_file()
print('FRESH_UNICODE_VENV_PASS')
"""
        environment = {**os.environ, "LFS_HOME": str(runtime), "PYTHONHOME": sys.base_prefix}
        environment.pop("PYTHONPATH", None)
        try:
            shutil.copyfile(sys._base_executable, executable)
            result = subprocess.run([str(executable), "-c", code], env=environment, capture_output=True,
                                    text=True, timeout=30, creationflags=subprocess.CREATE_NO_WINDOW)
            assert result.returncode == 0, result.stdout + result.stderr
            assert "FRESH_UNICODE_VENV_PASS" in result.stdout
        finally:
            executable.unlink(missing_ok=True)

    @pytest.mark.slow
    def test_init_creates_venv(self, lf):
        """Test init creates virtual environment."""
        if not lf.packages.is_uv_available():
            pytest.skip("UV not available")

        import os
        result = lf.packages.init()
        # Returns path to venv on success
        assert isinstance(result, str)
        assert os.path.exists(result)


class TestAsyncOperations:
    """Tests for async package operations."""

    def test_is_busy(self, lf):
        """Test is_busy function exists."""
        assert hasattr(lf.packages, "is_busy")
        result = lf.packages.is_busy()
        assert isinstance(result, bool)
        assert result is False  # Nothing should be running

    def test_install_async_exists(self, lf):
        """Test install_async function exists."""
        assert hasattr(lf.packages, "install_async")
