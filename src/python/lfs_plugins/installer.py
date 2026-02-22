# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Plugin dependency installer using uv."""

import logging
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional, Callable, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

from .plugin import PluginInstance
from .errors import PluginDependencyError, PluginError
try:
    import tomllib
except ImportError:
    import tomli as tomllib


class PluginInstaller:
    """Install plugin dependencies using uv."""

    def __init__(self, plugin: PluginInstance):
        self.plugin = plugin
        self._embedded_python_checked = False
        self._embedded_python_cache: Optional[Path] = None

    def _get_embedded_python(self) -> Optional[Path]:
        """Get path to the embedded Python executable."""
        if self._embedded_python_checked:
            return self._embedded_python_cache

        result: Optional[Path] = None
        try:
            import lichtfeld
            python_path = lichtfeld.packages.embedded_python_path()
            if python_path:
                python = self._normalize_path(Path(python_path))
                if python.exists():
                    result = python
                else:
                    logger.warning("embedded_python_path() returned missing path: %s", python)
            else:
                logger.warning("embedded_python_path() returned empty")
        except (ImportError, AttributeError):
            logger.warning("lichtfeld.packages not available while resolving bundled Python")

        self._embedded_python_cache = result
        self._embedded_python_checked = True
        return result

    def _require_bundled_python(self) -> Path:
        """Return the bundled Python path or raise with actionable guidance."""
        bundled_python = self._get_embedded_python()
        if bundled_python:
            return bundled_python

        raise PluginDependencyError(
            "Bundled Python not found. Plugin environments must use LichtFeld Studio's bundled "
            "Python interpreter. Refusing fallback to system or uv-managed Python."
        )

    def _is_portable_bundle(self) -> bool:
        """Detect portable runtime layout (bin/python.exe + bin/python312._pth)."""
        embedded = self._get_embedded_python()
        if not embedded:
            return False
        return (embedded.parent / "python312._pth").exists()

    @staticmethod
    def _uv_env(set_pythonhome: bool = False) -> dict:
        """Return env dict tailored for uv subprocesses."""
        env = os.environ.copy()
        env["UV_NO_MANAGED_PYTHON"] = "1"
        env["UV_PYTHON_DOWNLOADS"] = "never"
        env.pop("UV_MANAGED_PYTHON", None)
        if set_pythonhome:
            # Some runtimes (embedded/portable Python) need PYTHONHOME for stdlib discovery.
            env["PYTHONHOME"] = sys.prefix
        else:
            env.pop("PYTHONHOME", None)
        return env

    @staticmethod
    def _normalize_path(path: Path) -> Path:
        """Return an absolute path when possible."""
        try:
            return path.expanduser().resolve(strict=False)
        except OSError:
            return Path(os.path.abspath(str(path)))

    def _bundled_uv_candidates(self, portable_bundle: bool) -> list[Path]:
        """Build uv candidate paths near bundled/runtime locations."""
        candidates: list[Path] = []
        seen: set[str] = set()

        def add(path: Path) -> None:
            key = str(path)
            if key not in seen:
                seen.add(key)
                candidates.append(path)

        # Prefer C++-resolved bundled uv path.
        try:
            import lichtfeld
            uv_path = lichtfeld.packages.uv_path()
            if uv_path:
                add(self._normalize_path(Path(uv_path)))
        except (ImportError, AttributeError):
            pass

        embedded = self._get_embedded_python()
        base_dirs: list[Path] = []
        if embedded:
            base_dirs.append(embedded.parent)

        # lfs_plugins/installer.py lives in bin/lfs_plugins for portable builds.
        module_dir = self._normalize_path(Path(__file__)).parent
        base_dirs.append(module_dir.parent)

        for base in base_dirs:
            if os.name == "nt":
                add(base / "uv.exe")
            add(base / "uv")
            if os.name == "nt":
                add(base / "bin" / "uv.exe")
            add(base / "bin" / "uv")

        return candidates

    def _venv_creation_attempts(self) -> list[tuple[str, dict, str]]:
        """Build uv venv attempts (bundled Python only)."""
        bundled_python = self._require_bundled_python()
        return [(str(bundled_python), self._uv_env(set_pythonhome=True), "bundled")]

    def _venv_uses_bundled_python(self, venv_path: Path, bundled_python: Path) -> bool:
        """Best-effort check that an existing venv was created from bundled Python."""
        cfg_path = venv_path / "pyvenv.cfg"
        if not cfg_path.exists():
            return False

        try:
            cfg = cfg_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return False

        def normalize_str(path: Path) -> str:
            return os.path.normcase(str(self._normalize_path(path)))

        expected = {
            normalize_str(bundled_python),
            normalize_str(bundled_python.parent),
            normalize_str(bundled_python.parent.parent),
        }

        for line in cfg.splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip().lower()
            if key not in {"home", "executable", "base-executable"}:
                continue
            candidate = value.strip()
            if not candidate:
                continue
            candidate_path = os.path.normcase(str(self._normalize_path(Path(candidate))))
            if candidate_path in expected:
                return True
        return False

    def ensure_venv(self) -> bool:
        """Create plugin-specific venv using uv if needed."""
        venv_path = self.plugin.info.path / ".venv"
        self.plugin.venv_path = venv_path
        bundled_python = self._require_bundled_python()

        venv_python = self._get_venv_python()
        if venv_python.exists():
            if not self._venv_uses_bundled_python(venv_path, bundled_python):
                logger.warning(
                    "Existing plugin venv was not created from bundled Python, recreating: %s",
                    venv_path,
                )
                shutil.rmtree(venv_path, ignore_errors=True)
            else:
                logger.info("Plugin venv ready: %s", venv_python)
                return True

        if venv_path.exists():
            logger.warning("Broken venv (missing python), removing: %s", venv_path)
            shutil.rmtree(venv_path)

        uv = self._find_uv()
        if not uv:
            raise PluginDependencyError("uv not found - cannot create plugin venv")

        failures: list[str] = []
        portable_bundle = self._is_portable_bundle()

        for python_arg, env, label in self._venv_creation_attempts():
            cmd = [
                str(uv),
                "venv",
                str(venv_path),
                "--python",
                python_arg,
                "--no-managed-python",
                "--no-python-downloads",
            ]
            logger.info("Creating venv (%s): %s", label, " ".join(cmd))

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
            )

            if result.returncode == 0:
                logger.info("Plugin venv created (%s): %s", label, venv_path)
                return True

            stderr = (result.stderr or "").strip()
            stdout = (result.stdout or "").strip()
            detail = stderr or stdout or "no error output"
            logger.warning("uv venv failed using %s (exit %d): %s", label, result.returncode, detail)
            failures.append(f"[{label}] {detail}")

        if portable_bundle and os.name == "nt":
            embedded = self._get_embedded_python()
            if embedded:
                helper_dir = embedded.parent
                missing = [name for name in ("pythonw.exe", "venvlauncher.exe", "venvwlauncher.exe")
                           if not (helper_dir / name).exists()]
                if missing:
                    failures.append(
                        "[hint] Missing bundled Windows Python helpers in "
                        f"{helper_dir}: {', '.join(missing)}"
                    )

        raise PluginDependencyError("Failed to create venv:\n" + "\n".join(failures))

    DEPS_STAMP = ".deps_installed"

    def _deps_stamp_path(self) -> Path:
        assert self.plugin.venv_path is not None
        return self.plugin.venv_path / self.DEPS_STAMP

    def _deps_already_installed(self) -> bool:
        stamp = self._deps_stamp_path()
        if not stamp.exists():
            return False
        stamp_mtime = stamp.stat().st_mtime
        for name in ("pyproject.toml", "uv.lock"):
            src = self.plugin.info.path / name
            if src.exists() and src.stat().st_mtime > stamp_mtime:
                return False
        return True

    def install_dependencies(
        self, on_progress: Optional[Callable[[str], None]] = None
    ) -> bool:
        """Install plugin dependencies via uv sync."""
        self._require_bundled_python()

        plugin_path = self.plugin.info.path
        if not (plugin_path / "pyproject.toml").exists():
            return True

        if self._deps_already_installed():
            return True

        logger.info("Installing dependencies for %s...", self.plugin.info.name)

        uv = self._find_uv()
        if not uv:
            raise PluginDependencyError("uv not found")

        # Use the venv's python (created by ensure_venv)
        venv_python = self._get_venv_python()
        logger.info("uv sync python: %s", venv_python)

        cmd = [
            str(uv),
            "sync",
            "--project",
            str(plugin_path),
            "--python",
            str(venv_python),
            "--no-managed-python",
            "--no-python-downloads",
        ]

        logger.info("uv sync command: %s", " ".join(cmd))

        if on_progress:
            on_progress("Syncing dependencies with uv...")

        output_lines = []
        with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=self._uv_env(set_pythonhome=False),
        ) as proc:
            if proc.stdout is not None:
                for line in iter(proc.stdout.readline, ""):
                    line = line.rstrip()
                    if line and on_progress:
                        on_progress(line)
                    output_lines.append(line)
            proc.wait()

        if proc.returncode != 0:
            tail = "\n".join(output_lines[-10:])
            raise PluginDependencyError(f"uv sync failed:\n{tail}")

        self._deps_stamp_path().touch()
        logger.info("Dependencies installed for %s", self.plugin.info.name)
        return True

    def _find_uv(self) -> Optional[Path]:
        """Find uv binary."""
        portable_bundle = self._is_portable_bundle()

        for candidate in self._bundled_uv_candidates(portable_bundle):
            if candidate.exists():
                logger.info("uv resolved (bundled): %s", candidate)
                return candidate

        logger.error("uv not found in bundled runtime; refusing system uv fallback")
        return None

    def _get_venv_python(self) -> Path:
        """Get path to venv's Python interpreter."""
        assert self.plugin.venv_path is not None
        venv = self.plugin.venv_path

        # Linux/macOS
        python = venv / "bin" / "python"
        if python.exists():
            return python

        # Windows
        python = venv / "Scripts" / "python.exe"
        return python


def parse_github_url(url: str) -> Tuple[str, str, Optional[str]]:
    """Parse GitHub URL into (owner, repo, branch).

    Supports:
        - https://github.com/owner/repo
        - https://github.com/owner/repo.git
        - https://github.com/owner/repo/tree/branch
        - github:owner/repo
        - github:owner/repo@branch
        - owner/repo (assumes GitHub)
    """
    url = url.strip()

    # Handle github: shorthand
    if url.startswith("github:"):
        url = url[7:]  # Remove "github:"
        if "@" in url:
            repo_part, branch = url.rsplit("@", 1)
        else:
            repo_part, branch = url, None

        parts = repo_part.split("/")
        if len(parts) != 2:
            raise PluginError(f"Invalid GitHub shorthand: {url}")
        return parts[0], parts[1], branch

    # Handle owner/repo shorthand
    if "/" in url and not url.startswith("http"):
        parts = url.split("/")
        if len(parts) == 2 and not url.startswith("."):
            return parts[0], parts[1], None

    # Normalize URLs without scheme (github.com/owner/repo -> https://github.com/owner/repo)
    if url.startswith("github.com/") or url.startswith("www.github.com/"):
        url = "https://" + url

    # Handle full URLs
    parsed = urlparse(url)
    if parsed.netloc not in ("github.com", "www.github.com"):
        raise PluginError(f"Not a GitHub URL: {url}")

    path_parts = parsed.path.strip("/").split("/")
    if len(path_parts) < 2:
        raise PluginError(f"Invalid GitHub URL: {url}")

    owner = path_parts[0]
    repo = path_parts[1].removesuffix(".git")

    # Check for /tree/branch pattern
    branch = None
    if len(path_parts) >= 4 and path_parts[2] == "tree":
        branch = path_parts[3]

    return owner, repo, branch


def normalize_repo_name(repo: str) -> str:
    """Apply the same prefix/suffix stripping that clone_from_url uses."""
    repo_lower = repo.lower()
    if repo_lower.startswith("lichtfeld-plugin-"):
        return repo[17:]
    if repo_lower.startswith("lfs-plugin-"):
        return repo[11:]
    if repo_lower.startswith("lichtfeld-") and repo_lower.endswith("-plugin"):
        return repo[10:-7]
    return repo


def clone_from_url(
    url: str,
    plugins_dir: Path,
    on_progress: Optional[Callable[[str], None]] = None,
) -> Path:
    """Clone a plugin from GitHub URL.

    Args:
        url: GitHub URL or shorthand (github:owner/repo, owner/repo)
        plugins_dir: Directory to clone into
        on_progress: Optional progress callback

    Returns:
        Path to the cloned plugin directory
    """
    owner, repo, branch = parse_github_url(url)
    clone_url = f"https://github.com/{owner}/{repo}.git"

    plugin_name = normalize_repo_name(repo)

    plugins_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{repo}-", dir=plugins_dir))

    if on_progress:
        on_progress(f"Cloning {owner}/{repo}...")

    # Check if git is available
    git = shutil.which("git")
    if not git:
        raise PluginError("git not found in PATH")

    cmd = [git, "clone"]
    if branch:
        cmd.extend(["--branch", branch])
    cmd.extend([clone_url, str(temp_dir)])

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise PluginError(f"Failed to clone repository: {result.stderr}")

    # Verify it's a valid plugin
    manifest_path = temp_dir / "pyproject.toml"
    if not manifest_path.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise PluginError(f"Repository is not a valid plugin (missing pyproject.toml)")

    with open(manifest_path, "rb") as f:
        data = tomllib.load(f)
    lf_section = data.get("tool", {}).get("lichtfeld", {})
    if not lf_section:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise PluginError("Repository is not a valid plugin (missing [tool.lichtfeld])")
    manifest_name = str(data.get("project", {}).get("name", "")).strip()
    final_name = manifest_name or plugin_name
    target_dir = plugins_dir / final_name

    if target_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise PluginError(f"Plugin directory already exists: {target_dir}")

    if temp_dir != target_dir:
        temp_dir.replace(target_dir)

    if on_progress:
        on_progress(f"Cloned {final_name}")

    return target_dir


def update_plugin(
    plugin_dir: Path,
    on_progress: Optional[Callable[[str], None]] = None,
) -> bool:
    """Update a plugin by pulling latest changes.

    Args:
        plugin_dir: Plugin directory (must be a git repo)
        on_progress: Optional progress callback

    Returns:
        True if updated successfully
    """
    git_dir = plugin_dir / ".git"
    if not git_dir.exists():
        raise PluginError(f"Plugin is not a git repository: {plugin_dir}")

    git = shutil.which("git")
    if not git:
        raise PluginError("git not found in PATH")

    if on_progress:
        on_progress(f"Updating {plugin_dir.name}...")

    result = subprocess.run(
        [git, "pull", "--ff-only"],
        cwd=plugin_dir,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise PluginError(f"Failed to update plugin: {result.stderr}")

    if on_progress:
        on_progress(f"Updated {plugin_dir.name}")

    return True


def uninstall_plugin(plugin_dir: Path) -> bool:
    """Remove a plugin directory.

    Args:
        plugin_dir: Plugin directory to remove

    Returns:
        True if removed successfully
    """
    if not plugin_dir.exists():
        return False

    def _on_remove_error(func, path, exc):
        # Git marks object/pack files read-only; make writable and retry.
        if isinstance(exc, PermissionError):
            os.chmod(path, stat.S_IWRITE)
            func(path)
            return
        raise exc

    # Retry for transient Windows locks (indexer, AV, file scanner).
    last_error: Optional[Exception] = None
    for attempt in range(3):
        try:
            shutil.rmtree(plugin_dir, onexc=_on_remove_error)
            return True
        except PermissionError as e:
            last_error = e
            if os.name != "nt":
                raise
            # WinError 5 = access denied, 32 = sharing violation
            if getattr(e, "winerror", None) not in (5, 32):
                raise
            time.sleep(0.15 * (attempt + 1))

    raise PluginError(
        f"Failed to remove plugin directory '{plugin_dir}'. "
        "A file is likely locked by another process (or remains read-only)."
    ) from last_error
