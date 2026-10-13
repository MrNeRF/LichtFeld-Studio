# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Exercise formatting publication against a real, local Git remote."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "ci_format_push.sh"
GIT = shutil.which("git")
BASH = shutil.which("bash")
if os.name == "nt" and GIT:
    git_bash = Path(GIT).resolve().parents[1] / "bin" / "bash.exe"
    if git_bash.exists():
        BASH = str(git_bash)


@unittest.skipUnless(GIT and BASH, "Git and Bash are required")
class FormatPushTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.remote = self.root / "remote.git"
        self.repo = self.root / "formatter"
        self.env = dict(os.environ, GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)
        self.git(self.root, "init", "--bare", str(self.remote))
        self.git(self.root, "init", "-b", "master", str(self.repo))
        self.git(self.repo, "config", "user.name", "CI test")
        self.git(self.repo, "config", "user.email", "ci@example.invalid")
        self.git(self.repo, "remote", "add", "origin", str(self.remote))
        self.git(self.repo, "commit", "--allow-empty", "-m", "base")
        self.base = self.git(self.repo, "rev-parse", "HEAD")
        self.git(self.repo, "push", "origin", "master")
        self.git(self.repo, "commit", "--allow-empty", "-m", "format")
        self.formatted = self.git(self.repo, "rev-parse", "HEAD")

    def git(self, cwd, *args):
        return subprocess.check_output(
            [GIT, *args], cwd=cwd, env=self.env, text=True, stderr=subprocess.PIPE
        ).strip()

    def run_push(self, advance=None, push_error=None):
        env = self.env.copy()
        command = [BASH, SCRIPT.as_posix(), "master", self.base]
        if advance:
            env.update(ADVANCE_REMOTE=self.remote.as_posix(), ADVANCE_SHA=advance,
                       PUSH_ERROR=push_error or "")
            # Advance before Git advertises refs, so a real push reports its
            # explicit non-fast-forward porcelain status. Inject other errors
            # at the same boundary to ensure an advance cannot hide them.
            wrapper = '''git() {
                if [[ "$1" == push ]]; then
                    command git --git-dir="$ADVANCE_REMOTE" update-ref refs/heads/master "$ADVANCE_SHA"
                    if [[ -n "$PUSH_ERROR" ]]; then
                        printf '%s\\n' "$PUSH_ERROR" >&2
                        return 1
                    fi
                fi
                command git "$@"
            }
            export -f git
            bash "$@"'''
            command = [BASH, "-c", wrapper, "--", SCRIPT.as_posix(), "master", self.base]
        return subprocess.run(
            command, cwd=self.repo, env=env, text=True, capture_output=True, timeout=30,
        )

    def remote_head(self):
        return self.git(self.root, "--git-dir", str(self.remote), "rev-parse", "master")

    def advance_commit(self):
        self.git(self.repo, "checkout", "-b", "next", self.base)
        self.git(self.repo, "commit", "--allow-empty", "-m", "advance")
        advance = self.git(self.repo, "rev-parse", "HEAD")
        self.git(self.repo, "push", "origin", "next")
        self.git(self.repo, "checkout", "master")
        return advance

    def test_publishes_when_branch_is_unchanged(self):
        result = self.run_push()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.remote_head(), self.formatted)

    def test_skips_when_branch_already_advanced(self):
        advance = self.advance_commit()
        self.git(self.root, "--git-dir", str(self.remote), "update-ref", "refs/heads/master", advance)
        result = self.run_push()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("skipping obsolete", result.stdout)
        self.assertEqual(self.remote_head(), advance)

    def test_skips_when_branch_advances_between_fetch_and_push(self):
        advance = self.advance_commit()
        result = self.run_push(advance=advance)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("advanced during push", result.stdout)
        self.assertEqual(self.remote_head(), advance)

    def test_reports_real_push_failure(self):
        hook = self.remote / "hooks" / "pre-receive"
        hook.write_text("#!/usr/bin/env bash\nexit 1\n", encoding="utf-8", newline="\n")
        hook.chmod(0o755)
        result = self.run_push()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no verified non-fast-forward", result.stderr)
        self.assertEqual(self.remote_head(), self.base)

    def test_authentication_failure_stays_red_even_if_branch_advanced(self):
        advance = self.advance_commit()
        result = self.run_push(advance=advance, push_error="fatal: Authentication failed")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Authentication failed", result.stderr)
        self.assertEqual(self.remote_head(), advance)

    def test_server_rejection_stays_red_even_if_branch_advanced(self):
        advance = self.advance_commit()
        result = self.run_push(
            advance=advance, push_error="!\tHEAD:refs/heads/master\t[remote rejected] (server error)"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("server error", result.stderr)
        self.assertEqual(self.remote_head(), advance)


if __name__ == "__main__":
    unittest.main()
