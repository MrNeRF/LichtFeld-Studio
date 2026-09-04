---
sidebar_position: 5
---

# Windows build preflight

The Windows build preflight catches selected MSVC and DLL-boundary regressions
before a complete Windows build. It is a fast diagnostic layer, not a
replacement for building and testing the application.

## Source checks

The dependency-free check and its self-tests run on Windows or Linux with
Python 3.10 or newer:

```sh
python tools/windows_build_preflight.py --source-only
```

It checks a source contract that otherwise tends to fail only when the Windows
test executable is linked: a non-inline visualizer free function explicitly
imported or fully qualified by a test must declare `LFS_VIS_API` so it is
present in the DLL import library.

The rule is intentionally conservative. It ignores broad `using namespace`
imports, class methods, inline functions, and `constexpr` helpers rather than
guessing at C++ semantics.

In particular, `tests/test_project_lifecycle.cpp` and
`tests/test_p5_session_chapters.cpp` use unqualified calls after nested
`using namespace` directives. Removing `LFS_VIS_API` from
`capturePanelCameraProjectState` is not detected by this source-only rule.
The Windows link step is still needed to catch that missing export.

Run the checker self-tests after changing its matching rules:

```sh
python -m unittest tests/python/test_windows_build_preflight.py
```

## Configured MSVC check

For compiler-level coverage, first configure the normal Ninja/MSVC build so
that `compile_commands.json` exists. From the same Visual Studio developer
environment used for the build, run:

```powershell
python tools/windows_build_preflight.py `
  --compile-commands build/compile_commands.json `
  --changed-since upstream/master
```

The tool follows project-local quoted and angle-bracket includes from each changed header to the
affected C and C++ translation units. It then replays their existing commands
with MSVC's `/Zs` option. The Microsoft compiler therefore parses the same
sources with the configured defines and include paths, but does not generate
object files or link targets. Compiler-specific failures such as invalid use of
an incomplete type belong to this authoritative configured check, not to a
lexical Python rule.

Keep `compile_commands.json` in its original configured build directory. The
include index also considers headers under that directory's `include/`, so the
generated `config.h` does not alias Vulkan's private header of the same name.
Only project and generated headers are resolved; system headers are excluded.
Deleted headers and both sides of a rename participate in change discovery.

Configuration must already have generated headers needed by the selected
commands, including the ABI stamp for each configuration. The preflight reads
the compile database and existing headers; it does not configure CMake or run
build targets. Syntax-only replay bypasses `sccache` and removes `/showIncludes`,
`/Yc`, `/Yu`, and `/Fp` (including PCH filenames). It does not require a PCH build.

Build-system-only changes do not replay every translation unit automatically:
that would duplicate most of a full build and defeat the purpose of a fast
preflight. Add `--all-commands` when a change to CMake options or compile
definitions genuinely requires a complete configured syntax pass.

Use `--max-commands N` to impose a finite work budget. If a broad header change
exceeds that budget, the configured replay is skipped explicitly and the
normal full build remains authoritative. GitHub Actions also receives a
`::warning::` annotation so a green step does not hide the skipped replay.
This prevents the preflight from
adding an almost complete second compilation pass. Runs are unlimited by
default. `--all-commands` covers C/C++ sources under `src/`, `include/`, and
`tests/`; vendored, generated, `_deps`, and vcpkg sources are excluded.

In Windows PR CI, the default merge checkout is compared with `HEAD^1`, the
current target-branch parent. This excludes changes that arrived independently
on master. Push builds use the event's previous commit.

Before invoking MSVC, the configured pass lists every selected source using a
repository-relative path. The headline count refers to compile-database
commands; if the same source has more than one entry, the source list groups it
once and annotates the number of commands. This makes unexpectedly broad
selection visible without printing compiler command lines or machine-specific
paths.

Use the portable build directory to check portable compile definitions:

```powershell
python tools/windows_build_preflight.py `
  --compile-commands build-portable/compile_commands.json `
  --changed-since upstream/master
```

Use `--dry-run` to inspect the selected translation units without invoking the
compiler, or `--all-commands` when a complete configured
C/C++ syntax pass is needed:

```powershell
python tools/windows_build_preflight.py `
  --compile-commands build/compile_commands.json `
  --changed-since upstream/master `
  --dry-run

python tools/windows_build_preflight.py `
  --compile-commands build/compile_commands.json `
  --all-commands
```

## Coverage boundaries

The preflight does not compile CUDA translation units, perform general linker
resolution, generate optimized machine code, validate packaging, or exercise
runtime behavior. The DLL source rule catches the common missing-export case,
but it cannot prove that every library and executable will link. A successful
full Windows build and the relevant tests remain authoritative.

Include discovery is a lexical approximation: macro-generated includes and
ambiguous suffix matches are not resolved, and conditional branches are not
evaluated. Use the full build for definitive dependency coverage.

When no changed source or project header is reachable from the configured C/C++
graph, the tool reports that no affected translation unit was found. Use the
build directory for the configuration being validated, especially when
checking portable-only or optional code paths.
