# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The installed application layout (CMake: cmake --install, BUILD_PORTABLE).

    <bundle>/bin/                 LichtFeld-Studio.exe, all DLLs, python.exe, uv.exe
    <bundle>/Lib, <bundle>/DLLs   embedded Python 3.12 (stdlib, extension modules)
    <bundle>/lib/python/          lichtfeld.pyd, lfs_plugins, lichtfeld_sessions, stubs
    <bundle>/share/LichtFeld-Studio/  assets, locales, viewer, shaders, licenses
    <bundle>/extensions/          nvImageCodec extension DLLs

`lfs_files` places files at a destination; `lfs_python_runtime` takes the
embedded interpreter from the Python toolchain; `lfs_bundle` assembles
everything into one directory (a tree artifact) and creates `<name>.bat`, so
that `bazel run` starts the application from the bundle.
"""

load("@bazel_skylib//lib:paths.bzl", "paths")
load("@rules_python//python:py_runtime_info.bzl", "PyRuntimeInfo")

LfsFilesInfo = provider(
    doc = "Files of the bundle and their destination paths.",
    fields = {
        "files": "list of struct(src = File, dest = string); `src` may be a directory (tree artifact)",
    },
)

def _package_relative_path(file, label):
    base = label.workspace_root
    if base.startswith("external/"):
        base = "../" + base[len("external/"):]
    prefix = paths.join(base, label.package) if base else label.package
    if prefix and file.short_path.startswith(prefix + "/"):
        return file.short_path[len(prefix) + 1:]
    return file.short_path

def _lfs_files_impl(ctx):
    files = []
    strip = ctx.attr.strip_prefix.rstrip("/")
    for src in ctx.attr.srcs:
        for f in src[DefaultInfo].files.to_list():
            if ctx.attr.suffixes and not f.is_directory and not f.basename.endswith(tuple(ctx.attr.suffixes)):
                continue
            if ctx.attr.flatten:
                relative = "" if f.is_directory else f.basename
            else:
                relative = _package_relative_path(f, src.label)
                if strip:
                    if relative != strip and not relative.startswith(strip + "/"):
                        fail("%s is not below strip_prefix %s" % (f.short_path, strip))
                    relative = relative[len(strip) + 1:]
            dest = ctx.attr.renames.get(relative, relative)
            files.append(struct(src = f, dest = paths.normalize(paths.join(ctx.attr.prefix, dest))))
    return [LfsFilesInfo(files = files)]

lfs_files = rule(
    implementation = _lfs_files_impl,
    doc = """Files placed below `prefix` in the bundle, keeping their path relative to
their package (minus `strip_prefix`) unless `flatten` is set. Directories (tree
artifacts) are merged into the destination.""",
    attrs = {
        "flatten": attr.bool(doc = "Place files directly in `prefix` (by file name)."),
        "prefix": attr.string(doc = "Destination directory in the bundle."),
        "renames": attr.string_dict(doc = "Destination overrides: relative path -> new relative path."),
        "srcs": attr.label_list(allow_files = True),
        "strip_prefix": attr.string(doc = "Package-relative directory to remove from the paths."),
        "suffixes": attr.string_list(doc = "Only files with these suffixes (e.g. [\".dll\"])."),
    },
    provides = [LfsFilesInfo],
)

# CMake: install(DIRECTORY ${VCPKG_PYTHON_STDLIB}/ ... PATTERN ... EXCLUDE)
_PYTHON_EXCLUDED_DIRS = [
    "__pycache__",
    "ensurepip",
    "idle_test",
    "idlelib",
    "lib2to3",
    "test",
    "tests",
    "tkinter",
    "turtledemo",
]

def _lfs_python_runtime_impl(ctx):
    runtime = ctx.toolchains["@bazel_tools//tools/python:toolchain_type"].py3_runtime
    if not runtime or not runtime.interpreter:
        fail("lfs_python_runtime needs a hermetic Python toolchain")
    root = runtime.interpreter.dirname
    files = []
    for f in runtime.files.to_list() + [runtime.interpreter]:
        if not f.path.startswith(root + "/"):
            continue
        relative = f.path[len(root) + 1:]
        parts = relative.split("/")
        if len(parts) == 1:
            # Interpreter, its DLLs and venv helpers go next to the app (bin/).
            if relative in ctx.attr.bin_files:
                files.append(struct(src = f, dest = "bin/" + relative))
        elif parts[0] in ("Lib", "DLLs"):
            if f.basename.endswith(".pyc") or [p for p in parts[1:-1] if p in _PYTHON_EXCLUDED_DIRS]:
                continue
            files.append(struct(src = f, dest = relative))

    # Deduplicate (the interpreter is part of `files` as well).
    unique = {entry.dest: entry for entry in files}
    return [LfsFilesInfo(files = unique.values())]

lfs_python_runtime = rule(
    implementation = _lfs_python_runtime_impl,
    doc = """The embedded Python of the bundle, taken from the registered (hermetic)
Python toolchain: python.exe etc. in bin/, the standard library in Lib/ and the
extension modules in DLLs/ (getPythonHome(): <bin>/../Lib/os.py).""",
    attrs = {
        "bin_files": attr.string_list(
            default = [
                "python.exe",
                "python3.dll",
                "python312.dll",
                "pythonw.exe",
            ],
            doc = "Top-level files of the Python installation copied to bin/.",
        ),
    },
    toolchains = ["@bazel_tools//tools/python:toolchain_type"],
    provides = [LfsFilesInfo],
)

def _runtime_dlls(target):
    """DLLs that `target` (a cc_binary) needs at runtime: its runfiles' DLLs."""
    runfiles = target[DefaultInfo].default_runfiles
    files = runfiles.files.to_list() if runfiles else []
    return [f for f in files if f.basename.lower().endswith(".dll")]

def _add_entry(entries, src, dest):
    previous = entries.get(dest)
    if previous and previous.path != src.path:
        # Bazel copies the DLLs a binary loads next to it (feature
        # copy_dynamic_libraries_to_binary); those copies are the same DLLs.
        if dest.startswith("bin/") and previous.basename == src.basename and src.basename.lower().endswith(".dll"):
            return
        fail("%s and %s are both installed to %s" % (previous.short_path, src.short_path, dest))
    entries[dest] = src

def _lfs_bundle_impl(ctx):
    out = ctx.actions.declare_directory(ctx.label.name)
    entries = {}

    # The executable, the DLL binaries and everything they load at runtime go to bin/.
    for f in ctx.attr.executable[DefaultInfo].files.to_list():
        if f.basename.lower().endswith(".exe"):
            _add_entry(entries, f, "bin/" + f.basename)
    for target in [ctx.attr.executable] + ctx.attr.binaries:
        for f in target[DefaultInfo].files.to_list():
            if f.basename.lower().endswith(".dll"):
                _add_entry(entries, f, "bin/" + f.basename)
        for f in _runtime_dlls(target):
            _add_entry(entries, f, "bin/" + f.basename)

    for src in ctx.attr.srcs:
        for entry in src[LfsFilesInfo].files:
            _add_entry(entries, entry.src, entry.dest)

    manifest = ctx.actions.declare_file(ctx.label.name + ".manifest.json")
    ctx.actions.write(manifest, json.encode({
        "files": [[src.path, dest] for dest, src in sorted(entries.items()) if not src.is_directory],
        "out": out.path,
        "text": ctx.attr.text_files,
        "trees": [[src.path, dest] for dest, src in sorted(entries.items()) if src.is_directory],
    }))

    ctx.actions.run(
        executable = ctx.executable._assemble,
        arguments = [manifest.path],
        inputs = depset([manifest] + entries.values()),
        outputs = [out],
        mnemonic = "LfsBundle",
        progress_message = "Assembling %{label}",
    )

    executable_name = ctx.attr.executable[DefaultInfo].files_to_run.executable.basename
    launcher = ctx.actions.declare_file(ctx.label.name + ".bat")
    ctx.actions.write(
        launcher,
        "\r\n".join([
            "@echo off",
            "rem Starts the application from the bundle next to this script.",
            "if defined BUILD_WORKING_DIRECTORY cd /d \"%BUILD_WORKING_DIRECTORY%\"",
            "\"%~dp0{}\\bin\\{}\" %*".format(ctx.label.name, executable_name),
            "",
        ]),
        is_executable = True,
    )

    return [DefaultInfo(
        executable = launcher,
        files = depset([out, launcher]),
        runfiles = ctx.runfiles(files = [out]),
    )]

lfs_bundle = rule(
    implementation = _lfs_bundle_impl,
    doc = "The application directory (see the module docstring) plus a launcher for `bazel run`.",
    attrs = {
        "binaries": attr.label_list(
            doc = "Further binaries (DLLs, the Python module) whose runtime DLLs go to bin/.",
        ),
        "executable": attr.label(
            mandatory = True,
            doc = "The application's cc_binary; installed to bin/ with its runtime DLLs.",
        ),
        "srcs": attr.label_list(
            providers = [LfsFilesInfo],
            doc = "lfs_files / lfs_python_runtime targets.",
        ),
        "text_files": attr.string_dict(
            doc = "Generated text files: destination -> content.",
        ),
        "_assemble": attr.label(
            default = Label("//bazel/tools:assemble_bundle"),
            executable = True,
            cfg = "exec",
        ),
    },
    executable = True,
)
