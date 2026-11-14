# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Python port of cmake/GenerateUserShaderIR.cmake; writes identical files.

Writes manifest.json beside finished IR modules for the runtime compiler,
which loads modules from IR only and never reads Slang sources. The shipped
package lists each host's IR, reflection, dependencies and imported modules.
A test package instead holds one minimal host. The ABI hash covers every
public API source and is also written as a C++ header, so a binary rejects
packages built from a different API.

  generate_user_shader_ir.py --slangc <slangc> --source-dir <dir> --output-dir <dir>
      --reflection-dir <dir> (--hosts <name:define,...|...> | --test-host <module>)
      [--abi-header <file>]
  generate_user_shader_ir.py --source-dir <dir> --abi-header <file>

The options are the variables of the CMake script. Without --output-dir only
the ABI header is written (Bazel generates it apart from the IR modules).
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

API_SOURCES = ["reveal_noise.inc", "lfs_helpers.slang", "lfs.slang"]
API_MODULES = ["lfs_helpers", "lfs"]  # dependency order
# Modules the hosts may import beside the API, in load order.
HOST_MODULES = ["lfs_reveal", "spherical_harmonics", "utils"]

# CMake's string(STRIP) removes these (isspace) characters.
_WHITESPACE = " \t\n\v\f\r"


class Error(Exception):
    pass


def _read_text(path):
    # CMake handles file contents as bytes; surrogateescape round-trips them.
    return path.read_bytes().decode("utf-8", "surrogateescape")


def _sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _host_subgroup_size(source_dir):
    """The renderer pins every splat pipeline to config.slang's SUBGROUP_SIZE."""
    # file(STRINGS ... REGEX): lines without CR characters.
    lines = _read_text(source_dir / "config.slang").replace("\r", "").split("\n")
    matches = [line for line in lines if re.fullmatch(r"#define SUBGROUP_SIZE [0-9]+", line)]
    if len(matches) != 1:
        raise Error("config.slang must define SUBGROUP_SIZE once")
    return matches[0][len("#define SUBGROUP_SIZE "):]


def _abi_hash(source_dir):
    abi_input = "".join(f"{name}:{_sha256_file(source_dir / name)}\n" for name in API_SOURCES)
    return hashlib.sha256(abi_input.encode("utf-8")).hexdigest()


def _compiler_build(slangc):
    """slangc -version, escaped for a JSON string (stdout and stderr: slangc
    prints its build tag to stderr on some platforms)."""
    try:
        result = subprocess.run([str(Path(slangc)), "-version"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except OSError as e:
        raise Error(f"Failed to run {slangc}: {e}") from e
    if result.returncode != 0:
        raise Error(f"{slangc} -version failed with exit code {result.returncode}")
    # execute_process drops NUL characters and the CR of CR LF pairs.
    raw = result.stdout
    output = bytearray()
    for i, c in enumerate(raw):
        if c == 0 or (c == 0x0D and i + 1 < len(raw) and raw[i + 1] == 0x0A):
            continue
        output.append(c)
    build = output.decode("utf-8", "surrogateescape").strip(_WHITESPACE)
    return build.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _module_entry(output_dir, name, file):
    """Hash a finished module; compilation belongs to its depfile-backed build rule."""
    digest = _sha256_file(output_dir / f"{file}.slang-module")
    return f'{{"name": "{name}", "file": "{file}.slang-module", "sha256": "{digest}"}}'


def _get(value, *path):
    """string(JSON <out> GET <json> <path>...) without ERROR_VARIABLE: missing members fail."""
    for key in path:
        try:
            value = value[key]
        except (KeyError, IndexError, TypeError) as e:
            raise Error(f"reflection JSON has no {'/'.join(str(k) for k in path)}") from e
    return value


def _host_entry(output_dir, reflection_dir, subgroup_size, host, defines):
    """The manifest entry of a host IR module the build already compiled, with
    the interface its reflection reports."""
    digest = _sha256_file(output_dir / f"{host}.slang-module")
    reflection = json.loads(_read_text(reflection_dir / f"{host}.json"))

    bindings = []
    for parameter in _get(reflection, "parameters"):
        name = _get(parameter, "name")
        kind = _get(parameter, "binding", "kind")
        index = _get(parameter, "binding", "index")
        binding = _get(parameter, "binding")
        space = binding["space"] if isinstance(binding, dict) and "space" in binding else 0
        if kind != "descriptorTableSlot":
            raise Error(f"{host}: unsupported host parameter {name} ({kind})")
        bindings.append(f'{{"name": "{name}", "set": {space}, "binding": {index}, "count": 1}}')

    # Entry-point uniforms are the push constants; varyings have no binding.
    push_constant_bytes = 0
    for parameter in _get(reflection, "entryPoints", 0, "parameters"):
        binding = parameter.get("binding") if isinstance(parameter, dict) else None
        if isinstance(binding, dict) and binding.get("kind") == "uniform":
            push_constant_bytes += int(_get(binding, "size"))

    workgroup = ", ".join(str(_get(reflection, "entryPoints", 0, "threadGroupSize", axis)) for axis in range(3))

    # The modules whose sources the host's compile read, as REFLECTION_DIR/<host>.deps
    # lists them. Banded hosts import utils built for banded export.
    sources = _read_text(reflection_dir / f"{host}.deps")
    imports = []
    for module in HOST_MODULES:
        if not re.search(r"[/\\]" + re.escape(module) + r"\.slang", sources):
            continue
        if module == "utils" and "LFS_BANDED_EXPORT_RENDER=1" in defines:
            module = "utils_banded"
        imports.append(f'"{module}.slang-module"')

    quoted_defines = ", ".join(f'"{define}"' for define in defines)
    return (
        "{\n"
        f'      "module": {{"name": "{host}", "file": "{host}.slang-module", "sha256": "{digest}"}},\n'
        f'      "imports": [{", ".join(imports)}],\n'
        '      "stage": "compute",\n'
        '      "entry": "main",\n'
        f'      "defines": [{quoted_defines}],\n'
        f'      "bindings": [{", ".join(bindings)}],\n'
        f'      "push_constant_bytes": {push_constant_bytes},\n'
        f'      "workgroup": [{workgroup}],\n'
        f'      "subgroup_size": {subgroup_size}\n'
        "    }"
    )


def _manifest(args, abi_hash):
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    reflection_dir = Path(args.reflection_dir)
    subgroup_size = _host_subgroup_size(source_dir)
    build = _compiler_build(args.slangc)

    api = ",\n    ".join(_module_entry(output_dir, name, name) for name in API_MODULES)
    if args.test_host:
        modules = ""
        hosts = _host_entry(output_dir, reflection_dir, subgroup_size, args.test_host, [])
    else:
        # What the hosts import beside the API. Banded hosts import utils built
        # for banded export under the same module name.
        modules = ",\n    ".join([
            _module_entry(output_dir, "lfs_reveal", "lfs_reveal"),
            _module_entry(output_dir, "spherical_harmonics", "spherical_harmonics"),
            _module_entry(output_dir, "utils", "utils"),
            _module_entry(output_dir, "utils", "utils_banded"),
        ])
        entries = []
        for spec in args.hosts.split("|") if args.hosts else []:
            host, _, defines = spec.partition(":")
            entries.append(_host_entry(output_dir, reflection_dir, subgroup_size, host,
                                       re.split(r"[:,]", defines) if defines else []))
        hosts = ",\n    ".join(entries)

    # Options match the legacy SPIR-V recipes in
    # src/rendering/rasterizer/vulkan/CMakeLists.txt (-O -fp-mode fast, slangc's
    # SPIR-V 1.5 default and column-major matrices); the runtime applies them.
    # Each host lists the interface the renderer provides to it.
    manifest = (
        "{\n"
        '  "version": 1,\n'
        f'  "abi_hash": "{abi_hash}",\n'
        f'  "compiler_build": "{build}",\n'
        '  "profile": "spirv_1_5",\n'
        '  "optimization": 1,\n'
        '  "fast_math": true,\n'
        '  "column_major": true,\n'
        '  "api": [\n'
        f"    {api}\n"
        "  ],\n"
        '  "modules": [\n'
        f"    {modules}\n"
        "  ],\n"
        '  "hosts": [\n'
        f"    {hosts}\n"
        "  ]\n"
        "}\n"
    )
    try:
        json.loads(manifest)
    except ValueError as e:
        raise Error(f"Invalid shader manifest: {e}") from e
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_bytes(manifest.encode("utf-8", "surrogateescape"))


def _abi_header(path, abi_hash):
    header = (
        "// Generated by cmake/GenerateUserShaderIR.cmake. Do not edit.\n"
        "#pragma once\n"
        "\n"
        "namespace lfs::vis::shaders {\n"
        f'    inline constexpr char kAbiHash[] = "{abi_hash}";\n'
        "}\n"
    ).encode("utf-8")
    # Rewrite only on change so API-neutral host edits do not recompile C++.
    if path.exists() and path.read_bytes() == header:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header)


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--slangc", help="slangc executable (CMake: SLANGC)")
    parser.add_argument("--source-dir", required=True, help="Slang source directory (CMake: SOURCE_DIR)")
    parser.add_argument("--output-dir", help="package directory with the IR modules (CMake: OUTPUT_DIR)")
    parser.add_argument("--reflection-dir", help="host reflection and .deps files (CMake: REFLECTION_DIR)")
    hosts = parser.add_mutually_exclusive_group()
    hosts.add_argument("--hosts", help="host specs name:define,...|... (CMake: HOSTS)")
    hosts.add_argument("--test-host", help="the only host of a test package (CMake: TEST_HOST)")
    parser.add_argument("--abi-header", help="C++ header to write the ABI hash to (CMake: ABI_HEADER)")
    args = parser.parse_args(argv)
    if args.output_dir is None and args.abi_header is None:
        parser.error("nothing to do: give --output-dir and/or --abi-header")
    if args.output_dir is not None and (args.slangc is None or args.reflection_dir is None):
        parser.error("--output-dir requires --slangc and --reflection-dir")

    try:
        abi_hash = _abi_hash(Path(args.source_dir))
        if args.output_dir is not None:
            _manifest(args, abi_hash)
        if args.abi_header is not None:
            _abi_header(Path(args.abi_header), abi_hash)
    except (Error, OSError, ValueError) as e:
        print(f"generate_user_shader_ir: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
