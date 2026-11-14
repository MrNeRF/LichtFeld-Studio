# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Common build definitions for LichtFeld Studio.

The CMake build compiles LichtFeld's own code with a few global settings (see
the top-level CMakeLists.txt). They are collected here so every first-party
target uses the same flags:

* `lfs_cc_library` / `lfs_cc_binary`: C++ code (MSVC: /Zc:preprocessor, /bigobj).
* `lfs_cuda_library`: CUDA code (nvcc with -O3 -use_fast_math in opt builds).
* `lfs_dll`: a Windows DLL with CMake-like SHARED library semantics.
* `cc_headers_only`: forwards only the headers (no linking) of other targets.
"""

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_import.bzl", "cc_import")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load("@rules_cuda//cuda:defs.bzl", "CudaArchsInfo", "cuda_library")
load("//bazel/windows:resources.bzl", "lfs_windows_version_resource")

# CMake adds these only to LichtFeld's targets (after external/), so vendored
# third-party code keeps its upstream settings.
LFS_COPTS = select({
    "@platforms//os:windows": [
        "/Zc:preprocessor",
        "/bigobj",
    ],
    "//conditions:default": [],
})

LFS_DEFINES = [
    # CMake sets this on glm::glm: Vulkan-native clip-space depth in [0, 1].
    "GLM_FORCE_DEPTH_ZERO_TO_ONE",
] + select({
    # Bazel's MSVC toolchain defines _WIN32_WINNT=0x0601 (Windows 7); CMake builds
    # use the Windows SDK default (Windows 10), e.g. for FileCaseSensitiveInfo.
    "@platforms//os:windows": [
        "NOMINMAX",
        "_WIN32_WINNT=0x0A00",
        "WINVER=0x0A00",
    ],
    "//conditions:default": [],
}) + select({
    Label("//bazel/config:dbg"): ["DEBUG_BUILD"],
    "//conditions:default": ["RELEASE_BUILD"],
})

LFS_CUDA_DEFINES = LFS_DEFINES + select({
    # nvcc does not propagate /utf-8 reliably to fmt's Unicode check.
    "@platforms//os:windows": ["FMT_UNICODE=0"],
    "//conditions:default": [],
})

def _cc_headers_only_impl(ctx):
    contexts = [dep[CcInfo].compilation_context for dep in ctx.attr.deps]
    if ctx.attr.strip_defines:
        contexts = [
            cc_common.create_compilation_context(
                headers = context.headers,
                includes = context.includes,
                quote_includes = context.quote_includes,
                system_includes = context.system_includes,
                framework_includes = context.framework_includes,
            )
            for context in contexts
        ]
    return [CcInfo(compilation_context = cc_common.merge_compilation_contexts(compilation_contexts = contexts))]

cc_headers_only = rule(
    implementation = _cc_headers_only_impl,
    doc = """Provides the headers, include paths and (unless stripped) defines of `deps`,
but nothing to link. Lets header targets reference third-party headers without
forcing every consumer to link the library.""",
    attrs = {
        "deps": attr.label_list(providers = [CcInfo]),
        "strip_defines": attr.bool(
            default = False,
            doc = "Drop the defines of `deps` (e.g. @stb's public STB_IMAGE_IMPLEMENTATION).",
        ),
    },
    provides = [CcInfo],
)

def _cc_link_only_impl(ctx):
    merged = cc_common.merge_cc_infos(cc_infos = [dep[CcInfo] for dep in ctx.attr.deps])
    return [CcInfo(linking_context = merged.linking_context)]

cc_link_only = rule(
    implementation = _cc_link_only_impl,
    doc = "Provides only the link inputs of `deps` (no headers, include paths or defines).",
    attrs = {"deps": attr.label_list(providers = [CcInfo])},
    provides = [CcInfo],
)

def _cc_clean_headers_impl(ctx):
    include_dir = ctx.label.name + "/include"
    workspace_roots = {dep.label.workspace_root: True for dep in ctx.attr.deps}
    headers = []
    seen = {}
    for dep in ctx.attr.deps:
        for header in dep[CcInfo].compilation_context.headers.to_list():
            root = header.owner.workspace_root
            if root not in workspace_roots:
                continue
            prefix = (header.root.path + "/" + root + "/") if header.root.path else (root + "/")
            if not header.path.startswith(prefix):
                continue
            relative = header.path[len(prefix):]
            if relative in seen or not relative.startswith(tuple(ctx.attr.prefixes)):
                continue
            if not relative.endswith(tuple(ctx.attr.suffixes)):
                continue
            seen[relative] = True
            copy = ctx.actions.declare_file(include_dir + "/" + relative)
            ctx.actions.symlink(output = copy, target_file = header)
            headers.append(copy)
    include_path = ctx.bin_dir.path + "/" + ctx.label.package + "/" + include_dir
    return [
        DefaultInfo(files = depset(headers)),
        CcInfo(compilation_context = cc_common.create_compilation_context(
            headers = depset(headers),
            includes = depset([include_path]),
        )),
    ]

cc_clean_headers = rule(
    implementation = _cc_clean_headers_impl,
    doc = """Re-exposes the headers of `deps` (from the deps' own repository) in a clean
include directory, like an installed prefix. Used for libraries whose repository
root is an include path (FFmpeg: its VERSION file shadows <version> on Windows).""",
    attrs = {
        "deps": attr.label_list(providers = [CcInfo]),
        "prefixes": attr.string_list(doc = "Only headers below these repository-relative directories."),
        "suffixes": attr.string_list(default = [".h", ".hpp"], doc = "Only files with these suffixes."),
    },
    provides = [CcInfo],
)

def _lfs_cuda_min_sm_impl(ctx):
    sms = []
    for spec in ctx.attr._archs[CudaArchsInfo].arch_specs:
        # "89" or "90a": the virtual architecture the code was compiled for.
        digits = "".join([c for c in spec.stage1_arch.elems() if c.isdigit()])
        if digits:
            sms.append(int(digits))

    # CMake falls back to 70 when it cannot derive the floor.
    min_sm = min(sms) if sms else 70
    return [CcInfo(compilation_context = cc_common.create_compilation_context(
        defines = depset(["LFS_MIN_SM=%d" % min_sm]),
    ))]

lfs_cuda_min_sm = rule(
    implementation = _lfs_cuda_min_sm_impl,
    doc = """Defines LFS_MIN_SM, the lowest CUDA architecture compiled into the binary
(from --@rules_cuda//cuda:archs). CMake: LFS_MIN_SM=${LFS_RUNTIME_MIN_SM}; the app
refuses to start on GPUs below it.""",
    attrs = {
        "_archs": attr.label(default = Label("@rules_cuda//cuda:archs")),
    },
    provides = [CcInfo],
)

def lfs_cc_library(name, copts = [], local_defines = [], **kwargs):
    cc_library(
        name = name,
        copts = LFS_COPTS + copts,
        local_defines = LFS_DEFINES + local_defines,
        # No per-library DLL: LichtFeld's DLLs are explicit (lfs_dll). Also avoids
        # analysing a dynamic link of libraries that FFmpeg's with_defaults
        # transition builds in a second configuration (e.g. bzip2, zlib).
        linkstatic = True,
        **kwargs
    )

def lfs_cc_binary(name, copts = [], local_defines = [], **kwargs):
    cc_binary(
        name = name,
        copts = LFS_COPTS + copts,
        local_defines = LFS_DEFINES + local_defines,
        **kwargs
    )

def lfs_cuda_library(
        name,
        fast_math = True,
        extended_lambda = True,
        copts = [],
        host_copts = [],
        local_defines = [],
        **kwargs):
    """CUDA library with the nvcc flags of the CMake build.

    Args:
        name: Target name.
        fast_math: Add -use_fast_math in optimized builds (CMake Release).
        extended_lambda: Kept for documentation; --extended-lambda and
            --expt-relaxed-constexpr are set globally in .bazelrc.
        copts: Extra nvcc options.
        host_copts: Extra host compiler (MSVC) options.
        local_defines: Extra defines.
        **kwargs: Forwarded to cuda_library.
    """
    _ = extended_lambda  # @unused
    cuda_library(
        name = name,
        copts = select({
            Label("//bazel/config:dbg"): ["-O0", "-lineinfo"],
            "//conditions:default": ["-O3"] + (["-use_fast_math"] if fast_math else []),
        }) + ["--diag-suppress=27"] + copts,
        host_copts = select({
            "@platforms//os:windows": [
                "/D_CRT_SECURE_NO_WARNINGS",
                # rules_cuda passes -D_WIN32_WINNT=0x0601 to the host compiler.
                "/D_WIN32_WINNT=0x0A00",
            ],
            "//conditions:default": [],
        }) + host_copts,
        local_defines = LFS_CUDA_DEFINES + local_defines,
        **kwargs
    )

def lfs_dll(
        name,
        dll_name,
        export_define,
        srcs = [],
        deps = [],
        interface_deps = [],
        copts = [],
        local_defines = [],
        linkopts = [],
        description = None,
        visibility = None,
        **kwargs):
    """A LichtFeld shared library (Windows DLL), modelled after CMake's SHARED libraries.

    Creates:
      `<name>_objs`: the library's own objects (always linked, compiled with
          `export_define` so LFS_*_API symbols are dllexport'ed),
      `<dll_name>`: the DLL itself,
      `<name>`: what consumers depend on: `interface_deps` (CMake PUBLIC link
          libraries, usually header targets) plus the import library and the DLL.

    `deps` are linked privately into the DLL (CMake PRIVATE).

    Args:
        name: Name of the consumer-facing target.
        dll_name: File name of the DLL, e.g. "lfs_core.dll".
        export_define: Define that switches the module's API macro to dllexport.
        srcs: Sources of the DLL.
        deps: Private dependencies (linked into the DLL).
        interface_deps: Public dependencies (also propagated to consumers).
        copts: Extra C++ compiler options.
        local_defines: Extra defines for the DLL's sources.
        linkopts: Extra linker options for the DLL.
        description: FileDescription of the DLL's version resource (CMake:
            lfs_add_windows_version_resource); no version resource if None.
        visibility: Visibility of the consumer-facing target and the DLL.
        **kwargs: Forwarded to the `<name>_objs` cc_library.
    """
    cc_library(
        name = name + "_objs",
        srcs = srcs,
        copts = LFS_COPTS + copts,
        local_defines = LFS_DEFINES + [export_define] + local_defines,
        deps = deps + interface_deps,
        alwayslink = True,
        linkstatic = True,
        visibility = ["//visibility:private"],
        **kwargs
    )

    resources = []
    if description:
        lfs_windows_version_resource(
            name = name + "_version",
            description = description,
            filename = dll_name,
            target_compatible_with = ["@platforms//os:windows"],
            visibility = ["//visibility:private"],
        )
        resources = [":" + name + "_version"]

    cc_binary(
        name = dll_name,
        linkshared = True,
        linkopts = linkopts,
        deps = [":" + name + "_objs"] + resources,
        visibility = visibility,
    )

    native.filegroup(
        name = name + "_import_lib",
        srcs = [":" + dll_name],
        output_group = "interface_library",
        visibility = ["//visibility:private"],
    )

    cc_import(
        name = name + "_import",
        interface_library = ":" + name + "_import_lib",
        shared_library = ":" + dll_name,
        visibility = ["//visibility:private"],
    )

    cc_library(
        name = name,
        deps = interface_deps + [":" + name + "_import"],
        visibility = visibility,
    )
