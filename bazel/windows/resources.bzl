# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Windows resources (.rc) for cc_binary targets.

rules_cc cannot compile .rc files. `windows_resources` compiles one with rc.exe
from the Windows SDK of the configured MSVC toolchain and provides the
resulting .res file as a linker input (CcInfo), so it can simply be added to the
`deps` of a cc_binary. Only add it to final binaries: every binary that links
it gets its resources.

`lfs_windows_version_resource` is the port of cmake/WindowsVersionResource.cmake
(lfs_add_windows_version_resource): a VERSIONINFO resource for an exe or DLL.
"""

load("@bazel_skylib//rules:expand_template.bzl", "expand_template")
load("@rules_cc//cc:action_names.bzl", "ACTION_NAMES")
load("@rules_cc//cc:find_cc_toolchain.bzl", "find_cc_toolchain", "use_cc_toolchain")
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load("//bazel/gen:defs.bzl", "LFS_VERSION")

def _windows_resources_impl(ctx):
    cc_toolchain = find_cc_toolchain(ctx)
    feature_configuration = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )

    # PATH (with the Windows SDK bin directory) and INCLUDE of the MSVC toolchain.
    env = cc_common.get_environment_variables(
        feature_configuration = feature_configuration,
        action_name = ACTION_NAMES.cpp_compile,
        variables = cc_common.create_compile_variables(
            feature_configuration = feature_configuration,
            cc_toolchain = cc_toolchain,
        ),
    )

    res = ctx.actions.declare_file(ctx.label.name + ".res")
    include_dirs = {ctx.file.rc.dirname: True}
    for src in ctx.files.srcs:
        include_dirs[src.dirname] = True

    args = ctx.actions.args()
    args.add("/nologo")
    args.add_all(ctx.attr.defines, format_each = "/d%s")
    args.add_all(include_dirs.keys(), before_each = "/i")
    args.add("/fo", res)
    args.add(ctx.file.rc)

    ctx.actions.run(
        executable = ctx.executable._rc,
        arguments = [args],
        inputs = [ctx.file.rc] + ctx.files.srcs,
        outputs = [res],
        env = env,
        mnemonic = "WindowsResource",
        progress_message = "Compiling Windows resources %{label}",
    )

    linker_input = cc_common.create_linker_input(
        owner = ctx.label,
        user_link_flags = depset([res.path]),
        additional_inputs = depset([res]),
    )
    return [
        DefaultInfo(files = depset([res])),
        CcInfo(linking_context = cc_common.create_linking_context(
            linker_inputs = depset([linker_input]),
        )),
    ]

windows_resources = rule(
    implementation = _windows_resources_impl,
    doc = "Compiles a Windows resource script (.rc) and links the .res into dependent binaries.",
    attrs = {
        "defines": attr.string_list(doc = "Preprocessor defines for rc.exe."),
        "rc": attr.label(
            allow_single_file = [".rc"],
            mandatory = True,
            doc = "The resource script.",
        ),
        "srcs": attr.label_list(
            allow_files = True,
            doc = "Files referenced by the script (icons, manifests, ...); their directories are include paths.",
        ),
        "_cc_toolchain": attr.label(default = Label("@rules_cc//cc:current_cc_toolchain")),
        "_rc": attr.label(
            default = Label("//bazel/windows:rc"),
            executable = True,
            cfg = "exec",
        ),
    },
    fragments = ["cpp"],
    toolchains = use_cc_toolchain(),
    provides = [CcInfo],
)

_DEBUG_FLAGS = "$<IF:$<CONFIG:Debug>,0x1L,0x0L>"

def lfs_windows_version_resource(name, description, filename, executable = False, **kwargs):
    """A VERSIONINFO resource (cmake/windows_version.rc.in) for a LichtFeld binary.

    Args:
        name: Target name (a windows_resources target).
        description: FileDescription of the binary.
        filename: File name of the binary, e.g. "lfs_core.dll".
        executable: True for an .exe (VFT_APP), False for a DLL or Python module (VFT_DLL).
        **kwargs: Forwarded to windows_resources.
    """
    major, minor, patch = LFS_VERSION.split(".")
    substitutions = {
        "@LFS_WINDOWS_FILE_DESCRIPTION@": description,
        "@LFS_WINDOWS_FILE_TYPE@": "0x1L" if executable else "0x2L",
        "@LFS_WINDOWS_INTERNAL_NAME@": filename.split(".")[0],
        "@LFS_WINDOWS_ORIGINAL_FILENAME@": filename,
        "@LFS_WINDOWS_VERSION_TWEAK@": "0",
        "@PROJECT_VERSION@": LFS_VERSION,
        "@PROJECT_VERSION_MAJOR@": major,
        "@PROJECT_VERSION_MINOR@": minor,
        "@PROJECT_VERSION_PATCH@": patch,
    }
    expand_template(
        name = name + "_rc",
        out = name + ".rc",
        substitutions = select({
            Label("//bazel/config:dbg"): substitutions | {_DEBUG_FLAGS: "0x1L"},
            "//conditions:default": substitutions | {_DEBUG_FLAGS: "0x0L"},
        }),
        template = Label("//:cmake/windows_version.rc.in"),
        visibility = ["//visibility:private"],
    )
    windows_resources(
        name = name,
        rc = name + "_rc",
        **kwargs
    )
