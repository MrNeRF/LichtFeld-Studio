# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Rules for LichtFeld's generated headers (config.h, git_version.h, ABI stamp)."""

LFS_VERSION = "0.5.5"

def _configure_file_impl(ctx):
    ctx.actions.expand_template(
        template = ctx.file.template,
        output = ctx.outputs.out,
        substitutions = {"@" + key + "@": value for key, value in ctx.attr.substitutions.items()},
    )
    return [DefaultInfo(files = depset([ctx.outputs.out]))]

lfs_configure_file = rule(
    implementation = _configure_file_impl,
    doc = "CMake configure_file(... @ONLY) equivalent.",
    attrs = {
        "out": attr.output(mandatory = True),
        "substitutions": attr.string_dict(doc = "VARIABLE -> value for @VARIABLE@."),
        "template": attr.label(allow_single_file = True, mandatory = True),
    },
)

def _git_version_header_impl(ctx):
    args = ctx.actions.args()
    args.add("git_version")
    args.add("--status", ctx.info_file)
    args.add("--version", ctx.attr.version)
    args.add("--out", ctx.outputs.out)
    ctx.actions.run(
        executable = ctx.executable._tool,
        arguments = [args],
        inputs = [ctx.info_file],
        outputs = [ctx.outputs.out],
        mnemonic = "LfsGitVersion",
        progress_message = "Generating %{output}",
    )
    return [DefaultInfo(files = depset([ctx.outputs.out]))]

lfs_git_version_header = rule(
    implementation = _git_version_header_impl,
    doc = "git_version.h from the workspace status (bazel/gen/workspace_status.py).",
    attrs = {
        "out": attr.output(mandatory = True),
        "version": attr.string(default = LFS_VERSION),
        "_tool": attr.label(default = "//bazel/gen:gen_headers", executable = True, cfg = "exec"),
    },
)

def _core_abi_stamp_impl(ctx):
    args = ctx.actions.args()
    args.add("core_abi_stamp")
    args.add("--template", ctx.file.template)
    args.add("--out", ctx.outputs.out)
    for item in ctx.attr.material:
        args.add("--material", item)
    args.add_all(ctx.files.srcs)
    ctx.actions.run(
        executable = ctx.executable._tool,
        arguments = [args],
        inputs = ctx.files.srcs + [ctx.file.template],
        outputs = [ctx.outputs.out],
        mnemonic = "LfsCoreAbiStamp",
        progress_message = "Generating %{output}",
    )
    return [DefaultInfo(files = depset([ctx.outputs.out]))]

lfs_core_abi_stamp = rule(
    implementation = _core_abi_stamp_impl,
    doc = "lfs_core_abi_stamp.h: hash over build configuration and all core sources.",
    attrs = {
        "material": attr.string_list(doc = "Configuration values that are part of the ABI."),
        "out": attr.output(mandatory = True),
        "srcs": attr.label_list(allow_files = True),
        "template": attr.label(allow_single_file = True, mandatory = True),
        "_tool": attr.label(default = "//bazel/gen:gen_headers", executable = True, cfg = "exec"),
    },
)
