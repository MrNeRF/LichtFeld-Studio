# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Helpers for the Bazel build of the vendored nvImageCodec (see ../BUILD.bazel)."""

def _file_from_impl(ctx):
    matches = [f for f in ctx.files.srcs if f.basename == ctx.attr.basename]
    if len(matches) != 1:
        fail("%s: expected exactly one file named '%s' in %s, found %d." % (
            ctx.label,
            ctx.attr.basename,
            [str(src.label) for src in ctx.attr.srcs],
            len(matches),
        ))
    return [DefaultInfo(files = depset(matches))]

file_from = rule(
    implementation = _file_from_impl,
    doc = """Selects one file by its basename from the files of `srcs`.

Used to take a single library out of a hermetic CUDA component, e.g. cudart_static.lib
from @cuda//:cudart_all_files (rules_cuda has no working static cudart target on Windows).""",
    attrs = {
        "basename": attr.string(mandatory = True, doc = "File name to select, e.g. \"cudart_static.lib\"."),
        "srcs": attr.label_list(allow_files = True, mandatory = True),
    },
)
