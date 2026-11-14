# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Repository rules of the LichtFeld Studio build."""

_INDICATORS_BUILD = """\
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

# indicators (MIT), vendored as external/indicators.hpp.
cc_library(
    name = "indicators",
    hdrs = ["indicators/indicators.hpp"],
    includes = ["indicators"],
)
"""

def _lfs_vendored_impl(rctx):
    vendored = rctx.workspace_root.get_child("external")
    for directory in rctx.attr.dirs:
        rctx.symlink(vendored.get_child(directory), directory)
    rctx.symlink(vendored.get_child("indicators.hpp"), "indicators/indicators.hpp")
    rctx.file("BUILD.bazel", _INDICATORS_BUILD)
    rctx.file("REPO.bazel", "")

lfs_vendored = repository_rule(
    implementation = _lfs_vendored_impl,
    doc = """The vendored code below external/ (tree-sitter, Zep, spz, tinyusdz,
nvImageCodec, indicators) as a repository; each directory has its BUILD.bazel.

Bazel reserves the execroot path external/ for external repositories, so the
directory cannot be part of the main repository. It is not a local_repository
either: with MSVC every repository root becomes an include directory (/I, as
there is no -iquote), and the vendored single headers in external/ (tinyply.hpp,
nanoflann.hpp, stb_*.h, httplib/, ...) would shadow the BCR modules that replace
them. This repository therefore only contains the library directories, and
indicators.hpp in a directory of its own.""",
    attrs = {
        "dirs": attr.string_list(
            mandatory = True,
            doc = "Directories below external/ that are part of the repository.",
        ),
    },
    local = True,
)
