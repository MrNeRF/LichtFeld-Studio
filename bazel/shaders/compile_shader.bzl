# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""compile_shader: a GLSL shader as a C++ header (port of cmake/CompileShaders.cmake)."""

def _compile_shader_impl(ctx):
    src = ctx.file.src
    out = ctx.outputs.out
    args = ctx.actions.args()
    args.add("--input", src)
    args.add("--output", out)
    args.add("--symbol", ctx.attr.symbol)
    ctx.actions.run(
        executable = ctx.attr.compiler[DefaultInfo].files_to_run,
        arguments = [args],
        inputs = depset([src], transitive = [dep[DefaultInfo].files for dep in ctx.attr.hdrs]),
        outputs = [out],
        mnemonic = "CompileShader",
        progress_message = "Compiling shader %s" % src.short_path,
    )
    return [DefaultInfo(files = depset([out]))]

compile_shader = rule(
    implementation = _compile_shader_impl,
    doc = """Compiles a GLSL shader to a C++ header with its SPIR-V
(CMake: compile_shader() in cmake/CompileShaders.cmake):

lfs_shader_compiler --input <src> --output <out> --symbol <symbol>

The stage follows from the file extension (.vert, .frag, .geom, .comp). The
header defines `static constexpr std::uint32_t <symbol>[]` in
lfs::vis::viewport_shaders.""",
    attrs = {
        "compiler": attr.label(
            default = Label("//src/visualizer/rendering/resources/viewport:lfs_shader_compiler"),
            executable = True,
            cfg = "exec",
        ),
        "hdrs": attr.label_list(
            allow_files = True,
            doc = "Files the shader includes (CMake: the extra dependencies).",
        ),
        "out": attr.output(mandatory = True, doc = "The header, by convention <src>.spv.h."),
        "src": attr.label(allow_single_file = True, mandatory = True),
        "symbol": attr.string(mandatory = True, doc = "C++ name of the SPIR-V array."),
    },
)
