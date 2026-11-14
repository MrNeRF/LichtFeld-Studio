# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Rules for LichtFeld Studio's build-time shaders.

* `slang_spirv`: a Slang compute shader compiled to SPIR-V.
* `glsl_spirv`: a GLSL shader compiled to SPIR-V by glslangValidator.
* `slang_ir_module`: a Slang IR module (.slang-module) for the runtime effect
  compiler, plus slangc's dependency list and optionally its reflection JSON.
* `user_shader_manifest`: manifest.json of a user-shader package.
* `user_shader_abi_header`: lfs_shader_abi.hpp, the ABI hash of the user-shader API.
* `slang_recipes`: slang_recipes.json, the legacy SPIR-V recipes and their flags.
* `shader_directory`: copies shaders into one directory (a tree artifact).

The flags //bazel/shaders:debug_info and //bazel/shaders:spirv_validation
correspond to the CMake options LFS_ENABLE_VULKAN_SHADER_DEBUG_INFO and
LFS_ENABLE_VULKAN_VALIDATION; the tools are aliases in //bazel/shaders.
"""

load("@bazel_skylib//rules:common_settings.bzl", "BuildSettingInfo")

# CMake: _VULKAN_SLANG_FLAGS and _VULKAN_GLSL_FLAGS (src/rendering/rasterizer/vulkan).
SLANG_SPIRV_FLAGS = ["-stage", "compute", "-O", "-fp-mode", "fast", "-line-directive-mode", "none", "-target", "spirv"]
SLANG_DEBUG_INFO_FLAGS = ["-g3"]
GLSL_SPIRV_FLAGS = ["-V", "--target-env", "vulkan1.2"]
GLSL_DEBUG_INFO_FLAGS = ["-gVS"]

# CMake: lfs_add_shader_ir(). Every IR compile uses column-major matrices; a
# host's reflection comes from a second, codegen-free compile of its entry.
SLANG_IR_FLAGS = ["-matrix-layout-column-major"]
SLANG_REFLECTION_FLAGS = ["-target", "spirv", "-stage", "compute", "-entry", "main", "-skip-codegen"]

SlangIrModuleInfo = provider(
    doc = "A Slang IR module compiled by slang_ir_module.",
    fields = {
        "name": "Module name, the file name without .slang-module.",
        "module": "The .slang-module File.",
        "deps": "File: slangc's dependency list (depfile) of the compile.",
        "reflection": "File: reflection JSON of the entry point main, or None.",
        "defines": "list of strings: the preprocessor defines of the compile.",
    },
)

_SLANGC_ATTRS = {
    "_slangc": attr.label(
        default = Label("//bazel/shaders:slangc"),
        allow_single_file = True,
        executable = True,
        cfg = "exec",
    ),
    "_slangc_runtime": attr.label(
        default = Label("//bazel/shaders:slangc_runtime"),
        cfg = "exec",
    ),
}

_DEBUG_INFO_ATTRS = {
    "_debug_info": attr.label(default = Label("//bazel/shaders:debug_info")),
}

_SPIRV_VALIDATION_ATTRS = {
    "_run_and_stamp": attr.label(
        default = Label("//bazel/tools:run_and_stamp"),
        executable = True,
        cfg = "exec",
    ),
    "_spirv_val": attr.label(
        default = Label("//bazel/shaders:spirv_val"),
        allow_single_file = True,
        executable = True,
        cfg = "exec",
    ),
    "_spirv_validation": attr.label(default = Label("//bazel/shaders:spirv_validation")),
}

_SHADER_SOURCE_ATTRS = {
    "defines": attr.string_list(doc = "Preprocessor defines (NAME or NAME=VALUE)."),
    "hdrs": attr.label_list(
        allow_files = True,
        doc = "Files the shader may include or import. Their directories are not added to the search path.",
    ),
}

def _debug_info(ctx):
    return ctx.attr._debug_info[BuildSettingInfo].value

def _single_dirname(files, what):
    dirs = {f.dirname: None for f in files}
    if len(dirs) != 1:
        fail("%s must be in one directory, got: %s" % (what, ", ".join(sorted(dirs.keys()))))
    return dirs.keys()[0]

def _run_slangc(ctx, args, inputs, outputs, mnemonic, progress_message):
    ctx.actions.run(
        executable = ctx.executable._slangc,
        arguments = [args],
        inputs = inputs,
        tools = ctx.files._slangc_runtime,
        outputs = outputs,
        mnemonic = mnemonic,
        progress_message = progress_message,
    )

def _spirv_validation(ctx, spirv):
    """spirv-val as a validation action (CMake: LFS_ENABLE_VULKAN_VALIDATION)."""
    if not ctx.attr._spirv_validation[BuildSettingInfo].value:
        return []
    stamp = ctx.actions.declare_file(ctx.label.name + ".spirv-val")
    ctx.actions.run(
        executable = ctx.attr._run_and_stamp[DefaultInfo].files_to_run,
        arguments = [stamp.path, ctx.executable._spirv_val.path, spirv.path],
        inputs = [spirv],
        tools = [ctx.executable._spirv_val],
        outputs = [stamp],
        mnemonic = "SpirvVal",
        progress_message = "spirv-val %s" % spirv.basename,
    )
    return [OutputGroupInfo(_validation = depset([stamp]))]

def _slang_spirv_impl(ctx):
    src = ctx.file.src
    out = ctx.outputs.out
    args = ctx.actions.args()
    args.add(src)
    args.add_all(SLANG_SPIRV_FLAGS)
    if _debug_info(ctx):
        args.add_all(SLANG_DEBUG_INFO_FLAGS)
    args.add_all(ctx.attr.defines, format_each = "-D%s")
    args.add(src.dirname, format = "-I%s")
    args.add("-o", out)
    _run_slangc(
        ctx,
        args,
        inputs = depset([src], transitive = [dep[DefaultInfo].files for dep in ctx.attr.hdrs]),
        outputs = [out],
        mnemonic = "SlangSpirv",
        progress_message = "slangc %s" % out.basename,
    )
    return [DefaultInfo(files = depset([out]))] + _spirv_validation(ctx, out)

slang_spirv = rule(
    implementation = _slang_spirv_impl,
    doc = """Compiles a Slang compute shader to SPIR-V (CMake: _vulkan_add_slang_variant).

slangc <src> -stage compute -O -fp-mode fast -line-directive-mode none -target spirv
       [-g3] -D<define>... -I<directory of src> -o <out>""",
    attrs = dict(
        {
            "out": attr.output(mandatory = True, doc = "The SPIR-V file."),
            "src": attr.label(allow_single_file = [".slang"], mandatory = True),
        },
        **(_SHADER_SOURCE_ATTRS | _SLANGC_ATTRS | _DEBUG_INFO_ATTRS | _SPIRV_VALIDATION_ATTRS)
    ),
)

def _glsl_spirv_impl(ctx):
    src = ctx.file.src
    out = ctx.outputs.out
    args = ctx.actions.args()
    args.add_all(GLSL_SPIRV_FLAGS)
    if _debug_info(ctx):
        args.add_all(GLSL_DEBUG_INFO_FLAGS)
    args.add_all(ctx.attr.defines, format_each = "-D%s")
    args.add(src)
    args.add("-o", out)
    ctx.actions.run(
        executable = ctx.attr._glslang[DefaultInfo].files_to_run,
        arguments = [args],
        inputs = depset([src], transitive = [dep[DefaultInfo].files for dep in ctx.attr.hdrs]),
        outputs = [out],
        mnemonic = "GlslSpirv",
        progress_message = "glslang %s" % out.basename,
    )
    return [DefaultInfo(files = depset([out]))] + _spirv_validation(ctx, out)

glsl_spirv = rule(
    implementation = _glsl_spirv_impl,
    doc = """Compiles a GLSL shader to SPIR-V (CMake: _vulkan_add_glsl_shader).

glslangValidator -V --target-env vulkan1.2 [-gVS] -D<define>... <src> -o <out>""",
    attrs = dict(
        {
            "out": attr.output(mandatory = True, doc = "The SPIR-V file."),
            "src": attr.label(allow_single_file = True, mandatory = True),
            "_glslang": attr.label(
                default = Label("//bazel/shaders:glslang_validator"),
                allow_files = True,
                executable = True,
                cfg = "exec",
            ),
        },
        **(_SHADER_SOURCE_ATTRS | _DEBUG_INFO_ATTRS | _SPIRV_VALIDATION_ATTRS)
    ),
)

def _slang_ir_args(ctx):
    src = ctx.file.src
    args = ctx.actions.args()
    args.add(src)
    args.add_all(ctx.attr.defines, format_each = "-D%s")
    args.add_all(SLANG_IR_FLAGS)
    args.add(src.dirname, format = "-I%s")
    return args

def _slang_ir_module_impl(ctx):
    name = ctx.attr.module or ctx.label.name
    inputs = depset([ctx.file.src], transitive = [dep[DefaultInfo].files for dep in ctx.attr.hdrs])
    module = ctx.actions.declare_file("%s/%s.slang-module" % (ctx.attr.module_dir, name))

    # CMake keeps a copy of the depfile as <name>.deps; Bazel writes it there directly.
    deps = ctx.actions.declare_file("%s/%s.deps" % (ctx.attr.reflection_dir, name))
    args = _slang_ir_args(ctx)
    args.add("-o", module)
    args.add("-depfile", deps)
    _run_slangc(ctx, args, inputs, [module, deps], "SlangIr", "slangc %s" % module.basename)

    reflection = None
    if ctx.attr.reflect:
        reflection = ctx.actions.declare_file("%s/%s.json" % (ctx.attr.reflection_dir, name))
        args = _slang_ir_args(ctx)
        args.add_all(SLANG_REFLECTION_FLAGS)
        args.add("-reflection-json", reflection)
        _run_slangc(ctx, args, inputs, [reflection], "SlangReflection", "slangc %s" % reflection.basename)

    return [
        DefaultInfo(files = depset([module])),
        OutputGroupInfo(
            deps = depset([deps]),
            reflection = depset([reflection] if reflection else []),
        ),
        SlangIrModuleInfo(
            name = name,
            module = module,
            deps = deps,
            reflection = reflection,
            defines = ctx.attr.defines,
        ),
    ]

slang_ir_module = rule(
    implementation = _slang_ir_module_impl,
    doc = """Compiles a Slang module to IR (CMake: lfs_add_shader_ir).

slangc <src> -D<define>... -matrix-layout-column-major -I<directory of src>
       -o <module_dir>/<module>.slang-module -depfile <reflection_dir>/<module>.deps
and, with reflect = True, the reflection of its compute entry point main:
slangc <src> -D<define>... -matrix-layout-column-major -I<directory of src>
       -target spirv -stage compute -entry main -skip-codegen
       -reflection-json <reflection_dir>/<module>.json""",
    attrs = dict(
        {
            "module": attr.string(doc = "Module name (CMake: NAME). Defaults to the target name."),
            "module_dir": attr.string(mandatory = True, doc = "Directory of the .slang-module."),
            "reflect": attr.bool(doc = "Also write the reflection JSON (CMake: REFLECT)."),
            "reflection_dir": attr.string(mandatory = True, doc = "Directory of the .deps and .json files."),
            "src": attr.label(allow_single_file = [".slang"], mandatory = True),
        },
        **(_SHADER_SOURCE_ATTRS | _SLANGC_ATTRS)
    ),
    provides = [SlangIrModuleInfo],
)

def _user_shader_manifest_impl(ctx):
    manifest = ctx.outputs.out
    if manifest.basename != "manifest.json":
        fail("out must be <package dir>/manifest.json")
    modules = [dep[SlangIrModuleInfo] for dep in ctx.attr.modules]
    hosts = [dep[SlangIrModuleInfo] for dep in ctx.attr.hosts]
    if ctx.attr.test_host:
        if hosts:
            fail("hosts and test_host are exclusive")
        hosts = [ctx.attr.test_host[SlangIrModuleInfo]]
    if not hosts:
        fail("a user-shader package needs hosts or a test_host")
    for info in modules + hosts:
        if info.module.dirname != manifest.dirname:
            fail("module %s is not in the package directory %s" % (info.module.short_path, manifest.dirname))
    for info in hosts:
        if not info.reflection:
            fail("host %s has no reflection (slang_ir_module.reflect)" % info.name)
    reflection_dir = _single_dirname(
        [info.deps for info in hosts] + [info.reflection for info in hosts],
        "Host reflection files",
    )

    args = ctx.actions.args()
    args.add("--slangc", ctx.executable._slangc)
    args.add("--source-dir", _single_dirname(ctx.files.srcs, "srcs"))
    args.add("--output-dir", manifest.dirname)
    args.add("--reflection-dir", reflection_dir)
    if ctx.attr.test_host:
        args.add("--test-host", hosts[0].name)
    else:
        args.add("--hosts", "|".join(["%s:%s" % (info.name, ",".join(info.defines)) for info in hosts]))

    package_files = [info.module for info in modules + hosts]
    ctx.actions.run(
        executable = ctx.attr._generator[DefaultInfo].files_to_run,
        arguments = [args],
        inputs = depset(
            ctx.files.srcs + package_files + [info.deps for info in hosts] + [info.reflection for info in hosts],
        ),
        tools = [ctx.executable._slangc] + ctx.files._slangc_runtime,
        outputs = [manifest],
        mnemonic = "UserShaderManifest",
        progress_message = "Assembling %s manifest" % ctx.label.name,
    )
    package = package_files + [manifest]
    return [DefaultInfo(files = depset(package), runfiles = ctx.runfiles(package))]

user_shader_manifest = rule(
    implementation = _user_shader_manifest_impl,
    doc = """Writes manifest.json of a user-shader package beside its IR modules
(CMake: cmake/GenerateUserShaderIR.cmake, ported to //bazel/tools:generate_user_shader_ir).

Provides the package: all modules and the manifest.""",
    attrs = dict(
        {
            "hosts": attr.label_list(
                providers = [SlangIrModuleInfo],
                doc = "Custom effect hosts in manifest order (CMake: HOSTS).",
            ),
            "modules": attr.label_list(
                providers = [SlangIrModuleInfo],
                doc = "The API modules and what the hosts import.",
            ),
            "out": attr.output(mandatory = True, doc = "<package dir>/manifest.json"),
            "srcs": attr.label_list(
                allow_files = True,
                mandatory = True,
                doc = "Slang sources: config.slang and the user-shader API sources.",
            ),
            "test_host": attr.label(
                providers = [SlangIrModuleInfo],
                doc = "A test package's only host (CMake: TEST_HOST).",
            ),
            "_generator": attr.label(
                default = Label("//bazel/tools:generate_user_shader_ir"),
                executable = True,
                cfg = "exec",
            ),
        },
        **_SLANGC_ATTRS
    ),
)

def _user_shader_abi_header_impl(ctx):
    args = ctx.actions.args()
    args.add("--source-dir", _single_dirname(ctx.files.srcs, "srcs"))
    args.add("--abi-header", ctx.outputs.out)
    ctx.actions.run(
        executable = ctx.attr._generator[DefaultInfo].files_to_run,
        arguments = [args],
        inputs = ctx.files.srcs,
        outputs = [ctx.outputs.out],
        mnemonic = "UserShaderAbi",
        progress_message = "Generating %s" % ctx.outputs.out.basename,
    )
    return [DefaultInfo(files = depset([ctx.outputs.out]))]

user_shader_abi_header = rule(
    implementation = _user_shader_abi_header_impl,
    doc = """Writes the C++ header with the ABI hash of the user-shader API (kAbiHash).

CMake writes it as a byproduct of the package manifest. It only depends on the
API sources, so here it is a separate action: C++ compiles do not wait for the
IR modules.""",
    attrs = {
        "out": attr.output(mandatory = True),
        "srcs": attr.label_list(
            allow_files = True,
            mandatory = True,
            doc = "Slang sources including the user-shader API sources.",
        ),
        "_generator": attr.label(
            default = Label("//bazel/tools:generate_user_shader_ir"),
            executable = True,
            cfg = "exec",
        ),
    },
)

def _slang_recipes_impl(ctx):
    flags = SLANG_SPIRV_FLAGS + (SLANG_DEBUG_INFO_FLAGS if _debug_info(ctx) else [])
    ctx.actions.write(
        output = ctx.outputs.out,
        content = '{\n  "flags": ["%s"],\n  "recipes": [\n    %s\n  ]\n}\n' % (
            '", "'.join(flags),
            ",\n    ".join(ctx.attr.recipes),
        ),
    )
    return [DefaultInfo(files = depset([ctx.outputs.out]))]

slang_recipes = rule(
    implementation = _slang_recipes_impl,
    doc = "Writes slang_recipes.json: the slangc flags and the legacy SPIR-V recipes.",
    attrs = dict(
        {
            "out": attr.output(mandatory = True),
            "recipes": attr.string_list(doc = "JSON objects, one per recipe."),
        },
        **_DEBUG_INFO_ATTRS
    ),
)

def _shader_directory_impl(ctx):
    out = ctx.actions.declare_directory(ctx.label.name)
    prefix = ctx.label.package + "/"
    files = depset(transitive = [src[DefaultInfo].files for src in ctx.attr.srcs])
    args = ctx.actions.args()
    args.use_param_file("@%s", use_always = True)
    args.set_param_file_format("multiline")
    args.add(out.path)
    destinations = {}
    for f in files.to_list():
        if not f.short_path.startswith(prefix):
            fail("%s is not below %s" % (f.short_path, prefix))
        destination = f.short_path[len(prefix):]
        if destination in destinations:
            fail("%s and %s are both copied to %s" % (destinations[destination].short_path, f.short_path, destination))
        destinations[destination] = f
        args.add(f)
        args.add(destination)
    ctx.actions.run(
        executable = ctx.attr._copy_to_directory[DefaultInfo].files_to_run,
        arguments = [args],
        inputs = files,
        outputs = [out],
        mnemonic = "ShaderDirectory",
        progress_message = "Assembling %{label}",
    )
    return [DefaultInfo(files = depset([out]), runfiles = ctx.runfiles([out]))]

shader_directory = rule(
    implementation = _shader_directory_impl,
    doc = """Copies files into the directory <name> (a tree artifact), keeping their
paths relative to this package. srcs must be outputs of this package.""",
    attrs = {
        "srcs": attr.label_list(allow_files = True, mandatory = True),
        "_copy_to_directory": attr.label(
            default = Label("//bazel/tools:copy_to_directory"),
            executable = True,
            cfg = "exec",
        ),
    },
)
