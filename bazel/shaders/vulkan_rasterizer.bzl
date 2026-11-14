# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The Vulkan rasterizer's shader recipes (CMake: the "Shader compilation (Slang +
GLSL -> SPIR-V)" part of src/rendering/rasterizer/vulkan/CMakeLists.txt).

* vulkan_slang_shader() / vulkan_glsl_shader(): one _vulkan_add_slang_shader() /
  _vulkan_add_glsl_shader() call each, i.e. a recipe and the variants derived
  from it (banded, reveal, reveal_banded, custom host, custom SPIR-V).
* vulkan_rasterizer_shaders(): the SPIR-V of a list of such recipes plus
  shader_variants.inc and slang_recipes.json.
* custom_effect_hosts(): the custom effect hosts of such a list.
* lfs_user_shader_package(): a user-shader package (Slang IR + manifest.json).
"""

load("@bazel_skylib//rules:write_file.bzl", "write_file")
load(
    ":shaders.bzl",
    "glsl_spirv",
    "slang_ir_module",
    "slang_recipes",
    "slang_spirv",
    "user_shader_abi_header",
    "user_shader_manifest",
)

_BANDED = "LFS_BANDED_EXPORT_RENDER=1"
_REVEAL = "LFS_REVEAL=1"
_CUSTOM_EFFECT = "LFS_CUSTOM_EFFECT=1"

def _has_prefix(name, prefixes):
    for prefix in prefixes:
        if name.startswith(prefix):
            return True
    return False

def _variant_entry(program, banded, reveal, custom, custom_banded, custom_kind, chain):
    """A LFS_SHADER_VARIANTS line of shader_variants.inc."""
    return '{"%s", "%s", "%s", "%s", "%s", CustomKind::%s, CustomChain::%s},' % (
        program,
        banded,
        reveal,
        custom,
        custom_banded,
        custom_kind,
        chain,
    )

def _recipe_json(name, source, defines):
    """A slang_recipes.json recipe (CMake: _vulkan_record_slang_recipe)."""
    return '{"name": "%s", "source": "%s", "defines": [%s]}' % (
        name,
        source,
        ", ".join(['"%s"' % define for define in defines]),
    )

def vulkan_slang_shader(out_name, src_file, *defines):
    """A Slang recipe and its variants (CMake: _vulkan_add_slang_shader).

    Args:
        out_name: Name of the SPIR-V program, generated/<out_name>.spv.
        src_file: Slang source file in the shader source directory.
        *defines: Its preprocessor defines (NAME=VALUE).

    Returns:
        A struct for vulkan_rasterizer_shaders() and custom_effect_hosts() with,
        in the order the CMake function records them:
        spirv: (name, defines) of each SPIR-V variant,
        recipes: (name, defines) of the recipes listed in slang_recipes.json,
        hosts: (name, defines) of the custom effect hosts built as Slang IR,
        variants: its shader_variants.inc lines.
    """
    defines = list(defines)
    spirv = [(out_name, defines)]
    recipes = [(out_name, defines)]
    hosts = []
    variants = []
    banded = ""
    reveal = ""
    custom = ""
    custom_banded = ""
    custom_kind = "None"
    chain = "None"

    # Keep band-origin arithmetic out of interactive shaders. These variants
    # are loaded lazily when the viewer first exports a banded image.
    if (_has_prefix(out_name, ["projection_forward", "rasterize_forward", "macro_raster", "macro_compose"]) or
        out_name == "generate_keys_wave"):
        banded = out_name + "_banded"
        spirv.append((banded, defines + [_BANDED]))
        recipes.append((banded, defines + [_BANDED]))

    # Reveal shaders are selected only while a sequencer reveal is active.
    if (_has_prefix(out_name, ["projection_forward", "rasterize_forward_3dgut", "rasterize_outline_gut"]) or
        out_name == "cull_splats"):
        reveal = out_name + "_reveal"
        spirv.append((reveal, defines + [_REVEAL]))
        recipes.append((reveal, defines + [_REVEAL]))
        if _has_prefix(out_name, ["projection_forward", "rasterize_forward"]):
            reveal_banded = out_name + "_reveal_banded"
            spirv.append((reveal_banded, defines + [_REVEAL, _BANDED]))
            recipes.append((reveal_banded, defines + [_REVEAL, _BANDED]))
            variants.append(_variant_entry(reveal, reveal_banded, "", "", "", "None", "None"))

        # Projection calls the effect and ships as a host; the viewport
        # bands only survivor and 3DGUT chains. Cull and the 3DGUT rasters
        # only read what it wrote, so their custom variants are SPIR-V
        # built for every effect and stay out of slang_recipes.json.
        custom = out_name + "_custom"
        if out_name.startswith("projection_forward"):
            custom_kind = "Host"
            hosts.append((custom, defines + [_REVEAL, _CUSTOM_EFFECT]))
            if out_name.endswith("_survivors") or out_name.endswith("_3dgut"):
                hosts.append((out_name + "_custom_banded", defines + [_BANDED, _REVEAL, _CUSTOM_EFFECT]))
        else:
            custom_kind = "Spirv"
            spirv.append((custom, defines + [_REVEAL, _CUSTOM_EFFECT]))
            if out_name.startswith("rasterize_forward"):
                spirv.append((out_name + "_custom_banded", defines + [_REVEAL, _CUSTOM_EFFECT, _BANDED]))

    if custom:
        if "LFS_PROJECTION_SURVIVORS=1" in defines or out_name == "cull_splats":
            chain = "Survivors"
        elif "LFS_3DGUT_PROJECTION=1" in defines or "LFS_3DGUT_RASTERIZE=1" in defines:
            chain = "Gut"

        # A banded draw requests this label even when its host is not built.
        if banded:
            custom_banded = out_name + "_custom_banded"
    if banded or reveal or custom:
        variants.append(_variant_entry(out_name, banded, reveal, custom, custom_banded, custom_kind, chain))

    return struct(
        kind = "slang",
        name = out_name,
        src = src_file,
        spirv = spirv,
        recipes = recipes,
        hosts = hosts,
        variants = variants,
    )

def vulkan_glsl_shader(out_name, src_file, *defines):
    """A GLSL compute shader (CMake: _vulkan_add_glsl_shader).

    Args:
        out_name: Name of the SPIR-V program, radix_sort/<out_name>.spv.
        src_file: GLSL source file in the radix sort source directory.
        *defines: Its preprocessor defines (NAME=VALUE).

    Returns:
        A struct for vulkan_rasterizer_shaders().
    """
    return struct(
        kind = "glsl",
        name = out_name,
        src = src_file,
        spirv = [(out_name, list(defines))],
        recipes = [],
        hosts = [],
        variants = [],
    )

def custom_effect_hosts(shaders):
    """The custom effect hosts of recipes, in CMake order (LFS_USER_SHADER_HOSTS).

    Args:
        shaders: vulkan_slang_shader() / vulkan_glsl_shader() structs.

    Returns:
        Structs (name, src, defines) for lfs_user_shader_package(hosts = ...).
    """
    return [
        struct(name = name, src = shader.src, defines = defines)
        for shader in shaders
        for name, defines in shader.hosts
    ]

def vulkan_rasterizer_shaders(
        name,
        shaders,
        slang_srcs = ":slang_srcs",
        slang_dir = "src/slang",
        glsl_srcs = ":radix_sort_srcs",
        glsl_dir = "src/radix_sort",
        visibility = None):
    """The SPIR-V of the Vulkan rasterizer, shader_variants.inc and slang_recipes.json.

    Creates
      spv_<program>: compiles <slang_dir>/<src> to generated/<program>.spv, for
          every recipe and variant of the vulkan_slang_shader() structs,
      glsl_<program>: compiles <glsl_dir>/<src> to radix_sort/<program>.spv,
      <name>: all of these SPIR-V files (CMake: lfs_vulkan_rasterizer_shaders),
      shader_variants_inc: shader_variants.inc (the LFS_SHADER_VARIANTS entries),
      slang_recipes_json: slang_recipes.json (CMake: LFS_SLANG_RECIPES).

    The custom effect hosts are built by
    lfs_user_shader_package(hosts = custom_effect_hosts(shaders)).

    Args:
        name: Name of the filegroup with all SPIR-V files.
        shaders: vulkan_slang_shader() / vulkan_glsl_shader() structs, in CMake order.
        slang_srcs: All Slang sources (anything a shader may include or import).
        slang_dir: Package-relative directory of the Slang sources.
        glsl_srcs: All GLSL sources.
        glsl_dir: Package-relative directory of the GLSL sources.
        visibility: Visibility of the filegroup and the generated files.
    """
    spirv = []
    recipes = []
    variants = []
    for shader in shaders:
        if shader.kind == "glsl":
            for program, defines in shader.spirv:
                glsl_spirv(
                    name = "glsl_" + program,
                    src = "%s/%s" % (glsl_dir, shader.src),
                    hdrs = [glsl_srcs],
                    defines = defines,
                    out = "radix_sort/%s.spv" % program,
                )
                spirv.append(":glsl_" + program)
            continue
        for program, defines in shader.spirv:
            slang_spirv(
                name = "spv_" + program,
                src = "%s/%s" % (slang_dir, shader.src),
                hdrs = [slang_srcs],
                defines = defines,
                out = "generated/%s.spv" % program,
            )
            spirv.append(":spv_" + program)
        recipes += [_recipe_json(program, shader.src, defines) for program, defines in shader.recipes]
        variants += shader.variants

    native.filegroup(
        name = name,
        srcs = spirv,
        visibility = visibility,
    )

    write_file(
        name = "shader_variants_inc",
        out = "shader_variants.inc",
        content = variants + [""],
        newline = "unix",
        visibility = visibility,
    )

    slang_recipes(
        name = "slang_recipes_json",
        out = "slang_recipes.json",
        recipes = recipes,
        visibility = visibility,
    )

def _user_shader_ir(package, layout, module, src, defines, reflect):
    name = "%s_%s" % (package, module)
    slang_ir_module(
        name = name,
        module = module,
        src = "%s/%s" % (layout.slang_dir, src),
        hdrs = [layout.slang_srcs],
        defines = defines,
        reflect = reflect,
        module_dir = layout.dir,
        reflection_dir = layout.reflection_dir,
    )
    return ":" + name

def lfs_user_shader_package(
        name,
        dir,
        reflection_dir,
        hosts = [],
        test_host = None,
        abi_header = None,
        slang_srcs = ":slang_srcs",
        slang_dir = "src/slang",
        visibility = None):
    """A user-effect package for the runtime compiler in src/visualizer/shaders.

    CMake: one pass of foreach(_lfs_user_shader_package lfs_user_shader_ir
    lfs_user_shader_test_package). The API (lfs_helpers, lfs), the hosts and
    their imports (lfs_reveal, spherical_harmonics, utils, utils_banded) or a
    test host are compiled to Slang IR, <dir>/<module>.slang-module, beside
    <dir>/manifest.json. Reflection and dependency lists stay in reflection_dir.

    Creates <name>_<module> per module (slang_ir_module), <name> (the package:
    its modules and manifest.json) and, with abi_header, <name>_abi_header.

    Args:
        name: Name of the package target.
        dir: Package directory, relative to this package's output directory.
        reflection_dir: Directory of the reflection JSON and .deps files.
        hosts: Custom effect hosts, see custom_effect_hosts().
        test_host: Module name of a test package's only host (no hosts, no imports).
        abi_header: Also generate this C++ header with the API's ABI hash.
        slang_srcs: All Slang sources.
        slang_dir: Package-relative directory of the Slang sources.
        visibility: Visibility of the package and the ABI header.
    """
    layout = struct(dir = dir, reflection_dir = reflection_dir, slang_dir = slang_dir, slang_srcs = slang_srcs)
    host_modules = [
        _user_shader_ir(name, layout, host.name, host.src, host.defines, reflect = True)
        for host in hosts
    ]
    modules = []
    test_host_module = None
    if test_host:
        test_host_module = _user_shader_ir(name, layout, test_host, test_host + ".slang", [], reflect = True)
    else:
        modules += [
            _user_shader_ir(name, layout, module, module + ".slang", [], reflect = False)
            for module in ["lfs_reveal", "spherical_harmonics"]
        ]
        modules.append(_user_shader_ir(name, layout, "utils", "utils.slang", [_CUSTOM_EFFECT], reflect = False))
        modules.append(_user_shader_ir(name, layout, "utils_banded", "utils.slang", [_CUSTOM_EFFECT, _BANDED], reflect = False))
    modules += [
        _user_shader_ir(name, layout, module, module + ".slang", [], reflect = False)
        for module in ["lfs_helpers", "lfs"]
    ]

    user_shader_manifest(
        name = name,
        srcs = [slang_srcs],
        modules = modules,
        hosts = host_modules,
        test_host = test_host_module,
        out = dir + "/manifest.json",
        visibility = visibility,
    )

    if abi_header:
        user_shader_abi_header(
            name = name + "_abi_header",
            srcs = [slang_srcs],
            out = abi_header,
            visibility = visibility,
        )
