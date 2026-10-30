# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Reuse the shared point-filter predicates without importing Tensor or CUDA."""
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
source = args.source


def definition(text, marker, trailing=''):
    first = text.index(marker)
    body = text.index('{', first)
    depth, end = 1, body + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[first:end] + trailing


filters = (source / 'tensor/internal/point_filter.hpp').read_text(encoding='utf-8')
math = (source / 'tensor/internal/point_math.hpp').read_text(encoding='utf-8')
spatial = (source / 'include/core/tensor_spatial.hpp').read_text(encoding='utf-8')
spatial_math = (source / 'tensor/internal/point_spatial.hpp').read_text(encoding='utf-8')
tree_math = (source / 'tensor/internal/point_tree.hpp').read_text(encoding='utf-8')
triangle_math = (source / 'tensor/internal/triangle_tree.hpp').read_text(encoding='utf-8')
descriptors = (source / 'include/core/detail/descriptors.hpp').read_text(encoding='utf-8')
nearest = (source / 'tensor/internal/nearest_point.hpp').read_text(encoding='utf-8')
labels = (source / 'tensor/internal/point_labels.hpp').read_text(encoding='utf-8')
projection_source = (source / 'tensor/internal/point_projection.hpp').read_text(encoding='utf-8')
region_source = (source / 'tensor/internal/point_region.hpp').read_text(encoding='utf-8')
projection = definition(spatial, 'struct PointProjection {', ';')
for old, new in [('std::array<float, 9> rotation', 'float rotation[9]'),
                 ('std::array<float, 3> translation', 'float translation[3]')]:
    if projection.count(old) != 1:
        raise RuntimeError('Review changed point projection descriptor')
    projection = projection.replace(old, new)
math = math[math.index('    LFS_POINT_HD inline float roundedAdd'):math.rindex('} // namespace')]
math = math.replace('__CUDA_ARCH__', '__HIP_DEVICE_COMPILE__')
flags = filters[filters.index('#define LFS_FILTER_NODES'):filters.index('#ifndef LFS_POINT_FILTER_FLAGS_ONLY')]
code = '\n'.join([
    '// Generated from shared Core predicates; do not edit.\n#pragma once',
    '#include <cmath>\n#include <cstdint>\n#include <limits>\n#define LFS_POINT_HD __host__ __device__',
    flags, 'namespace lfs::core::internal {',
    definition(spatial, 'enum class PointProjectionModel', ';'),
    projection,
    definition(spatial, 'enum class PointRegion2DKind', ';'),
    definition(spatial, 'struct PointRegion2D {', ';'),
    definition(filters, 'struct PointFilterArgs', ';'), math,
    definition(filters, 'LFS_POINT_HD inline bool pointPassesFilter'),
    definition(labels, 'struct LabelUpdateParams', ';'),
    definition(labels, 'template <uint32_t Phase>'),
    definition(projection_source, 'LFS_POINT_HD inline void projectPoint'),
    definition(region_source, 'template <PointRegion2DKind Kind>'),
    descriptors[descriptors.index('        inline constexpr uint32_t kPointTreeFanoutBits'):descriptors.index('        struct PointTreeProgram')],
    definition(descriptors, 'struct PointTreeProgram {', ';'),
    spatial_math[spatial_math.index('    // Radius-wide cells'):spatial_math.rindex('} // namespace')],
    nearest[nearest.index('    LFS_NEAREST_HD int nearestPoint'):nearest.rindex('} // namespace')].replace('LFS_NEAREST_HD', 'LFS_POINT_HD').replace('float width) {', 'float width) {\n#pragma clang fp contract(off)\n').replace('float maximum) {', 'float maximum) {\n#pragma clang fp contract(off)\n'),
    tree_math[tree_math.index('    // Bounds on the squared'):tree_math.rindex('} // namespace')],
    '\n'.join(definition(triangle_math, 'LFS_POINT_HD inline ' + signature).replace('{', '{\n#pragma clang fp contract(off)', 1) for signature in ('void rayFrame(', 'bool rayCrossesTriangle(', 'int32_t triangleTreeParity(')), '}',
    '#undef LFS_POINT_HD\n',
])
# Keep the canonical PPISP math and parameter layout without Tensor headers.
ppisp = (source / 'tensor/internal/ppisp.hpp').read_text(encoding='utf-8')
params = (source / 'include/core/tensor_ppisp.hpp').read_text(encoding='utf-8')
code += '\n' + ppisp[:ppisp.index('#pragma once')] + '\nnamespace hip_ppisp {\n'
code += definition(params, 'struct PpispParams {', ';') + '\n'
code += definition(ppisp, 'LFS_PPISP_HD void ppisp_pixel(').replace('LFS_PPISP_HD', '__device__ __forceinline__')
code += '\n}\n'
# Reuse the compensated FP32 GPU factorization; no runtime CUDA dependency.
affine_source = (source / 'cuda/kernels/splat_transform.cu').read_text(encoding='utf-8')
affine = affine_source[affine_source.index('        // The float algorithm'):affine_source.index('    } // namespace')]
affine = affine.replace('splat_transform::LinearTransform', 'LinearTransform')
code += '\n' + affine_source[:affine_source.index('#include')] + '\nnamespace hip_affine {\nstruct LinearTransform { float rows[9]; };\n' + affine + '\n}\n'
code += '\n' + (source / 'include/core/morton.hpp').read_text(encoding='utf-8').replace('__CUDACC__', '__HIPCC__') + '\n'
# Lower the existing moment-matching kernel without its CUDA host launcher.
simplify_source = (source / 'tensor/backend/cuda/kernels/simplify_merge.cu').read_text(encoding='utf-8')
simplify_body = simplify_source[simplify_source.index('        constexpr float kEpsCov'):simplify_source.index('    } // namespace')]
code += '\n#include <cfloat>\nnamespace hip_simplify {\n' + simplify_body + '\n}\n'
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(code, encoding='utf-8')

# Native HIP owns the atomic parent accesses; the shared traversal kernels keep
# the canonical edge ordering and batched walk without depending on CUDA.
kernels = (source / 'tensor/backend/cuda/kernels/tensor_spatial.cu').read_text(encoding='utf-8')
components = '\n'.join(definition(kernels, '__global__ void ' + name + '(')
                        for name in ('union_components', 'flatten_components', 'point_tree_union'))
components = components.replace('ups[k] = base + k < count ? parent[nodes[k]] : mine;',
                                'ups[k] = base + k < count ? __hip_atomic_load(parent + nodes[k], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) : mine;')
components = components.replace('parent[i] = finalComponentRoot(parent, static_cast<int32_t>(i));',
                                'atomicExch(parent + i, finalComponentRoot(parent, static_cast<int32_t>(i)));')
atomic_helpers = r"""
__device__ inline int32_t componentRoot(int32_t* parent, int32_t node) {
    int32_t current=__hip_atomic_load(parent+node,__ATOMIC_RELAXED,__HIP_MEMORY_SCOPE_AGENT),previous=node;
    for (;;) {
        const int32_t ancestor=__hip_atomic_load(parent+current,__ATOMIC_RELAXED,__HIP_MEMORY_SCOPE_AGENT);
        if(current==ancestor) return current;
        atomicMin(parent+previous,ancestor);previous=current;current=ancestor;
    }
}
__device__ inline int32_t finalComponentRoot(const int32_t* parent,int32_t node) {
    for (;;) {const int32_t ancestor=__hip_atomic_load(parent+node,__ATOMIC_RELAXED,__HIP_MEMORY_SCOPE_AGENT);if(ancestor==node)return node;node=ancestor;}
}
"""
components = '/* Generated native HIP spatial components; do not edit. */\n#pragma once\nnamespace hip_spatial_components {\nusing namespace lfs::core::internal;\n'+atomic_helpers+components+'\n}\n'
args.output.with_name('hip_spatial_components.hpp').write_text(components,encoding='utf-8')
