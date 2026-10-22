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
    tree_math[tree_math.index('    // Bounds on the squared'):tree_math.rindex('} // namespace')], '}',
    '#undef LFS_POINT_HD\n',
])
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(code, encoding='utf-8')
