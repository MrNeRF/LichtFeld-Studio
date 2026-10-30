/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "slang_prelude.hpp"
#pragma clang fp contract(off)
namespace hip_export_kmeans {
    using namespace hip_slang;
#include "export_kmeans_kernel.cu"
} // namespace hip_export_kmeans
namespace hip_export_kmeans_accumulate {
    using namespace hip_slang;
#include "export_kmeans_accumulate_kernel.cu"
} // namespace hip_export_kmeans_accumulate
namespace hip_export_kmeans_group {
    using namespace hip_slang;
#include "export_kmeans_group_kernel.cu"
} // namespace hip_export_kmeans_group
namespace hip_export_radix {
    using namespace hip_slang;
#include "export_radix_kernel.cu"
} // namespace hip_export_radix
namespace hip_export_morton {
    using namespace hip_slang;
#include "export_morton_kernel.cu"
} // namespace hip_export_morton
namespace hip_export_decimate {
    using namespace hip_slang;
#include "export_decimate_kernel.cu"
} // namespace hip_export_decimate
namespace hip_export_decimate_cost {
    using namespace hip_slang;
#include "export_decimate_cost_kernel.cu"
} // namespace hip_export_decimate_cost
namespace hip_export_decimate_merge {
    using namespace hip_slang;
#include "export_decimate_merge_kernel.cu"
} // namespace hip_export_decimate_merge
namespace hip_point_raster {
    using namespace hip_slang;
#include "point_raster_kernel.cu"
} // namespace hip_point_raster

#pragma clang fp contract(fast)
