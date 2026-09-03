/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/point_cloud.hpp"

namespace lfs::core {
    namespace {
        bool matchesPointCount(const Tensor& tensor, const int64_t point_count) {
            return tensor.is_valid() &&
                   tensor.ndim() > 0 &&
                   tensor.size(0) == point_count;
        }

        Tensor compactRows(const Tensor& tensor, const Tensor& indices, const int64_t point_count) {
            return matchesPointCount(tensor, point_count) ? tensor.index_select(0, indices) : tensor;
        }
    } // namespace

    bool PointCloud::has_selection() const {
        return selection &&
               selection->is_valid() &&
               selection->numel() == static_cast<size_t>(size());
    }

    bool PointCloud::has_deleted() const {
        return deleted &&
               deleted->is_valid() &&
               deleted->numel() == static_cast<size_t>(size());
    }

    PointCloud remove_deleted_points(const PointCloud& pc) {
        if (!pc.has_deleted()) {
            return pc;
        }

        const auto device = pc.means.is_valid() ? pc.means.device() : pc.deleted->device();
        auto deleted_mask = pc.deleted->device() == device ? *pc.deleted : pc.deleted->to(device);
        if (deleted_mask.dtype() != DataType::Bool) {
            deleted_mask = deleted_mask.to(DataType::Bool);
        }

        const auto keep_indices = deleted_mask.logical_not().nonzero().squeeze(1);
        const int64_t point_count = pc.size();

        PointCloud result;
        result.means = compactRows(pc.means, keep_indices, point_count);
        result.colors = compactRows(pc.colors, keep_indices, point_count);
        result.normals = compactRows(pc.normals, keep_indices, point_count);
        result.sh0 = compactRows(pc.sh0, keep_indices, point_count);
        result.shN = compactRows(pc.shN, keep_indices, point_count);
        result.opacity = compactRows(pc.opacity, keep_indices, point_count);
        result.scaling = compactRows(pc.scaling, keep_indices, point_count);
        result.rotation = compactRows(pc.rotation, keep_indices, point_count);
        result.attribute_names = pc.attribute_names;
        return result;
    }
} // namespace lfs::core
