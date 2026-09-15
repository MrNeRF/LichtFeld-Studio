/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <algorithm>
#include <cstddef>
#include <span>
#include <vector>

namespace lfs::vis::gui {

    struct CameraThumbnailUploadCandidate {
        int page_index = -1;
        std::size_t byte_size = 0;
    };

    struct CameraThumbnailUploadBatch {
        int page_index = -1;
        std::vector<std::size_t> candidate_indices;
        std::size_t byte_size = 0;
    };

    // Select a prefix within the staging-byte budget and group it by atlas
    // page. Keeping candidate indices makes the helper independent of Vulkan
    // and lets callers update cache state only after a page upload succeeds.
    [[nodiscard]] inline std::vector<CameraThumbnailUploadBatch>
    batchCameraThumbnailUploads(const std::span<const CameraThumbnailUploadCandidate> candidates,
                                const std::size_t byte_budget) {
        std::vector<CameraThumbnailUploadBatch> batches;
        std::size_t total_bytes = 0;
        for (std::size_t candidate_index = 0; candidate_index < candidates.size(); ++candidate_index) {
            const auto& candidate = candidates[candidate_index];
            if (candidate.page_index < 0 || candidate.byte_size == 0 ||
                candidate.byte_size > byte_budget - total_bytes) {
                break;
            }
            total_bytes += candidate.byte_size;

            auto batch = std::find_if(batches.begin(), batches.end(), [&](const auto& value) {
                return value.page_index == candidate.page_index;
            });
            if (batch == batches.end()) {
                batches.push_back(CameraThumbnailUploadBatch{
                    .page_index = candidate.page_index,
                    .candidate_indices = {candidate_index},
                    .byte_size = candidate.byte_size,
                });
            } else {
                batch->candidate_indices.push_back(candidate_index);
                batch->byte_size += candidate.byte_size;
            }
        }
        return batches;
    }

} // namespace lfs::vis::gui
