/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/export.hpp"
#include "core/extracted_frame.hpp"
#include "core/job_registry.hpp"
#include "core/nn/weight_file.hpp"
#include "core/tensor.hpp"
#include "core/uuid.hpp"
#include "preprocessing/video_masking.hpp"
#include <array>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace lfs::vis::gui {
    struct AssistedMaskTracking {
        int first = 0, last = 0, seed = 0;
        preprocessing::VideoMaskDirection direction = preprocessing::VideoMaskDirection::Both;
    };
    struct AssistedMaskSemantic {
        // Acquisition adapter runs on the owned worker and retains no Python objects.
        std::function<lfs::Result<core::nn::WeightFile>(const std::function<bool()>&)> prepare_weights;
        std::vector<std::vector<int>> descriptions;
        // Default threshold in the official LLMDet Tiny detection example.
        float confidence = 0.4f;
    };
    struct AssistedMaskRequest {
        std::filesystem::path image;
        std::filesystem::path weights;
        std::string revision;
        int64_t mtime_ns = 0;
        uint64_t size_bytes = 0;
        std::vector<std::array<float, 3>> points;
        std::optional<std::array<float, 4>> box;
        uint64_t refine_request = 0;
        int refine_candidate = -1;
        int export_candidate = -1;
        bool exclude = false;
        std::optional<AssistedMaskTracking> tracking{};
        int review_frame = -1;
        core::Uuid run{};
        bool retain_frame = false;
        bool publish_approved = false;
        std::optional<AssistedMaskSemantic> semantic{};
    };
    struct AssistedMaskFrameSummary {
        // Proposal request ID; distinct from the sequence's change counter.
        uint64_t revision = 0;
        float score = 0;
        bool reliable = false, approved = false, keyframe = false;
        std::filesystem::path mask_path{};
        int matched_objects = -1;
        bool semantic_limited = false;
    };
    struct AssistedMaskApproval {
        size_t index = 0;
        uint64_t revision = 0;
        bool approved = false;
    };
    struct AssistedMaskSnapshot {
        uint64_t request = 0;
        std::string revision;
        std::string phase = "idle";
        std::string error;
        std::vector<core::Tensor> overlays;
        std::vector<float> scores;
        std::shared_ptr<const std::vector<std::byte>> mask;
        int width = 0, height = 0;
        std::filesystem::path image;
        std::filesystem::file_time_type source_time{};
        uint64_t source_bytes = 0;
        uint64_t sequence_revision = 0;
        size_t completed = 0, total = 0;
        size_t approved = 0;
        size_t empty_approved = 0;
        int matched_objects = -1;
        bool semantic_limited = false;
    };

    // App-owned worker. Closing an editor requests retirement; only app shutdown joins.
    // Inference produces disposable proposals; only explicit publication writes approved masks.
    class LFS_VIS_API AssistedMasking {
    public:
        using RevisionReader = std::function<std::optional<std::array<uint64_t, 2>>()>;
        AssistedMasking(JobRegistry& jobs, std::function<bool()> admission,
                        RevisionReader document_revision = {});
        ~AssistedMasking();
        uint64_t submit(AssistedMaskRequest request);
        AssistedMaskSnapshot poll();
        void close();
        void shutdown();
        void beginSequence(core::Uuid run, std::vector<core::ExtractedFrame> frames,
                           std::filesystem::path output, std::filesystem::path weights);
        std::vector<AssistedMaskFrameSummary> sequenceFrames(core::Uuid run, size_t first = 0, size_t count = 64);
        // Validate the complete group before changing any approval.
        bool approveFrames(core::Uuid run, std::span<const AssistedMaskApproval> approvals);
        std::weak_ptr<const std::vector<std::byte>> sequencePreview(std::string_view key);
        // Reject while work is in flight; saved mask assets survive draft edits.
        void invalidateFrame(core::Uuid run, size_t index);
        void endSequence(core::Uuid run);

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace lfs::vis::gui
