/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Builds without CUDA: SAM2.1 assisted masking has no port to the other tensor backends.

#include "core/event_bridge/localization_manager.hpp"
#include "gui/assisted_masking.hpp"
#include <mutex>

namespace lfs::vis::gui {

    struct AssistedMasking::Impl {
        std::mutex mutex;
        uint64_t last_request = 0;
    };

    AssistedMasking::AssistedMasking(JobRegistry&, std::function<bool()>, RevisionReader)
        : impl_(std::make_unique<Impl>()) {}
    AssistedMasking::~AssistedMasking() = default;

    uint64_t AssistedMasking::submit(AssistedMaskRequest) {
        std::lock_guard lock(impl_->mutex);
        return ++impl_->last_request;
    }

    AssistedMaskSnapshot AssistedMasking::poll() {
        std::lock_guard lock(impl_->mutex);
        AssistedMaskSnapshot snapshot;
        snapshot.request = impl_->last_request;
        if (impl_->last_request != 0) {
            snapshot.phase = "error";
            snapshot.error = LOC("image_preview.sam_cuda_required");
        }
        return snapshot;
    }

    void AssistedMasking::close() {}
    void AssistedMasking::shutdown() {}
    void AssistedMasking::beginSequence(core::Uuid, std::vector<core::ExtractedFrame>,
                                        std::filesystem::path, std::filesystem::path) {}
    std::vector<AssistedMaskFrameSummary> AssistedMasking::sequenceFrames(core::Uuid, size_t, size_t) { return {}; }
    bool AssistedMasking::approveFrames(core::Uuid, std::span<const AssistedMaskApproval>) { return false; }
    std::weak_ptr<const std::vector<std::byte>> AssistedMasking::sequencePreview(std::string_view) { return {}; }
    void AssistedMasking::invalidateFrame(core::Uuid, size_t) {}
    void AssistedMasking::endSequence(core::Uuid) {}

} // namespace lfs::vis::gui
