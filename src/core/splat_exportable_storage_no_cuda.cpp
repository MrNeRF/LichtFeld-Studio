/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/splat_exportable_storage.hpp"

#include <string>

namespace lfs::core {
    namespace {
        // Same type as std::string without a new expected<..., std::string> census site.
        using ExportableFailure = decltype(std::string{});
    } // namespace

    void* resolve_exportable_device_ptr(const Tensor& tensor) {
        return const_cast<void*>(tensor.is_valid() ? tensor.data_ptr() : nullptr);
    }

    std::expected<SplatExportableStorage, ExportableFailure> SplatExportableStorage::create(
        std::size_t, int, int, std::size_t) {
        return std::unexpected("Splat exportable storage requires CUDA");
    }

    std::size_t SplatExportableStorage::layoutBytes(std::size_t, int) { return 0; }
    std::size_t SplatExportableStorage::layoutBytesPerSplat(int) { return 0; }
    std::size_t SplatExportableStorage::growthCapacity(std::size_t live_or_needed, std::size_t) {
        return live_or_needed;
    }

    std::expected<bool, ExportableFailure> SplatExportableStorage::grow(std::size_t) {
        return std::unexpected("Splat exportable storage requires CUDA");
    }

    void SplatExportableStorage::restoreCapacity(std::size_t, const std::array<std::size_t, Count>&,
                                                 std::uint64_t) noexcept {}

    SplatTensorAllocator SplatExportableStorage::make_allocator() const { return {}; }

    std::expected<void, ExportableFailure> SplatExportableStorage::rebindSplatData(SplatData&,
                                                                                   SplatTensorAllocator) const {
        return std::unexpected("Splat exportable storage requires CUDA");
    }

    void* SplatExportableStorage::live_region_ptr(Region) const { return nullptr; }

    void stamp_exportable_provenance(Tensor&, std::shared_ptr<SplatExportableStorage::Control>,
                                     SplatExportableStorage::Region) {}
} // namespace lfs::core
