/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "core/error.hpp"
#include "core/export.hpp"
#include "core/splat_data.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace lfs::io::project {

    enum class SplatSourceKind : std::uint8_t {
        ImportedPly,
        ImportedSpz,
        ImportedSog, // reserved: SOG embed / bake-to-embedded (owner decision 2026-07-30)
        Generated,
        BakedRad, // reserved: SOG embed / bake-to-embedded (owner decision 2026-07-30)
        LiveRad,
    };

    class LFS_IO_API SplatChapterPayload {
    public:
        [[nodiscard]] static lfs::Result<SplatChapterPayload>
        from_lfsp(std::span<const std::byte> bytes);
        [[nodiscard]] static lfs::Result<SplatChapterPayload>
        from_lfsp(std::vector<std::byte>&& bytes);
        [[nodiscard]] static lfs::Result<SplatChapterPayload>
        capture(const lfs::core::SplatData& model, SplatSourceKind source_kind,
                bool is_training_model);

        [[nodiscard]] std::span<const std::byte> bytes() const noexcept {
            return bytes_;
        }
        [[nodiscard]] std::uint32_t lfsp_version() const noexcept {
            return lfsp_version_;
        }
        [[nodiscard]] lfs::Result<std::unique_ptr<lfs::core::SplatData>>
        hydrate(lfs::core::SplatTensorAllocator tensor_allocator = {}) const;

        [[nodiscard]] static bool must_reference_external(
            SplatSourceKind source_kind) noexcept;

    private:
        std::vector<std::byte> bytes_;
        std::uint32_t lfsp_version_ = 0;
    };

} // namespace lfs::io::project
