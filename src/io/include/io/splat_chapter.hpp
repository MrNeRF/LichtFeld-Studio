/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "core/error.hpp"
#include "core/export.hpp"
#include "core/splat_data.hpp"
#include "io/project_chapters.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <iosfwd>
#include <memory>
#include <optional>
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

    struct LFS_IO_API HydratedSplatStream {
        std::unique_ptr<lfs::core::SplatData> splat;
        Hash128 content_xxh3_128;
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

        // Parse and hydrate an LFSP v2 raw splat from a seekable logical-payload
        // stream without materializing the full payload. The returned hash is
        // XXH3-128 of the entire logical payload in order.
        [[nodiscard]] static lfs::Result<HydratedSplatStream>
        hydrate_lfsp_stream(
            std::istream& stream, std::uint64_t size,
            lfs::core::SplatTensorAllocator allocator = {},
            std::function<void(std::size_t, std::size_t)> progress = {});

        [[nodiscard]] static bool must_reference_external(
            SplatSourceKind source_kind) noexcept;

    private:
        std::vector<std::byte> bytes_;
        std::uint32_t lfsp_version_ = 0;
    };

    namespace detail {
        // Test-only override of the stream-hydrate window. nullopt restores 16 MiB.
        LFS_IO_API void set_splat_stream_window_bytes_for_testing(
            std::optional<std::size_t> window_bytes);
    } // namespace detail

} // namespace lfs::io::project
