/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera.hpp"
#include "core/error.hpp"
#include "core/export.hpp"
#include "core/scene.hpp"
#include "core/uuid.hpp"
#include "io/project_document.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace lfs::io::project {

    // SFMO chapter, little endian:
    //   header  magic "SFMO", u32 version, u32 camera_count, u32 reserved,
    //           u64 observation_count, u64 content_hash
    //   index   camera_count x { 16-byte camera node UUID, u64 count }, UUIDs ascending
    //   data    observation_count x { f32 u, v, x, y, z }
    // content_hash covers index and data, so an unchanged table is recognised without decoding.

    using SfmObservation = lfs::core::Camera::SfmObservation;

    struct UuidLess {
        [[nodiscard]] bool operator()(const lfs::core::Uuid& a, const lfs::core::Uuid& b) const noexcept {
            return a.bytes < b.bytes;
        }
    };

    using SfmObservationTable =
        std::map<lfs::core::Uuid, const std::vector<SfmObservation>*, UuidLess>;

    struct SfmObservationIndex {
        struct Entry {
            std::uint64_t first = 0;
            std::uint64_t count = 0;
        };
        std::map<lfs::core::Uuid, Entry, UuidLess> cameras;
        std::uint64_t observation_count = 0;
        std::uint64_t content_hash = 0;
    };

    [[nodiscard]] LFS_IO_API std::uint64_t sfm_observation_content_hash(const SfmObservationTable& table);
    [[nodiscard]] LFS_IO_API std::vector<std::byte> encode_sfm_observations(const SfmObservationTable& table);
    [[nodiscard]] LFS_IO_API lfs::Result<SfmObservationIndex>
    read_sfm_observation_index(const LazyChunkValue& chapter);

    // Camera nodes with observations. Observations are fixed after loading, so the handles can be
    // captured at a safe point and read later while the document is staged.
    using SfmObservationCameras = std::vector<std::pair<lfs::core::Uuid, std::shared_ptr<const lfs::core::Camera>>>;
    [[nodiscard]] LFS_IO_API SfmObservationCameras capture_sfm_observation_cameras(const lfs::core::Scene& scene);

    // Keeps the document's SFMO chapter in step with the cameras. A chapter the cameras still read
    // from, or one with the same content hash, is kept as is; otherwise a new one replaces it.
    [[nodiscard]] LFS_IO_API lfs::Result<void> sync_sfm_observations(ProjectDocument& document,
                                                                     const SfmObservationCameras& cameras);

    // Decodes the whole chapter on the first observations() call; later calls return the cache.
    class LFS_IO_API SfmObservationChapterSource final : public lfs::core::Camera::SfmObservationSource {
    public:
        [[nodiscard]] static lfs::Result<std::shared_ptr<const SfmObservationChapterSource>>
        open(const LazyChunkValue& chapter);

        [[nodiscard]] const std::vector<SfmObservation>& observations(
            const lfs::core::Uuid& camera_node) const override;
        [[nodiscard]] const lfs::core::Uuid& chapter_uuid() const noexcept { return chapter_uuid_; }
        [[nodiscard]] const SfmObservationIndex& index() const noexcept { return index_; }

    private:
        SfmObservationChapterSource(LazyChunkValue chapter, SfmObservationIndex index);

        LazyChunkValue chapter_;
        lfs::core::Uuid chapter_uuid_;
        SfmObservationIndex index_;
        mutable std::once_flag decoded_;
        mutable std::unordered_map<lfs::core::Uuid, std::vector<SfmObservation>> observations_;
    };

} // namespace lfs::io::project
