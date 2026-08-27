/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/error.hpp"
#include "core/export.hpp"
#include "core/scene.hpp"
#include "core/uuid.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace lfs::io::project {

    inline constexpr std::uint16_t SELM_CHAPTER_VERSION = 1;

    enum class SelectionMaskEncoding : std::uint8_t {
        RawU8 = 1,
        // Value 2 was assigned to DeltaBitpack, then withdrawn before
        // release. It is reserved and must never be reused.
    };

    struct SelectionMaskSlice {
        lfs::core::Uuid node_uuid;
        lfs::core::SelectionDomain domain =
            lfs::core::SelectionDomain::Splat;
        SelectionMaskEncoding encoding =
            SelectionMaskEncoding::RawU8;
        std::vector<std::uint8_t> mask;
    };

    class LFS_IO_API SelectionChapter {
    public:
        SelectionChapter() = default;

        [[nodiscard]] const std::vector<lfs::core::SelectionGroup>&
        groups() const noexcept {
            return groups_;
        }
        [[nodiscard]] std::uint8_t active_group_id() const noexcept {
            return active_group_id_;
        }
        [[nodiscard]] std::uint8_t next_group_id() const noexcept {
            return next_group_id_;
        }
        [[nodiscard]] const std::vector<SelectionMaskSlice>&
        slices() const noexcept {
            return slices_;
        }
        [[nodiscard]] const std::vector<lfs::core::Uuid>&
        selected_node_uuids() const noexcept {
            return selected_node_uuids_;
        }

        [[nodiscard]] lfs::Result<void> set_groups(
            std::vector<lfs::core::SelectionGroup> groups,
            std::uint8_t active_group_id,
            std::uint8_t next_group_id);
        [[nodiscard]] lfs::Result<void> upsert_slice(
            SelectionMaskSlice slice);
        [[nodiscard]] bool remove_slice(
            const lfs::core::Uuid& node_uuid,
            lfs::core::SelectionDomain domain);
        [[nodiscard]] lfs::Result<void> set_selected_node_uuids(
            std::vector<lfs::core::Uuid> uuids);

    private:
        friend LFS_IO_API lfs::Result<SelectionChapter>
            decode_selection_chapter(std::span<const std::byte>);

        std::vector<lfs::core::SelectionGroup> groups_;
        std::uint8_t active_group_id_ = 0;
        std::uint8_t next_group_id_ = 1;
        std::vector<SelectionMaskSlice> slices_;
        std::vector<lfs::core::Uuid> selected_node_uuids_;
    };

    struct SelectionHydrationReport {
        bool repaired_group_metadata = false;
        std::vector<lfs::core::Uuid> selected_node_uuids;
    };

    struct LFS_IO_API StagedSelectionChapter {
        lfs::core::Scene::RestoreSelectionState state;
        SelectionHydrationReport report;
    };

    // Detached value-only selection state. Tensor values are copied into
    // owned CPU byte vectors during capture; chapter validation and assembly
    // can then run after the optimizer safe point.
    struct CapturedSelectionState {
        std::vector<lfs::core::SelectionGroup> groups;
        std::uint8_t active_group_id = 0;
        std::uint8_t next_group_id = 1;
        std::vector<SelectionMaskSlice> slices;
        std::vector<lfs::core::Uuid> selected_node_uuids;
    };

    // The omit list prunes each listed node and its whole subtree,
    // matching capture_scene_graph.
    [[nodiscard]] LFS_IO_API lfs::Result<CapturedSelectionState>
    capture_selection_state(
        const lfs::core::Scene& scene,
        std::span<const lfs::core::Uuid> selected_node_uuids,
        std::span<const lfs::core::Uuid> omit_node_uuids = {});

    [[nodiscard]] LFS_IO_API lfs::Result<SelectionChapter>
    materialize_selection_chapter(CapturedSelectionState state);

    [[nodiscard]] LFS_IO_API lfs::Result<std::vector<std::byte>>
    encode_selection_chapter(const SelectionChapter& chapter);

    [[nodiscard]] LFS_IO_API lfs::Result<void>
    validate_selection_chapter(const SelectionChapter& chapter);

    [[nodiscard]] LFS_IO_API lfs::Result<SelectionChapter>
    decode_selection_chapter(std::span<const std::byte> payload);

    // The omit list prunes each listed node and its whole subtree,
    // matching capture_scene_graph.
    [[nodiscard]] LFS_IO_API lfs::Result<SelectionChapter>
    capture_selection_chapter(
        const lfs::core::Scene& scene,
        std::span<const lfs::core::Uuid> selected_node_uuids,
        std::span<const lfs::core::Uuid> omit_node_uuids = {});

    // Phase-A API. Every slice is validated against topology and copied into
    // its final contiguous CPU tensor. topology is not mutated.
    [[nodiscard]] LFS_IO_API lfs::Result<StagedSelectionChapter>
    stage_selection_chapter(
        const SelectionChapter& chapter,
        const lfs::core::Scene& topology);

    [[nodiscard]] LFS_IO_API lfs::Result<SelectionHydrationReport>
    hydrate_selection_chapter(
        const SelectionChapter& chapter,
        lfs::core::Scene& scene);

} // namespace lfs::io::project
