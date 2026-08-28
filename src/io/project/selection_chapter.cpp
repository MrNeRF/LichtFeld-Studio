/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/selection_chapter.hpp"

#include "chapter_binary_utils.hpp"
#include "io/capture_omit_filter.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <format>
#include <limits>
#include <ranges>
#include <set>
#include <string_view>
#include <type_traits>
#include <unordered_set>
#include <utility>

namespace lfs::io::project {

    namespace {

        constexpr std::size_t HEADER_BYTES = 64;
        constexpr std::size_t GROUP_ROW_BYTES = 48;
        constexpr std::size_t SLICE_ROW_BYTES = 64;
        constexpr std::size_t UUID_BYTES = 16;
        constexpr std::uint64_t DATA_ALIGNMENT = 64;

        using chapter_binary::align_up;
        using chapter_binary::all_zero;
        using chapter_binary::checked_add;
        using chapter_binary::checked_mul;
        using chapter_binary::read_f32;
        using chapter_binary::read_u16;
        using chapter_binary::read_u32;
        using chapter_binary::read_u64;
        using chapter_binary::valid_utf8;
        using chapter_binary::write_f32;
        using chapter_binary::write_u16;
        using chapter_binary::write_u32;
        using chapter_binary::write_u64;

        struct EncodedSlice {
            SelectionMaskSlice slice;
            std::vector<std::byte> bytes;
            std::uint64_t nonzero_count = 0;
        };

        struct SliceDescriptor {
            lfs::core::Uuid node_uuid;
            lfs::core::SelectionDomain domain =
                lfs::core::SelectionDomain::Splat;
            std::uint64_t element_count = 0;
            std::uint64_t nonzero_count = 0;
            std::uint64_t data_offset = 0;
            std::uint64_t data_bytes = 0;
        };

        lfs::Error selection_error(
            const lfs::ErrorCode code,
            std::string user_message,
            std::string detail,
            const std::string_view field = {}) {
            lfs::SmallFields fields;
            if (!field.empty()) {
                fields.add("field", field);
            }
            return lfs::make_error(lfs::ErrorInit{
                .code = code,
                .domain = lfs::ErrorDomain::IO,
                .severity = lfs::Severity::Error,
                .retryability = lfs::Retryability::NotRetryable,
                .operation_id = {},
                .user_message = std::move(user_message),
                .detail = std::move(detail),
                .detection = LFS_SOURCE_SITE_CURRENT(),
                .fields = std::move(fields),
                .native = std::nullopt,
            });
        }

        template <typename T>
        lfs::Result<T> corrupt(
            std::string detail,
            const std::string_view field = {}) {
            auto error = selection_error(
                lfs::ErrorCode::DataLoss,
                "The project contains invalid selection data.",
                std::move(detail),
                field);
            if constexpr (std::is_void_v<T>) {
                return lfs::Result<void>::failure(
                    std::move(error));
            } else {
                return error;
            }
        }

        lfs::Result<void> invalid(
            std::string detail,
            const std::string_view field = {}) {
            return lfs::Result<void>::failure(selection_error(
                lfs::ErrorCode::InvalidArgument,
                "The selection state cannot be saved.",
                std::move(detail),
                field));
        }

        bool valid_domain(
            const lfs::core::SelectionDomain domain) {
            return domain ==
                       lfs::core::SelectionDomain::Splat ||
                   domain ==
                       lfs::core::SelectionDomain::PointCloud;
        }

        lfs::Result<void> validate_groups(
            const std::span<const lfs::core::SelectionGroup> groups,
            const std::uint8_t active_group_id,
            const std::uint8_t next_group_id,
            const bool validate_allocator_metadata) {
            std::array<bool, 256> seen{};
            for (const auto& group : groups) {
                if (group.id == 0 || seen[group.id]) {
                    return invalid(
                        std::format(
                            "Selection group ID {} is zero or duplicated",
                            group.id),
                        "groups.id");
                }
                if (!valid_utf8(group.name)) {
                    return invalid(
                        std::format(
                            "Selection group {} name is not valid UTF-8",
                            group.id),
                        "groups.name");
                }
                if (!std::isfinite(group.color.x) ||
                    !std::isfinite(group.color.y) ||
                    !std::isfinite(group.color.z)) {
                    return invalid(
                        std::format(
                            "Selection group {} color is not finite",
                            group.id),
                        "groups.color");
                }
                seen[group.id] = true;
            }

            if (!validate_allocator_metadata) {
                return {};
            }
            if (active_group_id != 0 &&
                !seen[active_group_id]) {
                return invalid(
                    std::format(
                        "Active selection group {} does not exist",
                        active_group_id),
                    "active_group_id");
            }
            if (next_group_id == 0) {
                if (groups.size() != 255) {
                    return invalid(
                        "next_group_id is zero before all IDs are used",
                        "next_group_id");
                }
            } else if (seen[next_group_id]) {
                return invalid(
                    std::format(
                        "next_group_id {} is already allocated",
                        next_group_id),
                    "next_group_id");
            }
            return {};
        }

        lfs::Result<void> validate_slice_values(
            const SelectionMaskSlice& slice,
            const std::array<bool, 256>& group_ids) {
            if (slice.node_uuid.is_nil()) {
                return invalid(
                    "Selection slice has a nil node UUID",
                    "slices.node_uuid");
            }
            if (!valid_domain(slice.domain)) {
                return invalid(
                    "Selection slice has an unknown domain",
                    "slices.domain");
            }
            if (slice.encoding !=
                SelectionMaskEncoding::RawU8) {
                const auto value = static_cast<std::uint8_t>(
                    slice.encoding);
                return invalid(
                    value == 2
                        ? "Selection mask encoding 2 was withdrawn before "
                          "release"
                        : std::format(
                              "Selection slice has unknown encoding {}",
                              value),
                    "slices.encoding");
            }
            for (std::size_t index = 0;
                 index < slice.mask.size();
                 ++index) {
                const auto value = slice.mask[index];
                if (value != 0 && !group_ids[value]) {
                    return invalid(
                        std::format(
                            "Selection slice {}:{} contains unknown group "
                            "ID {} at element {}",
                            slice.node_uuid.to_string(),
                            static_cast<unsigned>(slice.domain),
                            value,
                            index),
                        "slices.mask");
                }
            }
            return {};
        }

        std::array<bool, 256> group_id_set(
            const std::span<const lfs::core::SelectionGroup> groups) {
            std::array<bool, 256> result{};
            for (const auto& group : groups) {
                result[group.id] = true;
            }
            return result;
        }

        EncodedSlice encode_slice(SelectionMaskSlice slice) {
            EncodedSlice result{
                .slice = std::move(slice),
                .bytes = {},
                .nonzero_count = 0,
            };
            result.nonzero_count =
                static_cast<std::uint64_t>(
                    std::ranges::count_if(
                        result.slice.mask,
                        [](const std::uint8_t value) {
                            return value != 0;
                        }));
            const auto bytes =
                std::as_bytes(std::span(result.slice.mask));
            result.bytes.assign(bytes.begin(), bytes.end());
            return result;
        }

        bool slice_key_less(
            const SelectionMaskSlice& lhs,
            const SelectionMaskSlice& rhs) {
            if (lhs.node_uuid.bytes != rhs.node_uuid.bytes) {
                return std::ranges::lexicographical_compare(
                    lhs.node_uuid.bytes,
                    rhs.node_uuid.bytes);
            }
            return static_cast<std::uint8_t>(lhs.domain) <
                   static_cast<std::uint8_t>(rhs.domain);
        }

        bool same_slice_key(
            const SelectionMaskSlice& lhs,
            const SelectionMaskSlice& rhs) {
            return lhs.node_uuid == rhs.node_uuid &&
                   lhs.domain == rhs.domain;
        }

        struct SelectionChapterPlan {
            std::vector<std::size_t> sorted_slice_order;
            std::uint64_t group_offset = 0;
            std::uint64_t group_bytes = 0;
            std::uint64_t slice_offset = 0;
            std::uint64_t slice_bytes = 0;
            std::uint64_t selected_offset = 0;
            std::uint64_t selected_bytes = 0;
            std::uint64_t data_offset = 0;
            std::vector<std::uint64_t> name_offsets;
            std::vector<std::uint64_t> slice_data_offsets;
            std::uint64_t total_bytes = 0;
        };

        lfs::Result<SelectionChapterPlan>
        plan_selection_chapter(const SelectionChapter& chapter) {
            if (auto result = validate_groups(
                    chapter.groups(),
                    chapter.active_group_id(),
                    chapter.next_group_id(),
                    true);
                !result) {
                return std::move(result).error();
            }
            const auto ids = group_id_set(chapter.groups());
            const auto& slices = chapter.slices();
            std::vector<std::size_t> sorted_slice_order(slices.size());
            for (std::size_t index = 0; index < slices.size(); ++index) {
                sorted_slice_order[index] = index;
            }
            std::ranges::sort(
                sorted_slice_order,
                [&](const std::size_t lhs, const std::size_t rhs) {
                    return slice_key_less(slices[lhs], slices[rhs]);
                });
            for (std::size_t index = 1; index < sorted_slice_order.size();
                 ++index) {
                if (same_slice_key(
                        slices[sorted_slice_order[index - 1]],
                        slices[sorted_slice_order[index]])) {
                    return selection_error(
                        lfs::ErrorCode::InvalidArgument,
                        "The selection state cannot be saved.",
                        "SELM contains duplicate node/domain slice keys",
                        "slices");
                }
            }
            for (const std::size_t index : sorted_slice_order) {
                if (auto result = validate_slice_values(slices[index], ids);
                    !result) {
                    return std::move(result).error();
                }
            }

            std::unordered_set<lfs::core::Uuid> selected_seen;
            for (const auto& uuid : chapter.selected_node_uuids()) {
                if (uuid.is_nil() || !selected_seen.emplace(uuid).second) {
                    return selection_error(
                        lfs::ErrorCode::InvalidArgument,
                        "The selection state cannot be saved.",
                        "Selected-node UUIDs contain nil or duplicate values",
                        "selected_node_uuids");
                }
            }

            if (chapter.groups().size() >
                    std::numeric_limits<std::uint32_t>::max() ||
                sorted_slice_order.size() >
                    std::numeric_limits<std::uint32_t>::max() ||
                chapter.selected_node_uuids().size() >
                    std::numeric_limits<std::uint32_t>::max()) {
                return selection_error(
                    lfs::ErrorCode::InvalidArgument,
                    "The selection state cannot be saved.",
                    "SELM table count exceeds u32");
            }

            SelectionChapterPlan plan;
            plan.sorted_slice_order = std::move(sorted_slice_order);
            plan.group_offset = HEADER_BYTES;
            if (!checked_mul<std::uint64_t>(
                    chapter.groups().size(),
                    GROUP_ROW_BYTES,
                    plan.group_bytes) ||
                !checked_add(
                    plan.group_offset, plan.group_bytes, plan.slice_offset) ||
                !checked_mul<std::uint64_t>(
                    plan.sorted_slice_order.size(),
                    SLICE_ROW_BYTES,
                    plan.slice_bytes) ||
                !checked_add(
                    plan.slice_offset,
                    plan.slice_bytes,
                    plan.selected_offset) ||
                !checked_mul<std::uint64_t>(
                    chapter.selected_node_uuids().size(),
                    UUID_BYTES,
                    plan.selected_bytes) ||
                !checked_add(
                    plan.selected_offset,
                    plan.selected_bytes,
                    plan.data_offset) ||
                !align_up(
                    plan.data_offset,
                    DATA_ALIGNMENT,
                    plan.data_offset)) {
                return selection_error(
                    lfs::ErrorCode::InvalidArgument,
                    "The selection state cannot be saved.",
                    "SELM table offsets overflow");
            }

            plan.name_offsets.reserve(chapter.groups().size());
            plan.slice_data_offsets.reserve(plan.sorted_slice_order.size());
            std::uint64_t cursor = plan.data_offset;
            for (const auto& group : chapter.groups()) {
                if (group.name.size() >
                    std::numeric_limits<std::uint32_t>::max()) {
                    return selection_error(
                        lfs::ErrorCode::InvalidArgument,
                        "The selection state cannot be saved.",
                        "Selection group name exceeds u32 bytes",
                        "groups.name");
                }
                if (!align_up(cursor, DATA_ALIGNMENT, cursor)) {
                    return selection_error(
                        lfs::ErrorCode::InvalidArgument,
                        "The selection state cannot be saved.",
                        "SELM group-name offset overflows",
                        "groups.name");
                }
                plan.name_offsets.push_back(cursor);
                if (!checked_add<std::uint64_t>(
                        cursor, group.name.size(), cursor)) {
                    return selection_error(
                        lfs::ErrorCode::InvalidArgument,
                        "The selection state cannot be saved.",
                        "SELM group-name range overflows",
                        "groups.name");
                }
            }
            for (const std::size_t index : plan.sorted_slice_order) {
                if (!align_up(cursor, DATA_ALIGNMENT, cursor)) {
                    return selection_error(
                        lfs::ErrorCode::InvalidArgument,
                        "The selection state cannot be saved.",
                        "SELM slice offset overflows",
                        "slices.data");
                }
                plan.slice_data_offsets.push_back(cursor);
                if (!checked_add<std::uint64_t>(
                        cursor, slices[index].mask.size(), cursor)) {
                    return selection_error(
                        lfs::ErrorCode::InvalidArgument,
                        "The selection state cannot be saved.",
                        "SELM slice range overflows",
                        "slices.data");
                }
            }
            if (!align_up(cursor, DATA_ALIGNMENT, cursor) ||
                cursor > std::numeric_limits<std::size_t>::max()) {
                return selection_error(
                    lfs::ErrorCode::ResourceExhausted,
                    "The selection state is too large for this process.",
                    "SELM payload size exceeds size_t");
            }
            plan.total_bytes = cursor;
            return plan;
        }

    } // namespace

    lfs::Result<void> SelectionChapter::set_groups(
        std::vector<lfs::core::SelectionGroup> groups,
        const std::uint8_t active_group_id,
        const std::uint8_t next_group_id) {
        if (auto result = validate_groups(
                groups,
                active_group_id,
                next_group_id,
                true);
            !result) {
            return result;
        }
        const auto ids = group_id_set(groups);
        for (const auto& slice : slices_) {
            if (auto result =
                    validate_slice_values(slice, ids);
                !result) {
                return result;
            }
        }
        for (auto& group : groups) {
            group.count = 0;
        }
        groups_ = std::move(groups);
        active_group_id_ = active_group_id;
        next_group_id_ = next_group_id;
        return {};
    }

    lfs::Result<void> SelectionChapter::upsert_slice(
        SelectionMaskSlice slice) {
        const auto ids = group_id_set(groups_);
        if (auto result =
                validate_slice_values(slice, ids);
            !result) {
            return result;
        }
        const auto found = std::ranges::find_if(
            slices_,
            [&](const SelectionMaskSlice& existing) {
                return same_slice_key(existing, slice);
            });
        if (found == slices_.end()) {
            slices_.push_back(std::move(slice));
        } else {
            *found = std::move(slice);
        }
        return {};
    }

    bool SelectionChapter::remove_slice(
        const lfs::core::Uuid& node_uuid,
        const lfs::core::SelectionDomain domain) {
        const auto before = slices_.size();
        std::erase_if(
            slices_,
            [&](const SelectionMaskSlice& slice) {
                return slice.node_uuid == node_uuid &&
                       slice.domain == domain;
            });
        return slices_.size() != before;
    }

    lfs::Result<void>
    SelectionChapter::set_selected_node_uuids(
        std::vector<lfs::core::Uuid> uuids) {
        std::unordered_set<lfs::core::Uuid> seen;
        for (const auto& uuid : uuids) {
            if (uuid.is_nil() || !seen.emplace(uuid).second) {
                return invalid(
                    "Selected-node UUIDs contain nil or duplicate values",
                    "selected_node_uuids");
            }
        }
        selected_node_uuids_ = std::move(uuids);
        return {};
    }

    lfs::Result<void>
    validate_selection_chapter(const SelectionChapter& chapter) {
        auto plan = plan_selection_chapter(chapter);
        if (!plan) {
            return lfs::Result<void>::failure(std::move(plan).error());
        }
        return {};
    }

    lfs::Result<std::vector<std::byte>>
    encode_selection_chapter(const SelectionChapter& chapter) {
        auto planned = plan_selection_chapter(chapter);
        if (!planned) {
            return std::move(planned).error();
        }
        const SelectionChapterPlan& plan = *planned;
        std::vector<EncodedSlice> encoded_slices;
        encoded_slices.reserve(plan.sorted_slice_order.size());
        for (const std::size_t index : plan.sorted_slice_order) {
            encoded_slices.push_back(
                encode_slice(chapter.slices()[index]));
        }

        const std::uint64_t group_offset = plan.group_offset;
        const std::uint64_t slice_offset = plan.slice_offset;
        const std::uint64_t selected_offset = plan.selected_offset;
        const std::uint64_t data_offset = plan.data_offset;
        const std::vector<std::uint64_t>& name_offsets = plan.name_offsets;
        const std::vector<std::uint64_t>& slice_data_offsets =
            plan.slice_data_offsets;

        std::vector<std::byte> result(
            static_cast<std::size_t>(plan.total_bytes),
            std::byte{0});
        std::memcpy(result.data(), "LSEL", 4);
        write_u16(result, 4, SELM_CHAPTER_VERSION);
        write_u16(
            result,
            6,
            static_cast<std::uint16_t>(HEADER_BYTES));
        assert(
            chapter.groups().size() <=
            std::numeric_limits<std::uint32_t>::max());
        write_u32(
            result,
            8,
            static_cast<std::uint32_t>(
                chapter.groups().size()));
        assert(
            encoded_slices.size() <=
            std::numeric_limits<std::uint32_t>::max());
        write_u32(
            result,
            12,
            static_cast<std::uint32_t>(
                encoded_slices.size()));
        assert(
            chapter.selected_node_uuids().size() <=
            std::numeric_limits<std::uint32_t>::max());
        write_u32(
            result,
            16,
            static_cast<std::uint32_t>(
                chapter.selected_node_uuids().size()));
        result[20] =
            static_cast<std::byte>(
                chapter.active_group_id());
        result[21] =
            static_cast<std::byte>(
                chapter.next_group_id());
        write_u64(result, 24, group_offset);
        write_u64(result, 32, slice_offset);
        write_u64(result, 40, selected_offset);
        write_u64(result, 48, data_offset);
        write_u64(result, 56, result.size());

        for (std::size_t index = 0;
             index < chapter.groups().size();
             ++index) {
            const auto& group = chapter.groups()[index];
            const auto offset =
                static_cast<std::size_t>(
                    group_offset +
                    index * GROUP_ROW_BYTES);
            result[offset] =
                static_cast<std::byte>(group.id);
            result[offset + 1] =
                static_cast<std::byte>(
                    group.locked ? 1 : 0);
            write_f32(result, offset + 4, group.color.x);
            write_f32(result, offset + 8, group.color.y);
            write_f32(result, offset + 12, group.color.z);
            write_u64(
                result, offset + 16, name_offsets[index]);
            write_u32(
                result,
                offset + 24,
                static_cast<std::uint32_t>(
                    group.name.size()));
            if (!group.name.empty()) {
                std::memcpy(
                    result.data() + name_offsets[index],
                    group.name.data(),
                    group.name.size());
            }
        }

        for (std::size_t index = 0;
             index < encoded_slices.size();
             ++index) {
            const auto& slice = encoded_slices[index];
            const auto offset =
                static_cast<std::size_t>(
                    slice_offset +
                    index * SLICE_ROW_BYTES);
            std::memcpy(
                result.data() + offset,
                slice.slice.node_uuid.bytes.data(),
                UUID_BYTES);
            result[offset + 16] =
                static_cast<std::byte>(
                    slice.slice.domain);
            result[offset + 17] =
                static_cast<std::byte>(
                    slice.slice.encoding);
            write_u64(
                result,
                offset + 24,
                slice.slice.mask.size());
            write_u64(
                result,
                offset + 32,
                slice.nonzero_count);
            write_u64(
                result,
                offset + 40,
                slice_data_offsets[index]);
            write_u64(
                result,
                offset + 48,
                slice.bytes.size());
            if (!slice.bytes.empty()) {
                std::memcpy(
                    result.data() +
                        slice_data_offsets[index],
                    slice.bytes.data(),
                    slice.bytes.size());
            }
        }

        for (std::size_t index = 0;
             index <
             chapter.selected_node_uuids().size();
             ++index) {
            std::memcpy(
                result.data() +
                    selected_offset +
                    index * UUID_BYTES,
                chapter.selected_node_uuids()[index]
                    .bytes.data(),
                UUID_BYTES);
        }
        return result;
    }

    lfs::Result<SelectionChapter> decode_selection_chapter(
        const std::span<const std::byte> payload) {
        if (payload.size() < HEADER_BYTES ||
            std::memcmp(payload.data(), "LSEL", 4) != 0) {
            return corrupt<SelectionChapter>(
                "SELM payload is truncated or has the wrong magic",
                "magic");
        }
        if (read_u16(payload, 4) !=
            SELM_CHAPTER_VERSION) {
            return selection_error(
                lfs::ErrorCode::Unsupported,
                "This selection chapter requires a newer LichtFeld version.",
                std::format(
                    "Unsupported SELM version {}",
                    read_u16(payload, 4)),
                "chapter_version");
        }
        if (read_u16(payload, 6) != HEADER_BYTES ||
            !all_zero(payload.subspan(22, 2))) {
            return corrupt<SelectionChapter>(
                "SELM header size or reserved bytes are invalid",
                "header");
        }
        const auto group_count = read_u32(payload, 8);
        const auto slice_count = read_u32(payload, 12);
        const auto selected_count = read_u32(payload, 16);
        const auto active_group_id =
            std::to_integer<std::uint8_t>(payload[20]);
        const auto next_group_id =
            std::to_integer<std::uint8_t>(payload[21]);
        const auto group_offset = read_u64(payload, 24);
        const auto slice_offset = read_u64(payload, 32);
        const auto selected_offset = read_u64(payload, 40);
        const auto data_offset = read_u64(payload, 48);
        const auto payload_bytes = read_u64(payload, 56);
        if (payload_bytes != payload.size() ||
            group_offset != HEADER_BYTES ||
            data_offset % DATA_ALIGNMENT != 0) {
            return corrupt<SelectionChapter>(
                "SELM header offsets or payload size are noncanonical",
                "header");
        }

        std::uint64_t group_bytes = 0;
        std::uint64_t expected_slice_offset = 0;
        std::uint64_t slice_bytes = 0;
        std::uint64_t expected_selected_offset = 0;
        std::uint64_t selected_bytes = 0;
        std::uint64_t expected_data_offset = 0;
        if (!checked_mul<std::uint64_t>(
                group_count,
                GROUP_ROW_BYTES,
                group_bytes) ||
            !checked_add(
                group_offset,
                group_bytes,
                expected_slice_offset) ||
            !checked_mul<std::uint64_t>(
                slice_count,
                SLICE_ROW_BYTES,
                slice_bytes) ||
            !checked_add(
                expected_slice_offset,
                slice_bytes,
                expected_selected_offset) ||
            !checked_mul<std::uint64_t>(
                selected_count,
                UUID_BYTES,
                selected_bytes) ||
            !checked_add(
                expected_selected_offset,
                selected_bytes,
                expected_data_offset) ||
            !align_up(
                expected_data_offset,
                DATA_ALIGNMENT,
                expected_data_offset) ||
            slice_offset != expected_slice_offset ||
            selected_offset != expected_selected_offset ||
            data_offset != expected_data_offset ||
            data_offset > payload.size()) {
            return corrupt<SelectionChapter>(
                "SELM table bounds or offsets are invalid",
                "header");
        }
        if (!all_zero(payload.subspan(
                static_cast<std::size_t>(
                    expected_selected_offset +
                    selected_bytes),
                static_cast<std::size_t>(
                    data_offset -
                    expected_selected_offset -
                    selected_bytes)))) {
            return corrupt<SelectionChapter>(
                "SELM table alignment padding is nonzero",
                "header");
        }

        SelectionChapter result;
        result.active_group_id_ = active_group_id;
        result.next_group_id_ = next_group_id;
        result.groups_.reserve(group_count);
        std::array<bool, 256> ids{};
        struct Range {
            std::uint64_t begin;
            std::uint64_t end;
            std::string label;
        };
        std::vector<Range> ranges;
        ranges.reserve(
            static_cast<std::size_t>(group_count) +
            static_cast<std::size_t>(slice_count));

        for (std::uint32_t index = 0;
             index < group_count;
             ++index) {
            const auto offset =
                static_cast<std::size_t>(
                    group_offset +
                    index * GROUP_ROW_BYTES);
            const auto id =
                std::to_integer<std::uint8_t>(
                    payload[offset]);
            const auto locked =
                std::to_integer<std::uint8_t>(
                    payload[offset + 1]);
            if (id == 0 || ids[id] || locked > 1 ||
                !all_zero(payload.subspan(offset + 2, 2)) ||
                !all_zero(payload.subspan(offset + 28, 20))) {
                return corrupt<SelectionChapter>(
                    std::format(
                        "SELM group row {} has invalid ID, boolean, or "
                        "reserved bytes",
                        index),
                    "groups");
            }
            const float red = read_f32(payload, offset + 4);
            const float green = read_f32(payload, offset + 8);
            const float blue = read_f32(payload, offset + 12);
            if (!std::isfinite(red) ||
                !std::isfinite(green) ||
                !std::isfinite(blue)) {
                return corrupt<SelectionChapter>(
                    std::format(
                        "SELM group {} color is not finite", id),
                    "groups.color");
            }
            const auto name_offset =
                read_u64(payload, offset + 16);
            const auto name_bytes =
                read_u32(payload, offset + 24);
            std::uint64_t name_end = 0;
            if (name_offset % DATA_ALIGNMENT != 0 ||
                name_offset < data_offset ||
                !checked_add<std::uint64_t>(
                    name_offset, name_bytes, name_end) ||
                name_end > payload.size()) {
                return corrupt<SelectionChapter>(
                    std::format(
                        "SELM group {} name range is invalid", id),
                    "groups.name");
            }
            const std::string_view name(
                reinterpret_cast<const char*>(
                    payload.data() + name_offset),
                name_bytes);
            if (!valid_utf8(name)) {
                return corrupt<SelectionChapter>(
                    std::format(
                        "SELM group {} name is not valid UTF-8", id),
                    "groups.name");
            }
            ids[id] = true;
            result.groups_.push_back(
                lfs::core::SelectionGroup{
                    .id = id,
                    .name = std::string(name),
                    .color = {red, green, blue},
                    .count = 0,
                    .locked = locked != 0,
                });
            ranges.push_back(Range{
                .begin = name_offset,
                .end = name_end,
                .label = std::format("group {} name", id),
            });
        }
        if (auto group_result = validate_groups(
                result.groups_,
                active_group_id,
                next_group_id,
                false);
            !group_result) {
            return selection_error(
                lfs::ErrorCode::DataLoss,
                "The project contains invalid selection data.",
                std::string(group_result.error().detail()),
                "groups");
        }

        std::vector<SliceDescriptor> descriptors;
        descriptors.reserve(slice_count);
        for (std::uint32_t index = 0;
             index < slice_count;
             ++index) {
            const auto offset =
                static_cast<std::size_t>(
                    slice_offset +
                    index * SLICE_ROW_BYTES);
            lfs::core::Uuid uuid;
            std::memcpy(
                uuid.bytes.data(),
                payload.data() + offset,
                UUID_BYTES);
            const auto domain =
                static_cast<lfs::core::SelectionDomain>(
                    std::to_integer<std::uint8_t>(
                        payload[offset + 16]));
            const auto encoding =
                std::to_integer<std::uint8_t>(
                    payload[offset + 17]);
            const auto element_count =
                read_u64(payload, offset + 24);
            const auto nonzero_count =
                read_u64(payload, offset + 32);
            const auto slice_data_offset =
                read_u64(payload, offset + 40);
            const auto slice_data_bytes =
                read_u64(payload, offset + 48);
            if (encoding != static_cast<std::uint8_t>(
                                SelectionMaskEncoding::RawU8)) {
                return corrupt<SelectionChapter>(
                    encoding == 2
                        ? std::format(
                              "SELM slice row {} uses encoding 2, which "
                              "was withdrawn before release",
                              index)
                        : std::format(
                              "SELM slice row {} uses unknown encoding {}",
                              index,
                              encoding),
                    "slices.encoding");
            }
            if (uuid.is_nil() || !valid_domain(domain) ||
                !all_zero(payload.subspan(offset + 18, 6)) ||
                !all_zero(payload.subspan(offset + 56, 8)) ||
                element_count >
                    std::numeric_limits<std::size_t>::max() ||
                nonzero_count > element_count ||
                slice_data_offset % DATA_ALIGNMENT != 0 ||
                slice_data_offset < data_offset) {
                return corrupt<SelectionChapter>(
                    std::format(
                        "SELM slice row {} has invalid metadata",
                        index),
                    "slices");
            }
            std::uint64_t slice_end = 0;
            if (!checked_add(
                    slice_data_offset,
                    slice_data_bytes,
                    slice_end) ||
                slice_end > payload.size()) {
                return corrupt<SelectionChapter>(
                    std::format(
                        "SELM slice row {} range is out of bounds",
                        index),
                    "slices.data");
            }
            descriptors.push_back(SliceDescriptor{
                .node_uuid = uuid,
                .domain = domain,
                .element_count = element_count,
                .nonzero_count = nonzero_count,
                .data_offset = slice_data_offset,
                .data_bytes = slice_data_bytes,
            });
            ranges.push_back(Range{
                .begin = slice_data_offset,
                .end = slice_end,
                .label = std::format("slice {}", index),
            });
        }
        std::ranges::sort(
            descriptors,
            [](const SliceDescriptor& lhs,
               const SliceDescriptor& rhs) {
                if (lhs.node_uuid.bytes != rhs.node_uuid.bytes) {
                    return std::ranges::lexicographical_compare(
                        lhs.node_uuid.bytes,
                        rhs.node_uuid.bytes);
                }
                return static_cast<std::uint8_t>(lhs.domain) <
                       static_cast<std::uint8_t>(rhs.domain);
            });
        for (std::size_t index = 1;
             index < descriptors.size();
             ++index) {
            if (descriptors[index - 1].node_uuid ==
                    descriptors[index].node_uuid &&
                descriptors[index - 1].domain ==
                    descriptors[index].domain) {
                return corrupt<SelectionChapter>(
                    "SELM contains duplicate node/domain slice keys",
                    "slices");
            }
        }

        std::ranges::sort(
            ranges,
            {},
            &Range::begin);
        for (std::size_t index = 1;
             index < ranges.size();
             ++index) {
            if (ranges[index].begin <
                ranges[index - 1].end) {
                return corrupt<SelectionChapter>(
                    std::format(
                        "SELM data ranges '{}' and '{}' overlap",
                        ranges[index - 1].label,
                        ranges[index].label),
                    "data");
            }
        }

        for (const auto& descriptor : descriptors) {
            const auto bytes = payload.subspan(
                static_cast<std::size_t>(
                    descriptor.data_offset),
                static_cast<std::size_t>(
                    descriptor.data_bytes));
            if (descriptor.data_bytes !=
                descriptor.element_count) {
                return corrupt<SelectionChapter>(
                    "SELM raw slice byte count does not equal "
                    "element_count",
                    "slices.data");
            }
            std::uint64_t nonzero_count = 0;
            for (const auto byte : bytes) {
                const auto value =
                    std::to_integer<std::uint8_t>(byte);
                if (value != 0) {
                    ++nonzero_count;
                    if (!ids[value]) {
                        return corrupt<SelectionChapter>(
                            std::format(
                                "SELM raw slice uses unknown group ID {}",
                                value),
                            "slices.data");
                    }
                }
            }
            if (nonzero_count != descriptor.nonzero_count) {
                return corrupt<SelectionChapter>(
                    "SELM raw slice nonzero count is incorrect",
                    "slices.nonzero_count");
            }
        }

        result.selected_node_uuids_.reserve(
            selected_count);
        std::unordered_set<lfs::core::Uuid> selected_seen;
        for (std::uint32_t index = 0;
             index < selected_count;
             ++index) {
            lfs::core::Uuid uuid;
            std::memcpy(
                uuid.bytes.data(),
                payload.data() +
                    selected_offset +
                    index * UUID_BYTES,
                UUID_BYTES);
            if (uuid.is_nil() ||
                !selected_seen.emplace(uuid).second) {
                return corrupt<SelectionChapter>(
                    "SELM selected-node table contains nil or duplicate "
                    "UUIDs",
                    "selected_node_uuids");
            }
            result.selected_node_uuids_.push_back(uuid);
        }

        result.slices_.reserve(descriptors.size());
        for (const auto& descriptor : descriptors) {
            const auto bytes = payload.subspan(
                static_cast<std::size_t>(
                    descriptor.data_offset),
                static_cast<std::size_t>(
                    descriptor.data_bytes));
            std::vector<std::uint8_t> mask(
                static_cast<std::size_t>(
                    descriptor.element_count));
            if (!mask.empty()) {
                std::memcpy(
                    mask.data(), bytes.data(), mask.size());
            }
            result.slices_.push_back(SelectionMaskSlice{
                .node_uuid = descriptor.node_uuid,
                .domain = descriptor.domain,
                .encoding = SelectionMaskEncoding::RawU8,
                .mask = std::move(mask),
            });
        }
        return result;
    }

    lfs::Result<CapturedSelectionState> capture_selection_state(
        const lfs::core::Scene& scene,
        const std::span<const lfs::core::Uuid>
            selected_node_uuids,
        const std::span<const lfs::core::Uuid>
            omit_node_uuids) {
        const CaptureOmitFilter omit_filter(
            scene, omit_node_uuids);

        CapturedSelectionState state;
        const auto metadata =
            scene.captureSelectionStateMetadata();
        state.groups = metadata.groups;
        for (auto& group : state.groups) {
            group.count = 0;
        }
        state.active_group_id = metadata.active_group_id;
        state.next_group_id = metadata.next_group_id;

        for (const auto domain :
             {lfs::core::SelectionDomain::Splat,
              lfs::core::SelectionDomain::PointCloud}) {
            const auto slices =
                scene.capturePerNodeSelectionSlices(domain);
            for (const auto& [uuid, tensor] : slices) {
                if (omit_filter.omits(uuid)) {
                    continue;
                }
                if (!tensor.is_valid() ||
                    tensor.ndim() != 1) {
                    return selection_error(
                        lfs::ErrorCode::ContractViolation,
                        "The selection state cannot be captured.",
                        std::format(
                            "Selection slice {}:{} is not a rank-one "
                            "tensor",
                            uuid.to_string(),
                            static_cast<unsigned>(domain)),
                        "slices.mask");
                }
                const auto cpu =
                    tensor.cpu()
                        .to(lfs::core::DataType::UInt8)
                        .contiguous();
                std::vector<std::uint8_t> mask(cpu.numel());
                if (!mask.empty()) {
                    std::memcpy(
                        mask.data(),
                        cpu.data_ptr(),
                        mask.size());
                }
                state.slices.push_back(
                    SelectionMaskSlice{
                        .node_uuid = uuid,
                        .domain = domain,
                        .encoding = SelectionMaskEncoding::RawU8,
                        .mask = std::move(mask),
                    });
            }
        }
        state.selected_node_uuids.reserve(
            selected_node_uuids.size());
        for (const auto& uuid : selected_node_uuids) {
            const auto* node = scene.getNodeByUuid(uuid);
            if (node != nullptr &&
                (node->type == lfs::core::NodeType::KEYFRAME ||
                 node->type == lfs::core::NodeType::KEYFRAME_GROUP)) {
                continue;
            }
            if (omit_filter.omits(uuid)) {
                continue;
            }
            state.selected_node_uuids.push_back(uuid);
        }
        return state;
    }

    lfs::Result<SelectionChapter>
    materialize_selection_chapter(
        CapturedSelectionState state) {
        SelectionChapter chapter;
        if (auto result = chapter.set_groups(
                std::move(state.groups),
                state.active_group_id,
                state.next_group_id);
            !result) {
            return std::move(result).error();
        }
        for (auto& slice : state.slices) {
            if (auto result =
                    chapter.upsert_slice(
                        std::move(slice));
                !result) {
                return std::move(result).error();
            }
        }
        if (auto result =
                chapter.set_selected_node_uuids(
                    std::move(
                        state.selected_node_uuids));
            !result) {
            return std::move(result).error();
        }
        return chapter;
    }

    lfs::Result<SelectionChapter> capture_selection_chapter(
        const lfs::core::Scene& scene,
        const std::span<const lfs::core::Uuid>
            selected_node_uuids,
        const std::span<const lfs::core::Uuid>
            omit_node_uuids) {
        auto state =
            capture_selection_state(
                scene, selected_node_uuids,
                omit_node_uuids);
        if (!state) {
            return std::move(state).error();
        }
        return materialize_selection_chapter(
            std::move(*state));
    }

    lfs::Result<StagedSelectionChapter>
    stage_selection_chapter(
        const SelectionChapter& chapter,
        const lfs::core::Scene& topology) {
        if (auto groups = validate_groups(
                chapter.groups(),
                chapter.active_group_id(),
                chapter.next_group_id(),
                false);
            !groups) {
            return std::move(groups).error();
        }
        const auto ids = group_id_set(chapter.groups());

        struct NodeRange {
            lfs::core::SelectionDomain domain;
            std::size_t offset = 0;
            std::size_t count = 0;
        };
        std::unordered_map<lfs::core::Uuid, NodeRange>
            ranges;
        const auto nodes = topology.getNodes();
        ranges.reserve(nodes.size());
        std::array<std::size_t, 2> capacities{};
        for (const auto* node : nodes) {
            std::optional<lfs::core::SelectionDomain> domain;
            std::size_t count = 0;
            if (node->type == lfs::core::NodeType::SPLAT) {
                domain = lfs::core::SelectionDomain::Splat;
                count = node->gaussian_count.load(
                    std::memory_order_acquire);
            } else if (
                node->type ==
                    lfs::core::NodeType::POINTCLOUD &&
                node->point_cloud) {
                domain =
                    lfs::core::SelectionDomain::PointCloud;
                const auto point_count =
                    node->point_cloud->size();
                count =
                    point_count > 0
                        ? static_cast<std::size_t>(
                              point_count)
                        : 0;
            }
            if (!domain) {
                continue;
            }
            const std::size_t domain_index =
                *domain ==
                        lfs::core::SelectionDomain::Splat
                    ? 0
                    : 1;
            std::size_t end = 0;
            if (!checked_add(
                    capacities[domain_index], count,
                    end)) {
                return selection_error(
                    lfs::ErrorCode::ResourceExhausted,
                    "The saved selection topology is too large.",
                    "Selection domain capacity overflows size_t",
                    "slices.mask");
            }
            const auto [ignored, inserted] =
                ranges.emplace(
                    node->uuid,
                    NodeRange{
                        .domain = *domain,
                        .offset =
                            capacities[domain_index],
                        .count = count,
                    });
            (void)ignored;
            if (!inserted) {
                return selection_error(
                    lfs::ErrorCode::DataLoss,
                    "The project contains invalid selection data.",
                    "SCNG contains duplicate selection owner UUIDs",
                    "slices.node_uuid");
            }
            capacities[domain_index] = end;
        }

        lfs::core::Tensor splat_mask;
        lfs::core::Tensor point_cloud_mask;
        try {
            if (capacities[0] != 0) {
                splat_mask = lfs::core::Tensor::zeros(
                    {capacities[0]},
                    lfs::core::Device::CPU,
                    lfs::core::DataType::UInt8);
            }
            if (capacities[1] != 0) {
                point_cloud_mask =
                    lfs::core::Tensor::zeros(
                        {capacities[1]},
                        lfs::core::Device::CPU,
                        lfs::core::DataType::UInt8);
            }
        } catch (const std::exception& error) {
            // LFS-CENSUS-OK(empty-catch): normalize tensor allocation failures at the chapter boundary.
            return selection_error(
                lfs::ErrorCode::ResourceExhausted,
                "The saved selection tensors could not be allocated.",
                error.what(), "slices.mask");
        }

        std::array<std::unordered_set<lfs::core::Uuid>, 2>
            seen_slices;
        std::array<std::size_t, 2> selected_counts{};
        lfs::core::Scene::SelectionGroupCounts group_counts{};
        for (const auto& slice : chapter.slices()) {
            if (auto result =
                    validate_slice_values(slice, ids);
                !result) {
                return std::move(result).error();
            }
            const auto range = ranges.find(slice.node_uuid);
            if (range == ranges.end()) {
                return selection_error(
                    lfs::ErrorCode::FailedPrecondition,
                    "A saved selection refers to a missing scene node.",
                    std::format(
                        "SELM slice {}:{} has no SCNG owner",
                        slice.node_uuid.to_string(),
                        static_cast<unsigned>(slice.domain)),
                    "slices.node_uuid");
            }
            if (range->second.domain != slice.domain ||
                slice.mask.size() != range->second.count) {
                return selection_error(
                    lfs::ErrorCode::FailedPrecondition,
                    "A saved selection no longer matches its geometry.",
                    std::format(
                        "SELM slice {}:{} has {} elements; SCNG/Payload "
                        "expects {}",
                        slice.node_uuid.to_string(),
                        static_cast<unsigned>(slice.domain),
                        slice.mask.size(),
                        range->second.count),
                    "slices.mask");
            }

            const std::size_t domain_index =
                slice.domain ==
                        lfs::core::SelectionDomain::Splat
                    ? 0
                    : 1;
            if (!seen_slices[domain_index]
                     .insert(slice.node_uuid)
                     .second) {
                return selection_error(
                    lfs::ErrorCode::DataLoss,
                    "The project contains invalid selection data.",
                    "SELM contains duplicate slice keys",
                    "slices");
            }
            auto& mask =
                domain_index == 0
                    ? splat_mask
                    : point_cloud_mask;
            std::uint8_t* destination = nullptr;
            if (!slice.mask.empty()) {
                destination =
                    mask.ptr<std::uint8_t>() +
                    range->second.offset;
            }
            for (std::size_t index = 0;
                 index < slice.mask.size(); ++index) {
                const auto value = slice.mask[index];
                destination[index] = value;
                if (value != 0) {
                    ++selected_counts[domain_index];
                    ++group_counts[value];
                }
            }
        }

        std::unordered_set<lfs::core::Uuid> selected_nodes;
        selected_nodes.reserve(
            chapter.selected_node_uuids().size());
        for (const auto& uuid :
             chapter.selected_node_uuids()) {
            if (!ranges.contains(uuid) &&
                topology.getNodeByUuid(uuid) == nullptr) {
                return selection_error(
                    lfs::ErrorCode::FailedPrecondition,
                    "A saved node selection refers to a missing scene node.",
                    std::format(
                        "SELM selected node {} has no SCNG owner",
                        uuid.to_string()),
                    "selected_node_uuids");
            }
            if (!selected_nodes.insert(uuid).second) {
                return selection_error(
                    lfs::ErrorCode::DataLoss,
                    "The project contains invalid selection data.",
                    "SELM selected node UUIDs are duplicated",
                    "selected_node_uuids");
            }
        }

        StagedSelectionChapter staged{
            .state =
                {
                    .splat_mask = nullptr,
                    .point_cloud_mask = nullptr,
                    .groups = chapter.groups(),
                    .active_group_id =
                        chapter.active_group_id(),
                    .next_group_id =
                        chapter.next_group_id(),
                },
            .report =
                {
                    .selected_node_uuids =
                        chapter.selected_node_uuids(),
                },
        };
        std::array<bool, 256> used{};
        for (auto& group : staged.state.groups) {
            used[group.id] = true;
            group.count = group_counts[group.id];
        }
        std::uint8_t active = staged.state.active_group_id;
        if (active != 0 && !used[active]) {
            active = staged.state.groups.empty()
                         ? 0
                         : staged.state.groups.front().id;
            staged.report.repaired_group_metadata = true;
        }
        std::uint8_t next = staged.state.next_group_id;
        const bool next_valid =
            (next == 0 &&
             staged.state.groups.size() == 255) ||
            (next != 0 && !used[next]);
        if (!next_valid) {
            next = 0;
            for (std::uint16_t candidate = 1;
                 candidate <= 255;
                 ++candidate) {
                if (!used[candidate]) {
                    next = static_cast<std::uint8_t>(
                        candidate);
                    break;
                }
            }
            staged.report.repaired_group_metadata = true;
        }
        staged.state.active_group_id = active;
        staged.state.next_group_id = next;
        if (selected_counts[0] != 0) {
            staged.state.splat_mask =
                std::make_shared<lfs::core::Tensor>(
                    std::move(splat_mask));
            staged.state.has_splat_selection = true;
        }
        if (selected_counts[1] != 0) {
            staged.state.point_cloud_mask =
                std::make_shared<lfs::core::Tensor>(
                    std::move(point_cloud_mask));
            staged.state.has_point_cloud_selection = true;
        }
        return staged;
    }

    lfs::Result<SelectionHydrationReport>
    hydrate_selection_chapter(
        const SelectionChapter& chapter,
        lfs::core::Scene& scene) {
        auto staged =
            stage_selection_chapter(chapter, scene);
        if (!staged) {
            return std::move(staged).error();
        }
        auto report = std::move(staged->report);
        scene.installRestoreSelectionState(
            std::move(staged->state));
        return report;
    }

} // namespace lfs::io::project
