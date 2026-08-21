/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/capture_omit_filter.hpp"
#include "io/scene_chapter_adapter.hpp"
#include "io/selection_chapter.hpp"
#include "licht_test_support.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <format>
#include <memory>
#include <span>
#include <stdexcept>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace {

    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::NodeType;
    using lfs::core::Scene;
    using lfs::core::SelectionDomain;
    using lfs::core::Tensor;
    using lfs::core::Uuid;
    using lfs::io::project::CaptureOmitFilter;
    using lfs::io::project::SceneGraphChapter;
    using lfs::io::project::SceneNodeRecord;
    using lfs::io::project::SelectionChapter;
    using lfs::io::project::SelectionMaskEncoding;
    using lfs::io::project::SelectionMaskSlice;
    using lfs::test::licht::fixed_uuid;

    std::shared_ptr<lfs::core::PointCloud> point_cloud(
        const std::size_t count) {
        auto result =
            std::make_shared<lfs::core::PointCloud>();
        result->means =
            Tensor::zeros(
                lfs::core::TensorShape{count, 3},
                Device::CPU,
                DataType::Float32);
        return result;
    }

    void add_splat_placeholder(
        Scene& scene,
        const Uuid& uuid,
        const std::size_t count,
        const std::string_view name) {
        const auto id = scene.restoreNodeWithUuid(
            Scene::RestoreNodeDesc{
                .uuid = uuid,
                .type = NodeType::SPLAT,
                .name = std::string(name),
                .gaussian_count = count,
            });
        ASSERT_NE(id, lfs::core::NULL_NODE);
    }

    void add_point_cloud(
        Scene& scene,
        const Uuid& uuid,
        const std::size_t count,
        const std::string_view name) {
        const auto id = scene.restoreNodeWithUuid(
            Scene::RestoreNodeDesc{
                .uuid = uuid,
                .type = NodeType::POINTCLOUD,
                .name = std::string(name),
                .point_cloud = point_cloud(count),
            });
        ASSERT_NE(id, lfs::core::NULL_NODE);
    }

    Tensor mask_tensor(
        const std::initializer_list<int> values) {
        return Tensor::from_vector(
                   values,
                   lfs::core::TensorShape{values.size()},
                   Device::CPU)
            .to(DataType::UInt8);
    }

    std::vector<std::uint8_t> mask_values(
        const Scene& scene,
        const SelectionDomain domain) {
        const auto mask = scene.getSelectionMask(domain);
        if (!mask) {
            return {};
        }
        const auto cpu =
            mask->cpu().to(DataType::UInt8).contiguous();
        const auto* begin = cpu.ptr<std::uint8_t>();
        return {begin, begin + cpu.numel()};
    }

    std::uint64_t read_u64(
        const std::span<const std::byte> bytes,
        const std::size_t offset) {
        std::uint64_t result = 0;
        for (std::size_t byte = 0; byte < 8; ++byte) {
            result |= static_cast<std::uint64_t>(
                          std::to_integer<std::uint8_t>(
                              bytes[offset + byte]))
                      << (byte * 8u);
        }
        return result;
    }

    void write_u64(
        const std::span<std::byte> bytes,
        const std::size_t offset,
        const std::uint64_t value) {
        for (std::size_t byte = 0; byte < 8; ++byte) {
            bytes[offset + byte] =
                static_cast<std::byte>(
                    value >> (byte * 8u));
        }
    }

#if !defined(LFS_FORMAT_TEST_TARGET)
    TEST(SelectionChapterTest,
         BothDomainsGroupsAndSelectedNodeOrderRoundTrip) {
        const Uuid splat_uuid = fixed_uuid(1);
        const Uuid point_uuid = fixed_uuid(2);
        Scene source;
        add_splat_placeholder(
            source, splat_uuid, 6, "edited splat");
        add_point_cloud(
            source, point_uuid, 5, "point cloud");
        const auto second_group =
            source.addSelectionGroup(
                "Details", {0.2f, 0.4f, 0.8f});
        ASSERT_EQ(second_group, 2);
        source.setSelectionGroupLocked(1, true);
        source.setActiveSelectionGroup(2);
        source.applyPerNodeSelectionSlices(
            SelectionDomain::Splat,
            {{splat_uuid,
              mask_tensor({0, 1, 1, 0, 2, 2})}});
        source.applyPerNodeSelectionSlices(
            SelectionDomain::PointCloud,
            {{point_uuid,
              mask_tensor({2, 0, 1, 0, 2})}});

        const std::array selected = {
            point_uuid, splat_uuid};
        auto chapter =
            lfs::io::project::capture_selection_chapter(
                source, selected);
        ASSERT_TRUE(chapter)
            << lfs::format_for_developer(chapter.error());
        ASSERT_EQ(chapter->slices().size(), 2);

        auto encoded =
            lfs::io::project::encode_selection_chapter(
                *chapter);
        ASSERT_TRUE(encoded)
            << lfs::format_for_developer(encoded.error());
        auto decoded =
            lfs::io::project::decode_selection_chapter(
                *encoded);
        ASSERT_TRUE(decoded)
            << lfs::format_for_developer(decoded.error());
        ASSERT_EQ(decoded->slices().size(), 2);
        EXPECT_EQ(
            decoded->slices()[0].encoding,
            SelectionMaskEncoding::RawU8);
        EXPECT_EQ(
            decoded->selected_node_uuids(),
            (std::vector<Uuid>{point_uuid, splat_uuid}));

        Scene restored;
        add_splat_placeholder(
            restored, splat_uuid, 6, "edited splat");
        add_point_cloud(
            restored, point_uuid, 5, "point cloud");
        auto hydration =
            lfs::io::project::hydrate_selection_chapter(
                *decoded, restored);
        ASSERT_TRUE(hydration)
            << lfs::format_for_developer(hydration.error());
        EXPECT_FALSE(hydration->repaired_group_metadata);
        EXPECT_EQ(
            hydration->selected_node_uuids,
            (std::vector<Uuid>{point_uuid, splat_uuid}));
        EXPECT_EQ(
            mask_values(restored, SelectionDomain::Splat),
            (std::vector<std::uint8_t>{0, 1, 1, 0, 2, 2}));
        EXPECT_EQ(
            mask_values(
                restored,
                SelectionDomain::PointCloud),
            (std::vector<std::uint8_t>{2, 0, 1, 0, 2}));

        ASSERT_EQ(restored.getSelectionGroups().size(), 2);
        EXPECT_EQ(restored.getSelectionGroups()[0].count, 3u);
        EXPECT_EQ(restored.getSelectionGroups()[1].count, 4u);
        EXPECT_TRUE(restored.getSelectionGroups()[0].locked);
        EXPECT_EQ(restored.getActiveSelectionGroup(), 2);
    }

    TEST(SelectionChapterTest,
         RawEncodingRemainsDecodableAndInvalidMetadataIsRepaired) {
        const Uuid splat_uuid = fixed_uuid(11);
        Scene source;
        add_splat_placeholder(source, splat_uuid, 4, "splat");
        source.applyPerNodeSelectionSlices(
            SelectionDomain::Splat,
            {{splat_uuid, mask_tensor({1, 0, 1, 0})}});
        auto chapter =
            lfs::io::project::capture_selection_chapter(
                source, {});
        ASSERT_TRUE(chapter)
            << lfs::format_for_developer(chapter.error());
        auto raw =
            lfs::io::project::encode_selection_chapter(
                *chapter);
        ASSERT_TRUE(raw)
            << lfs::format_for_developer(raw.error());
        (*raw)[20] = static_cast<std::byte>(99);
        (*raw)[21] = static_cast<std::byte>(1);

        auto decoded =
            lfs::io::project::decode_selection_chapter(*raw);
        ASSERT_TRUE(decoded)
            << lfs::format_for_developer(decoded.error());
        ASSERT_EQ(decoded->slices().size(), 1);
        EXPECT_EQ(
            decoded->slices()[0].encoding,
            SelectionMaskEncoding::RawU8);

        Scene restored;
        add_splat_placeholder(
            restored, splat_uuid, 4, "splat");
        auto hydration =
            lfs::io::project::hydrate_selection_chapter(
                *decoded, restored);
        ASSERT_TRUE(hydration)
            << lfs::format_for_developer(hydration.error());
        EXPECT_TRUE(hydration->repaired_group_metadata);
        EXPECT_EQ(restored.getActiveSelectionGroup(), 1);
        const auto metadata =
            restored.captureSelectionStateMetadata();
        EXPECT_EQ(metadata.next_group_id, 2);
    }

    TEST(SelectionChapterTest,
         CaptureOmitsRequestedNodesFromSelectionAndSlices) {
        const Uuid node_a = fixed_uuid(41);
        const Uuid node_b = fixed_uuid(42);
        Scene scene;
        add_splat_placeholder(
            scene, node_a, 3, "splat a");
        add_splat_placeholder(
            scene, node_b, 4, "splat b");
        scene.applyPerNodeSelectionSlices(
            SelectionDomain::Splat,
            {
                {node_a, mask_tensor({1, 0, 1})},
                {node_b, mask_tensor({0, 1, 0, 1})},
            });

        const std::array selected = {node_a, node_b};
        auto full =
            lfs::io::project::capture_selection_chapter(
                scene, selected);
        ASSERT_TRUE(full)
            << lfs::format_for_developer(full.error());

        const std::array omit = {node_b};
        auto omitted =
            lfs::io::project::capture_selection_chapter(
                scene, selected, omit);
        ASSERT_TRUE(omitted)
            << lfs::format_for_developer(omitted.error());
        ASSERT_EQ(omitted->slices().size(), 1);
        EXPECT_EQ(
            omitted->slices()[0].node_uuid, node_a);
        EXPECT_EQ(
            omitted->selected_node_uuids(),
            (std::vector<Uuid>{node_a}));
        ASSERT_EQ(
            omitted->groups().size(),
            full->groups().size());
        for (std::size_t i = 0;
             i < omitted->groups().size(); ++i) {
            EXPECT_EQ(
                omitted->groups()[i].id,
                full->groups()[i].id);
            EXPECT_EQ(
                omitted->groups()[i].name,
                full->groups()[i].name);
            EXPECT_EQ(
                omitted->groups()[i].count,
                full->groups()[i].count);
            EXPECT_EQ(
                omitted->groups()[i].locked,
                full->groups()[i].locked);
            EXPECT_EQ(
                omitted->groups()[i].color,
                full->groups()[i].color);
        }
        EXPECT_EQ(
            omitted->active_group_id(),
            full->active_group_id());
        EXPECT_EQ(
            omitted->next_group_id(),
            full->next_group_id());
    }

    TEST(SelectionChapterTest,
         CaptureOmitsRequestedSubtreeFromSelectionAndSlices) {
        const Uuid parent = fixed_uuid(51);
        const Uuid sibling = fixed_uuid(52);
        Scene scene;
        add_splat_placeholder(
            scene, parent, 3, "parent");
        add_splat_placeholder(
            scene, sibling, 4, "sibling");
        const auto parent_id =
            scene.getNodeIdByUuid(parent);
        ASSERT_NE(parent_id, lfs::core::NULL_NODE);
        const auto child_id =
            scene.addCropBox("box", parent_id);
        ASSERT_NE(child_id, lfs::core::NULL_NODE);
        const Uuid child = scene.getNodeUuid(child_id);
        scene.applyPerNodeSelectionSlices(
            SelectionDomain::Splat,
            {
                {parent, mask_tensor({1, 0, 1})},
                {sibling, mask_tensor({0, 1, 0, 1})},
            });

        const std::array selected = {
            parent, child, sibling};
        auto full =
            lfs::io::project::capture_selection_chapter(
                scene, selected);
        ASSERT_TRUE(full)
            << lfs::format_for_developer(full.error());
        EXPECT_EQ(
            full->selected_node_uuids(),
            (std::vector<Uuid>{parent, child, sibling}));

        const std::array omit = {parent};
        auto omitted =
            lfs::io::project::capture_selection_chapter(
                scene, selected, omit);
        ASSERT_TRUE(omitted)
            << lfs::format_for_developer(omitted.error());
        ASSERT_EQ(omitted->slices().size(), 1);
        EXPECT_EQ(
            omitted->slices()[0].node_uuid, sibling);
        EXPECT_EQ(
            omitted->selected_node_uuids(),
            (std::vector<Uuid>{sibling}));
        ASSERT_EQ(
            omitted->groups().size(),
            full->groups().size());
        for (std::size_t i = 0;
             i < omitted->groups().size(); ++i) {
            EXPECT_EQ(
                omitted->groups()[i].id,
                full->groups()[i].id);
            EXPECT_EQ(
                omitted->groups()[i].name,
                full->groups()[i].name);
            EXPECT_EQ(
                omitted->groups()[i].count,
                full->groups()[i].count);
            EXPECT_EQ(
                omitted->groups()[i].locked,
                full->groups()[i].locked);
            EXPECT_EQ(
                omitted->groups()[i].color,
                full->groups()[i].color);
        }
        EXPECT_EQ(
            omitted->active_group_id(),
            full->active_group_id());
        EXPECT_EQ(
            omitted->next_group_id(),
            full->next_group_id());
    }

    TEST(CaptureOmitFilterTest,
         ClosesRootsOverDescendantsAndKeepsUnresolvableUuids) {
        const Uuid parent = fixed_uuid(71);
        const Uuid child = fixed_uuid(72);
        const Uuid grandchild = fixed_uuid(73);
        const Uuid sibling = fixed_uuid(74);
        const Uuid missing = fixed_uuid(75);
        Scene scene;
        const auto parent_id = scene.restoreNodeWithUuid(
            Scene::RestoreNodeDesc{
                .uuid = parent,
                .type = NodeType::GROUP,
                .name = "parent",
            });
        ASSERT_NE(parent_id, lfs::core::NULL_NODE);
        const auto child_id = scene.restoreNodeWithUuid(
            Scene::RestoreNodeDesc{
                .uuid = child,
                .type = NodeType::GROUP,
                .name = "child",
                .parent = parent_id,
            });
        ASSERT_NE(child_id, lfs::core::NULL_NODE);
        ASSERT_NE(
            scene.restoreNodeWithUuid(
                Scene::RestoreNodeDesc{
                    .uuid = grandchild,
                    .type = NodeType::GROUP,
                    .name = "grandchild",
                    .parent = child_id,
                }),
            lfs::core::NULL_NODE);
        ASSERT_NE(
            scene.restoreNodeWithUuid(
                Scene::RestoreNodeDesc{
                    .uuid = sibling,
                    .type = NodeType::GROUP,
                    .name = "sibling",
                }),
            lfs::core::NULL_NODE);

        const std::array roots = {parent, missing};
        const CaptureOmitFilter filter(scene, roots);
        EXPECT_TRUE(filter.omits(parent));
        EXPECT_TRUE(filter.omits(child));
        EXPECT_TRUE(filter.omits(grandchild));
        EXPECT_TRUE(filter.omits(missing));
        EXPECT_FALSE(filter.omits(sibling));
    }

    TEST(SceneGraphChapterTest,
         CaptureWithPreClosedOmitSetMatchesOmitRoots) {
        const Uuid parent = fixed_uuid(81);
        const Uuid child = fixed_uuid(82);
        const Uuid grandchild = fixed_uuid(83);
        const Uuid sibling = fixed_uuid(84);
        Scene scene;
        const auto parent_id = scene.restoreNodeWithUuid(
            Scene::RestoreNodeDesc{
                .uuid = parent,
                .type = NodeType::GROUP,
                .name = "parent",
            });
        ASSERT_NE(parent_id, lfs::core::NULL_NODE);
        const auto child_id = scene.restoreNodeWithUuid(
            Scene::RestoreNodeDesc{
                .uuid = child,
                .type = NodeType::GROUP,
                .name = "child",
                .parent = parent_id,
            });
        ASSERT_NE(child_id, lfs::core::NULL_NODE);
        ASSERT_NE(
            scene.restoreNodeWithUuid(
                Scene::RestoreNodeDesc{
                    .uuid = grandchild,
                    .type = NodeType::GROUP,
                    .name = "grandchild",
                    .parent = child_id,
                }),
            lfs::core::NULL_NODE);
        ASSERT_NE(
            scene.restoreNodeWithUuid(
                Scene::RestoreNodeDesc{
                    .uuid = sibling,
                    .type = NodeType::GROUP,
                    .name = "sibling",
                }),
            lfs::core::NULL_NODE);

        const std::array roots = {parent};
        const std::array closed = {
            parent, child, grandchild};
        auto from_roots =
            lfs::io::project::capture_scene_graph(
                scene,
                lfs::io::project::ScenePayloadBindings{},
                roots);
        ASSERT_TRUE(from_roots)
            << lfs::format_for_developer(
                   from_roots.error());
        auto from_closed =
            lfs::io::project::capture_scene_graph(
                scene,
                lfs::io::project::ScenePayloadBindings{},
                closed);
        ASSERT_TRUE(from_closed)
            << lfs::format_for_developer(
                   from_closed.error());
        EXPECT_EQ(
            from_roots->to_bytes(),
            from_closed->to_bytes());
        const auto nodes = from_roots->nodes();
        ASSERT_TRUE(nodes);
        std::vector<Uuid> uuids;
        uuids.reserve(nodes->size());
        for (const auto& node : *nodes) {
            uuids.push_back(node.uuid);
        }
        EXPECT_EQ(uuids, (std::vector<Uuid>{sibling}));
    }

    TEST(SelectionChapterTest,
         PartialLoadMergeDropsSlicesAbsentFromCapturedSceneGraph) {
        const Uuid omitted = fixed_uuid(91);
        const Uuid kept = fixed_uuid(92);
        SelectionChapter previous;
        ASSERT_TRUE(previous.upsert_slice(
            SelectionMaskSlice{
                .node_uuid = omitted,
                .domain = SelectionDomain::Splat,
                .mask = {0, 0, 0},
            }));
        ASSERT_TRUE(previous.upsert_slice(
            SelectionMaskSlice{
                .node_uuid = kept,
                .domain = SelectionDomain::Splat,
                .mask = {0, 0},
            }));

        SceneGraphChapter captured_scene;
        ASSERT_TRUE(captured_scene.upsert_node(
            SceneNodeRecord{
                .uuid = kept,
                .type = "group",
                .name = "kept",
                .child_order = 0,
            }));
        const auto captured_nodes = captured_scene.nodes();
        ASSERT_TRUE(captured_nodes);
        std::unordered_set<Uuid> keep;
        keep.reserve(captured_nodes->size());
        for (const auto& node : *captured_nodes) {
            keep.insert(node.uuid);
        }

        const auto old_slices = previous.slices();
        for (const auto& slice : old_slices) {
            if (!keep.contains(slice.node_uuid)) {
                static_cast<void>(previous.remove_slice(
                    slice.node_uuid, slice.domain));
            }
        }
        ASSERT_EQ(previous.slices().size(), 1u);
        EXPECT_EQ(previous.slices()[0].node_uuid, kept);
    }
#endif

    TEST(SelectionChapterTest,
         WithdrawnEncodingTruncationAndOverlappingRangesAreRejected) {
        SelectionChapter chapter;
        ASSERT_TRUE(chapter.set_groups(
            {lfs::core::SelectionGroup{
                .id = 1,
                .name = "Group",
                .color = {1.0f, 0.0f, 0.0f},
                .count = 0,
                .locked = false,
            }},
            1,
            2));
        ASSERT_TRUE(chapter.upsert_slice(
            SelectionMaskSlice{
                .node_uuid = fixed_uuid(21),
                .domain = SelectionDomain::Splat,
                .mask = {0, 1, 0, 1, 1, 0},
            }));
        ASSERT_TRUE(chapter.upsert_slice(
            SelectionMaskSlice{
                .node_uuid = fixed_uuid(22),
                .domain = SelectionDomain::PointCloud,
                .mask = {1, 0, 0},
            }));
        auto encoded =
            lfs::io::project::encode_selection_chapter(
                chapter);
        ASSERT_TRUE(encoded)
            << lfs::format_for_developer(encoded.error());

        const auto slice_table = read_u64(*encoded, 32);

        auto withdrawn = *encoded;
        withdrawn[slice_table + 17] = std::byte{2};
        auto withdrawn_result =
            lfs::io::project::decode_selection_chapter(
                withdrawn);
        ASSERT_FALSE(withdrawn_result);
        EXPECT_EQ(
            withdrawn_result.error().code(),
            lfs::ErrorCode::DataLoss);
        EXPECT_NE(
            withdrawn_result.error().detail().find(
                "withdrawn before release"),
            std::string_view::npos);

        auto truncated = *encoded;
        const auto data_bytes =
            read_u64(truncated, slice_table + 48);
        ASSERT_GT(data_bytes, 0u);
        write_u64(
            truncated,
            slice_table + 48,
            data_bytes - 1);
        auto truncated_result =
            lfs::io::project::decode_selection_chapter(
                truncated);
        EXPECT_FALSE(truncated_result);

        auto overlap = *encoded;
        const auto first_data =
            read_u64(overlap, slice_table + 40);
        write_u64(
            overlap,
            slice_table + 64 + 40,
            first_data);
        auto overlap_result =
            lfs::io::project::decode_selection_chapter(
                overlap);
        EXPECT_FALSE(overlap_result);
    }

#if !defined(LFS_FORMAT_TEST_TARGET)
    TEST(SelectionChapterTest,
         PointCloudTopologyRemovalKeepsRemainingUuidSlice) {
        const Uuid first_uuid = fixed_uuid(31);
        const Uuid second_uuid = fixed_uuid(32);
        Scene scene;
        add_point_cloud(scene, first_uuid, 3, "first");
        add_point_cloud(scene, second_uuid, 4, "second");
        scene.applyPerNodeSelectionSlices(
            SelectionDomain::PointCloud,
            {
                {first_uuid, mask_tensor({1, 0, 1})},
                {second_uuid, mask_tensor({0, 1, 0, 1})},
            });

        scene.removeNodeById(
            scene.getNodeIdByUuid(first_uuid));
        EXPECT_EQ(
            scene.getSelectionCapacity(
                SelectionDomain::PointCloud),
            4u);
        EXPECT_EQ(
            mask_values(
                scene, SelectionDomain::PointCloud),
            (std::vector<std::uint8_t>{0, 1, 0, 1}));
        const auto slices =
            scene.capturePerNodeSelectionSlices(
                SelectionDomain::PointCloud);
        EXPECT_FALSE(slices.contains(first_uuid));
        EXPECT_TRUE(slices.contains(second_uuid));
    }
#endif

} // namespace
