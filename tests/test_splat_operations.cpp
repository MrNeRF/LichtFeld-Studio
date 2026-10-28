/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/splat_operations.hpp"

#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <regex>
#include <set>

namespace {
    using namespace lfs::core;

    class SplatOperationsTest : public ::testing::Test {
    protected:
        SceneNode owner;
        SceneNode element;

        void SetUp() override {
            owner.id = 4;
            owner.uuid.bytes[0] = 1;
            owner.type = NodeType::SPLAT;
            owner.children = {9};
            element.id = 9;
            element.uuid.bytes[0] = 2;
            element.parent_id = owner.id;
            element.type = NodeType::CROPBOX;
            element.cropbox = std::make_unique<CropBoxData>();
        }
    };

    TEST(SplatOperationCatalogTest, OnlySplatOwnersAndExistingCapabilitiesAreAdvertised) {
        std::set<std::string_view> identifiers;
        std::set<SplatOperationKind> kinds;
        for (const auto& descriptor : splatOperationDescriptors()) {
            EXPECT_TRUE(identifiers.insert(descriptor.identifier).second);
            EXPECT_TRUE(kinds.insert(descriptor.kind).second);
            EXPECT_EQ(splatOperationDescriptor(descriptor.kind), &descriptor);
            EXPECT_EQ(descriptor.supportsOwner(NodeType::SPLAT),
                      descriptor.storage != SplatOperationStorage::CommandOnly &&
                          descriptor.scope == SplatOperationScope::Object);
            for (const auto type : {NodeType::GROUP, NodeType::DATASET, NodeType::POINTCLOUD,
                                    NodeType::MESH, NodeType::CAMERA, NodeType::PLY_SEQUENCE})
                EXPECT_FALSE(descriptor.supportsOwner(type));
            EXPECT_FALSE(descriptor.capabilities.supports_group_owner);
            EXPECT_FALSE(descriptor.capabilities.reorder);
            if (descriptor.storage == SplatOperationStorage::SceneVolume) {
                EXPECT_TRUE(descriptor.capabilities.set_enabled);
                EXPECT_TRUE(descriptor.capabilities.edit_parameters);
                EXPECT_TRUE(descriptor.capabilities.show_helper);
            } else {
                EXPECT_FALSE(descriptor.capabilities.set_enabled);
                EXPECT_FALSE(descriptor.capabilities.edit_parameters);
                EXPECT_FALSE(descriptor.capabilities.show_helper);
            }
        }
        EXPECT_EQ(splatOperationDescriptors().size(), 32);
        EXPECT_EQ(splatOperationDescriptor(static_cast<SplatOperationKind>(99)), nullptr);
        EXPECT_EQ(splatOperationDescriptor(SplatOperationKind::Simplify)->storage, SplatOperationStorage::CommandOnly);
        EXPECT_TRUE(splatOperationDescriptor(SplatOperationKind::Collision)->capabilities.export_result);
        EXPECT_FALSE(splatOperationDescriptor(SplatOperationKind::Simplify)->capabilities.export_result);
    }

    TEST(SplatOperationCatalogTest, CoversEveryNativeSplatToolbarToolWithoutTurningCommandsIntoModifiers) {
        std::set<std::string_view> tools;
        for (const auto& descriptor : splatOperationDescriptors()) {
            if (!descriptor.tool_identifier.empty())
                tools.insert(descriptor.tool_identifier);
            if (descriptor.category == SplatOperationCategory::Selection ||
                descriptor.category == SplatOperationCategory::SelectionFilter ||
                descriptor.category == SplatOperationCategory::Transform ||
                descriptor.category == SplatOperationCategory::DirectEdit ||
                descriptor.category == SplatOperationCategory::SceneOrganization) {
                EXPECT_EQ(descriptor.storage, SplatOperationStorage::CommandOnly);
                EXPECT_FALSE(descriptor.capabilities.set_enabled);
                EXPECT_FALSE(descriptor.capabilities.reorder);
            }
        }
        // Compare against the actual toolbar definitions, so a newly registered
        // native tool cannot silently disappear from this classification.
        const auto path = std::filesystem::path(PROJECT_ROOT_PATH) / "src/python/lfs_plugins/tool_defs/builtin.py";
        std::ifstream input(path);
        ASSERT_TRUE(input) << path;
        const std::string source{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
        const std::regex pattern{R"re(id\s*=\s*"(builtin\.[a-z_]+)")re"};
        std::set<std::string> expected;
        for (auto it = std::sregex_iterator(source.begin(), source.end(), pattern); it != std::sregex_iterator(); ++it)
            expected.insert((*it)[1].str());
        ASSERT_FALSE(expected.empty());
        const std::set<std::string> actual(tools.begin(), tools.end());
        EXPECT_EQ(actual, expected);
    }

    TEST(SplatOperationCatalogTest, GlobalSelectionAndViewportStateCannotBeOwnedByASplat) {
        for (const auto kind : {SplatOperationKind::Select, SplatOperationKind::SelectAll,
                                SplatOperationKind::SelectNone, SplatOperationKind::SelectInvert,
                                SplatOperationKind::SelectGrow, SplatOperationKind::SelectShrink,
                                SplatOperationKind::SelectByOpacity, SplatOperationKind::SelectByScale,
                                SplatOperationKind::SelectionGroups, SplatOperationKind::Mirror,
                                SplatOperationKind::DeleteSelection, SplatOperationKind::CopySelection,
                                SplatOperationKind::CutSelection, SplatOperationKind::PasteSelection,
                                SplatOperationKind::GroupSelection}) {
            const auto* descriptor = splatOperationDescriptor(kind);
            ASSERT_NE(descriptor, nullptr);
            EXPECT_EQ(descriptor->scope, SplatOperationScope::SceneSelection);
            EXPECT_FALSE(descriptor->supportsOwner(NodeType::SPLAT));
        }
        for (const auto kind : {SplatOperationKind::SelectionDepthFilter, SplatOperationKind::SelectionCropFilter}) {
            const auto* descriptor = splatOperationDescriptor(kind);
            ASSERT_NE(descriptor, nullptr);
            EXPECT_EQ(descriptor->scope, SplatOperationScope::ViewportTool);
            EXPECT_FALSE(descriptor->supportsOwner(NodeType::SPLAT));
        }
        const auto* simplify = splatOperationDescriptor(SplatOperationKind::Simplify);
        EXPECT_EQ(simplify->scope, SplatOperationScope::Object);
        EXPECT_FALSE(simplify->supportsOwner(NodeType::SPLAT)); // Source target, not a persisted recipe.
        EXPECT_EQ(splatOperationDescriptor(SplatOperationKind::WriteGaussianField)->scope, SplatOperationScope::Object);
    }

    TEST(SplatOperationCatalogTest, VolumeActionsMatchExistingCommandsAndAreNotInventedForOtherTools) {
        for (const auto& descriptor : splatOperationDescriptors()) {
            const bool volume = descriptor.storage == SplatOperationStorage::SceneVolume;
            const auto& caps = descriptor.capabilities;
            EXPECT_EQ(caps.fit, volume);
            EXPECT_EQ(caps.reset, volume);
            EXPECT_EQ(caps.invert, volume);
            EXPECT_EQ(caps.apply_to_geometry, volume);
            EXPECT_EQ(caps.remove_element, volume || descriptor.kind == SplatOperationKind::Collision);
            if (volume) {
                EXPECT_EQ(descriptor.scope, SplatOperationScope::Object);
                EXPECT_TRUE(descriptor.supportsOwner(NodeType::SPLAT));
            }
        }
    }

    TEST_F(SplatOperationsTest, CropSnapshotPreservesPayloadAndSeparatesVisibilityFromEnabled) {
        element.cropbox->min = {-3, -2, -1};
        element.cropbox->max = {4, 5, 6};
        element.cropbox->inverse = true;
        element.cropbox->enabled = true;
        element.visible.setQuiet(false);
        element.locked.setQuiet(true);
        glm::mat4 transform{1};
        transform[3] = {7, 8, 9, 1};
        element.local_transform.setQuiet(transform);
        const auto result = readSplatOperation(owner, element);
        ASSERT_TRUE(result);
        EXPECT_EQ(result->owner_uuid, owner.uuid);
        EXPECT_EQ(result->element_uuid, element.uuid);
        EXPECT_EQ(result->local_transform, transform);
        EXPECT_FALSE(result->node_visible);
        EXPECT_TRUE(result->node_locked);
        EXPECT_EQ(result->enabled, true);
        const auto& data = std::get<CropBoxData>(result->parameters);
        EXPECT_EQ(data.min, element.cropbox->min);
        EXPECT_EQ(data.max, element.cropbox->max);
        EXPECT_TRUE(data.inverse);
        EXPECT_TRUE(element.cropbox->enabled);
        // Changing the scene afterwards cannot mutate an already returned snapshot.
        element.cropbox->min = {0, 0, 0};
        EXPECT_EQ(data.min, glm::vec3(-3, -2, -1));
    }

    TEST_F(SplatOperationsTest, EllipsoidUsesItsOwnPayloadRatherThanOtherPresentFields) {
        element.type = NodeType::ELLIPSOID;
        element.ellipsoid = std::make_unique<EllipsoidData>();
        element.ellipsoid->radii = {2, 3, 4};
        element.ellipsoid->inverse = true;
        element.ellipsoid->enabled = false;
        element.cropbox->enabled = true;
        const auto result = readSplatOperation(owner, element);
        ASSERT_TRUE(result);
        EXPECT_EQ(result->kind, SplatOperationKind::Ellipsoid);
        EXPECT_EQ(result->enabled, false);
        EXPECT_EQ(std::get<EllipsoidData>(result->parameters).radii, glm::vec3(2, 3, 4));
        EXPECT_TRUE(std::get<EllipsoidData>(result->parameters).inverse);
    }

    TEST_F(SplatOperationsTest, PolygonSnapshotOwnsItsOutlineAndSurvivesPayloadRemoval) {
        element.type = NodeType::CROP_POLYGON;
        element.crop_polygon = std::make_unique<CropPolygonData>();
        element.crop_polygon->vertices = {{-1, -1}, {1, -1}, {0, 1}};
        element.crop_polygon->half_depth = 3;
        element.crop_polygon->inverse = true;
        const auto result = readSplatOperation(owner, element);
        ASSERT_TRUE(result);
        element.crop_polygon.reset();
        const auto& data = std::get<CropPolygonData>(result->parameters);
        EXPECT_EQ(data.vertices.size(), 3);
        EXPECT_EQ(data.vertices.back(), glm::vec2(0, 1));
        EXPECT_EQ(data.half_depth, 3);
        EXPECT_TRUE(data.inverse);
    }

    TEST_F(SplatOperationsTest, CollisionExposesCompletedSettingsWithoutSyntheticRecipeOrEnableFlag) {
        element.type = NodeType::COLLISION;
        auto volume = std::make_shared<CollisionVolume>();
        volume->settings = {.voxel_size = .2f, .opacity = .3f, .max_gaussian_area = .4f, .auto_resolution = true, .cleanup = false};
        volume->words = {0, 1, 2};
        element.collision = volume;
        const auto result = readSplatOperation(owner, element);
        ASSERT_TRUE(result);
        EXPECT_EQ(result->kind, SplatOperationKind::Collision);
        EXPECT_FALSE(result->enabled.has_value());
        const auto& settings = std::get<CollisionSettings>(result->parameters);
        EXPECT_EQ(settings.voxel_size, .2f);
        EXPECT_EQ(settings.opacity, .3f);
        EXPECT_EQ(settings.max_gaussian_area, .4f);
        EXPECT_TRUE(settings.auto_resolution);
        EXPECT_FALSE(settings.cleanup);
        EXPECT_EQ(volume.use_count(), 2); // The snapshot does not retain the volume.
        element.collision.reset();
        EXPECT_EQ(settings.voxel_size, .2f);
    }

    TEST_F(SplatOperationsTest, RejectsGroupsAndOtherOwnersWithoutSelectionFallback) {
        for (const auto type : {NodeType::GROUP, NodeType::POINTCLOUD, NodeType::DATASET, NodeType::MESH}) {
            owner.type = type;
            const auto result = readSplatOperation(owner, element);
            ASSERT_FALSE(result);
            EXPECT_EQ(result.error(), SplatOperationReadError::UnsupportedOwner);
        }
    }

    TEST_F(SplatOperationsTest, RejectsWrongOwnerAndInconsistentChildListsEvenWithMatchingNames) {
        owner.name = element.name = "same";
        element.parent_id = 99;
        EXPECT_EQ(readSplatOperation(owner, element).error(), SplatOperationReadError::NotDirectChild);
        element.parent_id = owner.id;
        owner.children.clear();
        EXPECT_EQ(readSplatOperation(owner, element).error(), SplatOperationReadError::NotDirectChild);
    }

    TEST_F(SplatOperationsTest, RejectsMissingPayloadForEverySupportedElement) {
        element.cropbox.reset();
        for (const auto type : {NodeType::CROPBOX, NodeType::ELLIPSOID, NodeType::CROP_POLYGON, NodeType::COLLISION}) {
            element.type = type;
            EXPECT_EQ(readSplatOperation(owner, element).error(), SplatOperationReadError::MissingPayload);
        }
    }

    TEST_F(SplatOperationsTest, DoesNotInferOperationsFromOrdinaryChildren) {
        for (const auto type : {NodeType::SPLAT, NodeType::CAMERA, NodeType::GROUP, NodeType::MESH}) {
            element.type = type;
            EXPECT_EQ(readSplatOperation(owner, element).error(), SplatOperationReadError::UnsupportedElement);
        }
    }

    TEST_F(SplatOperationsTest, RequiresDistinctStableIdentitiesAndDoesNotGenerateThem) {
        const auto uuid = element.uuid;
        element.uuid = {};
        EXPECT_EQ(readSplatOperation(owner, element).error(), SplatOperationReadError::InvalidIdentity);
        EXPECT_EQ(element.uuid, Uuid{});
        element.uuid = owner.uuid;
        EXPECT_EQ(readSplatOperation(owner, element).error(), SplatOperationReadError::InvalidIdentity);
        element.uuid = uuid;
        owner.id = NULL_NODE;
        EXPECT_EQ(readSplatOperation(owner, element).error(), SplatOperationReadError::InvalidIdentity);
        owner.id = -2;
        EXPECT_EQ(readSplatOperation(owner, element).error(), SplatOperationReadError::InvalidIdentity);
    }

    TEST_F(SplatOperationsTest, MetadataReadDoesNotRequireHydratingOwnerGeometryOrDependOnNames) {
        ASSERT_FALSE(owner.model);
        owner.payload_hydration = PayloadHydrationState::Unloaded;
        const auto first = readSplatOperation(owner, element);
        ASSERT_TRUE(first);
        owner.name = "renamed owner";
        element.name = "renamed helper";
        const auto second = readSplatOperation(owner, element);
        ASSERT_TRUE(second);
        EXPECT_EQ(first->owner_uuid, second->owner_uuid);
        EXPECT_EQ(first->element_uuid, second->element_uuid);
        EXPECT_EQ(owner.payload_hydration, PayloadHydrationState::Unloaded);
        EXPECT_FALSE(owner.model);
    }
} // namespace
