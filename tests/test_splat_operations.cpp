/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/splat_operations.hpp"

#include <gtest/gtest.h>
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

    TEST(SplatOperationCatalogTest, DescribesOnlySceneElementsAndExistingOperatorPaths) {
        std::set<NodeType> types;
        for (const auto& descriptor : splatOperationDescriptors()) {
            EXPECT_TRUE(types.insert(descriptor.node_type).second);
            EXPECT_EQ(splatOperationDescriptor(descriptor.kind), &descriptor);
            EXPECT_EQ(splatOperationDescriptor(descriptor.node_type), &descriptor);
            if (descriptor.category == SplatOperationCategory::CropVolume)
                EXPECT_FALSE(descriptor.set_operator.empty());
            else {
                EXPECT_TRUE(descriptor.set_operator.empty());
                EXPECT_TRUE(descriptor.duplicate_operator.empty());
            }
        }
        EXPECT_EQ(splatOperationDescriptors().size(), 4);
        EXPECT_EQ(splatOperationDescriptor(NodeType::SPLAT), nullptr);
        EXPECT_EQ(splatOperationDescriptor(NodeType::GROUP), nullptr);
        EXPECT_EQ(splatOperationDescriptor(static_cast<SplatOperationKind>(99)), nullptr);
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
            EXPECT_EQ(result.error().code(), lfs::ErrorCode::Unsupported);
            EXPECT_EQ(result.error().domain(), lfs::ErrorDomain::Core);
            EXPECT_FALSE(result.error().detail().empty());
        }
    }

    TEST_F(SplatOperationsTest, RejectsWrongOwnerAndInconsistentChildListsEvenWithMatchingNames) {
        owner.name = element.name = "same";
        element.parent_id = 99;
        EXPECT_EQ(readSplatOperation(owner, element).error().code(), lfs::ErrorCode::FailedPrecondition);
        element.parent_id = owner.id;
        owner.children.clear();
        EXPECT_EQ(readSplatOperation(owner, element).error().code(), lfs::ErrorCode::FailedPrecondition);
    }

    TEST_F(SplatOperationsTest, RejectsMissingPayloadForEverySupportedElement) {
        element.cropbox.reset();
        for (const auto type : {NodeType::CROPBOX, NodeType::ELLIPSOID, NodeType::CROP_POLYGON, NodeType::COLLISION}) {
            element.type = type;
            EXPECT_EQ(readSplatOperation(owner, element).error().code(), lfs::ErrorCode::FailedPrecondition);
        }
    }

    TEST_F(SplatOperationsTest, DoesNotInferOperationsFromOrdinaryChildren) {
        for (const auto type : {NodeType::SPLAT, NodeType::CAMERA, NodeType::GROUP, NodeType::MESH}) {
            element.type = type;
            EXPECT_EQ(readSplatOperation(owner, element).error().code(), lfs::ErrorCode::Unsupported);
        }
    }

    TEST_F(SplatOperationsTest, RequiresDistinctStableIdentitiesAndDoesNotGenerateThem) {
        const auto uuid = element.uuid;
        element.uuid = {};
        EXPECT_EQ(readSplatOperation(owner, element).error().code(), lfs::ErrorCode::InvalidArgument);
        EXPECT_EQ(element.uuid, Uuid{});
        element.uuid = owner.uuid;
        EXPECT_EQ(readSplatOperation(owner, element).error().code(), lfs::ErrorCode::InvalidArgument);
        element.uuid = uuid;
        owner.id = NULL_NODE;
        EXPECT_EQ(readSplatOperation(owner, element).error().code(), lfs::ErrorCode::InvalidArgument);
        owner.id = -2;
        EXPECT_EQ(readSplatOperation(owner, element).error().code(), lfs::ErrorCode::InvalidArgument);
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
