/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "internal/viewport.hpp"
#include "rendering/rendering.hpp"

#include <gtest/gtest.h>

namespace {

    struct TestRasterRequest {
        glm::mat3 view_rotation{1.0f};
        glm::vec3 view_translation{0.0f};
        glm::ivec2 viewport_size{128, 128};
        float focal_length_mm = lfs::rendering::DEFAULT_FOCAL_LENGTH_MM;
        glm::vec3 background_color{0.0f};
        float far_plane = 100.0f;
        bool orthographic = false;
        float ortho_scale = lfs::rendering::DEFAULT_ORTHO_SCALE;
    };

    lfs::rendering::FrameView makeFrameView(const TestRasterRequest& request) {
        return lfs::rendering::FrameView{
            .rotation = request.view_rotation,
            .translation = request.view_translation,
            .size = request.viewport_size,
            .focal_length_mm = request.focal_length_mm,
            .far_plane = request.far_plane,
            .orthographic = request.orthographic,
            .ortho_scale = request.ortho_scale,
            .background_color = request.background_color};
    }

    std::unique_ptr<lfs::core::SplatData> makeSingleTestSplat(const glm::vec3& mean) {
        using lfs::core::DataType;
        using lfs::core::Device;
        using lfs::core::Tensor;

        auto means = Tensor::from_vector({mean.x, mean.y, mean.z}, {size_t{1}, size_t{3}}, Device::CUDA).to(DataType::Float32);
        auto sh0 = Tensor::from_vector(
                       {0.0f, 0.0f, 0.0f},
                       {size_t{1}, size_t{1}, size_t{3}},
                       Device::CUDA)
                       .to(DataType::Float32);
        auto shN = Tensor::zeros({size_t{1}, size_t{0}, size_t{3}}, Device::CUDA, DataType::Float32);
        auto scaling = Tensor::from_vector(
                           {-2.0f, -2.0f, -2.0f},
                           {size_t{1}, size_t{3}},
                           Device::CUDA)
                           .to(DataType::Float32);
        auto rotation = Tensor::from_vector(
                            {1.0f, 0.0f, 0.0f, 0.0f},
                            {size_t{1}, size_t{4}},
                            Device::CUDA)
                            .to(DataType::Float32);
        auto opacity = Tensor::from_vector(
                           {8.0f},
                           {size_t{1}},
                           Device::CUDA)
                           .to(DataType::Float32);

        return std::make_unique<lfs::core::SplatData>(
            0,
            std::move(means),
            std::move(sh0),
            std::move(shN),
            std::move(scaling),
            std::move(rotation),
            std::move(opacity),
            1.0f);
    }

    glm::vec2 renderCentroidForPoint(const glm::vec3& local_point,
                                     const TestRasterRequest& request,
                                     const std::vector<glm::mat4>& model_transforms = {}) {
        auto engine = lfs::rendering::RenderingEngine::createRasterOnly();
        const auto init_result = engine->initializeRasterOnly();
        EXPECT_TRUE(init_result.has_value()) << init_result.error();
        if (!init_result) {
            return {-1.0f, -1.0f};
        }
        const auto splat = makeSingleTestSplat(local_point);

        lfs::rendering::ViewportRenderRequest configured_request{
            .frame_view = makeFrameView(request),
            .sh_degree = 0,
            .scene = {.model_transforms = model_transforms.empty() ? nullptr : &model_transforms}};

        const auto result = engine->renderGaussiansImage(*splat, configured_request);
        EXPECT_TRUE(result.has_value()) << result.error();
        if (!result) {
            return {-1.0f, -1.0f};
        }

        auto image = result->image->cpu().contiguous();
        EXPECT_EQ(image.ndim(), 3);
        EXPECT_EQ(image.size(0), size_t{3});
        if (image.ndim() != 3 || image.size(0) != size_t{3}) {
            return {-1.0f, -1.0f};
        }

        auto acc = image.accessor<float, 3>();

        double weight_sum = 0.0;
        double weighted_x = 0.0;
        double weighted_y = 0.0;
        const int height = static_cast<int>(image.size(1));
        const int width = static_cast<int>(image.size(2));
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const float r = acc(0, y, x);
                const float g = acc(1, y, x);
                const float b = acc(2, y, x);
                const double weight = static_cast<double>(r + g + b);
                weight_sum += weight;
                weighted_x += weight * static_cast<double>(x);
                weighted_y += weight * static_cast<double>(y);
            }
        }

        EXPECT_GT(weight_sum, 0.0);
        return {
            static_cast<float>(weighted_x / weight_sum),
            static_cast<float>(weighted_y / weight_sum),
        };
    }

    std::optional<glm::vec2> renderScreenPositionForPoint(
        const glm::vec3& local_point,
        const TestRasterRequest& request,
        const std::vector<glm::mat4>& model_transforms = {}) {
        auto engine = lfs::rendering::RenderingEngine::createRasterOnly();
        const auto init_result = engine->initializeRasterOnly();
        EXPECT_TRUE(init_result.has_value()) << init_result.error();
        if (!init_result) {
            return std::nullopt;
        }
        const auto splat = makeSingleTestSplat(local_point);

        const lfs::rendering::ScreenPositionRenderRequest configured_request{
            .frame_view = makeFrameView(request),
            .scene = {.model_transforms = model_transforms.empty() ? nullptr : &model_transforms}};

        const auto result = engine->renderGaussianScreenPositions(*splat, configured_request);
        EXPECT_TRUE(result.has_value()) << result.error();
        if (!result) {
            return std::nullopt;
        }

        auto screen_positions = (*result)->cpu().contiguous();
        EXPECT_EQ(screen_positions.ndim(), 2);
        EXPECT_EQ(screen_positions.size(0), size_t{1});
        EXPECT_EQ(screen_positions.size(1), size_t{2});
        if (screen_positions.ndim() != 2 || screen_positions.size(0) != size_t{1} || screen_positions.size(1) != size_t{2}) {
            return std::nullopt;
        }

        auto acc = screen_positions.accessor<float, 2>();
        return glm::vec2(acc(0, 0), acc(0, 1));
    }

    bool previewSelectionHitsCursorAtWindowPosition(
        const glm::vec3& local_point,
        const TestRasterRequest& request,
        const glm::vec2& cursor,
        const float) {
        auto engine = lfs::rendering::RenderingEngine::createRasterOnly();
        const auto init_result = engine->initializeRasterOnly();
        EXPECT_TRUE(init_result.has_value()) << init_result.error();
        if (!init_result) {
            return false;
        }
        const auto splat = makeSingleTestSplat(local_point);

        const lfs::rendering::HoveredGaussianQueryRequest configured_request{
            .frame_view = makeFrameView(request),
            .cursor = cursor};
        const auto result = engine->queryHoveredGaussianId(*splat, configured_request);
        EXPECT_TRUE(result.has_value()) << result.error();
        if (!result) {
            return false;
        }

        return result->has_value() && **result == 0;
    }

    glm::vec2 tensorCentroidToWindowCoords(const glm::vec2& centroid,
                                           const TestRasterRequest& request) {
        return {
            centroid.x,
            static_cast<float>(request.viewport_size.y) - centroid.y,
        };
    }

    TestRasterRequest makeTestRasterRequest() {
        TestRasterRequest request;
        request.view_rotation = glm::mat3(1.0f);
        request.view_translation = glm::vec3(0.0f);
        request.viewport_size = {128, 128};
        request.focal_length_mm = lfs::rendering::DEFAULT_FOCAL_LENGTH_MM;
        request.background_color = glm::vec3(0.0f);
        request.far_plane = 100.0f;
        return request;
    }

    std::optional<glm::vec2> projectWithClipSpaceMatrices(const glm::mat3& rotation,
                                                          const glm::vec3& translation,
                                                          const glm::ivec2& viewport_size,
                                                          const glm::vec3& world_point,
                                                          const float focal_length_mm,
                                                          const bool orthographic = false,
                                                          const float ortho_scale = lfs::rendering::DEFAULT_ORTHO_SCALE) {
        const glm::mat4 view = lfs::rendering::makeViewMatrix(rotation, translation);
        const glm::mat4 projection = lfs::rendering::createProjectionMatrixFromFocal(
            viewport_size,
            focal_length_mm,
            orthographic,
            ortho_scale);
        const glm::vec4 clip = projection * view * glm::vec4(world_point, 1.0f);
        if (!orthographic && clip.w <= 1e-6f) {
            return std::nullopt;
        }

        const glm::vec3 ndc = glm::vec3(clip) / clip.w;
        return glm::vec2(
            (ndc.x * 0.5f + 0.5f) * static_cast<float>(viewport_size.x),
            (1.0f - (ndc.y * 0.5f + 0.5f)) * static_cast<float>(viewport_size.y));
    }

} // namespace

TEST(ViewportTest, InvalidWorldPositionUsesNamedSentinel) {
    Viewport viewport(100, 100);

    const glm::vec3 invalid = viewport.unprojectPixel(50.0f, 50.0f, -1.0f);

    EXPECT_FALSE(Viewport::isValidWorldPosition(invalid));
    EXPECT_FLOAT_EQ(invalid.x, Viewport::INVALID_WORLD_POS);
    EXPECT_FLOAT_EQ(invalid.y, Viewport::INVALID_WORLD_POS);
    EXPECT_FLOAT_EQ(invalid.z, Viewport::INVALID_WORLD_POS);
}

TEST(ViewportTest, DefaultCameraStartsAboveWorldYAxis) {
    Viewport viewport(100, 100);

    EXPECT_GT(viewport.camera.t.y, 0.0f);
}

TEST(ViewportTest, UnprojectPixelDependsOnScreenPixel) {
    Viewport viewport(100, 100);
    viewport.camera.R = glm::mat3(1.0f);
    viewport.camera.t = glm::vec3(0.0f);

    const glm::vec3 center = viewport.unprojectPixel(50.0f, 50.0f, 10.0f);
    const glm::vec3 top_left = viewport.unprojectPixel(0.0f, 0.0f, 10.0f);

    ASSERT_TRUE(Viewport::isValidWorldPosition(center));
    ASSERT_TRUE(Viewport::isValidWorldPosition(top_left));
    EXPECT_NEAR(center.x, 0.0f, 1e-4f);
    EXPECT_NEAR(center.y, 0.0f, 1e-4f);
    EXPECT_NEAR(center.z, -10.0f, 1e-4f);
    EXPECT_LT(top_left.x, center.x);
    EXPECT_GT(top_left.y, center.y);
    EXPECT_NEAR(top_left.z, center.z, 1e-4f);
}

TEST(ViewportTest, ViewportDataViewMatrixMatchesViewport) {
    Viewport viewport(160, 90);
    viewport.camera.R = lfs::rendering::makeVisualizerLookAtRotation(
        glm::vec3(3.0f, 2.0f, 5.0f),
        glm::vec3(0.0f, 0.0f, 0.0f));
    viewport.camera.t = glm::vec3(3.0f, 2.0f, 5.0f);

    const lfs::rendering::ViewportData data{
        .rotation = viewport.getRotationMatrix(),
        .translation = viewport.getTranslation(),
        .size = viewport.windowSize,
    };

    const glm::mat4 expected = viewport.getViewMatrix();
    const glm::mat4 actual = data.getViewMatrix();
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            EXPECT_NEAR(actual[col][row], expected[col][row], 1e-5f);
        }
    }
}

TEST(ViewportTest, ProjectWorldPointUsesVisualizerConvention) {
    Viewport viewport(100, 100);
    viewport.camera.R = glm::mat3(1.0f);
    viewport.camera.t = glm::vec3(0.0f);

    const auto center = lfs::rendering::projectWorldPoint(
        viewport.camera.R,
        viewport.camera.t,
        viewport.windowSize,
        glm::vec3(0.0f, 0.0f, -10.0f),
        lfs::rendering::DEFAULT_FOCAL_LENGTH_MM);
    const auto behind = lfs::rendering::projectWorldPoint(
        viewport.camera.R,
        viewport.camera.t,
        viewport.windowSize,
        glm::vec3(0.0f, 0.0f, 10.0f),
        lfs::rendering::DEFAULT_FOCAL_LENGTH_MM);

    ASSERT_TRUE(center.has_value());
    EXPECT_NEAR(center->x, 50.0f, 1e-4f);
    EXPECT_NEAR(center->y, 50.0f, 1e-4f);
    EXPECT_FALSE(behind.has_value());
}

TEST(ViewportTest, WasdAdvanceSupportsFlatAdditionalSpeedInVisualizerSpace) {
    Viewport viewport(100, 100);
    viewport.camera.R = glm::mat3(1.0f);
    viewport.camera.t = glm::vec3(0.0f);
    viewport.camera.pivot = glm::vec3(0.0f);

    viewport.camera.advance_forward(1.0f, 20.0f);

    EXPECT_FLOAT_EQ(viewport.camera.getWasdSpeed(), 6.0f);
    EXPECT_NEAR(viewport.camera.t.x, 0.0f, 1e-5f);
    EXPECT_NEAR(viewport.camera.t.y, 0.0f, 1e-5f);
    EXPECT_NEAR(viewport.camera.t.z, -26.0f, 1e-5f);
    EXPECT_NEAR(viewport.camera.pivot.z, -26.0f, 1e-5f);
}

TEST(ViewportTest, OrbitDraggingRightMovesCameraLeftAroundPivot) {
    Viewport viewport(100, 100);
    viewport.camera.t = glm::vec3(0.0f, 0.0f, 5.0f);
    viewport.camera.setPivot(glm::vec3(0.0f));
    viewport.camera.R = glm::mat3(1.0f);

    viewport.camera.startRotateAroundCenter(glm::vec2(0.0f, 0.0f), 0.0f);
    viewport.camera.updateRotateAroundCenter(glm::vec2(100.0f, 0.0f), 0.0f);
    viewport.camera.endRotateAroundCenter();

    EXPECT_LT(viewport.camera.t.x, 0.0f);
    EXPECT_NEAR(glm::length(viewport.camera.t - viewport.camera.getPivot()), 5.0f, 1e-4f);
}

TEST(ViewportTest, RotateDraggingUpTiltsCameraUp) {
    Viewport viewport(100, 100);
    viewport.camera.R = glm::mat3(1.0f);
    viewport.camera.initScreenPos(glm::vec2(0.0f, 0.0f));

    viewport.camera.rotate(glm::vec2(0.0f, -100.0f));

    EXPECT_GT(lfs::rendering::cameraForward(viewport.camera.R).y, 0.0f);
}

TEST(ViewportTest, FpvRotateKeepsPivotInFrontOfCamera) {
    Viewport viewport(100, 100);
    viewport.camera.t = glm::vec3(0.0f, 0.0f, 5.0f);
    viewport.camera.setPivot(glm::vec3(0.0f));
    viewport.camera.R = glm::mat3(1.0f);
    viewport.camera.initScreenPos(glm::vec2(0.0f, 0.0f));

    viewport.camera.rotateFpv(glm::vec2(0.0f, -100.0f));

    const glm::vec3 forward = lfs::rendering::cameraForward(viewport.camera.R);
    const glm::vec3 to_pivot = glm::normalize(viewport.camera.getPivot() - viewport.camera.t);
    EXPECT_GT(forward.y, 0.0f);
    EXPECT_NEAR(glm::length(viewport.camera.getPivot() - viewport.camera.t), 5.0f, 1e-4f);
    EXPECT_NEAR(glm::dot(forward, to_pivot), 1.0f, 1e-4f);
}

TEST(ViewportTest, OrbitDraggingUpMovesCameraDownAroundPivot) {
    Viewport viewport(100, 100);
    viewport.camera.t = glm::vec3(0.0f, 0.0f, 5.0f);
    viewport.camera.setPivot(glm::vec3(0.0f));
    viewport.camera.R = glm::mat3(1.0f);

    viewport.camera.startRotateAroundCenter(glm::vec2(0.0f, 0.0f), 0.0f);
    viewport.camera.updateRotateAroundCenter(glm::vec2(0.0f, -100.0f), 0.0f);
    viewport.camera.endRotateAroundCenter();

    EXPECT_LT(viewport.camera.t.y, 0.0f);
    EXPECT_NEAR(glm::length(viewport.camera.t - viewport.camera.getPivot()), 5.0f, 1e-4f);
}

TEST(ViewportTest, OrbitDraggingDownMovesCameraUpAroundPivot) {
    Viewport viewport(100, 100);
    viewport.camera.t = glm::vec3(0.0f, 0.0f, 5.0f);
    viewport.camera.setPivot(glm::vec3(0.0f));
    viewport.camera.R = glm::mat3(1.0f);

    viewport.camera.startRotateAroundCenter(glm::vec2(0.0f, 0.0f), 0.0f);
    viewport.camera.updateRotateAroundCenter(glm::vec2(0.0f, 100.0f), 0.0f);
    viewport.camera.endRotateAroundCenter();

    EXPECT_GT(viewport.camera.t.y, 0.0f);
    EXPECT_NEAR(glm::length(viewport.camera.t - viewport.camera.getPivot()), 5.0f, 1e-4f);
}

TEST(ViewportTest, PanDraggingRightMovesCameraLeftAndDraggingUpMovesCameraDown) {
    Viewport viewport(100, 100);
    viewport.camera.t = glm::vec3(0.0f, 0.0f, 5.0f);
    viewport.camera.setPivot(glm::vec3(0.0f, 0.0f, 0.0f));
    viewport.camera.R = glm::mat3(1.0f);
    viewport.camera.initScreenPos(glm::vec2(0.0f, 0.0f));

    viewport.camera.translate(glm::vec2(20.0f, -30.0f));

    EXPECT_LT(viewport.camera.t.x, 0.0f);
    EXPECT_LT(viewport.camera.t.y, 0.0f);
    EXPECT_LT(viewport.camera.getPivot().x, 0.0f);
    EXPECT_LT(viewport.camera.getPivot().y, 0.0f);
    EXPECT_NEAR(viewport.camera.t.z, 5.0f, 1e-4f);
    EXPECT_NEAR(viewport.camera.getPivot().z, 0.0f, 1e-4f);
}

TEST(ViewportTest, PanDraggingMovesProjectedContentWithCursorInVisualizerSpace) {
    Viewport viewport(100, 100);
    viewport.camera.t = glm::vec3(3.0f, 2.0f, 5.0f);
    viewport.camera.setPivot(glm::vec3(0.0f, 0.0f, 0.0f));
    viewport.camera.R = lfs::rendering::makeVisualizerLookAtRotation(
        viewport.camera.t, viewport.camera.getPivot());
    viewport.camera.initScreenPos(glm::vec2(0.0f, 0.0f));

    const glm::vec3 world_point(0.0f, 0.0f, 0.0f);
    const auto before = lfs::rendering::projectWorldPoint(
        viewport.camera.R,
        viewport.camera.t,
        viewport.windowSize,
        world_point,
        lfs::rendering::DEFAULT_FOCAL_LENGTH_MM);
    ASSERT_TRUE(before.has_value());

    viewport.camera.translate(glm::vec2(20.0f, -30.0f));

    const auto after = lfs::rendering::projectWorldPoint(
        viewport.camera.R,
        viewport.camera.t,
        viewport.windowSize,
        world_point,
        lfs::rendering::DEFAULT_FOCAL_LENGTH_MM);
    ASSERT_TRUE(after.has_value());

    EXPECT_GT(after->x, before->x);
    EXPECT_LT(after->y, before->y);
}

TEST(ViewportTest, GaussianRasterSourceMatchesVisualizerLeftRightConvention) {
    const auto request = makeTestRasterRequest();
    const glm::vec2 center = tensorCentroidToWindowCoords(
        renderCentroidForPoint(glm::vec3(0.0f, 0.0f, -5.0f), request), request);
    const glm::vec2 right = tensorCentroidToWindowCoords(
        renderCentroidForPoint(glm::vec3(1.0f, 0.0f, -5.0f), request), request);
    const glm::vec2 left = tensorCentroidToWindowCoords(
        renderCentroidForPoint(glm::vec3(-1.0f, 0.0f, -5.0f), request), request);

    EXPECT_GT(right.x, center.x);
    EXPECT_LT(left.x, center.x);
}

TEST(ViewportTest, GaussianRasterSourceMatchesVisualizerUpDownConvention) {
    const auto request = makeTestRasterRequest();
    const glm::vec2 center = tensorCentroidToWindowCoords(
        renderCentroidForPoint(glm::vec3(0.0f, 0.0f, -5.0f), request), request);
    const glm::vec2 up = tensorCentroidToWindowCoords(
        renderCentroidForPoint(glm::vec3(0.0f, 1.0f, -5.0f), request), request);
    const glm::vec2 down = tensorCentroidToWindowCoords(
        renderCentroidForPoint(glm::vec3(0.0f, -1.0f, -5.0f), request), request);

    EXPECT_LT(up.y, center.y);
    EXPECT_GT(down.y, center.y);
}

TEST(ViewportTest, GaussianScreenPositionOutputUsesWindowCoordinates) {
    const auto request = makeTestRasterRequest();
    const glm::vec3 world_point(0.35f, 0.75f, -5.0f);

    const auto expected = lfs::rendering::projectWorldPoint(
        request.view_rotation,
        request.view_translation,
        request.viewport_size,
        world_point,
        request.focal_length_mm);
    ASSERT_TRUE(expected.has_value());

    const auto actual = renderScreenPositionForPoint(world_point, request);
    ASSERT_TRUE(actual.has_value());
    EXPECT_NEAR(actual->x, expected->x, 1e-2f);
    EXPECT_NEAR(actual->y, expected->y, 1e-2f);
}

TEST(ViewportTest, GaussianPreviewSelectionUsesWindowCursorCoordinates) {
    const auto request = makeTestRasterRequest();
    const glm::vec3 world_point(0.25f, 0.75f, -5.0f);

    const auto projected = lfs::rendering::projectWorldPoint(
        request.view_rotation,
        request.view_translation,
        request.viewport_size,
        world_point,
        request.focal_length_mm);
    ASSERT_TRUE(projected.has_value());

    EXPECT_TRUE(previewSelectionHitsCursorAtWindowPosition(world_point, request, *projected, 4.0f));

    const glm::vec2 mirrored_y(
        projected->x,
        static_cast<float>(request.viewport_size.y) - projected->y);
    EXPECT_FALSE(previewSelectionHitsCursorAtWindowPosition(world_point, request, mirrored_y, 4.0f));
}

TEST(ViewportTest, GaussianRasterMatchesVisualizerProjectionWhenYawed) {
    auto request = makeTestRasterRequest();
    request.view_rotation = lfs::rendering::makeVisualizerLookAtRotation(
        glm::vec3(0.0f, 0.0f, 0.0f),
        glm::vec3(1.0f, 0.0f, -5.0f));

    const glm::vec3 world_point(0.0f, 0.25f, -5.0f);
    const auto expected = lfs::rendering::projectWorldPoint(
        request.view_rotation,
        request.view_translation,
        request.viewport_size,
        world_point,
        request.focal_length_mm);
    ASSERT_TRUE(expected.has_value());

    const glm::vec2 actual = tensorCentroidToWindowCoords(
        renderCentroidForPoint(world_point, request), request);
    EXPECT_NEAR(actual.x, expected->x, 2.0f);
    EXPECT_NEAR(actual.y, expected->y, 2.0f);
}

TEST(ViewportTest, GaussianRasterMatchesVisualizerProjectionWithModelTransform) {
    const auto request = makeTestRasterRequest();
    const glm::vec3 local_point(0.0f, 0.0f, -5.0f);
    const glm::mat4 model_transform =
        glm::translate(glm::mat4(1.0f), glm::vec3(0.4f, 0.25f, 0.0f)) *
        glm::rotate(glm::mat4(1.0f), glm::radians(10.0f), glm::normalize(glm::vec3(0.0f, 1.0f, 0.0f)));
    const glm::vec3 world_point = glm::vec3(model_transform * glm::vec4(local_point, 1.0f));

    const auto expected = lfs::rendering::projectWorldPoint(
        request.view_rotation,
        request.view_translation,
        request.viewport_size,
        world_point,
        request.focal_length_mm);
    ASSERT_TRUE(expected.has_value());

    const glm::vec2 actual = tensorCentroidToWindowCoords(
        renderCentroidForPoint(local_point, request, {model_transform}), request);
    EXPECT_NEAR(actual.x, expected->x, 2.0f);
    EXPECT_NEAR(actual.y, expected->y, 2.0f);
}

TEST(ViewportTest, ClipSpaceMatrixProjectionMatchesVisualizerProjectionWhenYawed) {
    const auto request = makeTestRasterRequest();
    const glm::mat3 rotation = lfs::rendering::makeVisualizerLookAtRotation(
        glm::vec3(0.0f, 0.0f, 0.0f),
        glm::vec3(1.0f, 0.25f, -5.0f));
    const glm::vec3 translation(0.0f, 0.0f, 0.0f);
    const glm::vec3 world_point(0.5f, -0.2f, -4.0f);

    const auto explicit_projection = lfs::rendering::projectWorldPoint(
        rotation, translation, request.viewport_size, world_point, request.focal_length_mm);
    const auto matrix_projection = projectWithClipSpaceMatrices(
        rotation, translation, request.viewport_size, world_point, request.focal_length_mm);

    ASSERT_TRUE(explicit_projection.has_value());
    ASSERT_TRUE(matrix_projection.has_value());
    EXPECT_NEAR(matrix_projection->x, explicit_projection->x, 1e-3f);
    EXPECT_NEAR(matrix_projection->y, explicit_projection->y, 1e-3f);
}

TEST(ViewportTest, OrthographicProjectionMatchesClipSpaceAndIgnoresDepth) {
    const auto request = makeTestRasterRequest();
    const glm::mat3 rotation = lfs::rendering::makeVisualizerLookAtRotation(
        glm::vec3(1.0f, 0.5f, 2.0f),
        glm::vec3(1.0f, 0.5f, -3.0f));
    const glm::vec3 translation(1.0f, 0.5f, 2.0f);
    constexpr float ortho_scale = 75.0f;

    const glm::vec3 near_point(1.5f, 0.8f, -4.0f);
    const glm::vec3 far_point(1.5f, 0.8f, -12.0f);

    const auto near_projection = lfs::rendering::projectWorldPoint(
        rotation, translation, request.viewport_size, near_point, request.focal_length_mm, true, ortho_scale);
    const auto far_projection = lfs::rendering::projectWorldPoint(
        rotation, translation, request.viewport_size, far_point, request.focal_length_mm, true, ortho_scale);
    const auto matrix_projection = projectWithClipSpaceMatrices(
        rotation, translation, request.viewport_size, near_point, request.focal_length_mm, true, ortho_scale);

    ASSERT_TRUE(near_projection.has_value());
    ASSERT_TRUE(far_projection.has_value());
    ASSERT_TRUE(matrix_projection.has_value());
    EXPECT_NEAR(near_projection->x, far_projection->x, 1e-4f);
    EXPECT_NEAR(near_projection->y, far_projection->y, 1e-4f);
    EXPECT_NEAR(matrix_projection->x, near_projection->x, 1e-3f);
    EXPECT_NEAR(matrix_projection->y, near_projection->y, 1e-3f);
}
