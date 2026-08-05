/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/formats/opf.hpp"

#include <fstream>
#include <gtest/gtest.h>

namespace fs = std::filesystem;

class OpfFormatTest : public ::testing::Test {
protected:
    fs::path root = fs::temp_directory_path() / "lfs_opf_format_test";

    void SetUp() override { fs::create_directories(root); }
    void TearDown() override { fs::remove_all(root); }

    void write(const fs::path& path, const std::string& text) {
        fs::create_directories(path.parent_path());
        std::ofstream out(path);
        out << text;
    }
};

TEST_F(OpfFormatTest, ParsesProjectGraphAndResolvesResources) {
    write(root / "resource.json", "{}");
    write(root / "project.opf", R"({
        "format":"application/opf-project+json", "version":"1.0",
        "id":"project", "name":"fixture", "description":"test", "extensions":{"vendor":{}},
        "items":[{"id":"camera", "type":"camera_list", "sources":[],
                   "resources":[{"uri":"./resource.json", "format":"application/opf-camera-list+json"}]}]
    })");

    auto result = lfs::io::opf::read_project(root / "project.opf");
    ASSERT_TRUE(result.has_value()) << result.error().format();
    EXPECT_EQ(result->items.size(), 1u);
    EXPECT_EQ(result->items[0].resources[0].resolved_path, root / "resource.json");
    EXPECT_FALSE(result->warnings.empty());
}

TEST_F(OpfFormatTest, RejectsMalformedJson) {
    write(root / "project.opf", "{not json");
    auto result = lfs::io::opf::read_project(root / "project.opf");
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, lfs::io::ErrorCode::MALFORMED_JSON);
}

TEST_F(OpfFormatTest, RejectsMissingResource) {
    write(root / "project.opf", R"({
        "format":"application/opf-project+json", "version":"1.0",
        "id":"project", "name":"fixture", "description":"test", "items":[{"id":"camera",
        "type":"camera_list", "sources":[], "resources":[{"uri":"missing.json",
        "format":"application/json"}]}]
    })");
    auto result = lfs::io::opf::read_project(root / "project.opf");
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, lfs::io::ErrorCode::MISSING_REQUIRED_FILES);
}

TEST_F(OpfFormatTest, RejectsResourceEscapingProjectRoot) {
    write(root / "project.opf", R"({
        "format":"application/opf-project+json", "version":"1.0",
        "id":"project", "name":"fixture", "description":"test", "items":[{"id":"camera",
        "type":"camera_list", "sources":[], "resources":[{"uri":"../outside.json",
        "format":"application/json"}]}]
    })");
    auto result = lfs::io::opf::read_project(root / "project.opf");
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, lfs::io::ErrorCode::INVALID_DATASET);
}

TEST_F(OpfFormatTest, RejectsMissingSourceAndCycles) {
    write(root / "project.opf", R"({
        "format":"application/opf-project+json", "version":"1.0",
        "id":"project", "name":"fixture", "description":"test",
        "items":[{"id":"a", "type":"camera_list", "sources":[{"id":"missing", "type":"camera_list"}], "resources":[]}]
    })");
    auto missing = lfs::io::opf::read_project(root / "project.opf");
    ASSERT_FALSE(missing.has_value());
    EXPECT_EQ(missing.error().code, lfs::io::ErrorCode::INVALID_DATASET);

    write(root / "project.opf", R"({
        "format":"application/opf-project+json", "version":"1.0",
        "id":"project", "name":"fixture", "description":"test",
        "items":[
            {"id":"a", "type":"camera_list", "sources":[{"id":"b", "type":"camera_list"}], "resources":[]},
            {"id":"b", "type":"camera_list", "sources":[{"id":"a", "type":"camera_list"}], "resources":[]}
        ]
    })");
    auto cycle = lfs::io::opf::read_project(root / "project.opf");
    ASSERT_FALSE(cycle.has_value());
    EXPECT_EQ(cycle.error().code, lfs::io::ErrorCode::INVALID_DATASET);
}

TEST_F(OpfFormatTest, RejectsInvalidItemResourceContract) {
    write(root / "resource.json", "{}");
    write(root / "project.opf", R"({
        "format":"application/opf-project+json", "version":"1.0",
        "id":"project", "name":"fixture", "description":"test",
        "items":[{"id":"camera", "type":"camera_list", "sources":[],
                   "resources":[{"uri":"resource.json", "format":"application/json"}]}]
    })");
    auto result = lfs::io::opf::read_project(root / "project.opf");
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, lfs::io::ErrorCode::INVALID_DATASET);
}

TEST_F(OpfFormatTest, ParsesCameraListAndResolvesImageUris) {
    write(root / "image.jpg", "test");
    write(root / "camera-list.json", R"({
        "format":"application/opf-camera-list+json", "version":"1.0",
        "cameras":[{"id":42, "uri":"image.jpg"}]
    })");
    lfs::io::opf::Resource resource{"camera-list.json",
                                    "application/opf-camera-list+json",
                                    root / "camera-list.json"};
    auto result = lfs::io::opf::read_camera_list(resource, root);
    ASSERT_TRUE(result.has_value()) << result.error().format();
    ASSERT_EQ(result->size(), 1u);
    EXPECT_EQ(result->front().id, 42u);
    EXPECT_EQ(result->front().resolved_path, root / "image.jpg");
}

TEST_F(OpfFormatTest, RejectsDuplicateCameraIds) {
    write(root / "image.jpg", "test");
    write(root / "camera-list.json", R"({
        "format":"application/opf-camera-list+json", "version":"1.0",
        "cameras":[{"id":42, "uri":"image.jpg"}, {"id":42, "uri":"image.jpg"}]
    })");
    lfs::io::opf::Resource resource{"camera-list.json",
                                    "application/opf-camera-list+json",
                                    root / "camera-list.json"};
    auto result = lfs::io::opf::read_camera_list(resource, root);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, lfs::io::ErrorCode::INVALID_DATASET);
}

TEST_F(OpfFormatTest, ParsesPerspectiveInputSensor) {
    write(root / "input-cameras.json", R"({
        "format":"application/opf-input-cameras+json", "version":"1.0",
        "sensors":[{"id":7, "name":"test", "image_size_px":[1920,1080],
        "internals":{"type":"perspective", "principal_point_px":[960,540],
        "focal_length_px":1200, "radial_distortion":[0,0,0], "tangential_distortion":[0,0]}}],
        "captures":[]
    })");
    lfs::io::opf::Resource resource{"input-cameras.json", "application/opf-input-cameras+json",
                                    root / "input-cameras.json"};
    auto result = lfs::io::opf::read_input_cameras(resource);
    ASSERT_TRUE(result.has_value()) << result.error().format();
    ASSERT_EQ(result->size(), 1u);
    EXPECT_EQ(result->front().model, "perspective");
    EXPECT_EQ(result->front().width, 1920u);
    EXPECT_DOUBLE_EQ(result->front().focal_length, 1200.0);
}

TEST_F(OpfFormatTest, RejectsUnsupportedInputCameraModel) {
    write(root / "input-cameras.json", R"({
        "format":"application/opf-input-cameras+json", "version":"1.0",
        "sensors":[{"id":7, "image_size_px":[10,10],
        "internals":{"type":"custom_distortion", "principal_point_px":[5,5]}}], "captures":[]
    })");
    lfs::io::opf::Resource resource{"input-cameras.json", "application/opf-input-cameras+json",
                                    root / "input-cameras.json"};
    auto result = lfs::io::opf::read_input_cameras(resource);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, lfs::io::ErrorCode::UNSUPPORTED_FORMAT);
}
