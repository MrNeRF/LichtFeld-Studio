/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "python/package_manager.hpp"
#include "python/runner.hpp"
#include "python_runtime_state_helper.hpp"
#include <gtest/gtest.h>
#include <string>

namespace {
    std::string output;
    bool stderr_output = false;
    void record_output(const char* text, bool error) {
        output = text;
        stderr_output = error;
    }
    class PythonRuntimeStateTest : public ::testing::Test {
        void SetUp() override {
            output.clear();
            stderr_output = false;
            lfs::python::reset_python_init_state_for_testing();
        }
        void TearDown() override {
            lfs::python::set_output_callback({});
            lfs::python::clear_scene_time_callback();
            lfs::python::clear_frame_callback();
            lfs::python::reset_python_init_state_for_testing();
        }
    };

    TEST_F(PythonRuntimeStateTest, OutputCallbackCrossesDllBoundaryInBothDirections) {
        runtime_state_set_output(record_output);
        lfs::python::write_output("from executable", false);
        EXPECT_EQ(output, "from executable");
        EXPECT_FALSE(stderr_output);
        lfs::python::set_output_callback([](const std::string& text, bool error) {
            record_output(text.c_str(), error);
        });
        runtime_state_write_output();
        EXPECT_EQ(output, "from helper");
        EXPECT_TRUE(stderr_output);
    }

    TEST_F(PythonRuntimeStateTest, SceneTimeCallbackIsShared) {
        runtime_state_set_scene_callback();
        EXPECT_TRUE(lfs::python::has_scene_time_callback());
        lfs::python::clear_scene_time_callback();
        EXPECT_FALSE(runtime_state_has_scene_callback());
    }

    TEST_F(PythonRuntimeStateTest, FrameCallbackIsShared) {
        runtime_state_set_frame_callback();
        EXPECT_TRUE(lfs::python::has_frame_callback());
        lfs::python::clear_frame_callback();
        EXPECT_FALSE(runtime_state_has_frame_callback());
    }

    TEST_F(PythonRuntimeStateTest, InitializationFailureIsSharedWithoutStartingPython) {
        runtime_state_force_init_failure();
        ASSERT_EQ(lfs::python::init_state().state, lfs::python::PyInitState::Failed);
        const auto status = lfs::python::ensure_initialized();
        EXPECT_FALSE(status);
        lfs::python::reset_python_init_state_for_testing();
        EXPECT_EQ(runtime_state_init_state(), static_cast<int>(lfs::python::PyInitState::Uninitialized));
    }

    TEST_F(PythonRuntimeStateTest, PackageManagerHasOneInstance) {
        EXPECT_EQ(runtime_state_package_manager(), &lfs::python::PackageManager::instance());
    }
} // namespace
