// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "python/python_runtime.hpp"
#include <gtest/gtest.h>
#include <memory>

TEST(PythonRuntimeCallbacks, SceneTimeCallbackDestructionCanReenterRegistry) {
    lfs::python::clear_scene_time_callback();
    int destroyed = 0;
    auto callback = [&](bool expected_present) {
        auto owner = std::shared_ptr<int>(new int(0), [&, expected_present](int* value) {
            // Python callback handles acquire the GIL here. Querying the registry
            // instead detects destruction while its nonrecursive mutex is held.
            EXPECT_EQ(lfs::python::has_scene_time_callback(), expected_present);
            ++destroyed;
            delete value;
        });
        return [owner](float) {};
    };
    lfs::python::set_scene_time_callback(callback(true));
    lfs::python::set_scene_time_callback(callback(false));
    EXPECT_EQ(destroyed, 1);
    lfs::python::clear_scene_time_callback();
    EXPECT_EQ(destroyed, 2);
    EXPECT_FALSE(lfs::python::has_scene_time_callback());
}
