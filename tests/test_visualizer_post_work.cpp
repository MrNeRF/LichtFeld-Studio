/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <SDL3/SDL.h>

#include "visualizer/include/visualizer/visualizer.hpp"

#include <gtest/gtest.h>

TEST(VisualizerPostWorkTest, QueuedWorkWakesEventLoop) {
    ASSERT_TRUE(SDL_Init(SDL_INIT_EVENTS));
    SDL_FlushEvents(SDL_EVENT_USER, SDL_EVENT_USER);

    lfs::vis::ViewerOptions options;
    options.show_startup_overlay = false;

    bool ran = false;
    {
        auto viewer = lfs::vis::Visualizer::create(options);

        EXPECT_FALSE(SDL_HasEvents(SDL_EVENT_USER, SDL_EVENT_USER));
        EXPECT_TRUE(viewer->postWork({
            .run = [&ran]() { ran = true; },
            .cancel = nullptr,
        }));

        EXPECT_FALSE(ran);
        EXPECT_TRUE(SDL_HasEvents(SDL_EVENT_USER, SDL_EVENT_USER));
    }
}
