/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/utils/drag_drop_native.hpp"
#include <SDL3/SDL.h>
#include <gtest/gtest.h>
#include <memory>

namespace {
    class MacNativeDragDrop : public testing::Test {
    protected:
        void SetUp() override {
            ASSERT_TRUE(SDL_Init(SDL_INIT_VIDEO)) << SDL_GetError();
            window = SDL_CreateWindow("drag-drop-contract", 64, 64, SDL_WINDOW_HIDDEN);
            other = SDL_CreateWindow("other-window", 64, 64, SDL_WINDOW_HIDDEN);
            ASSERT_NE(window, nullptr);
            ASSERT_NE(other, nullptr);
            ASSERT_TRUE(drop.init(window));
        }
        void TearDown() override {
            drop.shutdown();
            SDL_DestroyWindow(other);
            SDL_DestroyWindow(window);
            SDL_Quit();
        }
        void send(Uint32 type, SDL_Window* target) {
            SDL_Event event{};
            event.type = type;
            event.drop.windowID = SDL_GetWindowID(target);
            ASSERT_TRUE(SDL_PushEvent(&event)) << SDL_GetError();
        }
        SDL_Window* window = nullptr;
        SDL_Window* other = nullptr;
        lfs::vis::gui::NativeDragDrop drop;
    };

    TEST_F(MacNativeDragDrop, HoverEndsOnDropOrCancellation) {
        EXPECT_FALSE(drop.isDragHovering());
        send(SDL_EVENT_DROP_BEGIN, window);
        EXPECT_TRUE(drop.isDragHovering());
        send(SDL_EVENT_DROP_POSITION, window);
        EXPECT_TRUE(drop.isDragHovering());
        send(SDL_EVENT_DROP_COMPLETE, window);
        EXPECT_FALSE(drop.isDragHovering());
        send(SDL_EVENT_DROP_POSITION, window);
        EXPECT_TRUE(drop.isDragHovering());
        drop.resetHovering();
        EXPECT_FALSE(drop.isDragHovering());
    }

    TEST_F(MacNativeDragDrop, EventsRemainQueuedForSingleFileDelivery) {
        SDL_FlushEvents(SDL_EVENT_FIRST, SDL_EVENT_LAST);
        int native_deliveries = 0;
        drop.setFileDropCallback([&](const auto&) { ++native_deliveries; });
        send(SDL_EVENT_DROP_BEGIN, window);
        for (const char* path : {"/tmp/scene.ply", "/tmp/scène space.spz"}) {
            SDL_Event event{};
            event.type = SDL_EVENT_DROP_FILE;
            event.drop.windowID = SDL_GetWindowID(window);
            event.drop.data = path;
            ASSERT_TRUE(SDL_PushEvent(&event));
        }
        send(SDL_EVENT_DROP_COMPLETE, window);
        int files = 0;
        int completions = 0;
        SDL_Event event{};
        while (SDL_PollEvent(&event)) {
            files += event.type == SDL_EVENT_DROP_FILE;
            completions += event.type == SDL_EVENT_DROP_COMPLETE;
        }
        EXPECT_EQ(files, 2);
        EXPECT_EQ(completions, 1);
        EXPECT_EQ(native_deliveries, 0);
        EXPECT_FALSE(drop.isDragHovering());
    }

    TEST_F(MacNativeDragDrop, OtherWindowsCannotChangeHover) {
        send(SDL_EVENT_DROP_BEGIN, other);
        EXPECT_FALSE(drop.isDragHovering());
        send(SDL_EVENT_DROP_BEGIN, window);
        send(SDL_EVENT_DROP_COMPLETE, other);
        EXPECT_TRUE(drop.isDragHovering());
    }

    TEST_F(MacNativeDragDrop, ShutdownDetachesWatchAndCanReinitialize) {
        send(SDL_EVENT_DROP_BEGIN, window);
        drop.shutdown();
        EXPECT_FALSE(drop.isDragHovering());
        send(SDL_EVENT_DROP_BEGIN, window);
        EXPECT_FALSE(drop.isDragHovering());
        drop.shutdown();
        ASSERT_TRUE(drop.init(window));
        ASSERT_TRUE(drop.init(window));
        send(SDL_EVENT_DROP_BEGIN, window);
        EXPECT_TRUE(drop.isDragHovering());
    }

    TEST(MacNativeDragDropInitialization, RejectsMissingWindow) {
        lfs::vis::gui::NativeDragDrop drop;
        EXPECT_FALSE(drop.init(nullptr));
        drop.shutdown();
    }
} // namespace
