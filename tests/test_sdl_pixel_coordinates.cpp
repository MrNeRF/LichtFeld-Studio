/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "input/frame_input_buffer.hpp"
#include <gtest/gtest.h>

// This executable links SDL headers only. Supply deterministic SDL state so the
// production conversion and FrameInputBuffer::finalize run without a display,
// GPU, or compositor. Keep these doubles out of the application test binary.
struct SDL_Window {
    int width = 800;
    int height = 600;
    int pixel_width = 800;
    int pixel_height = 600;
};

bool SDLCALL SDL_GetWindowSize(SDL_Window* window, int* width, int* height) {
    *width = window->width;
    *height = window->height;
    return true;
}

bool SDLCALL SDL_GetWindowSizeInPixels(SDL_Window* window, int* width, int* height) {
    *width = window->pixel_width;
    *height = window->pixel_height;
    return true;
}

SDL_MouseButtonFlags SDLCALL SDL_GetMouseState(float* x, float* y) {
    if (x)
        *x = 123.5f;
    if (y)
        *y = 67.25f;
    return SDL_BUTTON_RMASK;
}

SDL_Keymod SDLCALL SDL_GetModState() {
    return SDL_KMOD_SHIFT;
}

namespace {
    struct ScaleCase {
        int pixel_width;
        int pixel_height;
        float expected_x;
        float expected_y;
    };

    class SdlPixelCoordinatesTest : public testing::TestWithParam<ScaleCase> {};

    TEST_P(SdlPixelCoordinatesTest, PollingConvertsMousePositionAndPreservesButtons) {
        const auto c = GetParam();
        SDL_Window window{800, 600, c.pixel_width, c.pixel_height};
        float x = 0.0f, y = 0.0f;
        const auto buttons = lfs::vis::input::mouseStateInPixels(&window, &x, &y);
        EXPECT_FLOAT_EQ(x, c.expected_x);
        EXPECT_FLOAT_EQ(y, c.expected_y);
        EXPECT_EQ(buttons, SDL_BUTTON_RMASK);

        EXPECT_EQ(lfs::vis::input::mouseStateInPixels(&window, &x, nullptr), SDL_BUTTON_RMASK);
        EXPECT_FLOAT_EQ(x, c.expected_x);
        EXPECT_EQ(lfs::vis::input::mouseStateInPixels(&window, nullptr, &y), SDL_BUTTON_RMASK);
        EXPECT_FLOAT_EQ(y, c.expected_y);
        EXPECT_EQ(lfs::vis::input::mouseStateInPixels(&window, nullptr, nullptr), SDL_BUTTON_RMASK);
    }

    TEST_P(SdlPixelCoordinatesTest, FrameFinalizationAlignsClicksWithPointerAndFramebuffer) {
        const auto c = GetParam();
        SDL_Window window{800, 600, c.pixel_width, c.pixel_height};
        lfs::vis::FrameInputBuffer buffer;
        buffer.beginFrame();
        SDL_Event event{};
        event.type = SDL_EVENT_MOUSE_BUTTON_DOWN;
        event.button.windowID = 7;
        event.button.button = SDL_BUTTON_LEFT;
        event.button.x = 123.5f;
        event.button.y = 67.25f;
        event.button.clicks = 2;
        event.button.timestamp = 42;
        buffer.processEvent(event, 7);
        event.type = SDL_EVENT_MOUSE_BUTTON_UP;
        event.button.timestamp = 43;
        buffer.processEvent(event, 7);
        buffer.finalize(&window);

        EXPECT_EQ(buffer.window_w, c.pixel_width);
        EXPECT_EQ(buffer.window_h, c.pixel_height);
        EXPECT_FLOAT_EQ(buffer.mouse_x, c.expected_x);
        EXPECT_FLOAT_EQ(buffer.mouse_y, c.expected_y);
        EXPECT_FALSE(buffer.mouse_down[0]);
        EXPECT_TRUE(buffer.mouse_down[1]);
        EXPECT_TRUE(buffer.mouse_clicked[0]);
        EXPECT_TRUE(buffer.mouse_released[0]);
        EXPECT_EQ(buffer.key_mods, SDL_KMOD_SHIFT);
        ASSERT_EQ(buffer.mouse_button_events.size(), 2u);
        for (const auto& click : buffer.mouse_button_events) {
            EXPECT_FLOAT_EQ(click.x, c.expected_x);
            EXPECT_FLOAT_EQ(click.y, c.expected_y);
            EXPECT_EQ(click.button, 0);
            EXPECT_EQ(click.clicks, 2);
        }
        EXPECT_TRUE(buffer.mouse_button_events[0].down);
        EXPECT_FALSE(buffer.mouse_button_events[1].down);
        EXPECT_EQ(buffer.mouse_button_events[0].timestamp, 42u);
        EXPECT_EQ(buffer.mouse_button_events[1].timestamp, 43u);
    }

    INSTANTIATE_TEST_SUITE_P(
        DisplayScale, SdlPixelCoordinatesTest,
        testing::Values(
            ScaleCase{800, 600, 123.5f, 67.25f},
            ScaleCase{1200, 900, 185.25f, 100.875f},
            ScaleCase{1600, 1200, 247.0f, 134.5f}));

    TEST(SdlPixelCoordinatesFallbackTest, MissingOrZeroSizedWindowKeepsLogicalCoordinates) {
        SDL_Window empty_window{0, 0, 0, 0};
        for (SDL_Window* window : {static_cast<SDL_Window*>(nullptr), &empty_window}) {
            float x = 0.0f, y = 0.0f;
            EXPECT_EQ(lfs::vis::input::mouseStateInPixels(window, &x, &y), SDL_BUTTON_RMASK);
            EXPECT_FLOAT_EQ(x, 123.5f);
            EXPECT_FLOAT_EQ(y, 67.25f);
        }
    }
} // namespace
