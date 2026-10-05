/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "python/lfs/py_ui.hpp"

#include <gtest/gtest.h>

#include <utility>
#include <vector>

TEST(PyModalRegistryRegression, MessageWithoutCallbackUsesNativeModalPath) {
    auto& registry = lfs::python::PyModalRegistry::instance();
    registry.clear_for_test();
    std::vector<lfs::core::ModalRequest> enqueued;
    registry.set_enqueue_callback(
        [&enqueued](lfs::core::ModalRequest request) {
            enqueued.push_back(std::move(request));
        });

    registry.show_message("Dropped Files", "Unsupported file type", lfs::python::MessageStyle::Error);

    EXPECT_TRUE(registry.has_open_modals());
    registry.draw_modals();

    EXPECT_FALSE(registry.has_open_modals());
    EXPECT_EQ(enqueued.size(), 1);
    if (enqueued.size() == 1) {
        EXPECT_EQ(enqueued[0].title, "Dropped Files");
        EXPECT_EQ(enqueued[0].body_rml, "Unsupported file type");
        EXPECT_EQ(enqueued[0].style, lfs::core::ModalStyle::Error);
        EXPECT_EQ(enqueued[0].buttons.size(), 1);
        if (enqueued[0].buttons.size() == 1)
            EXPECT_EQ(enqueued[0].buttons[0].label, "OK");
    }

    registry.set_enqueue_callback({});
    registry.clear_for_test();
}
