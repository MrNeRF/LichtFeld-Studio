/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "python/lfs/py_ui.hpp"

#include <gtest/gtest.h>

#include <utility>
#include <vector>

TEST(PyModalRegistryRegression, ConfirmCallbackCanRegisterNestedModalWithoutLockReentry) {
    auto& registry = lfs::python::PyModalRegistry::instance();
    registry.clear_for_test();

    bool callback_called = false;
    bool mutex_was_unlocked_in_callback = false;
    bool nested_modal_registered = false;

    registry.run_pending_callback_for_test(
        [&registry, &callback_called, &mutex_was_unlocked_in_callback, &nested_modal_registered]() {
            callback_called = true;
            mutex_was_unlocked_in_callback = registry.can_lock_mutex_for_test();
            if (mutex_was_unlocked_in_callback) {
                registry.show_confirm("Inner Modal", "Nested", {"OK"}, [](const std::string&) {});
                nested_modal_registered = true;
            }
        });

    EXPECT_TRUE(callback_called);
    EXPECT_TRUE(mutex_was_unlocked_in_callback);
    EXPECT_TRUE(nested_modal_registered);
    EXPECT_TRUE(registry.has_open_modals());

    registry.clear_for_test();
}

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
