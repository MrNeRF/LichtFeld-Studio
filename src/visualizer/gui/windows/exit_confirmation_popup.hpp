/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <functional>

namespace lfs::vis::gui {

    class ExitConfirmationPopup {
    public:
        using Callback = std::function<void()>;

        void show(Callback on_confirm, Callback on_cancel = nullptr);
        void render();
        [[nodiscard]] bool isOpen() const { return open_; }

    private:
        bool open_ = false;
        bool pending_open_ = false;
        Callback on_confirm_;
        Callback on_cancel_;
    };

} // namespace lfs::vis::gui
