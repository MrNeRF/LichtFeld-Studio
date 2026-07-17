/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/error_bus.hpp"
#include "core/modal_request.hpp"

#include <core/export.hpp>
#include <functional>

// Native RmlUi consumer for the ErrorBus (Phase 8, packet P1). Owned by
// GuiManager and subscribed before Python, it renders the Modal surface via the
// existing thread-safe RmlModalOverlay. on_error runs on the publishing worker
// thread and is enqueue-only: it builds a ModalRequest and hands it to the
// GuiManager modal sink (drained on the UI frame), never touching RmlUi
// documents off the UI thread.
//
// P1 renders Modal only. Toast and Panel fall back to Modal (never dropped);
// StatusOnly is a silent no-op until the P2 status/toast surfaces land.
namespace lfs::vis::gui {

    class LFS_VIS_API GuiErrorConsumer final : public lfs::NativeErrorConsumer {
    public:
        using ModalSink = std::function<void(lfs::core::ModalRequest)>;

        explicit GuiErrorConsumer(ModalSink modal_sink);

        void on_error(const lfs::ErrorNotification& notification) noexcept override;

    private:
        ModalSink modal_sink_;
    };

} // namespace lfs::vis::gui
