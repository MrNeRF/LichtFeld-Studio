/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/error_bus.hpp"
#include "core/modal_request.hpp"
#include "gui/error_surface_types.hpp"

#include <core/export.hpp>
#include <functional>
#include <string>

// Native RmlUi consumer for the ErrorBus (Phase 8). Owned by GuiManager and
// subscribed before Python, it routes each surface to its native sink. on_error
// runs on the publishing worker thread and is enqueue-only: it builds a request
// and hands it to the matching sink (drained on the UI frame), never touching
// RmlUi documents off the UI thread.
//
// Modal renders via RmlModalOverlay; Toast via RmlToastOverlay (falling back to
// Modal when no toast sink is wired, so nothing is dropped); StatusOnly via the
// status bar (a silent no-op without a status sink); Panel as a developer
// details modal.
namespace lfs::vis::gui {

    class LFS_VIS_API GuiErrorConsumer final : public lfs::NativeErrorConsumer {
    public:
        using ModalSink = std::function<void(lfs::core::ModalRequest)>;
        using ToastSink = std::function<void(ToastRequest)>;
        using StatusSink = std::function<void(std::string, ErrorNoticeLevel)>;

        struct Sinks {
            ModalSink modal;   // required
            ToastSink toast;   // empty => Toast falls back to Modal (never dropped)
            StatusSink status; // empty => StatusOnly is a no-op (Cancelled-class, silent is legal)
        };
        explicit GuiErrorConsumer(Sinks sinks);

        void on_error(const lfs::ErrorNotification& notification,
                      const lfs::ErrorDeliveryInfo& delivery) noexcept override;

    private:
        Sinks sinks_;
    };

} // namespace lfs::vis::gui
