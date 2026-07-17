/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/gui_error_consumer.hpp"

#include "core/error.hpp"
#include "core/error_codes.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "gui/error_event_bridge.hpp"
#include "gui/string_keys.hpp"

#include <format>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace lfs::vis::gui {

    namespace {

        namespace Keys = lichtfeld::Strings::ErrorModal;

        [[nodiscard]] bool isOutOfMemory(const lfs::Error& error) noexcept {
            const lfs::ErrorDomain domain = error.domain();
            return error.code() == lfs::ErrorCode::ResourceExhausted &&
                   (domain == lfs::ErrorDomain::Training || domain == lfs::ErrorDomain::IO);
        }

        [[nodiscard]] const char* titleKeyFor(const lfs::Error& error) {
            const lfs::ErrorDomain domain = error.domain();
            const lfs::ErrorCode code = error.code();
            std::string_view op;
            if (!error.frames().empty()) {
                op = error.frames().back().operation;
            }

            if (isOutOfMemory(error)) {
                return Keys::OUT_OF_GPU_MEMORY;
            }
            switch (domain) {
            case lfs::ErrorDomain::CUDA:
                return code == lfs::ErrorCode::Unavailable ? Keys::CUDA_UNAVAILABLE
                                                           : Keys::CUDA_UNSUPPORTED;
            case lfs::ErrorDomain::Training:
                return Keys::TRAINING_FAILED;
            case lfs::ErrorDomain::Rendering:
                return Keys::MESH2SPLAT_FAILED;
            case lfs::ErrorDomain::App:
                return op == error_op::kLoadConfig ? Keys::CONFIG_INVALID : Keys::FILE_OPEN_FAILED;
            case lfs::ErrorDomain::IO:
                if (op == error_op::kLoadDataset) {
                    return Keys::DATASET_LOAD_FAILED;
                }
                if (op == error_op::kSave) {
                    return Keys::SAVE_FAILED;
                }
                if (op == error_op::kExportVideo) {
                    return Keys::VIDEO_EXPORT_FAILED;
                }
                return Keys::EXPORT_FAILED;
            default:
                return Keys::GENERIC;
            }
        }

        [[nodiscard]] lfs::core::ModalStyle styleFor(const lfs::Severity severity) noexcept {
            switch (severity) {
            case lfs::Severity::Fatal:
            case lfs::Severity::Error:
                return lfs::core::ModalStyle::Error;
            case lfs::Severity::Warning:
                return lfs::core::ModalStyle::Warning;
            case lfs::Severity::Info:
                return lfs::core::ModalStyle::Info;
            }
            return lfs::core::ModalStyle::Error;
        }

        // The body carries arbitrary error text from loaders/exporters; escape
        // RML metacharacters and turn newlines into breaks so a message with '<'
        // or multiple lines cannot corrupt the document.
        [[nodiscard]] std::string toRmlText(std::string_view text) {
            std::string out;
            out.reserve(text.size() + 16);
            for (const char c : text) {
                switch (c) {
                case '&': out += "&amp;"; break;
                case '<': out += "&lt;"; break;
                case '>': out += "&gt;"; break;
                case '\n': out += "<br/>"; break;
                case '\r': break;
                default: out += c; break;
                }
            }
            return out;
        }

        [[nodiscard]] std::string bodyFor(const lfs::Error& error) {
            const std::string_view message = error.user_message();
            if (isOutOfMemory(error)) {
                std::string body = std::format(
                    "<div>{}</div>"
                    "<div class=\"content-row\" style=\"margin-top: 8dp;\">{}</div>",
                    toRmlText(LOC(Keys::OOM_HEADING)), toRmlText(LOC(Keys::OOM_SUGGESTIONS)));
                if (!message.empty()) {
                    body += std::format(
                        "<div class=\"content-row dim-text\" style=\"margin-top: 8dp;\">{}</div>",
                        toRmlText(message));
                }
                return body;
            }
            return std::format("<div class=\"content-row\">{}</div>", toRmlText(message));
        }

        [[nodiscard]] const char* buttonStyleFor(const lfs::ErrorActionKind kind) noexcept {
            switch (kind) {
            case lfs::ErrorActionKind::Retry:
            case lfs::ErrorActionKind::Dismiss:
                return "primary";
            case lfs::ErrorActionKind::ChoosePath:
                return "warning";
            case lfs::ErrorActionKind::StopRenderer:
                return "error";
            case lfs::ErrorActionKind::OpenLog:
            case lfs::ErrorActionKind::Custom:
                return "secondary";
            }
            return "secondary";
        }

        [[nodiscard]] std::string labelFor(const lfs::ErrorAction& action) {
            if (!action.label.empty()) {
                return action.label;
            }
            return LOC(lichtfeld::Strings::Common::OK);
        }

    } // namespace

    GuiErrorConsumer::GuiErrorConsumer(ModalSink modal_sink)
        : modal_sink_(std::move(modal_sink)) {}

    void GuiErrorConsumer::on_error(const lfs::ErrorNotification& notification) noexcept {
        try {
            // StatusOnly maps to Cancelled-class severity and has no P1 surface;
            // every exit-criterion failure surfaces at Modal. Toast/Panel fall
            // back to Modal so nothing is dropped before the P2 surfaces land.
            if (notification.surface == lfs::ErrorSurface::StatusOnly) {
                return;
            }
            if (!modal_sink_) {
                return;
            }

            struct WiredButton {
                std::string label;
                std::function<void()> on_invoke;
            };
            std::vector<WiredButton> wired;
            wired.reserve(notification.actions.size());

            lfs::core::ModalRequest request;
            request.title = LOC(titleKeyFor(notification.error));
            request.body_rml = bodyFor(notification.error);
            request.style = styleFor(notification.error.severity());
            request.width_dp = 520;

            if (notification.actions.empty()) {
                request.buttons.push_back(
                    lfs::core::ModalButtonSpec{.label = LOC(lichtfeld::Strings::Common::OK),
                                               .style = "primary"});
            } else {
                for (const lfs::ErrorAction& action : notification.actions) {
                    std::string label = labelFor(action);
                    request.buttons.push_back(lfs::core::ModalButtonSpec{
                        .label = label,
                        .style = buttonStyleFor(action.kind)});
                    wired.push_back(WiredButton{.label = std::move(label), .on_invoke = action.on_invoke});
                }
            }

            request.on_result = [wired = std::move(wired)](const lfs::core::ModalResult& result) {
                for (const WiredButton& button : wired) {
                    if (button.label == result.button_label && button.on_invoke) {
                        // The action starts a NEW operation; the source error is
                        // const and never mutated by an action (frozen contract).
                        (void)lfs::OperationId::generate();
                        button.on_invoke();
                        return;
                    }
                }
            };

            modal_sink_(std::move(request));
        } catch (...) {
            // LFS-CENSUS-OK(empty-catch): on_error is a noexcept enqueue-only
            // boundary run on the publishing worker thread; a formatting/alloc
            // failure must degrade to no modal rather than terminate the worker.
        }
    }

} // namespace lfs::vis::gui
