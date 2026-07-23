/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/gui_error_consumer.hpp"

#include "core/error.hpp"
#include "core/error_codes.hpp"
#include "core/error_reporter.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "gui/error_event_bridge.hpp"
#include "gui/string_keys.hpp"

#include <format>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace lfs::vis::gui {

    std::string escapeRmlText(std::string_view text) {
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
            case lfs::ErrorDomain::Vulkan:
                if (code == lfs::ErrorCode::DeviceLost) {
                    return Keys::RENDERER_DEVICE_LOST;
                }
                if (code == lfs::ErrorCode::DeadlineExceeded) {
                    return Keys::RENDERER_STALLED;
                }
                return Keys::RENDERER_FAILED;
            case lfs::ErrorDomain::Rendering:
                if (op == error_op::kRenderFrame) {
                    return code == lfs::ErrorCode::ResourceExhausted ? Keys::OUT_OF_GPU_MEMORY
                                                                     : Keys::RENDERER_FAILED;
                }
                return Keys::MESH2SPLAT_FAILED;
            case lfs::ErrorDomain::Python:
                return Keys::PLUGINS_DISABLED;
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

        [[nodiscard]] ErrorNoticeLevel levelFor(const lfs::Severity severity) noexcept {
            switch (severity) {
            case lfs::Severity::Fatal:
            case lfs::Severity::Error:
                return ErrorNoticeLevel::Error;
            case lfs::Severity::Warning:
                return ErrorNoticeLevel::Warning;
            case lfs::Severity::Info:
                return ErrorNoticeLevel::Info;
            }
            return ErrorNoticeLevel::Error;
        }

        [[nodiscard]] std::string_view firstLine(std::string_view text) noexcept {
            const auto pos = text.find('\n');
            return pos == std::string_view::npos ? text : text.substr(0, pos);
        }

        [[nodiscard]] std::string bodyFor(const lfs::Error& error) {
            const std::string_view message = error.user_message();
            if (isOutOfMemory(error)) {
                std::string body = std::format(
                    "<div>{}</div>"
                    "<div class=\"content-row\" style=\"margin-top: 8dp;\">{}</div>",
                    escapeRmlText(LOC(Keys::OOM_HEADING)), escapeRmlText(LOC(Keys::OOM_SUGGESTIONS)));
                if (!message.empty()) {
                    body += std::format(
                        "<div class=\"content-row dim-text\" style=\"margin-top: 8dp;\">{}</div>",
                        escapeRmlText(message));
                }
                return body;
            }
            return std::format("<div class=\"content-row\">{}</div>", escapeRmlText(message));
        }

        // Developer details modal (§3): the same localized title plus a monospace,
        // scrollable block carrying format_for_developer (ENGLISH, technical).
        [[nodiscard]] lfs::core::ModalRequest buildDetailsModal(const lfs::Error& error) {
            lfs::core::ModalRequest request;
            request.title = LOC(titleKeyFor(error));
            request.body_rml = std::format(
                "<div class=\"content-row\">{}</div>"
                "<div class=\"details-block\">{}</div>",
                escapeRmlText(error.user_message()),
                escapeRmlText(lfs::format_for_developer(error)));
            request.style = styleFor(error.severity());
            request.width_dp = 640;
            request.buttons.push_back(lfs::core::ModalButtonSpec{
                .label = LOC(lichtfeld::Strings::Common::CLOSE),
                .style = "primary"});
            return request;
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
            namespace Actions = lichtfeld::Strings::ErrorActions;
            switch (action.kind) {
            case lfs::ErrorActionKind::Retry:
                return LOC(Actions::RETRY);
            case lfs::ErrorActionKind::ChoosePath:
                return LOC(Actions::CHOOSE_PATH);
            case lfs::ErrorActionKind::OpenLog:
                return LOC(Actions::OPEN_LOG);
            case lfs::ErrorActionKind::StopRenderer:
                return LOC(Actions::STOP_RENDERER);
            case lfs::ErrorActionKind::Dismiss:
            case lfs::ErrorActionKind::Custom:
                return LOC(lichtfeld::Strings::Common::OK);
            }
            return LOC(lichtfeld::Strings::Common::OK);
        }

    } // namespace

    GuiErrorConsumer::GuiErrorConsumer(Sinks sinks)
        : sinks_(std::move(sinks)) {}

    void GuiErrorConsumer::on_error(const lfs::ErrorNotification& notification,
                                    const lfs::ErrorDeliveryInfo& delivery) noexcept {
        try {
            switch (notification.surface) {
            case lfs::ErrorSurface::Toast:
                if (sinks_.toast) {
                    sinks_.toast(ToastRequest{
                        .title = LOC(titleKeyFor(notification.error)),
                        .message = std::string(notification.error.user_message()),
                        .level = levelFor(notification.error.severity()),
                        .fingerprint = lfs::core::error_fingerprint(notification.error),
                    });
                    return;
                }
                break; // no toast sink: fall back to the Modal path (never dropped)
            case lfs::ErrorSurface::StatusOnly:
                if (sinks_.status) {
                    sinks_.status(std::string(firstLine(notification.error.user_message())),
                                  levelFor(notification.error.severity()));
                }
                return; // silent no-op without a status sink (Cancelled-class)
            case lfs::ErrorSurface::Panel:
                if (sinks_.modal) {
                    sinks_.modal(buildDetailsModal(notification.error));
                }
                return;
            case lfs::ErrorSurface::Modal:
                break;
            }

            if (!sinks_.modal) {
                return;
            }

            struct WiredButton {
                std::string label;
                std::function<void(lfs::OperationId)> on_invoke;
            };
            std::vector<WiredButton> wired;
            wired.reserve(notification.actions.size());

            lfs::core::ModalRequest request;
            request.title = LOC(titleKeyFor(notification.error));
            request.body_rml = bodyFor(notification.error);
            if (delivery.suppressed_repeats > 0) {
                request.body_rml += std::format(
                    "<div class=\"content-row dim-text\" style=\"margin-top: 8dp;\">{} x{}</div>",
                    escapeRmlText(LOC(Keys::REPEATED)), delivery.suppressed_repeats);
            }
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

            // Uniform opt-in Details affordance re-enqueuing the developer details
            // modal. Skip when a wired action already uses the localized Details
            // label so a real action is never shadowed.
            const std::string details_label = LOC(Keys::DETAILS);
            bool details_collision = false;
            for (const WiredButton& button : wired) {
                if (button.label == details_label) {
                    details_collision = true;
                    break;
                }
            }
            if (!details_collision) {
                request.buttons.push_back(
                    lfs::core::ModalButtonSpec{.label = details_label, .style = "secondary"});
            }

            request.on_result = [wired = std::move(wired), details_label, details_collision,
                                 details_request = buildDetailsModal(notification.error),
                                 modal_sink = sinks_.modal](const lfs::core::ModalResult& result) {
                if (!details_collision && result.button_label == details_label) {
                    if (modal_sink)
                        modal_sink(details_request);
                    return;
                }
                for (const WiredButton& button : wired) {
                    if (button.label == result.button_label && button.on_invoke) {
                        // The action starts a NEW operation; the source error is
                        // const and never mutated by an action (frozen contract).
                        button.on_invoke(lfs::OperationId::generate());
                        return;
                    }
                }
            };

            sinks_.modal(std::move(request));
        } catch (...) {
            // LFS-CENSUS-OK(empty-catch): on_error is a noexcept enqueue-only
            // boundary run on the publishing worker thread; a formatting/alloc
            // failure must degrade to no surface rather than terminate the worker.
        }
    }

} // namespace lfs::vis::gui
