/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "file_transfer_progress.hpp"
#include <cstdio>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <shlobj.h>
#include <windows.h>
#elif defined(__linux__)
#include <gtk/gtk.h>
#endif

namespace lfs::vis::gui {
    struct FileTransferProgress::Impl {
        std::shared_ptr<FileTransferProgressState> state;
        std::string title, message, cancel_label;
        std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now();
        bool attempted = false;
#ifdef _WIN32
        HRESULT initialized = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
        IProgressDialog* dialog = nullptr;
        static std::wstring wide(const std::string& value) {
            if (value.empty())
                return {};
            const int count = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.c_str(), -1, nullptr, 0);
            if (count <= 0)
                return {};
            std::wstring result(static_cast<size_t>(count), L'\0');
            MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.c_str(), -1, result.data(), count);
            result.resize(static_cast<size_t>(count - 1));
            return result;
        }
#elif defined(__linux__)
        GtkWidget* dialog = nullptr;
        GtkWidget* bar = nullptr;
#endif
        ~Impl() {
#ifdef _WIN32
            if (dialog) {
                dialog->StopProgressDialog();
                dialog->Release();
            }
            if (SUCCEEDED(initialized))
                CoUninitialize();
#elif defined(__linux__)
            if (dialog)
                gtk_widget_destroy(dialog);
#endif
        }
        void show() {
            attempted = true;
#ifdef _WIN32
            if (FAILED(initialized) || FAILED(CoCreateInstance(CLSID_ProgressDialog, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&dialog))))
                return;
            dialog->SetTitle(wide(title).c_str());
            dialog->SetLine(1, wide(message).c_str(), FALSE, nullptr);
            // The Shell maintains a separate UI thread. Do not pump the app's
            // frame or disable the source window during a receiver's GetData.
            if (FAILED(dialog->StartProgressDialog(nullptr, nullptr, PROGDLG_NORMAL | PROGDLG_NOMINIMIZE | PROGDLG_NOTIME, nullptr))) {
                dialog->Release();
                dialog = nullptr;
            }
#elif defined(__linux__)
            if (!gtk_init_check(nullptr, nullptr))
                return;
            dialog = gtk_dialog_new();
            gtk_window_set_title(GTK_WINDOW(dialog), title.c_str());
            gtk_window_set_default_size(GTK_WINDOW(dialog), 420, -1);
            gtk_window_set_resizable(GTK_WINDOW(dialog), FALSE);
            gtk_window_set_focus_on_map(GTK_WINDOW(dialog), FALSE);
            gtk_dialog_add_button(GTK_DIALOG(dialog), cancel_label.c_str(), GTK_RESPONSE_CANCEL);
            auto* area = gtk_dialog_get_content_area(GTK_DIALOG(dialog));
            gtk_container_set_border_width(GTK_CONTAINER(area), 16);
            gtk_box_set_spacing(GTK_BOX(area), 12);
            auto* label = gtk_label_new(message.c_str());
            gtk_label_set_line_wrap(GTK_LABEL(label), TRUE);
            gtk_box_pack_start(GTK_BOX(area), label, FALSE, FALSE, 0);
            bar = gtk_progress_bar_new();
            gtk_progress_bar_set_show_text(GTK_PROGRESS_BAR(bar), TRUE);
            gtk_box_pack_start(GTK_BOX(area), bar, FALSE, FALSE, 0);
            g_signal_connect(dialog, "response", G_CALLBACK(+[](GtkDialog* widget, gint, gpointer data) {
                                 auto& self = *static_cast<Impl*>(data);
                                 self.state->requestCancel();
                                 gtk_widget_hide(GTK_WIDGET(widget));
                             }),
                             this);
            // Keep the widget alive until the worker has stopped. Window close
            // requests cancel just like the button, rather than dangling bar.
            g_signal_connect(dialog, "delete-event", G_CALLBACK(+[](GtkWidget* widget, GdkEvent*, gpointer data) -> gboolean {
                                 static_cast<Impl*>(data)->state->requestCancel();
                                 gtk_widget_hide(widget);
                                 return TRUE;
                             }),
                             this);
            gtk_widget_show_all(dialog);
#endif
        }
        bool update(int value) {
            // Fast exports do not flash a window. This is a data-request delay,
            // not preparation on hover or a requirement for a second gesture.
            bool may_show = true;
#ifdef _WIN32
            // A receiver can inspect data during hover. Do not create a Shell
            // window that could take focus while the user still holds the drag.
            may_show = !(GetAsyncKeyState(VK_LBUTTON) & 0x8000);
#endif
            if (!attempted && may_show && std::chrono::steady_clock::now() - started >= std::chrono::milliseconds(250))
                show();
            value = std::clamp(value, 0, 1000);
#ifdef _WIN32
            if (dialog) {
                dialog->SetProgress64(static_cast<ULONGLONG>(value), 1000);
                if (dialog->HasUserCancelled())
                    state->requestCancel();
            }
#elif defined(__linux__)
            if (dialog) {
                gtk_progress_bar_set_fraction(GTK_PROGRESS_BAR(bar), value / 1000.0);
                char percent[16];
                std::snprintf(percent, sizeof(percent), "%d%%", value / 10);
                gtk_progress_bar_set_text(GTK_PROGRESS_BAR(bar), percent);
                // Bound work per tick. Do not dispatch SDL's Wayland queue or
                // reenter scene/render commands while the snapshot is in use.
                for (int i = 0; i < 16 && g_main_context_pending(nullptr); ++i)
                    g_main_context_iteration(nullptr, FALSE);
            }
#endif
#if defined(_WIN32) || defined(__linux__)
            return !attempted || dialog != nullptr;
#else
            return true;
#endif
        }
    };
    FileTransferProgress::FileTransferProgress(std::shared_ptr<FileTransferProgressState> state,
                                               std::string title, std::string message, std::string cancel_label)
        : impl_(std::make_unique<Impl>()) {
        impl_->state = std::move(state);
        impl_->title = std::move(title);
        impl_->message = std::move(message);
        impl_->cancel_label = std::move(cancel_label);
    }
    FileTransferProgress::~FileTransferProgress() = default;
    bool FileTransferProgress::update(int completed) { return impl_->update(completed); }
} // namespace lfs::vis::gui
