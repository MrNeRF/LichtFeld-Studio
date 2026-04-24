/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/panels/python_console_panel.hpp"
#include "core/events.hpp"
#include "core/path_utils.hpp"
#include "gui/editor/python_editor.hpp"
#include "gui/gui_focus_state.hpp"
#include "gui/panel_layout.hpp"
#include "gui/rmlui/elements/python_editor_element.hpp"
#include "gui/rmlui/elements/terminal_element.hpp"
#include "gui/rmlui/rml_panel_host.hpp"
#include "gui/rmlui/rmlui_manager.hpp"
#include "gui/terminal/terminal_widget.hpp"
#include "gui/ui_widgets.hpp"
#include "gui/utils/native_file_dialog.hpp"
#include "theme/theme.hpp"

#include <RmlUi/Core/ElementDocument.h>
#include <RmlUi/Core/Event.h>
#include <RmlUi/Core/EventListener.h>
#include <RmlUi/Core/StringUtilities.h>
#include <RmlUi/Core/Elements/ElementFormControlInput.h>
#include <SDL3/SDL_scancode.h>
#include <chrono>
#include <cfloat>
#include <fstream>
#include <future>
#include <optional>
#include <sstream>
#include <thread>
#include <utility>
#include <imgui.h>

#include "python/python_compat.hpp"
#include <filesystem>
#include <mutex>

#include "python/gil.hpp"

#include "core/executable_path.hpp"
#include "core/services.hpp"
#include "python/package_manager.hpp"
#include "python/python_runtime.hpp"
#include "python/runner.hpp"
#include "scene/scene_manager.hpp"

namespace {
    std::once_flag g_console_init_once;
    std::once_flag g_syspath_init_once;
    lfs::vis::gui::panels::PythonConsoleState* g_python_console_state = nullptr;

    bool should_block_editor_input(const lfs::vis::editor::PythonEditor* editor,
                                   lfs::vis::gui::panels::PythonConsoleState& state) {
        bool block_editor_input = false;

        if (const auto* terminal = state.getTerminal()) {
            block_editor_input |= terminal->isFocused();
        }

        // Ignore the editor's own capture state; only external text widgets should lock it out.
        if (!editor || !editor->isFocused()) {
            block_editor_input |= lfs::vis::gui::guiFocusState().want_text_input;
        }

        return block_editor_input;
    }

    void format_editor_script(lfs::vis::gui::panels::PythonConsoleState& state) {
        auto* editor = state.getEditor();
        if (!editor) {
            return;
        }

        const std::string original = editor->getText();
        const auto result = lfs::python::format_python_code(original);
        if (!result.success) {
            editor->refreshSyntaxDiagnostics();
            if (!result.error.empty()) {
                state.addError("[Format] " + result.error);
            }
            return;
        }

        if (result.code != original) {
            editor->setText(result.code);
            state.setModified(true);
        }

        editor->focus();
    }

    void clean_editor_script(lfs::vis::gui::panels::PythonConsoleState& state) {
        auto* editor = state.getEditor();
        if (!editor) {
            return;
        }

        const std::string original = editor->getText();
        const auto result = lfs::python::clean_python_code(original);
        if (!result.success) {
            editor->refreshSyntaxDiagnostics();
            if (!result.error.empty()) {
                state.addError("[Cleanup] " + result.error);
            }
            return;
        }

        if (result.code != original) {
            editor->setText(result.code);
            state.setModified(true);
        }

        editor->focus();
    }

    struct RmlTerminalPane {
        std::unique_ptr<lfs::vis::gui::RmlPanelHost> host;
        lfs::vis::gui::TerminalElement* view = nullptr;
        lfs::vis::gui::RmlUIManager* manager = nullptr;
    };

    struct RmlEditorPane {
        std::unique_ptr<lfs::vis::gui::RmlPanelHost> host;
        lfs::vis::gui::PythonEditorElement* view = nullptr;
        lfs::vis::gui::RmlUIManager* manager = nullptr;
    };

    RmlEditorPane g_editor_pane;
    RmlTerminalPane g_output_terminal_pane;
    RmlTerminalPane g_repl_terminal_pane;

    struct RmlPackagesPane;

    void handle_packages_event(RmlPackagesPane& pane, Rml::Event& event);

    struct PackagesPaneListener : Rml::EventListener {
        RmlPackagesPane* owner = nullptr;
        void ProcessEvent(Rml::Event& event) override {
            if (owner)
                handle_packages_event(*owner, event);
        }
    };

    struct RmlPackagesPane {
        RmlPackagesPane() { listener.owner = this; }

        std::unique_ptr<lfs::vis::gui::RmlPanelHost> host;
        lfs::vis::gui::RmlUIManager* manager = nullptr;
        Rml::ElementDocument* document = nullptr;
        Rml::Element* refresh_button = nullptr;
        Rml::ElementFormControlInput* search_input = nullptr;
        Rml::Element* status_label = nullptr;
        Rml::Element* table_el = nullptr;
        Rml::Element* body_el = nullptr;
        Rml::Element* empty_el = nullptr;
        PackagesPaneListener listener;

        std::vector<lfs::python::PackageInfo> packages;
        std::future<std::vector<lfs::python::PackageInfo>> pending_refresh;
        bool loading = false;
        bool loaded_once = false;
        bool listeners_attached = false;
        std::string search_filter;
        std::string last_body_rml;
        std::string last_status_text;
        bool last_empty_visible = false;
    };

    RmlPackagesPane g_packages_pane;

    void reset_rml_terminal_pane(RmlTerminalPane& pane) {
        pane.view = nullptr;
        pane.host.reset();
        pane.manager = nullptr;
    }

    void reset_rml_editor_pane(RmlEditorPane& pane) {
        pane.view = nullptr;
        pane.host.reset();
        pane.manager = nullptr;
    }

    void reset_rml_packages_pane(RmlPackagesPane& pane) {
        pane.refresh_button = nullptr;
        pane.search_input = nullptr;
        pane.status_label = nullptr;
        pane.table_el = nullptr;
        pane.body_el = nullptr;
        pane.empty_el = nullptr;
        pane.document = nullptr;
        pane.host.reset();
        pane.manager = nullptr;
        pane.listeners_attached = false;
        pane.last_body_rml.clear();
        pane.last_status_text.clear();
    }

    void reset_rml_terminal_panes() {
        reset_rml_editor_pane(g_editor_pane);
        reset_rml_terminal_pane(g_output_terminal_pane);
        reset_rml_terminal_pane(g_repl_terminal_pane);
        reset_rml_packages_pane(g_packages_pane);
    }

    std::optional<lfs::vis::terminal::TerminalKey> terminal_key_from_scancode(int scancode) {
        using lfs::vis::terminal::TerminalKey;
        switch (scancode) {
        case SDL_SCANCODE_RETURN:
        case SDL_SCANCODE_KP_ENTER:
            return TerminalKey::Enter;
        case SDL_SCANCODE_BACKSPACE:
            return TerminalKey::Backspace;
        case SDL_SCANCODE_TAB:
            return TerminalKey::Tab;
        case SDL_SCANCODE_ESCAPE:
            return TerminalKey::Escape;
        case SDL_SCANCODE_UP:
            return TerminalKey::Up;
        case SDL_SCANCODE_DOWN:
            return TerminalKey::Down;
        case SDL_SCANCODE_RIGHT:
            return TerminalKey::Right;
        case SDL_SCANCODE_LEFT:
            return TerminalKey::Left;
        case SDL_SCANCODE_HOME:
            return TerminalKey::Home;
        case SDL_SCANCODE_END:
            return TerminalKey::End;
        case SDL_SCANCODE_PAGEUP:
            return TerminalKey::PageUp;
        case SDL_SCANCODE_PAGEDOWN:
            return TerminalKey::PageDown;
        case SDL_SCANCODE_DELETE:
            return TerminalKey::Delete;
        case SDL_SCANCODE_INSERT:
            return TerminalKey::Insert;
        case SDL_SCANCODE_F1:
            return TerminalKey::F1;
        case SDL_SCANCODE_F2:
            return TerminalKey::F2;
        case SDL_SCANCODE_F3:
            return TerminalKey::F3;
        case SDL_SCANCODE_F4:
            return TerminalKey::F4;
        case SDL_SCANCODE_F5:
            return TerminalKey::F5;
        case SDL_SCANCODE_F6:
            return TerminalKey::F6;
        case SDL_SCANCODE_F7:
            return TerminalKey::F7;
        case SDL_SCANCODE_F8:
            return TerminalKey::F8;
        case SDL_SCANCODE_F9:
            return TerminalKey::F9;
        case SDL_SCANCODE_F10:
            return TerminalKey::F10;
        case SDL_SCANCODE_F11:
            return TerminalKey::F11;
        case SDL_SCANCODE_F12:
            return TerminalKey::F12;
        default:
            return std::nullopt;
        }
    }

    lfs::vis::gui::PythonEditorElement* ensure_rml_editor_view(
        RmlEditorPane& pane,
        lfs::vis::gui::RmlUIManager* manager) {
        if (!manager || !manager->isInitialized())
            return nullptr;

        if (pane.manager != manager) {
            reset_rml_editor_pane(pane);
            pane.manager = manager;
        }

        if (!pane.host) {
            pane.host = std::make_unique<lfs::vis::gui::RmlPanelHost>(
                manager, "python_console_editor", "rmlui/python_editor_pane.rml");
        }

        if (!pane.host->ensureDocumentLoaded())
            return nullptr;

        if (!pane.view) {
            auto* doc = pane.host->getDocument();
            pane.view = doc
                            ? dynamic_cast<lfs::vis::gui::PythonEditorElement*>(
                                  doc->GetElementById("python-editor-view"))
                            : nullptr;
        }
        return pane.view;
    }

    bool draw_rml_editor_pane(RmlEditorPane& pane,
                              lfs::vis::gui::RmlUIManager* manager,
                              lfs::vis::editor::PythonEditor& editor,
                              const lfs::vis::gui::PanelInputState* input,
                              ImFont* mono_font) {
        const ImVec2 size = ImGui::GetContentRegionAvail();
        if (size.x <= 0.0f || size.y <= 0.0f)
            return false;

        auto* view = ensure_rml_editor_view(pane, manager);
        if (!view) {
            ImGui::TextDisabled("Editor view unavailable");
            return false;
        }

        const float font_size = mono_font ? mono_font->LegacySize : ImGui::GetTextLineHeight();
        view->setEditor(&editor);
        view->setFontSizePx(font_size);
        view->SetProperty("font-size", std::format("{:.0f}px", font_size));

        const ImVec2 pos = ImGui::GetCursorScreenPos();
        pane.host->markContentDirty();
        pane.host->setInput(input);
        if (input) {
            pane.host->drawDirect(pos.x, pos.y, size.x, size.y);
        } else {
            lfs::vis::gui::PanelDrawContext draw_ctx;
            pane.host->draw(draw_ctx, size.x, size.y, pos.x, pos.y);
        }
        pane.host->setInput(nullptr);
        if (input) {
            ImGui::Dummy(size);
        }
        return editor.consumeExecuteRequested();
    }

    lfs::vis::gui::TerminalElement* ensure_rml_terminal_view(RmlTerminalPane& pane,
                                                              lfs::vis::gui::RmlUIManager* manager,
                                                              const char* context_name) {
        if (!manager || !manager->isInitialized())
            return nullptr;

        if (pane.manager != manager) {
            reset_rml_terminal_pane(pane);
            pane.manager = manager;
        }

        if (!pane.host) {
            pane.host = std::make_unique<lfs::vis::gui::RmlPanelHost>(
                manager, context_name, "rmlui/python_terminal_pane.rml");
        }

        if (!pane.host->ensureDocumentLoaded())
            return nullptr;

        if (!pane.view) {
            auto* doc = pane.host->getDocument();
            pane.view = doc
                            ? dynamic_cast<lfs::vis::gui::TerminalElement*>(
                                  doc->GetElementById("terminal-view"))
                            : nullptr;
        }
        return pane.view;
    }

    void process_rml_terminal_input(lfs::vis::terminal::TerminalWidget& terminal,
                                    const lfs::vis::gui::PanelInputState* input,
                                    const ImVec2& pos,
                                    const ImVec2& size,
                                    float char_w,
                                    float char_h) {
        if (!input || size.x <= 0.0f || size.y <= 0.0f || char_w <= 0.0f || char_h <= 0.0f)
            return;

        const bool hovered =
            input->mouse_x >= pos.x && input->mouse_x < pos.x + size.x &&
            input->mouse_y >= pos.y && input->mouse_y < pos.y + size.y;

        const auto mouse_cell = [&]() {
            const int col = static_cast<int>((input->mouse_x - pos.x) / char_w);
            const int row = static_cast<int>((input->mouse_y - pos.y) / char_h);
            return std::pair<int, int>{row, col};
        };

        if (input->mouse_clicked[0]) {
            terminal.setFocused(hovered);
            if (hovered) {
                const auto [row, col] = mouse_cell();
                terminal.beginSelection(row, col);
            }
        }

        if (terminal.isFocused() && input->mouse_down[0]) {
            const auto [row, col] = mouse_cell();
            terminal.updateSelection(row, col);
        }

        if (input->mouse_released[0]) {
            terminal.endSelection();
            if (terminal.hasSelection()) {
                const std::string selection = terminal.getSelection();
                if (!selection.empty())
                    ImGui::SetClipboardText(selection.c_str());
            }
        }

        if (hovered && input->mouse_wheel != 0.0f) {
            if (input->mouse_wheel > 0.0f)
                terminal.scrollUp(3);
            else
                terminal.scrollDown(3);
        }

        if (!terminal.isFocused() || terminal.isReadOnly())
            return;

        auto& focus = lfs::vis::gui::guiFocusState();
        focus.want_capture_keyboard = true;
        focus.want_text_input = true;

        if (input->key_ctrl && input->key_shift) {
            for (int sc : input->keys_pressed) {
                if (sc == SDL_SCANCODE_V) {
                    if (const char* clipboard = ImGui::GetClipboardText())
                        terminal.paste(clipboard);
                    return;
                }
            }
        }

        for (int sc : input->keys_pressed) {
            if (input->key_ctrl && sc >= SDL_SCANCODE_A && sc <= SDL_SCANCODE_Z) {
                const char letter = static_cast<char>('A' + (sc - SDL_SCANCODE_A));
                if (letter == 'C' && terminal.hasSelection()) {
                    const std::string selection = terminal.getSelection();
                    if (!selection.empty())
                        ImGui::SetClipboardText(selection.c_str());
                } else {
                    terminal.sendControl(letter);
                }
                continue;
            }

            if (const auto key = terminal_key_from_scancode(sc)) {
                terminal.sendKey(*key);
            }
        }

        if (!input->key_ctrl) {
            for (const uint32_t cp : input->text_codepoints)
                terminal.sendCodepoint(cp);
        }
    }

    void draw_rml_terminal_pane(RmlTerminalPane& pane,
                                lfs::vis::gui::RmlUIManager* manager,
                                const char* context_name,
                                lfs::vis::terminal::TerminalWidget& terminal,
                                const lfs::vis::gui::PanelInputState* input,
                                ImFont* mono_font) {
        const ImVec2 size = ImGui::GetContentRegionAvail();
        if (size.x <= 0.0f || size.y <= 0.0f)
            return;

        if (!manager) {
            ImGui::TextDisabled("Terminal view unavailable");
            return;
        }

        const float font_size = mono_font ? mono_font->LegacySize : ImGui::GetTextLineHeight();
        const float char_h = std::max(1.0f, font_size);
        const float char_w = mono_font
                                 ? std::max(1.0f, mono_font->CalcTextSizeA(font_size, FLT_MAX, 0.0f, "M").x)
                                 : std::max(1.0f, ImGui::CalcTextSize("M").x);
        const int cols = std::max(1, static_cast<int>(size.x / char_w));
        const int rows = std::max(1, static_cast<int>(size.y / char_h));

        terminal.resize(cols, rows);
        terminal.update();

        auto* view = ensure_rml_terminal_view(pane, manager, context_name);
        if (!view) {
            ImGui::TextDisabled("Terminal view unavailable");
            return;
        }

        const ImVec2 pos = ImGui::GetCursorScreenPos();
        process_rml_terminal_input(terminal, input, pos, size, char_w, char_h);

        const bool dirty = terminal.needsRedraw();
        view->SetProperty("font-size", std::format("{:.0f}px", font_size));
        view->SetProperty("line-height", std::format("{:.0f}px", char_h));
        view->setSnapshot(terminal.snapshot());
        if (dirty)
            pane.host->markContentDirty();
        terminal.markRendered();

        pane.host->setInput(input);
        if (input) {
            pane.host->drawDirect(pos.x, pos.y, size.x, size.y);
        } else {
            lfs::vis::gui::PanelDrawContext draw_ctx;
            pane.host->draw(draw_ctx, size.x, size.y, pos.x, pos.y);
        }
        pane.host->setInput(nullptr);
        if (input) {
            ImGui::Dummy(size);
        }
    }

    void request_packages_refresh(RmlPackagesPane& pane) {
        if (pane.loading)
            return;

        pane.loading = true;
        pane.loaded_once = true;
        pane.pending_refresh = std::async(std::launch::async, [] {
            return lfs::python::PackageManager::instance().list_installed();
        });
        if (pane.host)
            pane.host->markContentDirty();
    }

    void clear_packages_cache(RmlPackagesPane& pane) {
        pane.document = nullptr;
        pane.refresh_button = nullptr;
        pane.search_input = nullptr;
        pane.status_label = nullptr;
        pane.table_el = nullptr;
        pane.body_el = nullptr;
        pane.empty_el = nullptr;
        pane.listeners_attached = false;
        pane.last_body_rml.clear();
        pane.last_status_text.clear();
    }

    bool ensure_packages_pane(RmlPackagesPane& pane, lfs::vis::gui::RmlUIManager* manager) {
        if (!manager || !manager->isInitialized())
            return false;

        if (pane.manager != manager) {
            reset_rml_packages_pane(pane);
            pane.manager = manager;
        }

        if (!pane.host) {
            pane.host = std::make_unique<lfs::vis::gui::RmlPanelHost>(
                manager, "python_console_packages", "rmlui/python_packages_pane.rml");
        }

        if (!pane.host->ensureDocumentLoaded())
            return false;

        auto* doc = pane.host->getDocument();
        if (pane.document != doc) {
            clear_packages_cache(pane);
            pane.document = doc;
        }

        if (!pane.document)
            return false;

        if (!pane.refresh_button)
            pane.refresh_button = pane.document->GetElementById("packages-refresh");
        if (!pane.search_input) {
            pane.search_input = dynamic_cast<Rml::ElementFormControlInput*>(
                pane.document->GetElementById("packages-search"));
        }
        if (!pane.status_label)
            pane.status_label = pane.document->GetElementById("packages-status");
        if (!pane.table_el)
            pane.table_el = pane.document->GetElementById("packages-table");
        if (!pane.body_el)
            pane.body_el = pane.document->GetElementById("packages-body");
        if (!pane.empty_el)
            pane.empty_el = pane.document->GetElementById("packages-empty");

        if (!pane.listeners_attached) {
            if (pane.refresh_button)
                pane.refresh_button->AddEventListener(Rml::EventId::Click, &pane.listener);
            if (pane.search_input) {
                pane.search_input->AddEventListener("change", &pane.listener);
                pane.search_input->AddEventListener("input", &pane.listener);
            }
            pane.listeners_attached = true;
        }

        return true;
    }

    void handle_packages_event(RmlPackagesPane& pane, Rml::Event& event) {
        const std::string type = event.GetType();
        auto* current = event.GetCurrentElement();
        auto* target = event.GetTargetElement();
        const Rml::String current_id = current ? current->GetId() : "";
        const Rml::String target_id = target ? target->GetId() : "";

        if (type == "click" && (current_id == "packages-refresh" || target_id == "packages-refresh")) {
            request_packages_refresh(pane);
            event.StopPropagation();
            return;
        }

        if ((type == "change" || type == "input") && current_id == "packages-search") {
            if (pane.search_input)
                pane.search_filter = pane.search_input->GetValue();
            if (pane.host)
                pane.host->markContentDirty();
            event.StopPropagation();
        }
    }

    bool package_matches_filter(const lfs::python::PackageInfo& pkg, const std::string& filter) {
        if (filter.empty())
            return true;
        return pkg.name.find(filter) != std::string::npos ||
               pkg.version.find(filter) != std::string::npos ||
               pkg.path.find(filter) != std::string::npos;
    }

    void sync_packages_pane(RmlPackagesPane& pane) {
        if (pane.loading && pane.pending_refresh.valid() &&
            pane.pending_refresh.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
            pane.packages = pane.pending_refresh.get();
            pane.loading = false;
            if (pane.host)
                pane.host->markContentDirty();
        }

        if (pane.search_input) {
            pane.search_filter = pane.search_input->GetValue();
        }

        std::string rows;
        rows.reserve(pane.packages.size() * 192);
        std::size_t visible_count = 0;
        for (const auto& pkg : pane.packages) {
            if (!package_matches_filter(pkg, pane.search_filter))
                continue;
            ++visible_count;
            rows += std::format(
                R"(<div class="pkg-row"><span class="pkg-name">{}</span><span class="pkg-version">{}</span><span class="pkg-path">{}</span></div>)",
                Rml::StringUtilities::EncodeRml(pkg.name),
                Rml::StringUtilities::EncodeRml(pkg.version),
                Rml::StringUtilities::EncodeRml(pkg.path));
        }

        if (pane.body_el && rows != pane.last_body_rml) {
            pane.body_el->SetInnerRML(rows);
            pane.last_body_rml = std::move(rows);
            if (pane.host)
                pane.host->markContentDirty();
        }

        std::string status;
        if (pane.loading) {
            status = "Loading...";
        } else if (pane.search_filter.empty()) {
            status = std::format("({})", pane.packages.size());
        } else {
            status = std::format("({} / {})", visible_count, pane.packages.size());
        }

        if (pane.status_label && status != pane.last_status_text) {
            pane.status_label->SetInnerRML(Rml::StringUtilities::EncodeRml(status));
            pane.last_status_text = std::move(status);
            if (pane.host)
                pane.host->markContentDirty();
        }

        const bool empty_visible = !pane.loading && visible_count == 0;
        if (pane.empty_el && empty_visible != pane.last_empty_visible) {
            pane.empty_el->SetProperty("display", empty_visible ? "block" : "none");
            pane.last_empty_visible = empty_visible;
            if (pane.host)
                pane.host->markContentDirty();
        }
        if (pane.table_el) {
            pane.table_el->SetProperty("display", empty_visible ? "none" : "block");
        }
    }

    void draw_rml_packages_pane(RmlPackagesPane& pane,
                                lfs::vis::gui::RmlUIManager* manager,
                                const lfs::vis::gui::PanelInputState* input) {
        const ImVec2 size = ImGui::GetContentRegionAvail();
        if (size.x <= 0.0f || size.y <= 0.0f)
            return;

        if (!ensure_packages_pane(pane, manager)) {
            ImGui::TextDisabled("Packages view unavailable");
            return;
        }

        if (!pane.loaded_once)
            request_packages_refresh(pane);

        sync_packages_pane(pane);

        const ImVec2 pos = ImGui::GetCursorScreenPos();
        pane.host->setInput(input);
        if (input) {
            pane.host->drawDirect(pos.x, pos.y, size.x, size.y);
        } else {
            lfs::vis::gui::PanelDrawContext draw_ctx;
            pane.host->draw(draw_ctx, size.x, size.y, pos.x, pos.y);
        }
        pane.host->setInput(nullptr);
        if (input) {
            ImGui::Dummy(size);
        }
    }

    void draw_vim_mode_button(lfs::vis::gui::panels::PythonConsoleState& state,
                              const lfs::vis::Theme& t) {
        auto* editor = state.getEditor();
        const bool enabled = editor && editor->isVimModeEnabled();

        if (enabled) {
            ImGui::PushStyleColor(ImGuiCol_Button, t.button_selected());
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, t.button_selected_hovered());
            ImGui::PushStyleColor(ImGuiCol_ButtonActive,
                                  lfs::vis::darken(t.button_selected_hovered(), 0.05f));
        }
        if (!editor) {
            ImGui::BeginDisabled();
        }

        if (ImGui::Button("Vim") && editor) {
            editor->setVimModeEnabled(!enabled);
            editor->focus();
        }

        if (!editor) {
            ImGui::EndDisabled();
        }
        if (enabled) {
            ImGui::PopStyleColor(3);
        }

        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
            ImGui::SetTooltip(enabled ? "Disable Vim mode" : "Enable Vim mode");
        }
    }

    void draw_syntax_status(lfs::vis::gui::panels::PythonConsoleState& state,
                            const lfs::vis::Theme& t) {
        auto* editor = state.getEditor();
        if (editor == nullptr) {
            ImGui::TextColored(t.palette.text_dim, "Syntax");
            return;
        }

        const std::string summary = editor->syntaxSummary();
        if (editor->hasSyntaxErrors()) {
            ImGui::TextColored(t.palette.error, "Syntax error");
        } else if (editor->syntaxDiagnosticsAvailable()) {
            ImGui::TextColored(t.palette.success, "Syntax OK");
        } else {
            ImGui::TextColored(t.palette.text_dim, "Syntax");
        }

        if (ImGui::IsItemHovered() && ImGui::BeginTooltip()) {
            ImGui::TextUnformatted(summary.c_str());
            const std::string structure = editor->syntaxStructureSummary();
            if (!structure.empty()) {
                ImGui::TextColored(t.palette.text_dim, "%s", structure.c_str());
            }
            const std::string scope = editor->currentSyntaxScope();
            if (!scope.empty()) {
                ImGui::TextColored(t.palette.text_dim, "Scope: %s", scope.c_str());
            }
            ImGui::EndTooltip();
        }
    }

    void draw_syntax_outline_control(lfs::vis::gui::panels::PythonConsoleState& state,
                                     const lfs::vis::Theme&,
                                     const char* id) {
        auto* editor = state.getEditor();
        const auto symbols = editor != nullptr ? editor->syntaxSymbols()
                                               : std::vector<lfs::vis::editor::PythonEditorSymbol>{};
        const bool has_symbols = !symbols.empty();

        if (!has_symbols) {
            ImGui::BeginDisabled();
        }

        ImGui::SetNextItemWidth(180.0f);
        if (ImGui::BeginCombo(id, "Outline")) {
            for (std::size_t i = 0; i < symbols.size(); ++i) {
                const auto& symbol = symbols[i];
                if (ImGui::Selectable(symbol.label.c_str(), false)) {
                    editor->jumpToSyntaxSymbol(i);
                }
                if (ImGui::IsItemHovered() && !symbol.detail.empty()) {
                    ImGui::SetTooltip("%s", symbol.detail.c_str());
                }
            }
            ImGui::EndCombo();
        }

        if (!has_symbols) {
            ImGui::EndDisabled();
        }

        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
            if (editor != nullptr && !editor->syntaxStructureCurrent()) {
                ImGui::SetTooltip("Jump to Python symbol (partial syntax structure)");
            } else {
                ImGui::SetTooltip("Jump to Python symbol");
            }
        }
    }

    void draw_syntax_breadcrumb_control(lfs::vis::gui::panels::PythonConsoleState& state,
                                        const lfs::vis::Theme&,
                                        const char* id) {
        auto* editor = state.getEditor();
        const auto breadcrumbs = editor != nullptr
                                     ? editor->syntaxBreadcrumbs()
                                     : std::vector<lfs::vis::editor::PythonEditorSymbol>{};
        const std::string scope = editor != nullptr ? editor->currentSyntaxScope() : std::string{};
        const bool has_breadcrumbs = !breadcrumbs.empty();

        if (!has_breadcrumbs) {
            ImGui::BeginDisabled();
        }

        ImGui::SetNextItemWidth(150.0f);
        if (ImGui::BeginCombo(id, scope.empty() ? "Scope" : scope.c_str())) {
            for (std::size_t i = 0; i < breadcrumbs.size(); ++i) {
                const auto& breadcrumb = breadcrumbs[i];
                if (ImGui::Selectable(breadcrumb.label.c_str(), i + 1 == breadcrumbs.size())) {
                    editor->jumpToSyntaxBreadcrumb(i);
                }
                if (ImGui::IsItemHovered() && !breadcrumb.detail.empty()) {
                    ImGui::SetTooltip("%s", breadcrumb.detail.c_str());
                }
            }
            ImGui::EndCombo();
        }

        if (!has_breadcrumbs) {
            ImGui::EndDisabled();
        }

        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
            ImGui::SetTooltip("Jump within current Python scope");
        }
    }

    void draw_syntax_fold_control(lfs::vis::gui::panels::PythonConsoleState& state,
                                  const lfs::vis::Theme&,
                                  const char* id) {
        auto* editor = state.getEditor();
        const auto folds = editor != nullptr ? editor->syntaxFolds()
                                             : std::vector<lfs::vis::editor::PythonEditorFold>{};
        const bool has_folds = !folds.empty();

        if (!has_folds) {
            ImGui::BeginDisabled();
        }

        ImGui::SetNextItemWidth(120.0f);
        if (ImGui::BeginCombo(id, "Blocks")) {
            for (std::size_t i = 0; i < folds.size(); ++i) {
                const auto& fold = folds[i];
                if (ImGui::Selectable(fold.label.c_str(), false)) {
                    editor->toggleSyntaxFold(i);
                }
                if (ImGui::IsItemHovered() && !fold.detail.empty()) {
                    ImGui::SetTooltip("%s", fold.detail.c_str());
                }
            }
            ImGui::Separator();
            if (ImGui::Selectable("Fold all")) {
                editor->foldAllSyntaxBlocks();
            }
            if (ImGui::Selectable("Unfold all")) {
                editor->unfoldAllSyntaxBlocks();
            }
            ImGui::EndCombo();
        }

        if (!has_folds) {
            ImGui::EndDisabled();
        }

        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
            ImGui::SetTooltip("Fold or unfold Python blocks");
        }
    }

    void setup_sys_path() {
        std::call_once(g_syspath_init_once, [] {
            const lfs::python::GilAcquire gil;

            const auto python_module_dir = lfs::core::getPythonModuleDir();
            if (!python_module_dir.empty()) {
                PyObject* sys_path = PySys_GetObject("path");
                if (sys_path) {
                    PyObject* py_path = PyUnicode_FromString(python_module_dir.string().c_str());
                    if (py_path) {
                        PyList_Insert(sys_path, 0, py_path);
                        Py_DECREF(py_path);
                    }
                }
            }
        });
    }

    // Replace Braille (U+2800-28FF) with cycling block elements
    std::string replace_braille_with_blocks(const std::string& text) {
        static constexpr const char* BLOCKS[] = {"░", "▒", "▓", "█", "▓", "▒"};
        static constexpr size_t BLOCK_COUNT = 6;
        static constexpr uint8_t UTF8_BRAILLE_LEAD = 0xE2;
        static int cycle = 0;

        std::string result;
        result.reserve(text.size());

        for (size_t i = 0; i < text.size(); ++i) {
            const auto c = static_cast<uint8_t>(text[i]);
            if (c == UTF8_BRAILLE_LEAD && i + 2 < text.size()) {
                const auto b1 = static_cast<uint8_t>(text[i + 1]);
                const auto b2 = static_cast<uint8_t>(text[i + 2]);
                if (b1 >= 0xA0 && b1 <= 0xA3 && (b2 & 0xC0) == 0x80) {
                    result += BLOCKS[cycle++ % BLOCK_COUNT];
                    i += 2;
                    continue;
                }
            }
            result += text[i];
        }
        return result;
    }

    void setup_console_output_capture() {
        std::call_once(g_console_init_once, [] {
            lfs::python::set_output_callback([](const std::string& text, const bool is_error) {
                auto& state = lfs::vis::gui::panels::PythonConsoleState::getInstance();
                auto* output = state.getOutputTerminal();
                if (!output)
                    return;

                const std::string filtered = replace_braille_with_blocks(text);
                if (is_error) {
                    output->write("\033[31m");
                    output->write(filtered);
                    output->write("\033[0m");
                } else {
                    output->write(filtered);
                }
            });
        });
    }

    void execute_python_code(const std::string& code, lfs::vis::gui::panels::PythonConsoleState& state) {
        std::string cmd = code;

        while (!cmd.empty() && (cmd.back() == '\n' || cmd.back() == '\r' || cmd.back() == ' '))
            cmd.pop_back();

        const size_t start = cmd.find_first_not_of(" \t");
        if (start == std::string::npos)
            return;
        if (start > 0)
            cmd = cmd.substr(start);

        state.runScriptAsync(cmd);
    }

    void reset_python_state(lfs::vis::gui::panels::PythonConsoleState& state) {
        // Clear output terminal
        auto* output = state.getOutputTerminal();
        if (output) {
            output->clear();
        }
    }

    bool load_script(const std::filesystem::path& path, lfs::vis::gui::panels::PythonConsoleState& state) {
        std::ifstream file;
        if (!lfs::core::open_file_for_read(path, file)) {
            state.addError("Failed to open: " + lfs::core::path_to_utf8(path));
            return false;
        }

        std::string content((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());

        if (auto* editor = state.getEditor()) {
            editor->setText(content);
        }

        state.setScriptPath(path);
        state.setModified(false);
        state.addInfo("Loaded: " + lfs::core::path_to_utf8(path.filename()));
        return true;
    }

    bool save_script(const std::filesystem::path& path, lfs::vis::gui::panels::PythonConsoleState& state) {
        auto* editor = state.getEditor();
        if (!editor) {
            return false;
        }

        std::ofstream file;
        if (!lfs::core::open_file_for_write(path, file)) {
            state.addError("Failed to save: " + lfs::core::path_to_utf8(path));
            return false;
        }

        file << editor->getTextStripped();
        file.close();

        state.setScriptPath(path);
        state.setModified(false);
        state.addInfo("Saved: " + lfs::core::path_to_utf8(path.filename()));
        return true;
    }

    void open_script_dialog(lfs::vis::gui::panels::PythonConsoleState& state) {
        const auto& current = state.getScriptPath();
        const auto start_dir = current.empty() ? std::filesystem::path{} : current.parent_path();
        const auto path = lfs::vis::gui::OpenPythonFileDialog(start_dir);
        if (!path.empty()) {
            load_script(path, state);
        }
    }

    void save_script_dialog(lfs::vis::gui::panels::PythonConsoleState& state) {
        const auto& current = state.getScriptPath();
        const std::string default_name = current.empty() ? "script" : current.stem().string();
        const auto path = lfs::vis::gui::SavePythonFileDialog(default_name);
        if (!path.empty()) {
            save_script(path, state);
        }
    }

    void save_current_script(lfs::vis::gui::panels::PythonConsoleState& state) {
        const auto& current = state.getScriptPath();
        if (current.empty()) {
            save_script_dialog(state);
        } else {
            save_script(current, state);
        }
    }

} // namespace

namespace lfs::vis::gui::panels {

    PythonConsoleState::PythonConsoleState()
        : terminal_(std::make_unique<terminal::TerminalWidget>(80, 24)),
          output_terminal_(std::make_unique<terminal::TerminalWidget>(80, 24)),
          editor_(std::make_unique<editor::PythonEditor>()) {
        g_python_console_state = this;
    }

    PythonConsoleState::~PythonConsoleState() {
        interruptScript();
        if (script_thread_.joinable()) {
            script_thread_.join();
        }
        g_python_console_state = nullptr;
    }

    PythonConsoleState& PythonConsoleState::getInstance() {
        static PythonConsoleState instance;
        return instance;
    }

    PythonConsoleState* PythonConsoleState::tryGetInstance() {
        return g_python_console_state;
    }

    void PythonConsoleState::addOutput(const std::string& text, uint32_t /*color*/) {
        std::lock_guard lock(mutex_);
        if (output_terminal_) {
            output_terminal_->write(text);
            output_terminal_->write("\n");
        }
    }

    void PythonConsoleState::addError(const std::string& text) {
        std::lock_guard lock(mutex_);
        if (output_terminal_) {
            output_terminal_->write("\033[31m"); // Red
            output_terminal_->write(text);
            output_terminal_->write("\033[0m\n"); // Reset + newline
        }
    }

    void PythonConsoleState::addInput(const std::string& text) {
        std::lock_guard lock(mutex_);
        if (output_terminal_) {
            output_terminal_->write("\033[32m>>> "); // Green prompt
            output_terminal_->write(text);
            output_terminal_->write("\033[0m\n"); // Reset + newline
        }
    }

    void PythonConsoleState::addInfo(const std::string& text) {
        std::lock_guard lock(mutex_);
        if (output_terminal_) {
            output_terminal_->write("\033[36m"); // Cyan
            output_terminal_->write(text);
            output_terminal_->write("\033[0m\n"); // Reset + newline
        }
    }

    void PythonConsoleState::clear() {
        std::lock_guard lock(mutex_);
        if (output_terminal_) {
            output_terminal_->clear();
        }
    }

    void PythonConsoleState::interruptScript() {
        const unsigned long tid = script_thread_id_.load();
        if (tid != 0 && script_running_.load()) {
            const python::GilAcquire gil;
            PyThreadState_SetAsyncExc(tid, PyExc_KeyboardInterrupt);
        }
    }

    void PythonConsoleState::runScriptAsync(const std::string& code) {
        if (script_running_.load()) {
            addError("A script is already running");
            return;
        }

        if (script_thread_.joinable()) {
            script_thread_.join();
        }

        setup_console_output_capture();

        addToHistory(code);
        clear();
        setActiveTab(0);

        const auto script_path = script_path_;
        const auto code_chars = code.size();

        script_running_ = true;
        script_thread_id_ = 0;
        core::events::state::EditorScriptStarted{
            .path = script_path,
            .code_chars = code_chars,
        }
            .emit();

        script_thread_ = std::thread([this, code, script_path, code_chars]() {
            bool success = true;
            bool interrupted = false;

            {
                const python::GilAcquire gil;

                lfs::python::install_output_redirect();

                script_thread_id_ = PyThreadState_Get()->thread_id;

                lfs::core::Scene* scene = nullptr;
                if (auto* sm = lfs::vis::services().sceneOrNull()) {
                    scene = &sm->getScene();
                }

                lfs::python::SceneContextGuard ctx(scene);
                const int result = PyRun_SimpleString(code.c_str());
                if (result != 0) {
                    success = false;
                    interrupted = PyErr_ExceptionMatches(PyExc_KeyboardInterrupt);
                    PyErr_Print();
                }

                script_thread_id_ = 0;
            }
            script_running_ = false;

            core::events::state::EditorScriptCompleted{
                .path = script_path,
                .code_chars = code_chars,
                .output_chars = getOutputText().size(),
                .success = success,
                .interrupted = interrupted,
            }
                .emit();
        });
    }

    void PythonConsoleState::increaseFontScale() {
        for (int i = 0; i < FONT_STEP_COUNT; ++i) {
            if (FONT_STEPS[i] > font_scale_ + 0.01f) {
                font_scale_ = FONT_STEPS[i];
                return;
            }
        }
    }

    void PythonConsoleState::decreaseFontScale() {
        for (int i = FONT_STEP_COUNT - 1; i >= 0; --i) {
            if (FONT_STEPS[i] < font_scale_ - 0.01f) {
                font_scale_ = FONT_STEPS[i];
                return;
            }
        }
    }

    void PythonConsoleState::addToHistory(const std::string& cmd) {
        std::lock_guard lock(mutex_);
        if (!cmd.empty() && (command_history_.empty() || command_history_.back() != cmd)) {
            command_history_.push_back(cmd);
        }
        history_index_ = -1;
        if (editor_) {
            editor_->addToHistory(cmd);
        }
    }

    void PythonConsoleState::historyUp() {
        std::lock_guard lock(mutex_);
        if (command_history_.empty())
            return;
        if (history_index_ < 0) {
            history_index_ = static_cast<int>(command_history_.size()) - 1;
        } else if (history_index_ > 0) {
            history_index_--;
        }
    }

    void PythonConsoleState::historyDown() {
        std::lock_guard lock(mutex_);
        if (history_index_ < 0)
            return;
        if (history_index_ < static_cast<int>(command_history_.size()) - 1) {
            history_index_++;
        } else {
            history_index_ = -1;
        }
    }

    terminal::TerminalWidget* PythonConsoleState::getTerminal() {
        return terminal_.get();
    }

    terminal::TerminalWidget* PythonConsoleState::getOutputTerminal() {
        return output_terminal_.get();
    }

    editor::PythonEditor* PythonConsoleState::getEditor() {
        return editor_.get();
    }

    void PythonConsoleState::setEditorText(const std::string& text) {
        if (editor_) {
            editor_->setText(text);
        }
    }

    void PythonConsoleState::focusEditor() {
        if (editor_) {
            editor_->focus();
        }
    }

    std::string PythonConsoleState::getEditorText() const {
        if (!editor_) {
            return {};
        }
        return editor_->getText();
    }

    std::string PythonConsoleState::getEditorTextStripped() const {
        if (!editor_) {
            return {};
        }
        return editor_->getTextStripped();
    }

    std::string PythonConsoleState::getOutputText() const {
        if (!output_terminal_) {
            return {};
        }
        return output_terminal_->getAllText();
    }

    namespace {
        float g_splitter_ratio = 0.6f;
        constexpr float MIN_PANE_HEIGHT = 100.0f;
        constexpr float SPLITTER_THICKNESS = 6.0f;

    } // namespace

    void ShutdownPythonConsoleRml() {
        reset_rml_terminal_panes();
    }

    void DrawPythonConsole(const UIContext& ctx, bool* open) {
        if (!open || !*open)
            return;

        // Initialize Python and set up output capture
        lfs::python::ensure_initialized();
        lfs::python::install_output_redirect();
        setup_sys_path();
        setup_console_output_capture();

        auto& state = PythonConsoleState::getInstance();
        const auto& t = theme();

        // Build window title with script name and modified indicator
        std::string window_title = "Python Console";
        if (!state.getScriptPath().empty()) {
            window_title += " - " + lfs::core::path_to_utf8(state.getScriptPath().filename());
        }
        if (state.isModified()) {
            window_title += " *";
        }
        window_title += "###python_console";

        ImGui::SetNextWindowSize(ImVec2(700, 600), ImGuiCond_FirstUseEver);
        if (!ImGui::Begin(window_title.c_str(), open, ImGuiWindowFlags_MenuBar)) {
            ImGui::End();
            return;
        }

        // Menu bar
        if (ImGui::BeginMenuBar()) {
            if (ImGui::BeginMenu("File")) {
                if (ImGui::MenuItem("New Script", "Ctrl+N")) {
                    if (auto* editor = state.getEditor()) {
                        editor->clear();
                    }
                    state.setScriptPath({});
                    state.setModified(false);
                }
                ImGui::Separator();
                if (ImGui::MenuItem("Open...", "Ctrl+O")) {
                    open_script_dialog(state);
                }
                if (ImGui::MenuItem("Reload", "Ctrl+Shift+O", false, !state.getScriptPath().empty())) {
                    load_script(state.getScriptPath(), state);
                }
                if (ImGui::MenuItem("Save", "Ctrl+S")) {
                    save_current_script(state);
                }
                if (ImGui::MenuItem("Save As...")) {
                    save_script_dialog(state);
                }
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("Edit")) {
                if (ImGui::MenuItem("Clear Output", "Ctrl+L")) {
                    state.clear();
                }
                if (ImGui::MenuItem("Format Script", "Ctrl+Shift+F")) {
                    format_editor_script(state);
                }
                if (ImGui::MenuItem("Clean Pasted Code", "Ctrl+Shift+I")) {
                    clean_editor_script(state);
                }
                ImGui::Separator();
                if (ImGui::MenuItem("Copy Selection")) {
                    if (auto* output = state.getOutputTerminal()) {
                        ImGui::SetClipboardText(output->getSelection().c_str());
                    }
                }
                if (ImGui::MenuItem("Copy All")) {
                    if (auto* output = state.getOutputTerminal()) {
                        ImGui::SetClipboardText(output->getAllText().c_str());
                    }
                }
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("Run")) {
                if (ImGui::MenuItem("Run Script", "F5")) {
                    if (auto* editor = state.getEditor()) {
                        execute_python_code(editor->getTextStripped(), state);
                    }
                }
                ImGui::Separator();
                if (ImGui::MenuItem("Reset Python State", "Ctrl+R")) {
                    reset_python_state(state);
                }
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("Help")) {
                ImGui::MenuItem("Ctrl+Enter to execute", nullptr, false, false);
                ImGui::MenuItem("F5 to run script", nullptr, false, false);
                ImGui::MenuItem("Ctrl+R to reset state", nullptr, false, false);
                ImGui::EndMenu();
            }
            ImGui::EndMenuBar();
        }

        // Toolbar
        {
            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(4, 4));
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(8, 4));

            // Run button
            ImGui::PushStyleColor(ImGuiCol_Button, t.palette.success);
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, lighten(t.palette.success, 0.1f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, darken(t.palette.success, 0.1f));
            if (ImGui::Button("Run") || ImGui::IsKeyPressed(ImGuiKey_F5, false)) {
                if (auto* editor = state.getEditor()) {
                    execute_python_code(editor->getTextStripped(), state);
                }
            }
            ImGui::PopStyleColor(3);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Run script (F5)");
            }

            ImGui::SameLine();

            // Stop button (for animations, running scripts, and UV operations)
            const bool has_animation = python::has_frame_callback();
            const bool has_running_script = state.isScriptRunning();
            const bool has_running_terminal = state.getOutputTerminal() && state.getOutputTerminal()->is_running();
            const bool has_uv_operation = python::PackageManager::instance().has_running_operation();
            const bool can_stop = has_animation || has_running_script || has_running_terminal || has_uv_operation;
            if (!can_stop) {
                ImGui::BeginDisabled();
            }
            ImGui::PushStyleColor(ImGuiCol_Button, t.palette.error);
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, lighten(t.palette.error, 0.1f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, darken(t.palette.error, 0.1f));
            if (ImGui::Button("Stop")) {
                if (has_animation) {
                    python::clear_frame_callback();
                }
                if (has_running_script) {
                    state.interruptScript();
                }
                if (has_uv_operation) {
                    python::PackageManager::instance().cancel_async();
                }
                if (auto* output = state.getOutputTerminal()) {
                    output->interrupt();
                }
            }
            ImGui::PopStyleColor(3);
            if (!can_stop) {
                ImGui::EndDisabled();
            }
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
                ImGui::SetTooltip("Stop running script (Ctrl+C)");
            }

            ImGui::SameLine();

            // Reset button
            if (ImGui::Button("Reset")) {
                reset_python_state(state);
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Reset Python state (Ctrl+R)");
            }

            ImGui::SameLine();

            // Clear button
            if (ImGui::Button("Clear")) {
                state.clear();
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Clear console output (Ctrl+L)");
            }

            ImGui::SameLine();
            draw_vim_mode_button(state, t);

            ImGui::SameLine();
            ImGui::Separator();
            ImGui::SameLine();

            // Status indicator
            if (can_stop) {
                ImGui::TextColored(t.palette.warning, "Running...");
            } else {
                ImGui::TextColored(t.palette.text_dim, "Python");
            }

            ImGui::SameLine();
            ImGui::TextColored(t.palette.text_dim, "|");
            ImGui::SameLine();
            draw_syntax_status(state, t);
            ImGui::SameLine();
            draw_syntax_outline_control(state, t, "##python_outline");
            ImGui::SameLine();
            draw_syntax_breadcrumb_control(state, t, "##python_breadcrumb");
            ImGui::SameLine();
            draw_syntax_fold_control(state, t, "##python_blocks");

            ImGui::PopStyleVar(2);
        }

        ImGui::Spacing();
        ImGui::Separator();

        // Calculate pane sizes
        const ImVec2 content_avail = ImGui::GetContentRegionAvail();
        const float total_height = content_avail.y;

        float top_height = total_height * g_splitter_ratio - SPLITTER_THICKNESS / 2;
        float bottom_height = total_height * (1.0f - g_splitter_ratio) - SPLITTER_THICKNESS / 2;
        bool editor_has_active_completion = false;

        top_height = std::max(top_height, MIN_PANE_HEIGHT);
        bottom_height = std::max(bottom_height, MIN_PANE_HEIGHT);

        // Script Editor (top pane)
        ImGui::BeginChild("##script_editor_pane", ImVec2(content_avail.x, top_height), false);
        {
            ImGui::TextColored(t.palette.text_dim, "Script Editor");
            ImGui::Spacing();

            const ImVec2 editor_size(ImGui::GetContentRegionAvail().x,
                                     ImGui::GetContentRegionAvail().y);

            // Use monospace font for code editor
            if (ctx.fonts.monospace) {
                ImGui::PushFont(ctx.fonts.monospace);
            }

            if (auto* editor = state.getEditor()) {
                editor->setReadOnly(should_block_editor_input(editor, state));

                (void)editor_size;
                if (draw_rml_editor_pane(g_editor_pane, ctx.rml_manager, *editor, nullptr,
                                         ctx.fonts.monospace)) {
                    // Ctrl+Enter was pressed - execute
                    execute_python_code(editor->getTextStripped(), state);
                }
                editor_has_active_completion = editor->hasActiveCompletion();
                if (editor->consumeTextChanged()) {
                    state.setModified(true);
                }
            }

            if (ctx.fonts.monospace) {
                ImGui::PopFont();
            }
        }
        ImGui::EndChild();

        // Horizontal splitter
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, t.palette.primary_dim);
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, t.palette.primary);
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 0);

        ImGui::Button("##splitter", ImVec2(content_avail.x, SPLITTER_THICKNESS));

        if (ImGui::IsItemHovered()) {
            ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
        }

        if (ImGui::IsItemActive()) {
            const float delta = ImGui::GetIO().MouseDelta.y;
            if (delta != 0.0f) {
                g_splitter_ratio += delta / total_height;
                g_splitter_ratio = std::clamp(g_splitter_ratio, 0.2f, 0.8f);
            }
            ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
        }

        ImGui::PopStyleVar();
        ImGui::PopStyleColor(3);

        // Bottom pane with tabs
        const ImGuiWindowFlags bottom_pane_flags =
            editor_has_active_completion ? ImGuiWindowFlags_NoNav : ImGuiWindowFlags_None;
        ImGui::BeginChild("##bottom_pane", ImVec2(content_avail.x, bottom_height), false,
                          bottom_pane_flags);
        {
            const bool terminal_has_focus = state.getTerminal() && state.getTerminal()->isFocused();
            const ImGuiTabItemFlags terminal_tab_flags =
                terminal_has_focus ? ImGuiTabItemFlags_SetSelected : ImGuiTabItemFlags_None;

            if (ImGui::BeginTabBar("##console_tabs")) {
                // Output tab (read-only terminal for script output)
                if (ImGui::BeginTabItem("Output")) {
                    state.setActiveTab(0);

                    if (auto* output = state.getOutputTerminal()) {
                        output->setReadOnly(true);
                        draw_rml_terminal_pane(g_output_terminal_pane, ctx.rml_manager,
                                               "python_console_output_terminal_floating",
                                               *output, nullptr, ctx.fonts.monospace);
                    }

                    ImGui::EndTabItem();
                }

                // Terminal tab (interactive Python REPL)
                if (ImGui::BeginTabItem("Terminal", nullptr, terminal_tab_flags)) {
                    state.setActiveTab(1);

                    if (auto* terminal = state.getTerminal()) {
                        if (!terminal->is_running()) {
                            const auto fds = terminal->spawnEmbedded();
                            if (fds.valid())
                                lfs::python::start_embedded_repl(fds.read_fd, fds.write_fd);
                        }
                        draw_rml_terminal_pane(g_repl_terminal_pane, ctx.rml_manager,
                                               "python_console_repl_terminal_floating",
                                               *terminal, nullptr, ctx.fonts.monospace);
                    }

                    ImGui::EndTabItem();
                }

                // Packages tab - shows installed packages
                if (ImGui::BeginTabItem("Packages")) {
                    state.setActiveTab(2);
                    draw_rml_packages_pane(g_packages_pane, ctx.rml_manager, nullptr);
                    ImGui::EndTabItem();
                }

                ImGui::EndTabBar();
            }
        }
        ImGui::EndChild();

        // Handle keyboard shortcuts
        if (ImGui::GetIO().KeyCtrl) {
            if (ImGui::IsKeyPressed(ImGuiKey_L, false)) {
                state.clear();
            }
            if (ImGui::IsKeyPressed(ImGuiKey_R, false)) {
                reset_python_state(state);
            }
            if (ImGui::IsKeyPressed(ImGuiKey_N, false)) {
                if (auto* editor = state.getEditor()) {
                    editor->clear();
                }
                state.setScriptPath({});
                state.setModified(false);
            }
            if (ImGui::IsKeyPressed(ImGuiKey_O, false)) {
                open_script_dialog(state);
            }
            if (ImGui::IsKeyPressed(ImGuiKey_S, false)) {
                save_current_script(state);
            }
            if (ImGui::GetIO().KeyShift && ImGui::IsKeyPressed(ImGuiKey_F, false)) {
                format_editor_script(state);
            }
            if (ImGui::GetIO().KeyShift && ImGui::IsKeyPressed(ImGuiKey_I, false)) {
                clean_editor_script(state);
            }
        }

        ImGui::End();
    }

    void DrawDockedPythonConsole(const UIContext& ctx, float x, float y, float w, float h,
                                 const PanelInputState* input) {
        lfs::python::ensure_initialized();
        lfs::python::install_output_redirect();
        setup_sys_path();
        setup_console_output_capture();

        auto& state = PythonConsoleState::getInstance();
        const auto& t = theme();

        ImGui::SetNextWindowPos(ImVec2(x, y), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(w, h), ImGuiCond_Always);

        ImGui::PushStyleColor(ImGuiCol_WindowBg, t.palette.background);

        constexpr ImGuiWindowFlags PANEL_FLAGS =
            ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoDocking |
            ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoScrollbar |
            ImGuiWindowFlags_NoScrollWithMouse;

        if (!ImGui::Begin("##DockedPythonConsole", nullptr, PANEL_FLAGS)) {
            ImGui::End();
            ImGui::PopStyleColor();
            return;
        }

        // Toolbar
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(8, 4));
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(4, 4));

        if (ImGui::Button("New")) {
            if (auto* editor = state.getEditor()) {
                editor->clear();
            }
            state.setScriptPath({});
            state.setModified(false);
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Clear editor (Ctrl+N)");

        ImGui::SameLine();

        // Load button
        if (ImGui::Button("Load")) {
            open_script_dialog(state);
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Load script (Ctrl+O)");

        ImGui::SameLine();

        // Reload button
        const bool has_script = !state.getScriptPath().empty();
        if (!has_script)
            ImGui::BeginDisabled();
        if (ImGui::Button("Reload")) {
            if (has_script) {
                load_script(state.getScriptPath(), state);
            }
        }
        if (!has_script)
            ImGui::EndDisabled();
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
            if (has_script) {
                const std::string filename_utf8 =
                    lfs::core::path_to_utf8(state.getScriptPath().filename());
                ImGui::SetTooltip("Reload: %s", filename_utf8.c_str());
            } else {
                ImGui::SetTooltip("No script loaded");
            }
        }

        ImGui::SameLine();

        // Save button
        if (ImGui::Button("Save")) {
            save_current_script(state);
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Save script (Ctrl+S)");

        ImGui::SameLine();

        // Save As button
        if (ImGui::Button("Save As")) {
            save_script_dialog(state);
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Save script as...");

        ImGui::SameLine();

        // Format button
        if (ImGui::Button("Format")) {
            format_editor_script(state);
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Format code (Ctrl+Shift+F)");

        ImGui::SameLine();

        if (ImGui::Button("Clean")) {
            clean_editor_script(state);
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Clean pasted code (Ctrl+Shift+I)");

        ImGui::SameLine();
        draw_vim_mode_button(state, t);

        ImGui::SameLine();
        ImGui::TextColored(t.palette.text_dim, "|");
        ImGui::SameLine();

        // Run button
        ImGui::PushStyleColor(ImGuiCol_Button, t.palette.success);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, lighten(t.palette.success, 0.1f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, darken(t.palette.success, 0.1f));
        if (ImGui::Button("Run") || ImGui::IsKeyPressed(ImGuiKey_F5, false)) {
            if (auto* editor = state.getEditor()) {
                execute_python_code(editor->getTextStripped(), state);
            }
        }
        ImGui::PopStyleColor(3);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Run script (F5)");

        ImGui::SameLine();

        // Stop button
        const bool has_animation = python::has_frame_callback();
        const bool has_running_script = state.isScriptRunning();
        const bool has_running_terminal = state.getOutputTerminal() && state.getOutputTerminal()->is_running();
        const bool has_uv_operation = python::PackageManager::instance().has_running_operation();
        const bool can_stop = has_animation || has_running_script || has_running_terminal || has_uv_operation;
        {
            if (!can_stop) {
                ImGui::BeginDisabled();
            }
            ImGui::PushStyleColor(ImGuiCol_Button, t.palette.error);
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, lighten(t.palette.error, 0.1f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, darken(t.palette.error, 0.1f));
            if (ImGui::Button("Stop")) {
                if (has_animation) {
                    python::clear_frame_callback();
                }
                if (has_running_script) {
                    state.interruptScript();
                }
                if (has_uv_operation) {
                    python::PackageManager::instance().cancel_async();
                }
                if (auto* output = state.getOutputTerminal()) {
                    output->interrupt();
                }
            }
            ImGui::PopStyleColor(3);
            if (!can_stop) {
                ImGui::EndDisabled();
            }
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
                ImGui::SetTooltip("Stop running script (Ctrl+C)");
        }

        ImGui::SameLine();

        // Reset button
        if (ImGui::Button("Reset")) {
            reset_python_state(state);
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Reset Python state (Ctrl+R)");

        ImGui::SameLine();

        // Clear button
        if (ImGui::Button("Clear")) {
            state.clear();
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Clear console (Ctrl+L)");

        ImGui::SameLine();
        ImGui::TextColored(t.palette.text_dim, "|");
        ImGui::SameLine();

        // Status indicator
        if (can_stop) {
            ImGui::TextColored(t.palette.warning, "Running...");
        } else {
            ImGui::TextColored(t.palette.text_dim, "Python");
        }

        ImGui::SameLine();
        ImGui::TextColored(t.palette.text_dim, "|");
        ImGui::SameLine();
        draw_syntax_status(state, t);
        ImGui::SameLine();
        draw_syntax_outline_control(state, t, "##docked_python_outline");
        ImGui::SameLine();
        draw_syntax_breadcrumb_control(state, t, "##docked_python_breadcrumb");
        ImGui::SameLine();
        draw_syntax_fold_control(state, t, "##docked_python_blocks");

        ImGui::PopStyleVar(2);

        ImGui::Spacing();
        ImGui::Separator();

        // Calculate pane sizes
        const ImVec2 content_avail = ImGui::GetContentRegionAvail();
        const float total_height = content_avail.y;

        float top_height = total_height * g_splitter_ratio - SPLITTER_THICKNESS / 2;
        float bottom_height = total_height * (1.0f - g_splitter_ratio) - SPLITTER_THICKNESS / 2;
        bool editor_has_active_completion = false;

        top_height = std::max(top_height, MIN_PANE_HEIGHT);
        bottom_height = std::max(bottom_height, MIN_PANE_HEIGHT);

        // Script Editor (top pane)
        ImGui::BeginChild("##docked_script_editor_pane", ImVec2(content_avail.x, top_height), false,
                          ImGuiWindowFlags_HorizontalScrollbar);
        {
            ImFont* const scaled_mono = ctx.fonts.monoForScale(state.getFontScale());
            if (scaled_mono) {
                ImGui::PushFont(scaled_mono);
            }

            const ImVec2 editor_size(ImGui::GetContentRegionAvail().x,
                                     ImGui::GetContentRegionAvail().y);

            if (auto* editor = state.getEditor()) {
                editor->setReadOnly(should_block_editor_input(editor, state));

                (void)editor_size;
                if (draw_rml_editor_pane(g_editor_pane, ctx.rml_manager, *editor, input,
                                         scaled_mono)) {
                    execute_python_code(editor->getTextStripped(), state);
                }
                editor_has_active_completion = editor->hasActiveCompletion();
                if (editor->consumeTextChanged()) {
                    state.setModified(true);
                }
            }

            if (scaled_mono) {
                ImGui::PopFont();
            }
        }
        ImGui::EndChild();

        // Horizontal splitter
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, t.palette.primary_dim);
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, t.palette.primary);
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 0);

        ImGui::Button("##docked_splitter", ImVec2(content_avail.x, SPLITTER_THICKNESS));

        if (ImGui::IsItemHovered()) {
            ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
        }

        if (ImGui::IsItemActive()) {
            const float delta = ImGui::GetIO().MouseDelta.y;
            if (delta != 0.0f) {
                g_splitter_ratio += delta / total_height;
                g_splitter_ratio = std::clamp(g_splitter_ratio, 0.2f, 0.8f);
            }
            ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
        }

        ImGui::PopStyleVar();
        ImGui::PopStyleColor(3);

        // Bottom pane with tabs
        const ImGuiWindowFlags bottom_pane_flags =
            editor_has_active_completion ? ImGuiWindowFlags_NoNav : ImGuiWindowFlags_None;
        ImGui::BeginChild("##docked_bottom_pane", ImVec2(content_avail.x, bottom_height), false,
                          bottom_pane_flags);
        {
            ImFont* const scaled_mono_bottom = ctx.fonts.monoForScale(state.getFontScale());
            const bool terminal_has_focus = state.getTerminal() && state.getTerminal()->isFocused();
            const ImGuiTabItemFlags terminal_tab_flags =
                terminal_has_focus ? ImGuiTabItemFlags_SetSelected : ImGuiTabItemFlags_None;

            if (ImGui::BeginTabBar("##docked_console_tabs")) {
                // Output tab (read-only terminal for script output)
                if (ImGui::BeginTabItem("Output")) {
                    state.setActiveTab(0);

                    if (auto* output = state.getOutputTerminal()) {
                        output->setReadOnly(true);
                        draw_rml_terminal_pane(g_output_terminal_pane, ctx.rml_manager,
                                               "python_console_output_terminal",
                                               *output, input, scaled_mono_bottom);
                    }

                    ImGui::EndTabItem();
                }

                // Terminal tab (interactive Python REPL)
                if (ImGui::BeginTabItem("Terminal", nullptr, terminal_tab_flags)) {
                    state.setActiveTab(1);

                    if (auto* terminal = state.getTerminal()) {
                        if (!terminal->is_running()) {
                            const auto fds = terminal->spawnEmbedded();
                            if (fds.valid())
                                lfs::python::start_embedded_repl(fds.read_fd, fds.write_fd);
                        }
                        draw_rml_terminal_pane(g_repl_terminal_pane, ctx.rml_manager,
                                               "python_console_repl_terminal",
                                               *terminal, input, scaled_mono_bottom);
                    }

                    ImGui::EndTabItem();
                }

                // Packages tab - shows installed packages
                if (ImGui::BeginTabItem("Packages")) {
                    state.setActiveTab(2);
                    draw_rml_packages_pane(g_packages_pane, ctx.rml_manager, input);
                    ImGui::EndTabItem();
                }

                ImGui::EndTabBar();
            }
        }
        ImGui::EndChild();

        // Keyboard shortcuts
        if (ImGui::GetIO().KeyCtrl) {
            if (ImGui::IsKeyPressed(ImGuiKey_L, false)) {
                state.clear();
            }
            if (ImGui::IsKeyPressed(ImGuiKey_R, false)) {
                reset_python_state(state);
            }
            if (ImGui::IsKeyPressed(ImGuiKey_O, false)) {
                open_script_dialog(state);
            }
            if (ImGui::IsKeyPressed(ImGuiKey_S, false)) {
                save_current_script(state);
            }
            if (ImGui::GetIO().KeyShift && ImGui::IsKeyPressed(ImGuiKey_F, false)) {
                format_editor_script(state);
            }
            if (ImGui::GetIO().KeyShift && ImGui::IsKeyPressed(ImGuiKey_I, false)) {
                clean_editor_script(state);
            }
            // Font scaling: Ctrl++ / Ctrl+= to increase, Ctrl+- to decrease, Ctrl+0 to reset
            if (ImGui::IsKeyPressed(ImGuiKey_Equal, false) ||
                ImGui::IsKeyPressed(ImGuiKey_KeypadAdd, false)) {
                state.increaseFontScale();
            }
            if (ImGui::IsKeyPressed(ImGuiKey_Minus, false) ||
                ImGui::IsKeyPressed(ImGuiKey_KeypadSubtract, false)) {
                state.decreaseFontScale();
            }
            if (ImGui::IsKeyPressed(ImGuiKey_0, false) ||
                ImGui::IsKeyPressed(ImGuiKey_Keypad0, false)) {
                state.resetFontScale();
            }
        }

        ImGui::End();
        ImGui::PopStyleColor();
    }

} // namespace lfs::vis::gui::panels
