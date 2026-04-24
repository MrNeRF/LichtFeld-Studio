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
#include "gui/utils/native_file_dialog.hpp"

#include <RmlUi/Core.h>
#include <RmlUi/Core/ElementDocument.h>
#include <RmlUi/Core/Event.h>
#include <RmlUi/Core/EventListener.h>
#include <RmlUi/Core/StringUtilities.h>
#include <RmlUi/Core/Elements/ElementFormControlInput.h>
#include <SDL3/SDL_clipboard.h>
#include <SDL3/SDL_scancode.h>
#include <chrono>
#include <cmath>
#include <fstream>
#include <future>
#include <optional>
#include <sstream>
#include <thread>
#include <utility>

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

    void execute_python_code(const std::string& code,
                             lfs::vis::gui::panels::PythonConsoleState& state);
    void reset_python_state(lfs::vis::gui::panels::PythonConsoleState& state);
    bool load_script(const std::filesystem::path& path,
                     lfs::vis::gui::panels::PythonConsoleState& state);
    void open_script_dialog(lfs::vis::gui::panels::PythonConsoleState& state);
    void save_script_dialog(lfs::vis::gui::panels::PythonConsoleState& state);
    void save_current_script(lfs::vis::gui::panels::PythonConsoleState& state);

    struct RmlPythonConsolePane;

    void handle_console_event(RmlPythonConsolePane& pane, Rml::Event& event);

    struct ConsolePaneListener : Rml::EventListener {
        RmlPythonConsolePane* owner = nullptr;
        void ProcessEvent(Rml::Event& event) override {
            if (owner)
                handle_console_event(*owner, event);
        }
    };

    enum class ConsolePopover {
        None,
        Outline,
        Breadcrumbs,
        Folds,
    };

    struct ElementBounds {
        float x = 0.0f;
        float y = 0.0f;
        float width = 0.0f;
        float height = 0.0f;
    };

    struct RmlPythonConsolePane {
        RmlPythonConsolePane() { listener.owner = this; }

        std::unique_ptr<lfs::vis::gui::RmlPanelHost> host;
        lfs::vis::gui::RmlUIManager* manager = nullptr;
        Rml::ElementDocument* document = nullptr;
        lfs::vis::gui::PythonEditorElement* editor_view = nullptr;
        lfs::vis::gui::TerminalElement* output_view = nullptr;
        lfs::vis::gui::TerminalElement* repl_view = nullptr;

        Rml::Element* toolbar_el = nullptr;
        Rml::Element* script_label_el = nullptr;
        Rml::Element* reload_button_el = nullptr;
        Rml::Element* vim_button_el = nullptr;
        Rml::Element* stop_button_el = nullptr;
        Rml::Element* run_status_el = nullptr;
        Rml::Element* syntax_status_el = nullptr;
        Rml::Element* outline_button_el = nullptr;
        Rml::Element* breadcrumb_button_el = nullptr;
        Rml::Element* fold_button_el = nullptr;
        Rml::Element* font_status_el = nullptr;
        Rml::Element* editor_panel_el = nullptr;
        Rml::Element* splitter_el = nullptr;
        Rml::Element* bottom_panel_el = nullptr;
        Rml::Element* output_panel_el = nullptr;
        Rml::Element* repl_panel_el = nullptr;
        Rml::Element* packages_panel_el = nullptr;
        Rml::Element* output_tab_el = nullptr;
        Rml::Element* repl_tab_el = nullptr;
        Rml::Element* packages_tab_el = nullptr;
        Rml::Element* outline_menu_el = nullptr;
        Rml::Element* breadcrumb_menu_el = nullptr;
        Rml::Element* fold_menu_el = nullptr;

        Rml::Element* packages_refresh_button = nullptr;
        Rml::ElementFormControlInput* packages_search_input = nullptr;
        Rml::Element* packages_status_label = nullptr;
        Rml::Element* packages_table_el = nullptr;
        Rml::Element* packages_body_el = nullptr;
        Rml::Element* packages_empty_el = nullptr;

        ConsolePaneListener listener;
        bool listeners_attached = false;
        bool splitter_dragging = false;
        ConsolePopover active_popover = ConsolePopover::None;
        float panel_x = 0.0f;
        float panel_y = 0.0f;
        float panel_w = 0.0f;
        float panel_h = 0.0f;
        float last_editor_h = -1.0f;
        float last_bottom_h = -1.0f;
        float last_font_size = -1.0f;

        std::vector<lfs::python::PackageInfo> packages;
        std::future<std::vector<lfs::python::PackageInfo>> pending_packages_refresh;
        bool packages_loading = false;
        bool packages_loaded_once = false;
        std::string packages_search_filter;
        std::string last_packages_body_rml;
        std::string last_outline_rml;
        std::string last_breadcrumb_rml;
        std::string last_fold_rml;
    };

    RmlPythonConsolePane g_console_pane;

    constexpr float MIN_PANE_HEIGHT = 100.0f;
    constexpr float SPLITTER_THICKNESS = 6.0f;
    float g_splitter_ratio = 0.6f;

    void clear_console_document_cache(RmlPythonConsolePane& pane) {
        pane.document = nullptr;
        pane.editor_view = nullptr;
        pane.output_view = nullptr;
        pane.repl_view = nullptr;
        pane.toolbar_el = nullptr;
        pane.script_label_el = nullptr;
        pane.reload_button_el = nullptr;
        pane.vim_button_el = nullptr;
        pane.stop_button_el = nullptr;
        pane.run_status_el = nullptr;
        pane.syntax_status_el = nullptr;
        pane.outline_button_el = nullptr;
        pane.breadcrumb_button_el = nullptr;
        pane.fold_button_el = nullptr;
        pane.font_status_el = nullptr;
        pane.editor_panel_el = nullptr;
        pane.splitter_el = nullptr;
        pane.bottom_panel_el = nullptr;
        pane.output_panel_el = nullptr;
        pane.repl_panel_el = nullptr;
        pane.packages_panel_el = nullptr;
        pane.output_tab_el = nullptr;
        pane.repl_tab_el = nullptr;
        pane.packages_tab_el = nullptr;
        pane.outline_menu_el = nullptr;
        pane.breadcrumb_menu_el = nullptr;
        pane.fold_menu_el = nullptr;
        pane.packages_refresh_button = nullptr;
        pane.packages_search_input = nullptr;
        pane.packages_status_label = nullptr;
        pane.packages_table_el = nullptr;
        pane.packages_body_el = nullptr;
        pane.packages_empty_el = nullptr;
        pane.listeners_attached = false;
        pane.splitter_dragging = false;
        pane.active_popover = ConsolePopover::None;
        pane.last_editor_h = -1.0f;
        pane.last_bottom_h = -1.0f;
        pane.last_font_size = -1.0f;
        pane.last_packages_body_rml.clear();
        pane.last_outline_rml.clear();
        pane.last_breadcrumb_rml.clear();
        pane.last_fold_rml.clear();
    }

    void reset_rml_python_console_pane(RmlPythonConsolePane& pane) {
        clear_console_document_cache(pane);
        pane.host.reset();
        pane.manager = nullptr;
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

    float console_font_size(const lfs::vis::gui::panels::PythonConsoleState& state) {
        return std::round(std::clamp(14.0f * state.getFontScale(), 10.0f, 34.0f));
    }

    bool has_key(const lfs::vis::gui::PanelInputState& input, const int scancode) {
        return std::find(input.keys_pressed.begin(), input.keys_pressed.end(), scancode) !=
               input.keys_pressed.end();
    }

    void set_disabled(Rml::Element* el, const bool disabled) {
        if (!el)
            return;
        el->SetClass("disabled", disabled);
        if (disabled)
            el->SetAttribute("disabled", "disabled");
        else
            el->RemoveAttribute("disabled");
    }

    void set_display(Rml::Element* el, const bool visible, const char* display = "block") {
        if (el)
            el->SetProperty("display", visible ? display : "none");
    }

    void set_text(Rml::Element* el, const std::string& text) {
        if (el)
            el->SetInnerRML(Rml::StringUtilities::EncodeRml(text));
    }

    void mark_dirty(RmlPythonConsolePane& pane) {
        if (pane.host)
            pane.host->markContentDirty();
    }

    std::string get_clipboard_text() {
        char* text = SDL_GetClipboardText();
        std::string result = text ? text : "";
        SDL_free(text);
        return result;
    }

    void set_clipboard_text(const std::string& text) {
        SDL_SetClipboardText(text.c_str());
    }

    bool can_stop_python_work(lfs::vis::gui::panels::PythonConsoleState& state) {
        return lfs::python::has_frame_callback() ||
               state.isScriptRunning() ||
               (state.getOutputTerminal() && state.getOutputTerminal()->is_running()) ||
               lfs::python::PackageManager::instance().has_running_operation();
    }

    void stop_python_work(lfs::vis::gui::panels::PythonConsoleState& state) {
        if (lfs::python::has_frame_callback())
            lfs::python::clear_frame_callback();
        if (state.isScriptRunning())
            state.interruptScript();
        if (lfs::python::PackageManager::instance().has_running_operation())
            lfs::python::PackageManager::instance().cancel_async();
        if (auto* output = state.getOutputTerminal())
            output->interrupt();
    }

    void new_script(lfs::vis::gui::panels::PythonConsoleState& state) {
        if (auto* editor = state.getEditor())
            editor->clear();
        state.setScriptPath({});
        state.setModified(false);
    }

    bool package_matches_filter(const lfs::python::PackageInfo& pkg, const std::string& filter) {
        if (filter.empty())
            return true;
        return pkg.name.find(filter) != std::string::npos ||
               pkg.version.find(filter) != std::string::npos ||
               pkg.path.find(filter) != std::string::npos;
    }

    Rml::Element* find_action_target(Rml::Element* target) {
        while (target) {
            if (!target->GetAttribute<Rml::String>("data-action", "").empty())
                return target;
            target = target->GetParentNode();
        }
        return nullptr;
    }

    int data_index(Rml::Element* el) {
        return el ? el->GetAttribute<int>("data-index", -1) : -1;
    }

    void request_packages_refresh(RmlPythonConsolePane& pane) {
        if (pane.packages_loading)
            return;
        pane.packages_loading = true;
        pane.packages_loaded_once = true;
        pane.pending_packages_refresh = std::async(std::launch::async, [] {
            return lfs::python::PackageManager::instance().list_installed();
        });
        mark_dirty(pane);
    }

    void cache_console_elements(RmlPythonConsolePane& pane) {
        auto* doc = pane.document;
        if (!doc)
            return;

        pane.toolbar_el = doc->GetElementById("python-console-toolbar");
        pane.script_label_el = doc->GetElementById("script-label");
        pane.reload_button_el = doc->GetElementById("reload-button");
        pane.vim_button_el = doc->GetElementById("vim-button");
        pane.stop_button_el = doc->GetElementById("stop-button");
        pane.run_status_el = doc->GetElementById("run-status");
        pane.syntax_status_el = doc->GetElementById("syntax-status");
        pane.outline_button_el = doc->GetElementById("outline-button");
        pane.breadcrumb_button_el = doc->GetElementById("breadcrumb-button");
        pane.fold_button_el = doc->GetElementById("fold-button");
        pane.font_status_el = doc->GetElementById("font-status");
        pane.editor_panel_el = doc->GetElementById("editor-panel");
        pane.splitter_el = doc->GetElementById("python-splitter");
        pane.bottom_panel_el = doc->GetElementById("bottom-panel");
        pane.output_panel_el = doc->GetElementById("output-panel");
        pane.repl_panel_el = doc->GetElementById("terminal-panel");
        pane.packages_panel_el = doc->GetElementById("packages-panel");
        pane.output_tab_el = doc->GetElementById("tab-output");
        pane.repl_tab_el = doc->GetElementById("tab-terminal");
        pane.packages_tab_el = doc->GetElementById("tab-packages");
        pane.outline_menu_el = doc->GetElementById("outline-menu");
        pane.breadcrumb_menu_el = doc->GetElementById("breadcrumb-menu");
        pane.fold_menu_el = doc->GetElementById("fold-menu");
        pane.editor_view = dynamic_cast<lfs::vis::gui::PythonEditorElement*>(
            doc->GetElementById("python-editor-view"));
        pane.output_view = dynamic_cast<lfs::vis::gui::TerminalElement*>(
            doc->GetElementById("python-output-terminal"));
        pane.repl_view = dynamic_cast<lfs::vis::gui::TerminalElement*>(
            doc->GetElementById("python-repl-terminal"));

        pane.packages_refresh_button = doc->GetElementById("packages-refresh");
        pane.packages_search_input = dynamic_cast<Rml::ElementFormControlInput*>(
            doc->GetElementById("packages-search"));
        pane.packages_status_label = doc->GetElementById("packages-status");
        pane.packages_table_el = doc->GetElementById("packages-table");
        pane.packages_body_el = doc->GetElementById("packages-body");
        pane.packages_empty_el = doc->GetElementById("packages-empty");
    }

    bool ensure_console_pane(RmlPythonConsolePane& pane, lfs::vis::gui::RmlUIManager* manager) {
        if (!manager || !manager->isInitialized())
            return false;

        if (pane.manager != manager) {
            reset_rml_python_console_pane(pane);
            pane.manager = manager;
        }

        if (!pane.host) {
            pane.host = std::make_unique<lfs::vis::gui::RmlPanelHost>(
                manager, "python_console_panel", "rmlui/python_console_panel.rml");
        }

        if (!pane.host->ensureDocumentLoaded())
            return false;

        auto* doc = pane.host->getDocument();
        if (pane.document != doc) {
            clear_console_document_cache(pane);
            pane.document = doc;
            cache_console_elements(pane);
        }

        if (!pane.document)
            return false;

        if (!pane.listeners_attached) {
            pane.document->AddEventListener(Rml::EventId::Click, &pane.listener);
            if (pane.packages_search_input) {
                pane.packages_search_input->AddEventListener("input", &pane.listener);
                pane.packages_search_input->AddEventListener("change", &pane.listener);
            }
            pane.listeners_attached = true;
        }

        return pane.editor_view && pane.output_view && pane.repl_view;
    }

    void handle_console_action(RmlPythonConsolePane& pane,
                               lfs::vis::gui::panels::PythonConsoleState& state,
                               Rml::Element* action_el) {
        if (!action_el || action_el->HasAttribute("disabled"))
            return;

        const std::string action = action_el->GetAttribute<Rml::String>("data-action", "");
        auto* editor = state.getEditor();
        const auto close_popover = [&] {
            pane.active_popover = ConsolePopover::None;
            mark_dirty(pane);
        };

        if (action == "new") {
            new_script(state);
        } else if (action == "load") {
            open_script_dialog(state);
        } else if (action == "reload") {
            if (!state.getScriptPath().empty())
                load_script(state.getScriptPath(), state);
        } else if (action == "save") {
            save_current_script(state);
        } else if (action == "save-as") {
            save_script_dialog(state);
        } else if (action == "format") {
            format_editor_script(state);
        } else if (action == "clean") {
            clean_editor_script(state);
        } else if (action == "toggle-vim") {
            if (editor) {
                editor->setVimModeEnabled(!editor->isVimModeEnabled());
                editor->focus();
            }
        } else if (action == "run") {
            if (editor)
                execute_python_code(editor->getTextStripped(), state);
        } else if (action == "stop") {
            stop_python_work(state);
        } else if (action == "reset") {
            reset_python_state(state);
        } else if (action == "clear") {
            state.clear();
        } else if (action == "tab-output") {
            state.setActiveTab(0);
            if (auto* terminal = state.getTerminal())
                terminal->setFocused(false);
        } else if (action == "tab-terminal") {
            state.setActiveTab(1);
        } else if (action == "tab-packages") {
            state.setActiveTab(2);
            if (auto* terminal = state.getTerminal())
                terminal->setFocused(false);
        } else if (action == "font-inc") {
            state.increaseFontScale();
        } else if (action == "font-dec") {
            state.decreaseFontScale();
        } else if (action == "font-reset") {
            state.resetFontScale();
        } else if (action == "toggle-outline") {
            pane.active_popover = pane.active_popover == ConsolePopover::Outline
                                      ? ConsolePopover::None
                                      : ConsolePopover::Outline;
        } else if (action == "toggle-breadcrumbs") {
            pane.active_popover = pane.active_popover == ConsolePopover::Breadcrumbs
                                      ? ConsolePopover::None
                                      : ConsolePopover::Breadcrumbs;
        } else if (action == "toggle-folds") {
            pane.active_popover = pane.active_popover == ConsolePopover::Folds
                                      ? ConsolePopover::None
                                      : ConsolePopover::Folds;
        } else if (action == "jump-symbol") {
            if (editor) {
                const int index = data_index(action_el);
                if (index >= 0)
                    editor->jumpToSyntaxSymbol(static_cast<std::size_t>(index));
            }
            close_popover();
        } else if (action == "jump-breadcrumb") {
            if (editor) {
                const int index = data_index(action_el);
                if (index >= 0)
                    editor->jumpToSyntaxBreadcrumb(static_cast<std::size_t>(index));
            }
            close_popover();
        } else if (action == "toggle-fold") {
            if (editor) {
                const int index = data_index(action_el);
                if (index >= 0)
                    editor->toggleSyntaxFold(static_cast<std::size_t>(index));
            }
            close_popover();
        } else if (action == "fold-all") {
            if (editor)
                editor->foldAllSyntaxBlocks();
            close_popover();
        } else if (action == "unfold-all") {
            if (editor)
                editor->unfoldAllSyntaxBlocks();
            close_popover();
        } else if (action == "packages-refresh") {
            request_packages_refresh(pane);
        }

        mark_dirty(pane);
    }

    void handle_console_event(RmlPythonConsolePane& pane, Rml::Event& event) {
        const std::string type = event.GetType();
        if ((type == "input" || type == "change") && event.GetCurrentElement() == pane.packages_search_input) {
            if (pane.packages_search_input)
                pane.packages_search_filter = pane.packages_search_input->GetValue();
            mark_dirty(pane);
            event.StopPropagation();
            return;
        }

        if (type != "click")
            return;

        auto& state = lfs::vis::gui::panels::PythonConsoleState::getInstance();
        if (auto* action_el = find_action_target(event.GetTargetElement())) {
            handle_console_action(pane, state, action_el);
            event.StopPropagation();
            return;
        }

        if (pane.active_popover != ConsolePopover::None) {
            pane.active_popover = ConsolePopover::None;
            mark_dirty(pane);
        }
    }

    bool measure_element_screen_bounds(RmlPythonConsolePane& pane,
                                       Rml::Element* el,
                                       ElementBounds& out) {
        if (!pane.document || !el)
            return false;
        const auto document_offset = pane.document->GetAbsoluteOffset(Rml::BoxArea::Border);
        const auto offset = el->GetAbsoluteOffset(Rml::BoxArea::Border);
        const auto size = el->GetBox().GetSize(Rml::BoxArea::Border);
        if (size.x <= 0.0f || size.y <= 0.0f)
            return false;
        out.x = pane.panel_x + offset.x - document_offset.x;
        out.y = pane.panel_y + offset.y - document_offset.y;
        out.width = size.x;
        out.height = size.y;
        return true;
    }

    void process_rml_terminal_input(lfs::vis::terminal::TerminalWidget& terminal,
                                    const lfs::vis::gui::PanelInputState* input,
                                    const ElementBounds& bounds,
                                    float char_w,
                                    float char_h) {
        if (!input || bounds.width <= 0.0f || bounds.height <= 0.0f ||
            char_w <= 0.0f || char_h <= 0.0f)
            return;

        const bool hovered =
            input->mouse_x >= bounds.x && input->mouse_x < bounds.x + bounds.width &&
            input->mouse_y >= bounds.y && input->mouse_y < bounds.y + bounds.height;

        const auto mouse_cell = [&]() {
            const int col = static_cast<int>((input->mouse_x - bounds.x) / char_w);
            const int row = static_cast<int>((input->mouse_y - bounds.y) / char_h);
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
                    set_clipboard_text(selection);
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

        if (input->key_ctrl && input->key_shift && has_key(*input, SDL_SCANCODE_V)) {
            terminal.paste(get_clipboard_text());
            return;
        }

        for (int sc : input->keys_pressed) {
            if (input->key_ctrl && sc >= SDL_SCANCODE_A && sc <= SDL_SCANCODE_Z) {
                const char letter = static_cast<char>('A' + (sc - SDL_SCANCODE_A));
                if (letter == 'C' && terminal.hasSelection()) {
                    const std::string selection = terminal.getSelection();
                    if (!selection.empty())
                        set_clipboard_text(selection);
                } else {
                    terminal.sendControl(letter);
                }
                continue;
            }

            if (const auto key = terminal_key_from_scancode(sc))
                terminal.sendKey(*key);
        }

        if (!input->key_ctrl) {
            for (const uint32_t cp : input->text_codepoints)
                terminal.sendCodepoint(cp);
        }
    }

    void sync_terminal_view(RmlPythonConsolePane& pane,
                            lfs::vis::terminal::TerminalWidget& terminal,
                            lfs::vis::gui::TerminalElement* view,
                            Rml::Element* view_el,
                            const lfs::vis::gui::PanelInputState* input,
                            const float font_size,
                            const bool process_input) {
        terminal.update();
        if (!view || !view_el)
            return;

        ElementBounds bounds;
        if (!measure_element_screen_bounds(pane, view_el, bounds))
            return;

        const float char_h = std::max(1.0f, font_size);
        const float char_w = std::max(1.0f, font_size * 0.62f);
        const int cols = std::max(1, static_cast<int>(bounds.width / char_w));
        const int rows = std::max(1, static_cast<int>(bounds.height / char_h));

        terminal.resize(cols, rows);
        terminal.update();
        if (process_input)
            process_rml_terminal_input(terminal, input, bounds, char_w, char_h);

        const bool dirty = terminal.needsRedraw();
        view->SetProperty("font-size", std::format("{:.0f}px", font_size));
        view->SetProperty("line-height", std::format("{:.0f}px", char_h));
        view->setSnapshot(terminal.snapshot());
        if (dirty)
            mark_dirty(pane);
        terminal.markRendered();
    }

    std::string menu_item_rml(const std::string& action,
                              const std::size_t index,
                              const std::string& label,
                              const std::string& detail = {}) {
        std::string row = std::format(
            R"(<button class="menu-item" data-action="{}" data-index="{}"><span class="menu-label">{}</span>)",
            action,
            index,
            Rml::StringUtilities::EncodeRml(label));
        if (!detail.empty()) {
            row += std::format(R"(<span class="menu-detail">{}</span>)",
                               Rml::StringUtilities::EncodeRml(detail));
        }
        row += "</button>";
        return row;
    }

    void sync_syntax_menus(RmlPythonConsolePane& pane,
                           lfs::vis::gui::panels::PythonConsoleState& state) {
        auto* editor = state.getEditor();
        const auto symbols = editor ? editor->syntaxSymbols()
                                    : std::vector<lfs::vis::editor::PythonEditorSymbol>{};
        const auto breadcrumbs = editor ? editor->syntaxBreadcrumbs()
                                        : std::vector<lfs::vis::editor::PythonEditorSymbol>{};
        const auto folds = editor ? editor->syntaxFolds()
                                  : std::vector<lfs::vis::editor::PythonEditorFold>{};

        std::string outline_rml;
        if (symbols.empty()) {
            outline_rml = R"(<div class="menu-empty">No symbols</div>)";
        } else {
            for (std::size_t i = 0; i < symbols.size(); ++i)
                outline_rml += menu_item_rml("jump-symbol", i, symbols[i].label, symbols[i].detail);
        }
        if (outline_rml != pane.last_outline_rml && pane.outline_menu_el) {
            pane.outline_menu_el->SetInnerRML(outline_rml);
            pane.last_outline_rml = std::move(outline_rml);
            mark_dirty(pane);
        }

        std::string breadcrumb_rml;
        if (breadcrumbs.empty()) {
            breadcrumb_rml = R"(<div class="menu-empty">No scope</div>)";
        } else {
            for (std::size_t i = 0; i < breadcrumbs.size(); ++i)
                breadcrumb_rml += menu_item_rml("jump-breadcrumb", i, breadcrumbs[i].label, breadcrumbs[i].detail);
        }
        if (breadcrumb_rml != pane.last_breadcrumb_rml && pane.breadcrumb_menu_el) {
            pane.breadcrumb_menu_el->SetInnerRML(breadcrumb_rml);
            pane.last_breadcrumb_rml = std::move(breadcrumb_rml);
            mark_dirty(pane);
        }

        std::string fold_rml;
        if (folds.empty()) {
            fold_rml = R"(<div class="menu-empty">No blocks</div>)";
        } else {
            for (std::size_t i = 0; i < folds.size(); ++i) {
                const std::string label = folds[i].collapsed ? ("+ " + folds[i].label)
                                                             : ("- " + folds[i].label);
                fold_rml += menu_item_rml("toggle-fold", i, label, folds[i].detail);
            }
            fold_rml += R"(<div class="menu-separator"></div>)";
            fold_rml += R"(<button class="menu-item" data-action="fold-all"><span class="menu-label">Fold all</span></button>)";
            fold_rml += R"(<button class="menu-item" data-action="unfold-all"><span class="menu-label">Unfold all</span></button>)";
        }
        if (fold_rml != pane.last_fold_rml && pane.fold_menu_el) {
            pane.fold_menu_el->SetInnerRML(fold_rml);
            pane.last_fold_rml = std::move(fold_rml);
            mark_dirty(pane);
        }

        set_disabled(pane.outline_button_el, symbols.empty());
        set_disabled(pane.breadcrumb_button_el, breadcrumbs.empty());
        set_disabled(pane.fold_button_el, folds.empty());
        set_display(pane.outline_menu_el, pane.active_popover == ConsolePopover::Outline);
        set_display(pane.breadcrumb_menu_el, pane.active_popover == ConsolePopover::Breadcrumbs);
        set_display(pane.fold_menu_el, pane.active_popover == ConsolePopover::Folds);
    }

    void sync_packages(RmlPythonConsolePane& pane) {
        if (pane.packages_loading && pane.pending_packages_refresh.valid() &&
            pane.pending_packages_refresh.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
            pane.packages = pane.pending_packages_refresh.get();
            pane.packages_loading = false;
            mark_dirty(pane);
        }

        if (pane.packages_search_input)
            pane.packages_search_filter = pane.packages_search_input->GetValue();

        std::string rows;
        rows.reserve(pane.packages.size() * 192);
        std::size_t visible_count = 0;
        for (const auto& pkg : pane.packages) {
            if (!package_matches_filter(pkg, pane.packages_search_filter))
                continue;
            ++visible_count;
            rows += std::format(
                R"(<div class="pkg-row"><span class="pkg-name">{}</span><span class="pkg-version">{}</span><span class="pkg-path">{}</span></div>)",
                Rml::StringUtilities::EncodeRml(pkg.name),
                Rml::StringUtilities::EncodeRml(pkg.version),
                Rml::StringUtilities::EncodeRml(pkg.path));
        }

        if (pane.packages_body_el && rows != pane.last_packages_body_rml) {
            pane.packages_body_el->SetInnerRML(rows);
            pane.last_packages_body_rml = std::move(rows);
            mark_dirty(pane);
        }

        std::string status;
        if (pane.packages_loading)
            status = "Loading...";
        else if (pane.packages_search_filter.empty())
            status = std::format("({})", pane.packages.size());
        else
            status = std::format("({} / {})", visible_count, pane.packages.size());
        set_text(pane.packages_status_label, status);

        const bool empty_visible = !pane.packages_loading && visible_count == 0;
        set_display(pane.packages_empty_el, empty_visible);
        set_display(pane.packages_table_el, !empty_visible);
    }

    void sync_console_dom(RmlPythonConsolePane& pane,
                          lfs::vis::gui::panels::PythonConsoleState& state,
                          const float panel_h) {
        auto* editor = state.getEditor();
        const bool has_script = !state.getScriptPath().empty();
        const bool can_stop = can_stop_python_work(state);
        const int active_tab = std::clamp(state.getActiveTab(), 0, 2);

        std::string script_label = "Untitled";
        if (has_script)
            script_label = lfs::core::path_to_utf8(state.getScriptPath().filename());
        if (state.isModified())
            script_label += " *";
        set_text(pane.script_label_el, script_label);

        set_disabled(pane.reload_button_el, !has_script);
        set_disabled(pane.stop_button_el, !can_stop);
        set_text(pane.run_status_el, can_stop ? "Running..." : "Python");
        if (pane.run_status_el)
            pane.run_status_el->SetClass("running", can_stop);

        const bool vim_enabled = editor && editor->isVimModeEnabled();
        if (pane.vim_button_el)
            pane.vim_button_el->SetClass("active", vim_enabled);

        if (editor == nullptr) {
            set_text(pane.syntax_status_el, "Syntax");
        } else if (editor->hasSyntaxErrors()) {
            set_text(pane.syntax_status_el, "Syntax error");
        } else if (editor->syntaxDiagnosticsAvailable()) {
            set_text(pane.syntax_status_el, "Syntax OK");
        } else {
            set_text(pane.syntax_status_el, "Syntax");
        }
        if (pane.syntax_status_el) {
            pane.syntax_status_el->SetClass("status-error", editor && editor->hasSyntaxErrors());
            pane.syntax_status_el->SetClass("status-ok", editor && !editor->hasSyntaxErrors() &&
                                                          editor->syntaxDiagnosticsAvailable());
        }

        const std::string scope = editor ? editor->currentSyntaxScope() : "";
        set_text(pane.breadcrumb_button_el, scope.empty() ? "Scope" : scope);
        set_text(pane.fold_button_el, editor ? std::format("Blocks ({})", editor->syntaxFoldCount()) : "Blocks");
        set_text(pane.font_status_el, std::format("{}%", static_cast<int>(std::round(state.getFontScale() * 100.0f))));

        set_display(pane.output_panel_el, active_tab == 0);
        set_display(pane.repl_panel_el, active_tab == 1);
        set_display(pane.packages_panel_el, active_tab == 2);
        if (pane.output_tab_el)
            pane.output_tab_el->SetClass("active", active_tab == 0);
        if (pane.repl_tab_el)
            pane.repl_tab_el->SetClass("active", active_tab == 1);
        if (pane.packages_tab_el)
            pane.packages_tab_el->SetClass("active", active_tab == 2);

        if (active_tab == 2 && !pane.packages_loaded_once)
            request_packages_refresh(pane);
        sync_syntax_menus(pane, state);
        sync_packages(pane);

        const float toolbar_h = pane.toolbar_el
                                    ? std::max(34.0f, pane.toolbar_el->GetBox().GetSize(Rml::BoxArea::Border).y)
                                    : 38.0f;
        const float available_h = std::max(0.0f, panel_h - toolbar_h - SPLITTER_THICKNESS);
        const float min_h = std::min(MIN_PANE_HEIGHT, available_h * 0.45f);
        const float bottom_h = available_h > 0.0f
                                   ? std::clamp(available_h * (1.0f - g_splitter_ratio),
                                                min_h, std::max(min_h, available_h - min_h))
                                   : 0.0f;
        const float editor_h = std::max(0.0f, available_h - bottom_h);
        if (std::abs(editor_h - pane.last_editor_h) > 0.5f && pane.editor_panel_el) {
            pane.editor_panel_el->SetProperty("height", std::format("{:.0f}px", editor_h));
            pane.last_editor_h = editor_h;
            mark_dirty(pane);
        }
        if (std::abs(bottom_h - pane.last_bottom_h) > 0.5f && pane.bottom_panel_el) {
            pane.bottom_panel_el->SetProperty("height", std::format("{:.0f}px", bottom_h));
            pane.last_bottom_h = bottom_h;
            mark_dirty(pane);
        }
    }

    void process_splitter(RmlPythonConsolePane& pane,
                          const lfs::vis::gui::PanelInputState* input) {
        if (!input || !pane.splitter_el)
            return;

        ElementBounds bounds;
        if (!measure_element_screen_bounds(pane, pane.splitter_el, bounds))
            return;

        const bool hovered =
            input->mouse_x >= bounds.x && input->mouse_x < bounds.x + bounds.width &&
            input->mouse_y >= bounds.y && input->mouse_y < bounds.y + bounds.height;
        if (input->mouse_clicked[0] && hovered)
            pane.splitter_dragging = true;
        if (!input->mouse_down[0])
            pane.splitter_dragging = false;

        if (!pane.splitter_dragging)
            return;

        const float toolbar_h = pane.toolbar_el
                                    ? std::max(34.0f, pane.toolbar_el->GetBox().GetSize(Rml::BoxArea::Border).y)
                                    : 38.0f;
        const float available_h = std::max(1.0f, pane.panel_h - toolbar_h - SPLITTER_THICKNESS);
        const float local_y = std::clamp(input->mouse_y - pane.panel_y - toolbar_h,
                                         0.0f, available_h);
        g_splitter_ratio = std::clamp(local_y / available_h, 0.2f, 0.8f);
        mark_dirty(pane);
    }

    void process_console_shortcuts(lfs::vis::gui::panels::PythonConsoleState& state,
                                   const lfs::vis::gui::PanelInputState* input) {
        if (!input)
            return;

        const bool terminal_focused =
            (state.getTerminal() && state.getTerminal()->isFocused()) ||
            (state.getOutputTerminal() && state.getOutputTerminal()->isFocused());
        auto* editor = state.getEditor();

        if (!terminal_focused && has_key(*input, SDL_SCANCODE_F5) && editor) {
            execute_python_code(editor->getTextStripped(), state);
        }

        if (!input->key_ctrl || terminal_focused)
            return;

        if (input->key_shift && has_key(*input, SDL_SCANCODE_O)) {
            if (!state.getScriptPath().empty())
                load_script(state.getScriptPath(), state);
            return;
        }
        if (has_key(*input, SDL_SCANCODE_L)) {
            state.clear();
        } else if (has_key(*input, SDL_SCANCODE_R)) {
            reset_python_state(state);
        } else if (has_key(*input, SDL_SCANCODE_N)) {
            new_script(state);
        } else if (has_key(*input, SDL_SCANCODE_O)) {
            open_script_dialog(state);
        } else if (has_key(*input, SDL_SCANCODE_S)) {
            save_current_script(state);
        } else if (input->key_shift && has_key(*input, SDL_SCANCODE_F)) {
            format_editor_script(state);
        } else if (input->key_shift && has_key(*input, SDL_SCANCODE_I)) {
            clean_editor_script(state);
        } else if (has_key(*input, SDL_SCANCODE_EQUALS) ||
                   has_key(*input, SDL_SCANCODE_KP_PLUS)) {
            state.increaseFontScale();
        } else if (has_key(*input, SDL_SCANCODE_MINUS) ||
                   has_key(*input, SDL_SCANCODE_KP_MINUS)) {
            state.decreaseFontScale();
        } else if (has_key(*input, SDL_SCANCODE_0) ||
                   has_key(*input, SDL_SCANCODE_KP_0)) {
            state.resetFontScale();
        } else if (has_key(*input, SDL_SCANCODE_C) && can_stop_python_work(state)) {
            stop_python_work(state);
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

    void ShutdownPythonConsoleRml() {
        reset_rml_python_console_pane(g_console_pane);
    }

    void DrawPythonConsole(const UIContext& ctx, bool* open) {
        (void)ctx;
        if (open)
            *open = false;
    }

    void DrawDockedPythonConsole(const UIContext& ctx, float x, float y, float w, float h,
                                 const PanelInputState* input) {
        lfs::python::ensure_initialized();
        lfs::python::install_output_redirect();
        setup_sys_path();
        setup_console_output_capture();

        auto& state = PythonConsoleState::getInstance();
        auto& pane = g_console_pane;
        if (!ensure_console_pane(pane, ctx.rml_manager))
            return;

        pane.panel_x = x;
        pane.panel_y = y;
        pane.panel_w = w;
        pane.panel_h = h;

        const float font_size = console_font_size(state);
        if (auto* editor = state.getEditor()) {
            editor->setReadOnly(should_block_editor_input(editor, state));
            pane.editor_view->setEditor(editor);
            pane.editor_view->setFontSizePx(font_size);
            pane.editor_view->SetProperty("font-size", std::format("{:.0f}px", font_size));
        }

        sync_console_dom(pane, state, h);
        pane.host->syncDirectLayout(w, h);
        process_splitter(pane, input);
        sync_console_dom(pane, state, h);
        pane.host->syncDirectLayout(w, h);

        if (auto* output = state.getOutputTerminal()) {
            output->setReadOnly(true);
            sync_terminal_view(pane, *output, pane.output_view, pane.output_view, input,
                               font_size, state.getActiveTab() == 0);
        }

        if (auto* terminal = state.getTerminal()) {
            terminal->setReadOnly(false);
            if (state.getActiveTab() == 1 && !terminal->is_running()) {
                const auto fds = terminal->spawnEmbedded();
                if (fds.valid())
                    lfs::python::start_embedded_repl(fds.read_fd, fds.write_fd);
            } else {
                terminal->update();
            }
            sync_terminal_view(pane, *terminal, pane.repl_view, pane.repl_view, input,
                               font_size, state.getActiveTab() == 1);
            state.setTerminalFocused(terminal->isFocused());
        }

        process_console_shortcuts(state, input);

        pane.host->markContentDirty();
        pane.host->setInput(input);
        if (input)
            pane.host->drawDirect(x, y, w, h);
        else
            pane.host->prepareDirect(w, h);
        pane.host->setInput(nullptr);

        if (auto* editor = state.getEditor()) {
            if (editor->consumeExecuteRequested())
                execute_python_code(editor->getTextStripped(), state);
            if (editor->consumeTextChanged())
                state.setModified(true);
        }
    }

} // namespace lfs::vis::gui::panels
