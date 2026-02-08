/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/panels/python_console_panel.hpp"
#include "gui/editor/python_editor.hpp"
#include "gui/terminal/terminal_widget.hpp"
#include "gui/ui_widgets.hpp"
#include "gui/utils/windows_utils.hpp"
#include "theme/theme.hpp"

#include <chrono>
#include <fstream>
#include <future>
#include <sstream>
#include <thread>
#include <imgui.h>

#include <Python.h>
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
        std::ifstream file(path);
        if (!file.is_open()) {
            state.addError("Failed to open: " + path.string());
            return false;
        }

        std::string content((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());

        if (auto* editor = state.getEditor()) {
            editor->setText(content);
        }

        state.setScriptPath(path);
        state.setModified(false);
        state.addInfo("Loaded: " + path.filename().string());
        return true;
    }

    bool save_script(const std::filesystem::path& path, lfs::vis::gui::panels::PythonConsoleState& state) {
        auto* editor = state.getEditor();
        if (!editor) {
            return false;
        }

        std::ofstream file(path);
        if (!file.is_open()) {
            state.addError("Failed to save: " + path.string());
            return false;
        }

        file << editor->getTextStripped();
        file.close();

        state.setScriptPath(path);
        state.setModified(false);
        state.addInfo("Saved: " + path.filename().string());
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
    }

    PythonConsoleState::~PythonConsoleState() {
        interruptScript();
        if (script_thread_.joinable()) {
            script_thread_.join();
        }
    }

    PythonConsoleState& PythonConsoleState::getInstance() {
        static PythonConsoleState instance;
        return instance;
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

        script_running_ = true;
        script_thread_id_ = 0;
        script_thread_ = std::thread([this, code]() {
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
                    PyErr_Print();
                }

                script_thread_id_ = 0;
            }
            script_running_ = false;
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

    namespace {
        float g_splitter_ratio = 0.6f;
        constexpr float MIN_PANE_HEIGHT = 100.0f;
        constexpr float SPLITTER_THICKNESS = 6.0f;

        constexpr float PKG_NAME_COL_WIDTH = 120.0f;
        constexpr float PKG_VERSION_COL_WIDTH = 60.0f;
        constexpr float PKG_SEARCH_WIDTH = 150.0f;
    } // namespace

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
            window_title += " - " + state.getScriptPath().filename().string();
        }
        if (state.isModified()) {
            window_title += " *";
        }

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
                ImGui::MenuItem("Ctrl+Space for autocomplete", nullptr, false, false);
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
            ImGui::Separator();
            ImGui::SameLine();

            // Status indicator
            if (can_stop) {
                ImGui::TextColored(t.palette.warning, "Running...");
            } else {
                ImGui::TextColored(t.palette.text_dim, "Python");
            }

            ImGui::PopStyleVar(2);
        }

        ImGui::Spacing();
        ImGui::Separator();

        // Calculate pane sizes
        const ImVec2 content_avail = ImGui::GetContentRegionAvail();
        const float total_height = content_avail.y;

        float top_height = total_height * g_splitter_ratio - SPLITTER_THICKNESS / 2;
        float bottom_height = total_height * (1.0f - g_splitter_ratio) - SPLITTER_THICKNESS / 2;

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

            // Block editor input when terminal has focus or other widget wants text input
            bool block_editor_input = ImGui::GetIO().WantTextInput;
            if (auto* terminal = state.getTerminal()) {
                block_editor_input |= terminal->isFocused();
            }

            if (auto* editor = state.getEditor()) {
                editor->setReadOnly(block_editor_input);

                if (editor->render(editor_size)) {
                    // Ctrl+Enter was pressed - execute
                    execute_python_code(editor->getTextStripped(), state);
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
        ImGui::BeginChild("##bottom_pane", ImVec2(content_avail.x, bottom_height), false);
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
                        output->render(ctx.fonts.monospace);
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
                        terminal->render(ctx.fonts.monospace);
                    }

                    ImGui::EndTabItem();
                }

                // Packages tab - shows installed packages
                if (ImGui::BeginTabItem("Packages")) {
                    state.setActiveTab(2);

                    static std::vector<python::PackageInfo> cached_packages;
                    static std::future<std::vector<python::PackageInfo>> pending_refresh;
                    static bool loading = false;
                    static char search_filter[128] = "";

                    if (!loading && ImGui::Button("Refresh")) {
                        loading = true;
                        pending_refresh = std::async(std::launch::async, []() {
                            return python::PackageManager::instance().list_installed();
                        });
                    }

                    if (loading && pending_refresh.valid() &&
                        pending_refresh.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                        cached_packages = pending_refresh.get();
                        loading = false;
                    }

                    ImGui::SameLine();
                    ImGui::SetNextItemWidth(PKG_SEARCH_WIDTH);
                    ImGui::InputTextWithHint("##search", "Search...", search_filter, sizeof(search_filter));

                    ImGui::SameLine();
                    if (loading) {
                        ImGui::TextColored(t.palette.text_dim, "Loading...");
                    } else {
                        ImGui::TextColored(t.palette.text_dim, "(%zu)", cached_packages.size());
                    }

                    if (cached_packages.empty() && !loading) {
                        ImGui::TextColored(t.palette.text_dim, "No packages installed");
                    } else {
                        constexpr auto TABLE_FLAGS =
                            ImGuiTableFlags_ScrollY | ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable;
                        if (ImGui::BeginTable("##pkg_table", 3, TABLE_FLAGS, ImGui::GetContentRegionAvail())) {
                            ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthFixed, PKG_NAME_COL_WIDTH);
                            ImGui::TableSetupColumn("Version", ImGuiTableColumnFlags_WidthFixed, PKG_VERSION_COL_WIDTH);
                            ImGui::TableSetupColumn("Path", ImGuiTableColumnFlags_WidthStretch);
                            ImGui::TableHeadersRow();
                            for (const auto& pkg : cached_packages) {
                                if (search_filter[0] != '\0' &&
                                    pkg.name.find(search_filter) == std::string::npos)
                                    continue;
                                ImGui::TableNextRow();
                                ImGui::TableNextColumn();
                                ImGui::Text("%s", pkg.name.c_str());
                                ImGui::TableNextColumn();
                                ImGui::TextColored(t.palette.text_dim, "%s", pkg.version.c_str());
                                ImGui::TableNextColumn();
                                ImGui::TextColored(t.palette.text_dim, "%s", pkg.path.c_str());
                            }
                            ImGui::EndTable();
                        }
                    }

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
        }

        ImGui::End();
    }

    void DrawDockedPythonConsole(const UIContext& ctx, const ImVec2& pos, const ImVec2& size) {
        lfs::python::ensure_initialized();
        lfs::python::install_output_redirect();
        setup_sys_path();
        setup_console_output_capture();

        auto& state = PythonConsoleState::getInstance();
        const auto& t = theme();

        ImGui::SetNextWindowPos(pos, ImGuiCond_Always);
        ImGui::SetNextWindowSize(size, ImGuiCond_Always);

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
            if (has_script)
                ImGui::SetTooltip("Reload: %s", state.getScriptPath().filename().string().c_str());
            else
                ImGui::SetTooltip("No script loaded");
        }

        ImGui::SameLine();

        // Save button
        if (ImGui::Button("Save")) {
            save_current_script(state);
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Save script (Ctrl+S)");

        ImGui::SameLine();

        // Format button
        if (ImGui::Button("Format")) {
            if (auto* editor = state.getEditor()) {
                const auto result = lfs::python::format_python_code(editor->getTextStripped());
                if (result.success) {
                    editor->setText(result.code);
                    state.setModified(true);
                } else if (!result.error.empty()) {
                    state.addError("[Format] " + result.error);
                }
            }
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Format code (Ctrl+Shift+F)");

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

        ImGui::PopStyleVar(2);

        ImGui::Spacing();
        ImGui::Separator();

        // Calculate pane sizes
        const ImVec2 content_avail = ImGui::GetContentRegionAvail();
        const float total_height = content_avail.y;

        float top_height = total_height * g_splitter_ratio - SPLITTER_THICKNESS / 2;
        float bottom_height = total_height * (1.0f - g_splitter_ratio) - SPLITTER_THICKNESS / 2;

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

            // Block editor input when terminal has focus or other widget wants text input
            bool block_editor_input = ImGui::GetIO().WantTextInput;
            if (auto* terminal = state.getTerminal()) {
                block_editor_input |= terminal->isFocused();
            }

            if (auto* editor = state.getEditor()) {
                editor->setReadOnly(block_editor_input);

                if (editor->render(editor_size)) {
                    execute_python_code(editor->getTextStripped(), state);
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
        ImGui::BeginChild("##docked_bottom_pane", ImVec2(content_avail.x, bottom_height), false);
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
                        output->render(scaled_mono_bottom);
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
                        terminal->render(scaled_mono_bottom);
                    }

                    ImGui::EndTabItem();
                }

                // Packages tab - shows installed packages
                if (ImGui::BeginTabItem("Packages")) {
                    state.setActiveTab(2);

                    static std::vector<python::PackageInfo> cached_packages;
                    static std::future<std::vector<python::PackageInfo>> pending_refresh;
                    static bool loading = false;
                    static char search_filter[128] = "";

                    if (!loading && ImGui::Button("Refresh##docked")) {
                        loading = true;
                        pending_refresh = std::async(std::launch::async, []() {
                            return python::PackageManager::instance().list_installed();
                        });
                    }

                    if (loading && pending_refresh.valid() &&
                        pending_refresh.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                        cached_packages = pending_refresh.get();
                        loading = false;
                    }

                    ImGui::SameLine();
                    ImGui::SetNextItemWidth(PKG_SEARCH_WIDTH);
                    ImGui::InputTextWithHint("##search_docked", "Search...", search_filter, sizeof(search_filter));

                    ImGui::SameLine();
                    if (loading) {
                        ImGui::TextColored(t.palette.text_dim, "Loading...");
                    } else {
                        ImGui::TextColored(t.palette.text_dim, "(%zu)", cached_packages.size());
                    }

                    if (cached_packages.empty() && !loading) {
                        ImGui::TextColored(t.palette.text_dim, "No packages installed");
                    } else {
                        constexpr auto TABLE_FLAGS =
                            ImGuiTableFlags_ScrollY | ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable;
                        if (ImGui::BeginTable("##docked_pkg_table", 3, TABLE_FLAGS, ImGui::GetContentRegionAvail())) {
                            ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthFixed, PKG_NAME_COL_WIDTH);
                            ImGui::TableSetupColumn("Version", ImGuiTableColumnFlags_WidthFixed, PKG_VERSION_COL_WIDTH);
                            ImGui::TableSetupColumn("Path", ImGuiTableColumnFlags_WidthStretch);
                            ImGui::TableHeadersRow();
                            for (const auto& pkg : cached_packages) {
                                if (search_filter[0] != '\0' &&
                                    pkg.name.find(search_filter) == std::string::npos)
                                    continue;
                                ImGui::TableNextRow();
                                ImGui::TableNextColumn();
                                ImGui::Text("%s", pkg.name.c_str());
                                ImGui::TableNextColumn();
                                ImGui::TextColored(t.palette.text_dim, "%s", pkg.version.c_str());
                                ImGui::TableNextColumn();
                                ImGui::TextColored(t.palette.text_dim, "%s", pkg.path.c_str());
                            }
                            ImGui::EndTable();
                        }
                    }

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
                if (auto* editor = state.getEditor()) {
                    const auto result = lfs::python::format_python_code(editor->getTextStripped());
                    if (result.success) {
                        editor->setText(result.code);
                        state.setModified(true);
                    } else if (!result.error.empty()) {
                        state.addError("[Format] " + result.error);
                    }
                }
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
