/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <expected>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace lfs::python {

    /**
     * @brief Execute a list of Python script files. Each script is expected to import `lichtfeld`
     *        and register its callbacks (e.g., with register_opacity_scaler or Session hooks).
     *
     * @return std::expected<void, std::string> error on failure (file missing, execution error,
     *         or interpreter unavailable when bindings are disabled).
     */
    std::expected<void, std::string> run_scripts(const std::vector<std::filesystem::path>& scripts);

    /**
     * @brief Set the callback for Python stdout/stderr capture.
     * @param callback Function(text, is_error) called when Python prints output.
     */
    void set_output_callback(std::function<void(const std::string&, bool)> callback);

    /**
     * @brief Write text to the output callback (used by package manager async output).
     */
    void write_output(const std::string& text, bool is_error = false);

    /**
     * @brief Initialize Python interpreter if not already done.
     */
    void ensure_initialized();

    /**
     * @brief Load user plugins configured for startup.
     *        This requires a ready Python runtime.
     */
    void ensure_plugins_loaded();

    /**
     * @brief Schedule asynchronous plugin autoload after startup.
     *        This keeps first-frame startup responsive.
     */
    void preload_user_plugins_async();

    /**
     * @brief Join the plugin preload thread if running.
     *        Called from finalize() to ensure clean shutdown.
     */
    void join_plugin_preload();

    /**
     * @brief Start an embedded Python REPL on a background thread.
     * @param read_fd File descriptor for stdin. Ownership transferred.
     * @param write_fd File descriptor for stdout/stderr. Ownership transferred.
     */
    void start_embedded_repl(int read_fd, int write_fd);

    /// @brief Stop the embedded REPL thread if running.
    void stop_embedded_repl();

    bool start_debugpy(int port = 5678);

    /**
     * @brief Install Python stdout/stderr redirect. Call after Python is initialized.
     */
    void install_output_redirect();

    /**
     * @brief Finalize Python interpreter. Call before program exit to avoid cleanup issues.
     */
    void finalize();

    /**
     * @brief Check if Python was used in this session.
     * @return true if Python scripts were executed.
     */
    bool was_python_used();

    struct FormatResult {
        std::string code;
        std::string error;
        bool success = false;
    };

    /**
     * @brief Format Python code using black.
     * @param code The Python code to format.
     * @return FormatResult with formatted code or error message.
     */
    FormatResult format_python_code(const std::string& code);

    /**
     * @brief Set a callback to be called each frame. Used for animations.
     * @param callback Function(delta_time) called each frame.
     */
    void set_frame_callback(std::function<void(float)> callback);

    /**
     * @brief Clear the frame callback.
     */
    void clear_frame_callback();

    /**
     * @brief Call the frame callback if set. Called by the visualizer each frame.
     * @param dt Delta time since last frame in seconds.
     */
    void tick_frame_callback(float dt);

    /**
     * @brief Check if a frame callback is set.
     */
    bool has_frame_callback();

    std::filesystem::path get_user_packages_dir();

    void update_python_path();

    struct CapabilityResult {
        bool success = false;
        std::string result_json;
        std::string error;
    };

    struct CapabilityInfo {
        std::string name;
        std::string description;
        std::string plugin_name;
    };

    /**
     * @brief Invoke a registered plugin capability by name.
     * @param name Capability name (e.g., "selection.by_text").
     * @param args_json JSON string of arguments.
     * @return Result with JSON result or error.
     */
    CapabilityResult invoke_capability(const std::string& name, const std::string& args_json);

    /**
     * @brief Check if a capability is registered.
     * @param name Capability name.
     * @return true if the capability exists.
     */
    bool has_capability(const std::string& name);

    /**
     * @brief List all registered capabilities.
     * @return Vector of capability info.
     */
    std::vector<CapabilityInfo> list_capabilities();

} // namespace lfs::python
