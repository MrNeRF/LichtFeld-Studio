/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "python_runtime.hpp"

#include <core/error.hpp>

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace lfs::python {

    // Phase 9 Section 3.1: latched Python-init state. Terminal and monotone in
    // production: Ready never becomes Failed and vice versa. Failed retains the
    // Error (interpreter init OR the lichtfeld-bridge init failed). Pure C++
    // (atomic + mutex + lfs::Error); readable/writable without the GIL.
    enum class PyInitState : std::uint8_t { Uninitialized,
                                            Initializing,
                                            Ready,
                                            Failed };

    struct PyInitStatus {
        PyInitState state = PyInitState::Uninitialized;
        std::optional<lfs::Error> error; // engaged iff state == Failed
    };

    // Cheap, thread-safe query of the latched init state. Lock-free fast path;
    // takes the error mutex only to copy the Error when Failed.
    [[nodiscard]] PyInitStatus init_state() noexcept;

    /**
     * @brief Execute a list of Python script files. Each script is expected to import `lichtfeld`
     *        and register its callbacks (e.g., with register_opacity_scaler or Session hooks).
     *
     * @return lfs::Result<void>: the latched init failure, or a typed IO/Python error on a
     *         missing script or execution failure. Success is a default Status.
     */
    [[nodiscard]] lfs::Result<void> run_scripts(const std::vector<std::filesystem::path>& scripts);

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
     * @brief Initialize the Python interpreter if not already done.
     * @return the latched Status: success once Ready, or the same latched Error on
     *         every call once Failed (interpreter or lichtfeld-bridge init failed).
     *         Native application services must not wait for this state.
     */
    [[nodiscard]] lfs::Status ensure_initialized();

    // Test-only (process-isolated). While armed, ensure_initialized() latches the
    // Failed state without touching the real interpreter or the init once-flag.
    // reset restores the latch to its pre-forced value (Ready if a real init
    // already succeeded, else Uninitialized) and disarms. Documented tests-only.
    void force_python_init_failure_for_testing(bool should_fail) noexcept;
    void reset_python_init_state_for_testing() noexcept;

    /**
     * @brief Register built-in Python UI once the retained GUI runtime is available.
     */
    void ensure_builtin_ui_registered();

    /**
     * @brief Load user plugins configured for startup.
     *        This requires a ready Python runtime.
     */
    [[nodiscard]] bool ensure_plugins_loaded();

    /**
     * @brief Schedule plugin autoload after startup.
     *        The complete load pipeline runs on one owned background worker.
     */
    void preload_user_plugins_async();

    /**
     * @brief True while startup plugin preload is running.
     *
     * UI code uses this to avoid blocking Python calls while startup imports
     * are in progress.
     */
    bool is_plugin_preload_running();

    /// @brief True while startup plugin preload may block Python calls.
    bool is_plugin_preload_blocking_python();

    /**
     * @brief Request cooperative cancellation of startup plugin loading.
     *        Safe to call from the render thread without acquiring the GIL.
     */
    void request_plugin_preload_stop();

    /**
     * @brief Stop and join startup plugin loading before Python teardown.
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
     * @brief Validate and format Python code using black.
     * @param code The Python code to format.
     * @return FormatResult with formatted code or error message.
     */
    FormatResult format_python_code(const std::string& code);

    /**
     * @brief Best-effort cleanup for pasted Python snippets, then format using black.
     * @param code The Python code to clean and format.
     * @return FormatResult with cleaned code or error message.
     */
    FormatResult clean_python_code(const std::string& code);

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
