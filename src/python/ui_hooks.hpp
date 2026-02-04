/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"

#include <functional>
#include <string>
#include <vector>

namespace lfs::python {

    enum class HookPosition {
        Prepend,
        Append
    };

    using UIHookCallback = std::function<void(void* layout)>;

    void register_ui_hook(const std::string& panel,
                          const std::string& section,
                          UIHookCallback callback,
                          HookPosition position = HookPosition::Append);

    void remove_ui_hook(const std::string& panel,
                        const std::string& section,
                        UIHookCallback callback);

    void clear_ui_hooks(const std::string& panel, const std::string& section = "");
    void clear_all_ui_hooks();
    bool has_ui_hooks(const std::string& panel, const std::string& section);

    void invoke_ui_hooks(const std::string& panel,
                         const std::string& section,
                         HookPosition position);

    std::vector<std::string> get_registered_hook_points();

    using PythonHookInvoker = void (*)(const char* panel, const char* section, bool prepend);
    using PythonHookChecker = bool (*)(const char* panel, const char* section);

    LFS_VIS_API void set_python_hook_invoker(PythonHookInvoker invoker);
    LFS_VIS_API void set_python_hook_checker(PythonHookChecker checker);
    LFS_VIS_API void clear_python_hook_invoker();
    LFS_VIS_API void invoke_python_hooks(const std::string& panel, const std::string& section, bool prepend);
    LFS_VIS_API bool has_python_hooks(const std::string& panel, const std::string& section);

} // namespace lfs::python
