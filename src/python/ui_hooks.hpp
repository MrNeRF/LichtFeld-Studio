/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"

#include <string>

namespace lfs::python {

    using PythonHookInvoker = void (*)(const char* panel, const char* section, bool prepend);
    using PythonDocumentHookInvoker = bool (*)(const char* panel, const char* section,
                                               void* document, bool prepend);
    using PythonHookChecker = bool (*)(const char* panel, const char* section, bool prepend);

    LFS_VIS_API void set_python_hook_invoker(PythonHookInvoker invoker);
    LFS_VIS_API void set_python_document_hook_invoker(PythonDocumentHookInvoker invoker);
    LFS_VIS_API void set_python_hook_checker(PythonHookChecker checker);
    LFS_VIS_API void invoke_python_hooks(const std::string& panel, const std::string& section, bool prepend);
    LFS_VIS_API bool invoke_python_document_hooks(const std::string& panel, const std::string& section,
                                                  void* document, bool prepend);
    LFS_VIS_API bool has_python_hooks(const std::string& panel, const std::string& section);
    LFS_VIS_API bool has_python_hooks(const std::string& panel, const std::string& section, bool prepend);

} // namespace lfs::python
