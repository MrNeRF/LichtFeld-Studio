/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "ui_hooks.hpp"
#include "python_runtime.hpp"

#include <atomic>

namespace lfs::python {

    namespace {
        std::atomic<PythonHookInvoker> g_hook_invoker{nullptr};
        std::atomic<PythonDocumentHookInvoker> g_document_hook_invoker{nullptr};
        std::atomic<PythonHookChecker> g_hook_checker{nullptr};
    } // namespace

    void set_python_hook_invoker(const PythonHookInvoker invoker) {
        g_hook_invoker.store(invoker, std::memory_order_release);
    }

    void set_python_document_hook_invoker(const PythonDocumentHookInvoker invoker) {
        g_document_hook_invoker.store(invoker, std::memory_order_release);
    }

    void set_python_hook_checker(const PythonHookChecker checker) {
        g_hook_checker.store(checker, std::memory_order_release);
    }

    void invoke_python_hooks(const std::string& panel, const std::string& section, const bool prepend) {
        const auto invoker = g_hook_invoker.load(std::memory_order_acquire);
        if (!invoker)
            return;
        if (bridge().prepare_ui)
            bridge().prepare_ui();
        invoker(panel.c_str(), section.c_str(), prepend);
    }

    bool invoke_python_document_hooks(const std::string& panel, const std::string& section,
                                      void* document, const bool prepend) {
        const auto invoker = g_document_hook_invoker.load(std::memory_order_acquire);
        if (!invoker)
            return false;
        if (bridge().prepare_ui)
            bridge().prepare_ui();
        return invoker(panel.c_str(), section.c_str(), document, prepend);
    }

    bool has_python_hooks(const std::string& panel, const std::string& section) {
        const auto checker = g_hook_checker.load(std::memory_order_acquire);
        return checker &&
               (checker(panel.c_str(), section.c_str(), true) ||
                checker(panel.c_str(), section.c_str(), false));
    }

    bool has_python_hooks(const std::string& panel, const std::string& section, const bool prepend) {
        const auto checker = g_hook_checker.load(std::memory_order_acquire);
        return checker && checker(panel.c_str(), section.c_str(), prepend);
    }

} // namespace lfs::python
