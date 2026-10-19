/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "splat_export_request.hpp"
#include <mutex>
#include <utility>
namespace lfs::vis::gui {
    namespace {
        std::mutex request_mutex;
        std::vector<std::string> requested_node;
    } // namespace
    void requestSplatExport(std::vector<std::string> name) {
        std::scoped_lock lock(request_mutex);
        requested_node = std::move(name);
    }
    std::vector<std::string> takeSplatExportRequest() {
        std::scoped_lock lock(request_mutex);
        return std::exchange(requested_node, {});
    }
} // namespace lfs::vis::gui
