/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

namespace nanobind {
    class module_;
}

namespace lfs::python {

    void register_store(nanobind::module_& ui_module);
    void shutdown_store_bridge();

} // namespace lfs::python
