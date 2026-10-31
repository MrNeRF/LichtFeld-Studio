/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace lfs::python {

    // lf.video: container probing and frame extraction over lfs::io::VideoFrameExtractor.
    void register_video(nb::module_& m);

} // namespace lfs::python
