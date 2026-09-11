/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstddef>

namespace fastgs_fisheye_test {

    struct DeviceProjectSample {
        float Px, Py, Pz;
        float fx, fy, cx, cy;
        float k1, k2, k3, k4;
        float px, py;
        float J00, J01, J02, J10, J11, J12;
        float theta, D;
        int finite;
    };

    // Returns false if a CUDA allocation, launch, or copy failed.
    bool project_on_device(DeviceProjectSample* samples, std::size_t n);

} // namespace fastgs_fisheye_test
