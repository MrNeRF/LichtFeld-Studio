/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "py_diagnostics.hpp"

#include "core/cuda_version.hpp"
#include "core/system_info.hpp"
#include "gui/gpu_memory_query.hpp"

#include <nanobind/stl/string.h>

#include <cstdint>
#include <string>

namespace lfs::python {

    void register_diagnostics(nb::module_& m) {
        m.def(
            "collect",
            [] {
                lfs::core::system_info::SystemInfo system;
                lfs::vis::gui::GpuMemoryInfo gpu;
                lfs::core::CudaVersionInfo driver;
                {
                    nb::gil_scoped_release release;
                    system = lfs::core::system_info::collect();
                    gpu = lfs::vis::gui::queryGpuMemory();
                    driver = lfs::core::check_cuda_version();
                }

                constexpr std::uint64_t bytes_per_mb = 1024ULL * 1024ULL;
                nb::dict result;
                result["os"] = system.os;
                result["os_build"] = system.os_build;
                result["cpu"] = system.cpu;
                result["ram_mb"] = system.ram_mb;
                result["cuda_runtime"] = system.cuda_runtime;
                result["gpu"] = gpu.device_name;
                result["vram_mb"] = static_cast<std::uint64_t>(gpu.total) / bytes_per_mb;
                result["vram_used_mb"] =
                    static_cast<std::uint64_t>(gpu.total_used) / bytes_per_mb;
                result["gpu_driver"] = driver.query_failed
                                           ? std::string{}
                                           : std::to_string(driver.major) + "." +
                                                 std::to_string(driver.minor);
                return result;
            },
            "Collect best-effort system, CUDA, and GPU diagnostics.");
    }

} // namespace lfs::python
