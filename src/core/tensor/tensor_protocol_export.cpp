/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/tensor_protocol_export.hpp"
#include "backend/backend_driver.hpp"
#include "backend/native_export.hpp"
#include "core/tensor.hpp"
namespace lfs::core {
    TensorExportCapabilities gpu_backend_export_capabilities(GpuBackend backend) {
        return internal::backend_driver(backend).export_capabilities();
    }
    namespace {
        internal::BackendDriver& export_driver(const Tensor& tensor) {
            auto& driver = internal::backend_driver(*gpu_backend_of(tensor));
            if (!driver.export_capabilities().dlpack)
                internal::reject_dlpack_export(driver.name());
            return driver;
        }
    } // namespace
    TensorExportDevice tensor_dlpack_device(const Tensor& tensor) {
        return tensor.device() == Device::CPU ? TensorExportDevice{} : export_driver(tensor).dlpack_device(tensor);
    }
    Tensor prepare_tensor_dlpack_export(const Tensor& tensor, std::optional<int64_t> stream) {
        return tensor.device() == Device::CPU ? tensor : export_driver(tensor).prepare_dlpack_export(tensor, stream);
    }
} // namespace lfs::core
