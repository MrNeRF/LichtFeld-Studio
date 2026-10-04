/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "graphics_external_tensor.hpp"

#include "core/exportable_storage.hpp"

namespace lfs::vis {

#if LFS_BUILD_TRAINER
    namespace {
        [[nodiscard]] lfs::Error unavailableError() {
            return lfs::make_error({
                .code = lfs::ErrorCode::Unsupported,
                .domain = lfs::ErrorDomain::Rendering,
                .user_message =
                    "External graphics-memory import is unavailable on the native Metal path; "
                    "training and viewing share Metal tensors directly",
                .detection = LFS_SOURCE_SITE_CURRENT(),
            });
        }
    } // namespace

    struct GraphicsExternalTensorStorage::Impl {};

    GraphicsExternalTensorStorage::GraphicsExternalTensorStorage(
        GraphicsContext&, std::shared_ptr<lfs::core::ExportableBlock>)
        : impl_(std::make_unique<Impl>()) {}

    GraphicsExternalTensorStorage::~GraphicsExternalTensorStorage() = default;

    bool GraphicsExternalTensorStorage::bindNewExportableChunks(
        const lfs::core::ExportableBlock&) {
        return false;
    }

    lfs::Result<lfs::core::Tensor> makeGraphicsExternalTensor(
        GraphicsContext&, lfs::core::TensorShape, lfs::core::DataType,
        std::size_t, const char*) {
        return unavailableError();
    }

    lfs::Result<lfs::core::SplatTensorAllocator>
    makeSplatExportableInteropAllocator(
        GraphicsContext&, const lfs::core::SplatExportableStorage&,
        std::shared_ptr<GraphicsExternalTensorStorage>*) {
        return unavailableError();
    }
#endif

} // namespace lfs::vis
