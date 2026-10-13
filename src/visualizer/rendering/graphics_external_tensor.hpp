/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/error.hpp"
#include "core/sh_value_quant.hpp"
#include "core/splat_data.hpp"
#include "core/splat_exportable_storage.hpp"
#include "core/tensor.hpp"

#include <memory>
#include <string_view>

namespace lfs::vis {
    class GraphicsContext;

#if LFS_BUILD_TRAINER
    class GraphicsExternalTensorStorage final {
    public:
        GraphicsExternalTensorStorage(GraphicsContext& context,
                                      std::shared_ptr<lfs::core::ExportableBlock> block);
        ~GraphicsExternalTensorStorage();
        GraphicsExternalTensorStorage(const GraphicsExternalTensorStorage&) = delete;
        GraphicsExternalTensorStorage& operator=(const GraphicsExternalTensorStorage&) = delete;

        [[nodiscard]] bool bindNewExportableChunks(const lfs::core::ExportableBlock& block);

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

    [[nodiscard]] lfs::Result<lfs::core::Tensor> makeGraphicsExternalTensor(
        GraphicsContext& context, lfs::core::TensorShape shape,
        lfs::core::DataType dtype, std::size_t capacity, const char* debug_name);

    [[nodiscard]] lfs::Result<lfs::core::SplatTensorAllocator>
    makeSplatExportableInteropAllocator(
        GraphicsContext& context,
        const lfs::core::SplatExportableStorage& storage,
        std::shared_ptr<GraphicsExternalTensorStorage>* parent_keep = nullptr);

    [[nodiscard]] inline bool keepFloatShNInPooledCuda(const std::string_view name,
                                                       const lfs::core::DataType dtype) {
        return name == "SplatData.shN" && dtype == lfs::core::DataType::Float32 &&
               lfs::core::sh_value_quant::enabled();
    }
#endif

    [[nodiscard]] LFS_VIS_API lfs::core::SplatTensorAllocator
    makeViewerSplatTensorAllocator(bool preserve_float_shN = false);
} // namespace lfs::vis
