/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "model.hpp"
#include "vulkan_runtime.hpp"

#include <expected>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace lfs::onnx_vulkan::detail {

    struct DeviceTensor {
        BufferBinding binding;
        ElementType type = ElementType::Float32;
        std::vector<std::int64_t> shape;
        std::vector<std::int64_t> strides;
        std::int64_t element_offset = 0;
    };

    struct WeightStore {
        Buffer buffer;
        Buffer packed_buffer;
        std::unordered_map<std::string, DeviceTensor> tensors;
        std::unordered_map<std::string, DeviceTensor> packed_tensors;
    };

    [[nodiscard]] std::expected<WeightStore, Error>
    upload_weights(const Model& model, const VulkanRuntime& runtime);

    class ExecutionPlan final {
    public:
        ExecutionPlan(const ExecutionPlan&) = delete;
        ExecutionPlan& operator=(const ExecutionPlan&) = delete;
        ~ExecutionPlan();

        [[nodiscard]] static std::expected<std::unique_ptr<ExecutionPlan>, Error>
        create(const Model& model,
               const WeightStore& weights,
               VulkanRuntime& runtime,
               std::span<const NamedTensorView> inputs,
               std::span<const std::string_view> requested_outputs);

        [[nodiscard]] std::expected<std::vector<NamedTensor>, Error>
        run(std::span<const NamedTensorView> inputs);

    private:
        explicit ExecutionPlan(VulkanRuntime& runtime);
        struct Impl;
        VulkanRuntime& runtime_;
        std::unique_ptr<Impl> impl_;
    };

    [[nodiscard]] std::string plan_signature(std::span<const NamedTensorView> inputs,
                                             std::span<const std::string_view> requested_outputs);

} // namespace lfs::onnx_vulkan::detail
