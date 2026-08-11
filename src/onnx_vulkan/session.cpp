/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs_onnx_vulkan/onnx_vulkan.hpp"

#include "execution_plan.hpp"
#include "model.hpp"
#include "vulkan_runtime.hpp"

#include <algorithm>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

namespace lfs::onnx_vulkan {
    namespace {
        [[nodiscard]] std::expected<std::size_t, Error>
        checked_bytes(const TensorView& tensor) {
            std::size_t count = 1;
            for (const auto extent : tensor.shape) {
                if (extent < 0)
                    return std::unexpected(Error{ErrorCode::InvalidInput, "input tensor has a negative extent"});
                if (extent == 0) {
                    count = 0;
                    break;
                }
                if (count > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(extent))
                    return std::unexpected(Error{ErrorCode::InvalidInput, "input tensor size overflows"});
                count *= static_cast<std::size_t>(extent);
            }
            if (count > std::numeric_limits<std::size_t>::max() / element_size(tensor.type))
                return std::unexpected(Error{ErrorCode::InvalidInput, "input tensor byte size overflows"});
            return count * element_size(tensor.type);
        }
    }

    struct VulkanSession::Impl {
        detail::Model model;
        std::unique_ptr<detail::VulkanRuntime> runtime;
        detail::WeightStore weights;
        std::vector<ValueInfo> inputs;
        std::vector<ValueInfo> outputs;
        std::unordered_map<std::string, std::unique_ptr<detail::ExecutionPlan>> plans;
        std::mutex mutex;
    };

    VulkanSession::VulkanSession(std::unique_ptr<Impl> impl) noexcept
        : impl_(std::move(impl)) {}
    VulkanSession::VulkanSession(VulkanSession&&) noexcept = default;
    VulkanSession& VulkanSession::operator=(VulkanSession&&) noexcept = default;
    VulkanSession::~VulkanSession() = default;

    std::expected<VulkanSession, Error>
    VulkanSession::create(const std::filesystem::path& model_path, SessionOptions options) {
        auto impl = std::make_unique<Impl>();
        auto model = detail::parse_model(model_path, options);
        if (!model)
            return std::unexpected(model.error());
        if (auto valid = detail::validate_model(*model); !valid)
            return std::unexpected(valid.error());
        impl->model = std::move(*model);
        impl->inputs = impl->model.graph.inputs;
        impl->outputs = impl->model.graph.outputs;
        auto runtime = detail::VulkanRuntime::create(options);
        if (!runtime)
            return std::unexpected(runtime.error());
        impl->runtime = std::move(*runtime);
        auto weights = detail::upload_weights(impl->model, *impl->runtime);
        if (!weights)
            return std::unexpected(weights.error());
        impl->weights = std::move(*weights);
        return VulkanSession(std::move(impl));
    }

    std::span<const ValueInfo> VulkanSession::inputs() const noexcept { return impl_->inputs; }
    std::span<const ValueInfo> VulkanSession::outputs() const noexcept { return impl_->outputs; }
    std::string_view VulkanSession::device_name() const noexcept { return impl_->runtime->device_name(); }

    std::expected<std::vector<NamedTensor>, Error>
    VulkanSession::run(const std::span<const NamedTensorView> named_inputs,
                       const std::span<const std::string_view> requested_outputs) {
        std::scoped_lock lock(impl_->mutex);
        if (named_inputs.size() != impl_->inputs.size())
            return std::unexpected(Error{ErrorCode::InvalidInput, "named input count does not match the model"});
        std::unordered_set<std::string_view> names;
        for (const auto& input : named_inputs) {
            if (!names.emplace(input.name).second)
                return std::unexpected(Error{ErrorCode::InvalidInput, "duplicate named input '" + std::string(input.name) + "'"});
            const auto expected = std::ranges::find(impl_->inputs, input.name, &ValueInfo::name);
            if (expected == impl_->inputs.end())
                return std::unexpected(Error{ErrorCode::InvalidInput, "unknown named input '" + std::string(input.name) + "'"});
            if (expected->type != input.tensor.type)
                return std::unexpected(Error{ErrorCode::InvalidInput, "input type mismatch for '" + std::string(input.name) + "'"});
            if (expected->shape.size() != input.tensor.shape.size())
                return std::unexpected(Error{ErrorCode::InvalidInput, "input rank mismatch for '" + std::string(input.name) + "'"});
            for (std::size_t axis = 0; axis < expected->shape.size(); ++axis)
                if (expected->shape[axis] >= 0 && expected->shape[axis] != input.tensor.shape[axis])
                    return std::unexpected(Error{ErrorCode::InvalidInput, "input shape mismatch for '" + std::string(input.name) + "'"});
            auto bytes = checked_bytes(input.tensor);
            if (!bytes)
                return std::unexpected(bytes.error());
            if (*bytes != input.tensor.bytes.size())
                return std::unexpected(Error{ErrorCode::InvalidInput, "input byte length mismatch for '" + std::string(input.name) + "'"});
        }
        for (const auto output : requested_outputs)
            if (std::ranges::find(impl_->outputs, output, &ValueInfo::name) == impl_->outputs.end())
                return std::unexpected(Error{ErrorCode::InvalidInput, "unknown requested output '" + std::string(output) + "'"});

        const auto signature = detail::plan_signature(named_inputs, requested_outputs);
        auto plan = impl_->plans.find(signature);
        if (plan == impl_->plans.end()) {
            auto created = detail::ExecutionPlan::create(impl_->model, impl_->weights, *impl_->runtime,
                                                         named_inputs, requested_outputs);
            if (!created)
                return std::unexpected(created.error());
            plan = impl_->plans.emplace(signature, std::move(*created)).first;
        }
        return plan->second->run(named_inputs);
    }

} // namespace lfs::onnx_vulkan
