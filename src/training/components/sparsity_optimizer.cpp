/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "sparsity_optimizer.hpp"
#include "core/logger.hpp"
#include "core/tensor/internal/tensor_serialization.hpp"
#include <cuda_runtime.h>
#include <format>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace lfs::training {

    namespace {
        constexpr uint32_t SPARSITY_CHECKPOINT_MAGIC = 0x4C464153; // "LFAS"
        constexpr uint32_t SPARSITY_CHECKPOINT_VERSION = 1;

        using SerializedState = std::tuple<lfs::core::Tensor, lfs::core::Tensor, lfs::core::Tensor>;

        void validate_state_tensors(
            const lfs::core::Tensor& z,
            const lfs::core::Tensor& u,
            const lfs::core::Tensor& opa_sigmoid,
            const std::optional<size_t> expected_rows) {
            const auto valid_state_tensor = [](const lfs::core::Tensor& tensor) {
                return tensor.is_valid() && tensor.dtype() == lfs::core::DataType::Float32 &&
                       tensor.ndim() == 2 && tensor.size(0) > 0 && tensor.size(1) == 1;
            };
            if (!valid_state_tensor(z) || !valid_state_tensor(u) || !valid_state_tensor(opa_sigmoid) ||
                u.shape() != z.shape() || opa_sigmoid.shape() != z.shape()) {
                throw std::runtime_error("Invalid sparsity ADMM checkpoint tensor schema");
            }
            if (expected_rows && z.size(0) != *expected_rows) {
                throw std::runtime_error("Invalid checkpoint: sparsity ADMM state does not match model count");
            }
        }

        SerializedState read_serialized_state(std::istream& is) {
            uint32_t magic = 0;
            uint32_t version = 0;
            lfs::core::serialization_detail::read_exact(is, &magic, sizeof(magic), "sparsity-ADMM magic");
            lfs::core::serialization_detail::read_exact(is, &version, sizeof(version), "sparsity-ADMM version");
            if (magic != SPARSITY_CHECKPOINT_MAGIC) {
                throw std::runtime_error("Invalid sparsity ADMM checkpoint");
            }
            if (version != SPARSITY_CHECKPOINT_VERSION) {
                throw std::runtime_error("Unsupported sparsity ADMM checkpoint version");
            }

            lfs::core::Tensor z;
            lfs::core::Tensor u;
            lfs::core::Tensor opa_sigmoid;
            is >> z >> u >> opa_sigmoid;
            validate_state_tensors(z, u, opa_sigmoid, std::nullopt);
            return {std::move(z), std::move(u), std::move(opa_sigmoid)};
        }

        size_t checked_add(const size_t lhs, const size_t rhs) {
            if (rhs > std::numeric_limits<size_t>::max() - lhs) {
                throw std::runtime_error("Sparsity ADMM checkpoint byte size overflows");
            }
            return lhs + rhs;
        }

        size_t serialized_tensor_size(const lfs::core::Tensor& tensor) {
            auto bytes = checked_add(sizeof(lfs::core::TensorFileHeader), tensor.bytes());
            if (tensor.ndim() > std::numeric_limits<size_t>::max() / sizeof(uint64_t)) {
                throw std::runtime_error("Sparsity ADMM tensor rank byte size overflows");
            }
            return checked_add(bytes, tensor.ndim() * sizeof(uint64_t));
        }
    } // namespace

    // Forward declaration of CUDA kernel launcher (defined in sparsity_optimizer_kernels.cu)
    void launch_admm_backward_fused(
        float* grad_opacities,
        const float* opa_sigmoid,
        const float* z,
        const float* u,
        float rho,
        float grad_loss,
        size_t n,
        bool accumulate);

    ADMMSparsityOptimizer::ADMMSparsityOptimizer(const Config& config)
        : config_(config) {
    }

    void ADMMSparsityOptimizer::reset() {
        u_ = {};
        z_ = {};
        opa_sigmoid_ = {};
        initialized_ = false;
    }

    void ADMMSparsityOptimizer::serialize(std::ostream& os) const {
        validate_checkpoint_state(std::nullopt);
        os.write(reinterpret_cast<const char*>(&SPARSITY_CHECKPOINT_MAGIC), sizeof(SPARSITY_CHECKPOINT_MAGIC));
        os.write(reinterpret_cast<const char*>(&SPARSITY_CHECKPOINT_VERSION), sizeof(SPARSITY_CHECKPOINT_VERSION));
        os << z_ << u_ << opa_sigmoid_;
    }

    void ADMMSparsityOptimizer::deserialize(std::istream& is) {
        auto [z, u, opa_sigmoid] = read_serialized_state(is);

        z_ = z.cuda();
        u_ = u.cuda();
        opa_sigmoid_ = opa_sigmoid.cuda();
        initialized_ = true;
    }

    size_t ADMMSparsityOptimizer::checkpoint_size_bytes(const size_t expected_rows) const {
        validate_checkpoint_state(expected_rows);
        auto bytes = checked_add(sizeof(SPARSITY_CHECKPOINT_MAGIC), sizeof(SPARSITY_CHECKPOINT_VERSION));
        bytes = checked_add(bytes, serialized_tensor_size(z_));
        bytes = checked_add(bytes, serialized_tensor_size(u_));
        return checked_add(bytes, serialized_tensor_size(opa_sigmoid_));
    }

    size_t ADMMSparsityOptimizer::consume_checkpoint(std::istream& is) {
        auto state = read_serialized_state(is);
        return std::get<0>(state).size(0);
    }

    void ADMMSparsityOptimizer::validate_checkpoint_state(
        const std::optional<size_t> expected_rows) const {
        if (!initialized_) {
            throw std::runtime_error("Cannot serialize uninitialized sparsity ADMM state");
        }
        validate_state_tensors(z_, u_, opa_sigmoid_, expected_rows);
    }

    void ADMMSparsityOptimizer::adopt_checkpoint_state(ADMMSparsityOptimizer& loaded) noexcept {
        std::swap(config_, loaded.config_);
        std::swap(z_, loaded.z_);
        std::swap(u_, loaded.u_);
        std::swap(opa_sigmoid_, loaded.opa_sigmoid_);
        std::swap(initialized_, loaded.initialized_);
    }

    std::expected<void, std::string>
    ADMMSparsityOptimizer::ensure_state_matches(const lfs::core::Tensor& opacities, const char* phase) {
        if (!initialized_) {
            return initialize(opacities);
        }

        if (!u_.is_valid() || !z_.is_valid() || !opa_sigmoid_.is_valid()) {
            LOG_WARN("Sparsity: resetting ADMM state during {} because cached tensors are invalid", phase);
            reset();
            return initialize(opacities);
        }

        if (u_.shape() != opacities.shape() ||
            z_.shape() != opacities.shape() ||
            opa_sigmoid_.shape() != opacities.shape()) {
            LOG_WARN("Sparsity: resetting ADMM state during {} after topology change (opacity={}, z={}, u={}, opa={})",
                     phase,
                     opacities.shape().str(),
                     z_.shape().str(),
                     u_.shape().str(),
                     opa_sigmoid_.shape().str());
            reset();
            return initialize(opacities);
        }

        return {};
    }

    std::expected<void, std::string> ADMMSparsityOptimizer::initialize(const lfs::core::Tensor& opacities) {
        try {
            if (opacities.numel() == 0) {
                return std::unexpected("Invalid opacity tensor for initialization");
            }

            // Initialize ADMM variables
            // opa = sigmoid(opacities)
            opa_sigmoid_ = opacities.sigmoid();

            // u = zeros_like(opa)
            u_ = lfs::core::Tensor::zeros(opa_sigmoid_.shape(),
                                          lfs::core::Device::CUDA,
                                          lfs::core::DataType::Float32);

            // z = prune_z(opa + u)
            auto opa_plus_u = opa_sigmoid_ + u_;
            z_ = prune_z(opa_plus_u);

            initialized_ = true;
            return {};
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to initialize ADMM sparsity optimizer: {}", e.what());
            return std::unexpected(std::format("Failed to initialize ADMM optimizer: {}", e.what()));
        }
    }

    std::expected<std::pair<lfs::core::Tensor, SparsityLossContext>, std::string>
    ADMMSparsityOptimizer::compute_loss_forward(const lfs::core::Tensor& opacities) {
        try {
            if (opacities.numel() == 0) {
                return std::unexpected("Invalid opacity tensor for loss computation");
            }

            if (auto result = ensure_state_matches(opacities, "loss forward"); !result) {
                return std::unexpected(result.error());
            }

            // Compute ADMM sparsity loss (manual - no autograd)
            // Loss: L = 0.5 * rho * ||opa - z + u||^2
            // ALL ON GPU - NO CPU SYNC!

            // opa = sigmoid(opacities)
            opa_sigmoid_ = opacities.sigmoid();

            // diff = opa - z + u
            auto diff = opa_sigmoid_ - z_ + u_;

            // ||diff||^2 = sum(diff^2) - stays on GPU as scalar tensor
            auto diff_sq_sum = diff.square().sum();

            // loss = 0.5 * rho * ||diff||^2 - GPU scalar
            auto loss_tensor = diff_sq_sum * (0.5f * config_.init_rho);

            // Create minimal context (pointers only, no tensor copies)
            SparsityLossContext ctx{
                .opacities_ptr = opacities.template ptr<const float>(),
                .opa_sigmoid_ptr = opa_sigmoid_.template ptr<const float>(),
                .z_ptr = z_.template ptr<const float>(),
                .u_ptr = u_.template ptr<const float>(),
                .n = static_cast<size_t>(opacities.numel()),
                .rho = config_.init_rho};

            return std::make_pair(std::move(loss_tensor), ctx);
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to compute ADMM sparsity loss (manual forward): {}", e.what());
            return std::unexpected(std::format("Failed to compute sparsity loss: {}", e.what()));
        }
    }

    std::expected<void, std::string>
    ADMMSparsityOptimizer::compute_loss_backward(const SparsityLossContext& ctx,
                                                 float grad_loss,
                                                 lfs::core::Tensor& grad_opacities) {
        try {
            // Use FUSED CUDA KERNEL for maximum performance
            // Computes: grad_opacities = rho * (opa - z + u) * opa * (1 - opa) * grad_loss
            // Single kernel launch, zero intermediate tensor allocations!
            // accumulate=true: adds to existing gradients (if grad_opacities already has values)
            // accumulate=false: overwrites (no need to zero first - saves 6 μs!)

            const size_t n = ctx.n;

            // Launch fused kernel via wrapper function
            // Note: Use accumulate=true in production if other losses contribute gradients
            launch_admm_backward_fused(
                grad_opacities.ptr<float>(), // Output: gradients
                ctx.opa_sigmoid_ptr,         // Input: sigmoid(opacities)
                ctx.z_ptr,                   // Input: ADMM auxiliary variable
                ctx.u_ptr,                   // Input: ADMM dual variable
                ctx.rho,                     // ADMM penalty parameter
                grad_loss,                   // Gradient from upstream
                n,                           // Number of elements
                true                         // accumulate: add to existing grads
            );

            // Check for kernel errors
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                return std::unexpected(std::format("CUDA kernel error: {}", cudaGetErrorString(err)));
            }

            return {};
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to compute ADMM sparsity loss backward: {}", e.what());
            return std::unexpected(std::format("Failed to compute sparsity loss backward: {}", e.what()));
        }
    }

    std::expected<void, std::string> ADMMSparsityOptimizer::update_state(const lfs::core::Tensor& opacities) {
        try {
            if (opacities.numel() == 0) {
                return std::unexpected("Invalid opacity tensor for state update");
            }

            if (auto result = ensure_state_matches(opacities, "state update"); !result) {
                return std::unexpected(result.error());
            }

            // ADMM update step
            // opa = sigmoid(opacities)
            opa_sigmoid_ = opacities.sigmoid();

            // z_temp = opa + u
            auto z_temp = opa_sigmoid_ + u_;

            // z = prune_z(z_temp)
            z_ = prune_z(z_temp);

            // u += opa - z
            auto opa_minus_z = opa_sigmoid_ - z_;
            u_.add_(opa_minus_z);

            return {};
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to update ADMM state: {}", e.what());
            return std::unexpected(std::format("Failed to update ADMM state: {}", e.what()));
        }
    }

    std::expected<lfs::core::Tensor, std::string>
    ADMMSparsityOptimizer::get_prune_mask(const lfs::core::Tensor& opacities) {
        try {
            if (opacities.numel() == 0) {
                return std::unexpected("Invalid opacity tensor for pruning");
            }

            // opa = sigmoid(opacities.flatten())
            auto opa = opacities.flatten().sigmoid();
            int n_prune = static_cast<int>(config_.prune_ratio * opa.shape()[0]);

            if (n_prune == 0) {
                return lfs::core::Tensor::zeros_bool({opa.shape()[0]}, lfs::core::Device::CUDA);
            }

            // Find indices of smallest opacities using sort
            // sort returns (values, indices) - we want smallest so ascending=true
            auto [sorted_values, sorted_indices] = opa.sort(0, /*descending=*/false);

            // Take first n_prune elements (indices only, we don't need the values)
            auto prune_indices = sorted_indices.slice(0, 0, n_prune).to(lfs::core::DataType::Int64);

            // Create boolean mask and use proper index_put_ (now that it's fixed!)
            auto mask = lfs::core::Tensor::zeros_bool({opa.shape()[0]}, lfs::core::Device::CUDA);
            auto true_values = lfs::core::Tensor::ones_bool({static_cast<size_t>(n_prune)}, lfs::core::Device::CUDA);

            // Use index_put_ to set mask values
            mask.index_put_({prune_indices}, true_values);

            return mask;
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to generate prune mask: {}", e.what());
            return std::unexpected(std::format("Failed to generate prune mask: {}", e.what()));
        }
    }

    int ADMMSparsityOptimizer::get_num_to_prune(const lfs::core::Tensor& opacities) {
        if (opacities.numel() == 0) {
            return 0;
        }
        return static_cast<int>(config_.prune_ratio * opacities.flatten().shape()[0]);
    }

    lfs::core::Tensor ADMMSparsityOptimizer::prune_z(const lfs::core::Tensor& z) {
        if (z.numel() == 0) {
            return lfs::core::Tensor::zeros(z.shape(), lfs::core::Device::CUDA);
        }

        int index = static_cast<int>(config_.prune_ratio * z.shape()[0]);
        if (index == 0) {
            return lfs::core::Tensor::zeros(z.shape(), lfs::core::Device::CUDA);
        }

        // Sort to find threshold
        auto [z_sorted, _] = z.flatten().sort(0, /*descending=*/false);

        // Get threshold - create tensor containing single element, then extract
        auto z_threshold_tensor = z_sorted.slice(0, index - 1, index);
        float z_threshold = z_threshold_tensor.item<float>();

        // Apply soft thresholding: result = (z > threshold) * z
        // This keeps values above threshold, zeros out values below
        auto threshold_mask = (z > z_threshold);
        auto result = lfs::core::Tensor::where(threshold_mask, z,
                                               lfs::core::Tensor::zeros(z.shape(), lfs::core::Device::CUDA));

        return result;
    }

    // Factory implementation
    std::unique_ptr<ISparsityOptimizer> SparsityOptimizerFactory::create(
        const std::string& method,
        const ADMMSparsityOptimizer::Config& config) {
        if (method == "admm") {
            return std::make_unique<ADMMSparsityOptimizer>(config);
        }
        LOG_ERROR("Unknown sparsity optimization method: {}", method);
        return nullptr;
    }

} // namespace lfs::training
