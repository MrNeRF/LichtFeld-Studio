// SPDX-License-Identifier: GPL-3.0-or-later
#include "sharpness_score.cuh"
#include "core/tensor_execution.hpp"
#include "sharpness_tensor.hpp"
namespace lfs::io {
    struct SharpnessScorer::State {
        TensorSharpnessScorer scorer;
    };
    SharpnessScorer::SharpnessScorer() = default;
    SharpnessScorer::~SharpnessScorer() = default;
    std::optional<double> SharpnessScorer::scoreDevice(const uint8_t* rgb, int w, int h, SharpnessAlgorithm algorithm) {
        if (w < 3 || h < 3)
            return 0.0;
        if (!state_)
            state_ = std::make_unique<State>();
        const auto target = core::TensorExecutionTarget::default_queue(core::GpuBackend::CUDA);
        const auto view = core::Tensor::from_blob(const_cast<uint8_t*>(rgb), {size_t(h), size_t(w), 3},
                                                  core::Device::GPU, core::DataType::UInt8, target);
        return state_->scorer.score(view, w, h, algorithm);
    }
} // namespace lfs::io
