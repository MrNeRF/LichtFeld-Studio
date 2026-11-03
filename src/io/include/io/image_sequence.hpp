// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "core/tensor.hpp"
#include "io/image_sequence_options.hpp"
#include <functional>

namespace lfs::io {
    // Thin Tensor producer adapter. Shared sinks own filenames and publication;
    // shared transfer curves own SDR conversion. At most one frame is read back.
    class TensorImageSequence {
    public:
        [[nodiscard]] media::SinkResult open(const TensorImageSequenceOptions&);
        [[nodiscard]] media::SinkResult write(const core::Tensor&);
        [[nodiscard]] media::SinkResult close();
        void abort() noexcept { sequence_.abort(); }
        [[nodiscard]] bool isOpen() const noexcept { return sequence_.isOpen(); }

    private:
        media::FrameLayout layout_;
        bool input_linear_ = true;
        std::function<bool()> cancelled_;
        media::ImageSequence sequence_;
        std::vector<float> linear_;
        std::vector<uint8_t> bytes_;
        bool in_call_ = false;
    };
} // namespace lfs::io
