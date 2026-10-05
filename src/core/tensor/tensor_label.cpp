/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor_label.hpp"

#include <utility>

namespace lfs::core {
    namespace {
        thread_local std::string current_label;
    }

    std::string_view current_tensor_label() noexcept {
        return current_label;
    }

    std::string exchange_tensor_label(std::string label) noexcept {
        current_label.swap(label);
        return label;
    }

    TensorLabelScope::TensorLabelScope(const std::string_view label)
        : previous_(exchange_tensor_label(std::string(label))) {}

    TensorLabelScope::~TensorLabelScope() {
        static_cast<void>(exchange_tensor_label(std::move(previous_)));
    }
} // namespace lfs::core
