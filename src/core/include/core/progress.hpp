/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <functional>
#include <stdexcept>
#include <stop_token>
#include <string>
namespace lfs::core {
    struct OperationCancelled : std::runtime_error {
        OperationCancelled() : std::runtime_error("Operation cancelled") {}
    };
    using ProgressCallback = std::function<bool(float, const std::string&)>;
    inline void reportProgress(std::stop_token stop, const ProgressCallback& callback, float value, const std::string& stage) {
        if (stop.stop_requested() || (callback && !callback(value, stage)) || stop.stop_requested())
            throw OperationCancelled();
    }
} // namespace lfs::core
