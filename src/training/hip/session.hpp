/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "training_api.h"
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>

namespace lfs::training::hip {
    class TrainingError : public std::runtime_error {
    public:
        const int status;
        const int native_code;
        TrainingError(int status, const LfsHipError& error);
    };

    // Optional bounded host staging, independent of session lifetime. Allocation
    // failure returns an empty owner and logs the ordinary-RAM fallback.
    class HostTargets {
    public:
        HostTargets(const std::filesystem::path& module, int device, size_t budget);
        std::shared_ptr<void> allocate(size_t bytes);
        size_t allocated_bytes() const;
    private:
        struct Impl;
        std::shared_ptr<Impl> impl_;
    };

    // Host compiler boundary: no HIP/CUDA headers or native streams.
    // The module remains loaded until the session has finished destruction.
    // Operations serialize; cancellation may interrupt an operation on another
    // thread. Join callers before destroying the C++ object itself.
    class Session {
    public:
        Session(const std::filesystem::path& module, const LfsHipTrainConfig&, const LfsHipTrainModel&);
        Session(const std::filesystem::path& module, const LfsHipTrainConfig&, const LfsHipTrainModel&, const LfsHipTrainSharedModel&);
        Session(const std::filesystem::path& module, const LfsHipTrainConfig&, const LfsHipTrainModel&, const LfsHipTrainSharedModel&, const LfsHipTrainModelLayout&);
        // Attach resident base parameters; starts fresh optimizer state.
        Session(const std::filesystem::path& module, const LfsHipTrainConfig&, const LfsHipTrainSharedModel&, const LfsHipTrainModelLayout&);
        // Owner is retained until successful teardown, including on failures.
        Session(const std::filesystem::path& module, const LfsHipTrainConfig&, const LfsHipTrainDeviceModel&, const LfsHipTrainModelLayout&, std::shared_ptr<void> owner, std::function<void()> validate = {}, std::function<void(int)> invalidate = {});
        static LfsHipTrainModelLayout model_layout(const std::filesystem::path& module, uint64_t count);
        ~Session();
        Session(const Session&) = delete;
        Session& operator=(const Session&) = delete;
        void set_frame(const LfsHipTrainFrame&);
        void set_rasterizer(uint32_t kind);
        void set_optimizer(uint32_t kind);
        void step(const LfsHipTrainRates&);
        void render(std::span<float> output);
        void read_parameter(uint32_t family, std::span<float> output);
        LfsHipTrainInfo info();
        void set_budget(uint64_t bytes);
        void set_loss(const LfsHipTrainLoss&);
        float loss();
        void set_sh(const LfsHipTrainSh&);
        void attach_resident_sh(const LfsHipTrainResidentSh&);
        void set_sh_degree(uint32_t degree, float rest_learning_rate);
        void set_cancel(bool cancelled);
        // Explicit teardown reports errors and permits retry. Destructor failure
        // logs a diagnostic and pins the module rather than unloading live work.
        void close();

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace lfs::training::hip
