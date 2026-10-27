/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "runtime_context.hpp"
#include "runtime_fault_control.hpp"
#include "runtime_test_directory.hpp"
#include <array>
#include <exception>
#include <iostream>
#include <thread>
#ifndef _WIN32
#include <dlfcn.h>
#endif

using namespace lfs::core::internal;
namespace {
    void require(bool condition, const char* message) {
        if (!condition)
            throw std::runtime_error(message);
    }
    template <class Exception, class F>
    bool rejects(F&& function) {
        try {
            function();
        } catch (const Exception&) {
            return true;
        }
        return false;
    }
    using lfs::test::FaultControl;
    void recovery(const std::filesystem::path& path) {
        FaultControl faults(path);
        {
            auto module = std::make_shared<HipRuntimeModule>(path);
            HipRuntimeContext context(module, 0);
            faults.arm(8);
            require(rejects<HipRuntimeError>([&] { context.allocate(16); }), "Allocation fault was ignored");
            auto data = context.allocate(16);
            faults.arm(9);
            require(rejects<HipRuntimeError>([&] { data.close(); }), "Failed release lost its retryable error");
            require(rejects<std::logic_error>([&] { context.close(); }), "Failed release removed its live lease");
            data.close();
            context.close();
        }
        require(faults.resources() == 0, "Explicit retry leaked native resources");
        {
            auto module = std::make_shared<HipRuntimeModule>(path);
            HipRuntimeContext context(module, 0);
            {
                auto retained = context.allocate(16);
                faults.arm(9);
            }
            require(faults.resources() != 0, "Failed buffer destruction lost its owned allocation");
            context.close();
        }
        require(faults.resources() == 0, "Context failed to release retained buffer storage");
        for (const bool timing : {false, true}) {
            auto module = std::make_shared<HipRuntimeModule>(path);
            HipRuntimeContext context(module, 0);
            auto retirement = std::make_unique<HipRetiredResource>();
            LfsHipReadback* readback = nullptr;
            LfsHipTiming* events = nullptr;
            {
                auto queue = context.lock();
                LfsHipError error{};
                if (timing)
                    hip_runtime_check(queue.api().timing_create(queue.handle(), 2, &events, &error), error, "timing_create");
                else
                    hip_runtime_check(queue.api().readback_create(queue.handle(), &readback, &error), error, "readback_create");
            }
            faults.arm(4);
            if (timing)
                context.release_timing(events, std::move(retirement));
            else
                context.release_readback(readback, std::move(retirement));
            require(faults.resources() != 0, "Failed transfer/timing cleanup lost its resources");
            faults.arm(4);
            require(rejects<HipRuntimeError>([&] { context.close(); }), "Failed resource retry lost its error");
            context.close();
            require(faults.resources() == 0, "Resource cleanup retry leaked native events");
        }
        std::weak_ptr<HipRuntimeModule> weak;
        {
            auto module = std::make_shared<HipRuntimeModule>(path);
            weak = module;
            HipRuntimeContext context(module, 0);
            faults.arm(7);
        }
        require(!weak.expired(), "Failed context cleanup unloaded its module");
        faults.arm(7);
        require(HipRuntimeContext::retry_retired_contexts() == 1, "Failed retry dropped native ownership");
        require(!weak.expired(), "Failed retry released module ownership");
        require(HipRuntimeContext::retry_retired_contexts() == 0, "Context cleanup retry failed");
        require(weak.expired(), "Successful cleanup retained its module");
        require(faults.resources() == 0, "Context cleanup left native resources");
    }
    void native_math(HipRuntimeContext& context) {
        constexpr std::array<float, 5> values{1, -2, 3, 4, -5};
        auto input = context.allocate(sizeof(values));
        auto output = context.allocate(sizeof(values));
        input.upload(0, values.data(), sizeof(values));
        {
            auto queue = context.lock();
            LfsHipError error{};
            hip_runtime_check(queue.api().pointwise_f32(queue.handle(), LFS_HIP_OP_Square,
                                                        queue.view(input, 0, sizeof(values)), {}, queue.view(output, 0, sizeof(values)),
                                                        values.size(), 0, &error),
                              error, "square");
        }
        std::array<float, 5> actual{};
        output.download(0, actual.data(), sizeof(actual));
        require(actual == std::array<float, 5>{1, 4, 9, 16, 25}, "Native dispatch result differs");
        require(rejects<std::logic_error>([&] { context.close(); }), "Context accepted live buffer leases");
        require(rejects<std::out_of_range>([&] { output.download(sizeof(actual), actual.data(), 4); }), "Host range validation failed");
        input.close();
        input.close();
        output.close();
    }
} // namespace

#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
#else
int main(int argc, char** argv) {
#endif
    try {
        if (argc != 3 && argc != 4)
            throw std::runtime_error("Expected explicit module and SDK directory");
        RuntimeDirectory directory{std::filesystem::path(argv[2])};
        const auto path = std::filesystem::path(argv[1]);
        auto module = std::make_shared<HipRuntimeModule>(path);
        HipRuntimeContext context(module, 0);
        native_math(context);
        HipRuntimeContext foreign(module, 0);
        auto other = foreign.allocate(16);
        {
            auto queue = context.lock();
            require(rejects<std::invalid_argument>([&] { queue.view(other, 0, 4); }), "Foreign context buffer accepted");
        }
        other.close();
        foreign.close();

        std::exception_ptr failures[2];
        auto worker = [&](int index) {
            try {
                for (int iteration = 0; iteration < 8; ++iteration) {
                    auto data = context.allocate(sizeof(float));
                    const float input = static_cast<float>(index * 100 + iteration);
                    data.upload(0, &input, sizeof(input));
                    float output = -1;
                    data.download(0, &output, sizeof(output));
                    require(output == input, "Serialized host calls corrupted data");
                }
            } catch (...) {
                failures[index] = std::current_exception();
            }
        };
        std::thread first(worker, 0), second(worker, 1);
        first.join();
        second.join();
        for (const auto& failure : failures)
            if (failure)
                std::rethrow_exception(failure);
        context.close();
        context.close();
        require(rejects<std::logic_error>([&] { context.allocate(4); }), "Closed context accepted allocation");

        HipRuntimeBuffer survivor;
        std::weak_ptr<HipRuntimeModule> weak;
        {
            auto owned_module = std::make_shared<HipRuntimeModule>(path);
            weak = owned_module;
            HipRuntimeContext owner(owned_module, 0);
            survivor = owner.allocate(4);
            const float value = 17;
            survivor.upload(0, &value, 4);
        }
        require(!weak.expired(), "Live buffer failed to pin context/module");
        float result = 0;
        survivor.download(0, &result, 4);
        require(result == 17, "Surviving buffer was invalidated");
        survivor = {};
        require(weak.expired(), "Released buffer leaked its module ownership");
        require(HipRuntimeContext::retry_retired_contexts() == 0, "Unexpected failed context teardown");
        if (argc == 4)
            recovery(path);
        std::cout << "HIP host ownership and serialized dispatch passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
