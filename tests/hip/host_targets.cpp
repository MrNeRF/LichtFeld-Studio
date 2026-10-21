/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../../src/training/hip/session.hpp"
#include "training_session_fixture.hpp"
#include <atomic>
#include <limits>
#include <cstring>
#include <thread>
#include <vector>
#include <cstdio>
using lfs::training::hip::HostTargets;
static void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
int main(int argc, char** argv) try {
    require(argc == 2, "Pass a training module");
    const std::filesystem::path module(argv[1]);
    std::shared_ptr<void> survivor;
    {
        HostTargets pool(module, 0, 8192);
        auto first = pool.allocate(4096);
        require(bool(first), "HIP host allocation failed");
        std::memset(first.get(), 0x37, 4096);
        survivor = pool.allocate(4096);
        require(bool(survivor) && pool.allocated_bytes() == 8192, "Host budget accounting");
        require(!pool.allocate(1), "Host budget did not reject excess");
        auto copy = first;
        first.reset();
        require(pool.allocated_bytes() == 8192, "Copied owner released early");
        require(static_cast<unsigned char*>(copy.get())[4095] == 0x37, "Host contents changed");
        copy.reset();
        require(pool.allocated_bytes() == 4096, "Host release did not restore budget");
        auto replacement = pool.allocate(4096);
        require(bool(replacement), "Host budget not reusable");
        std::memset(survivor.get(), 0x42, 4096);
    }
    require(static_cast<unsigned char*>(survivor.get())[4095] == 0x42, "Owner did not outlive pool/module");
    survivor.reset();
    HostTargets pool(module, 0, 4 * 4096);
    std::atomic<bool> ok{true};
    std::vector<std::thread> threads;
    for (int i = 0; i < 4; ++i) threads.emplace_back([&] {
        try {
            for (int j = 0; j < 32; ++j) {
                auto allocation = pool.allocate(4096);
                if (!allocation) { ok = false; break; }
                std::memset(allocation.get(), 0x73, 4096);
                if (static_cast<unsigned char*>(allocation.get())[4095] != 0x73) ok = false;
            }
        } catch (...) { ok = false; }
    });
    for (auto& thread : threads) thread.join();
    require(ok && pool.allocated_bytes() == 0, "Concurrent host owners/accounting failed");
    HostTargets invalid(module, -1, 4096);
    require(!invalid.allocate(4096) && invalid.allocated_bytes() == 0, "Failed native allocation consumed budget");
    require(bool(pool.allocate(4096)), "Optional failure affected healthy allocation");
    {
        using namespace hip_fixture;
        Fixture source;
        const LfsHipTrainConfig config{sizeof(config), 0, n, 64 * 1024 * 1024};
        lfs::training::hip::Session session(module, config, source.model());
        const size_t bytes = source.target.size() * sizeof(float);
        HostTargets targets(module, 0, bytes);
        auto storage = targets.allocate(bytes);
        require(bool(storage), "Pinned frame allocation failed");
        std::memcpy(storage.get(), source.target.data(), bytes);
        auto frame = source.frame();
        frame.target_chw = static_cast<float*>(storage.get());
        session.set_frame(frame);
        const float original = session.loss();
        static_cast<float*>(storage.get())[0] = std::numeric_limits<float>::quiet_NaN();
        bool rejected = false;
        try { session.set_frame(frame); }
        catch (const lfs::training::hip::TrainingError& error) { rejected = error.status == LFS_HIP_TRAIN_INVALID; }
        require(rejected && session.loss() == original, "Invalid pinned target damaged the prepared frame");
        storage.reset();
        require(targets.allocated_bytes() == 0 && session.loss() == original,
                "set_frame retained a borrowed host pointer");
        session.close();
    }
    std::puts("HIP owned host targets: budget, reuse, copied ownership, lifetime, concurrency and native fallback passed");
    return 0;
} catch (const std::exception& error) { std::fprintf(stderr, "%s\n", error.what()); return 1; }
