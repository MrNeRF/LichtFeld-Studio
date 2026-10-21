/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../../src/training/hip/session.hpp"
#include "training_session_fixture.hpp"
#include <algorithm>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <string_view>

using lfs::training::hip::Session;
using lfs::training::hip::TrainingError;

static void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

int main(int argc, char** argv) {
    try {
        require(argc == 2, "Pass an absolute HIP training module path");
        const std::filesystem::path module(argv[1]);
        using namespace hip_fixture;
        {
            Fixture source;
            const LfsHipTrainConfig config{sizeof(config), 0, n, 64 * 1024 * 1024};
            Session session(module, config, source.model());
            const auto original = session.info().allocated_bytes;
            const char* value = std::getenv("LFS_HIP_FAST_TRIM_UNUSED");
            const bool trim = value && std::string_view(value) == "1";
            session.set_rasterizer(1);
            const auto trimmed = session.info().allocated_bytes;
            require(trimmed == original - (trim ? size_t(n) * 56 : 0),
                    "FastGS unused geometry accounting differs");
            if (trim) {
                // Reject both immediately and after some temporary fields have
                // been allocated, exercising transactional rollback.
                for (uint64_t limit : {trimmed, trimmed + uint64_t(n) * 40}) {
                    session.set_budget(limit);
                    bool exhausted = false;
                    try {
                        session.set_rasterizer(0);
                    } catch (const TrainingError& error) {
                        exhausted = error.status == LFS_HIP_TRAIN_RESOURCE_EXHAUSTED;
                    }
                    require(exhausted && session.info().allocated_bytes == trimmed && !session.info().poisoned,
                            "Failed 3DGUT geometry restoration changed the allocation state");
                }
                session.set_budget(config.allocation_budget_bytes);
            }
            session.set_rasterizer(0);
            require(session.info().allocated_bytes == original, "3DGUT geometry restoration changed retained bytes");
            session.set_frame(source.frame());
            session.step({});
            require(session.info().completed_steps == 1 && !session.info().poisoned,
                    "3DGUT failed after restoring unused geometry");
            std::puts("FastGS unused geometry: byte accounting, failed restoration and subsequent 3DGUT training pass");
        }
        {
            Fixture source;
            const LfsHipTrainConfig config{sizeof(config), 0, n, 64 * 1024 * 1024};
            Session session(module, config, source.model());
            session.set_frame(source.frame());
            const auto original = source.target;
            for (uint32_t bits : {0u, 0x80000000u, 1u, 0x007fffffu, 0x00800000u, 0x7f7fffffu, 0xff7fffffu}) {
                source.target[0] = source.target.back() = std::bit_cast<float>(bits);
                session.set_frame(source.frame()); // Every finite IEEE-754 value remains accepted.
            }
            source.target = original;
            session.set_frame(source.frame());
            const float loss = session.loss();
            const auto live = session.info().allocated_bytes;
            for (uint32_t bits : {0x7f800000u, 0xff800000u, 0x7fc00001u, 0xffc00001u, 0x7f800001u})
                for (size_t index : {size_t(0), size_t(5), source.target.size() / 2, source.target.size() - 1}) {
                    source.target[index] = std::bit_cast<float>(bits);
                    bool rejected = false;
                    try {
                        session.set_frame(source.frame());
                    } catch (const TrainingError& error) { rejected = error.status == LFS_HIP_TRAIN_INVALID; }
                    source.target[index] = original[index];
                    const auto state = session.info();
                    require(rejected && state.allocated_bytes == live && state.completed_steps == 0 && !state.poisoned,
                            "Nonfinite input changed frame storage, optimizer or session health");
                    require(std::abs(session.loss() - loss) < 1e-7f, "Nonfinite input changed the prepared GPU target");
                }
            session.set_frame(source.frame());
            std::puts("HIP input preflight: IEEE finite extremes accepted; infinities and quiet/signaling NaNs preserve previous frame");
        }
        for (bool fused : {false, true})
            for (unsigned degree = 0; degree <= 3; ++degree) {
                Fixture source;
                const LfsHipTrainConfig config{sizeof(config), 0, n, 64 * 1024 * 1024};
                Session teacher(module, config, source.model());
                teacher.set_rasterizer(1);
                std::vector<float> rest(n * 45);
                for (size_t i = 0; i < rest.size(); ++i)
                    rest[i] = 0.02f * std::sin(float(i));
                const LfsHipTrainSh sh{sizeof(sh), 3, degree, 0.002f, rest.data(), rest.size()};
                teacher.set_sh(sh);
                teacher.set_frame(source.frame());
                std::vector<float> target(channels);
                teacher.render(target);
                teacher.close();
                source.perturb();
                source.target = target;
                Session learner(module, config, source.model());
                learner.set_rasterizer(1);
                if (fused)
                    learner.set_optimizer(1);
                learner.set_sh(sh);
                learner.set_frame(source.frame());
                const LfsHipTrainRates rates{0.001f, 0.001f, 0.001f, 0.002f, 0.001f};
                std::vector<float> means_before(n * 3), means_after(n * 3);
                learner.read_parameter(0, means_before);
                learner.set_budget(learner.info().allocated_bytes);
                bool exhausted = false;
                try {
                    learner.step(rates);
                } catch (const TrainingError& error) {
                    exhausted = error.status == LFS_HIP_TRAIN_RESOURCE_EXHAUSTED;
                }
                learner.read_parameter(0, means_after);
                require(exhausted && means_before == means_after && learner.info().completed_steps == 0 && !learner.info().poisoned,
                        "FastGS HIP pre-optimizer budget failure mutated model or poisoned session");
                learner.set_budget(config.allocation_budget_bytes);
                const float before = learner.loss();
                bool rejected = false;
                try {
                    learner.set_rasterizer(0);
                } catch (const TrainingError& error) {
                    rejected = error.status == LFS_HIP_TRAIN_INVALID;
                }
                require(rejected, "FastGS HIP accepted a rasterizer switch after frame binding");
                for (unsigned step = 0; step < 100; ++step)
                    learner.step(rates);
                const float after = learner.loss();
                require(std::isfinite(after) && after < before * 0.35f, "FastGS HIP training failed to converge");
                require(learner.info().completed_steps == 100 && !learner.info().poisoned, "FastGS HIP step accounting failed");
                learner.set_cancel(true);
                bool cancelled = false;
                try {
                    learner.step(rates);
                } catch (const TrainingError& error) {
                    cancelled = error.status == LFS_HIP_TRAIN_CANCELLED;
                }
                require(cancelled && learner.info().completed_steps == 100, "FastGS HIP cancellation mutated optimizer step");
                learner.set_cancel(false);
                learner.step(rates);
                require(learner.info().completed_steps == 101, "FastGS HIP failed after cancellation reset");
                std::printf("FastGS HIP SH%u optimizer=%s: loss %.9g -> %.9g; budget and cancellation recovered\n", degree,
                            fused ? "joint-fused" : "fp32", before, after);
            }
        {
            Fixture source;
            const LfsHipTrainConfig config{sizeof(config), 0, n, 64 * 1024 * 1024};
            Session session(module, config, source.model());
            session.set_rasterizer(1);
            const LfsHipTrainRates zero_rates{};
            std::vector<float> image(channels), original_means(n * 3), final_means(n * 3);
            session.read_parameter(0, original_means);
            for (unsigned iteration = 0; iteration < 8; ++iteration) {
                auto frame = source.frame();
                const bool empty = (iteration % 2) == 0;
                if (empty)
                    frame.world_to_camera[11] = -100;
                session.set_frame(frame);
                session.step(zero_rates);
                session.render(image);
                for (float value : image)
                    require(std::isfinite(value) && (!empty || value == 0),
                            "FastGS phase scratch reuse corrupted an empty/visible transition");
                require(session.info().completed_steps == iteration + 1 && !session.info().poisoned,
                        "FastGS phase scratch transition poisoned the session");
            }
            session.read_parameter(0, final_means);
            require(original_means == final_means, "Zero-rate scratch transitions changed model means");
            std::puts("FastGS phase scratch: repeated empty/visible worksets preserve output, parameters and session health");
        }
        {
            // Many visible rows in one tile make compact helper storage larger
            // than the temporary sort allocation, exercising dedicated fallback.
            Fixture source;
            constexpr size_t count = 10007;
            constexpr std::array<size_t, 5> widths{3, 4, 3, 3, 1};
            std::array<std::vector<float>, 5> values;
            for (size_t family = 0; family < widths.size(); ++family) {
                values[family].resize(count * widths[family]);
                for (size_t row = 0; row < count; ++row)
                    std::copy_n(source.values[family].data() + (row % n) * widths[family], widths[family],
                                values[family].data() + row * widths[family]);
            }
            for (size_t row = 0; row < count; ++row) {
                values[0][row * 3] = values[0][row * 3 + 1] = 0;
                values[0][row * 3 + 2] = 3;
            }
            const LfsHipTrainConfig config{sizeof(config), 0, count, 64 * 1024 * 1024};
            const LfsHipTrainModel model{values[0].data(), values[1].data(), values[2].data(), values[3].data(), values[4].data()};
            Session session(module, config, model);
            session.set_rasterizer(1);
            auto frame = source.frame();
            std::array<float, 3> target{}, image{};
            frame.width = frame.height = 1;
            frame.fx = frame.fy = 1;
            frame.cx = frame.cy = 0.5f;
            frame.target_chw = target.data();
            session.set_frame(frame);
            session.render(image);
            const auto before = session.info();
            require(before.intersection_count == count, "Dedicated helper fixture must put every visible row in one tile");
            session.set_budget(before.allocated_bytes);
            bool exhausted = false;
            try {
                session.step({});
            } catch (const TrainingError& error) {
                exhausted = error.status == LFS_HIP_TRAIN_RESOURCE_EXHAUSTED;
            }
            require(exhausted && session.info().completed_steps == 0 && !session.info().poisoned,
                    "Dedicated compact helper fallback failed to enforce budget before Adam");
            for (size_t family = 0; family < widths.size(); ++family) {
                std::vector<float> actual(values[family].size());
                session.read_parameter(static_cast<uint32_t>(family), actual);
                require(actual == values[family], "Helper fallback budget failure mutated a parameter family");
            }
            session.set_budget(config.allocation_budget_bytes);
            session.step({});
            require(session.info().allocated_bytes >= before.allocated_bytes + count * 16 * sizeof(float),
                    "Single-tile workset did not exercise dedicated helper fallback");
            session.set_budget(session.info().allocated_bytes);
            session.step({});
            require(session.info().completed_steps == 2 && !session.info().poisoned,
                    "Dedicated helper fallback failed to reuse storage after budget recovery");
            std::puts("FastGS dedicated helper fallback: budget rejection preserves all parameters; retry and exact-budget reuse succeed");
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
