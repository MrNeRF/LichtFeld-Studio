/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Reproduces the trainer↔metrics arena/lock contention that the bounded
// begin_frame (ScopedBeginFrameTimeout) must keep deadlock-free:
//   trainer:  holds the arena frame, then wants the exclusive render lock
//   metrics:  holds the shared render lock, then wants the arena frame
// With a wait-forever metrics begin_frame this cycles; bounded, the metrics
// acquisition bails and the cycle breaks.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cuda_runtime.h>
#include <future>
#include <gtest/gtest.h>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <thread>
#include <vector>

#include "core/cuda/memory_arena.hpp"
#include "core/logger.hpp"
#include "visualizer/rendering/stale_frame_guard.hpp"
#include "visualizer/rendering/vksplat_shared_scratch_install.hpp"

using lfs::core::RasterizerMemoryArena;

namespace {
    // Runs fn on a thread and fails (rather than hanging the suite) if it does
    // not finish within the budget — a deadlock manifests as the timeout.
    template <typename Fn>
    bool completes_within(std::chrono::milliseconds budget, Fn&& fn) {
        std::packaged_task<void()> task(std::forward<Fn>(fn));
        auto future = task.get_future();
        std::thread runner(std::move(task));
        const bool ok = future.wait_for(budget) == std::future_status::ready;
        if (ok) {
            runner.join();
        } else {
            runner.detach(); // leak the wedged thread; the test already failed
        }
        return ok;
    }

    // Keeps a stream busy until the flag is set, like kernels still running on
    // the GPU after the CPU side has moved on.
    void CUDART_CB hold_stream_until_released(void* flag) {
        const auto* const released = static_cast<const std::atomic<bool>*>(flag);
        while (!released->load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    // Counts arena tenants (CPU side and the GPU tail of training frames) and
    // remembers the most that were ever inside at once.
    struct TenantCounter {
        std::atomic<int> inside{0};
        std::atomic<int> most{0};

        void enter() {
            const int now = inside.fetch_add(1, std::memory_order_acq_rel) + 1;
            int seen = most.load(std::memory_order_relaxed);
            while (now > seen && !most.compare_exchange_weak(seen, now, std::memory_order_relaxed)) {
            }
        }
        void leave() { inside.fetch_sub(1, std::memory_order_acq_rel); }
    };

    // Stand in for a training step's kernels: the step enters the arena when its
    // GPU work starts and leaves 2 ms later, after the CPU released the frame.
    void CUDART_CB start_training_step_on_gpu(void* tenants) {
        static_cast<TenantCounter*>(tenants)->enter();
    }

    void CUDART_CB finish_training_step_on_gpu(void* tenants) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        static_cast<TenantCounter*>(tenants)->leave();
    }

    // Training loop as the trainer runs it: a blocking begin per step, GPU work
    // queued on its stream, the frame released before that work finishes.
    class TrainingLoop {
    public:
        TrainingLoop(RasterizerMemoryArena& arena, TenantCounter& tenants)
            : arena_(arena),
              tenants_(tenants),
              thread_([this] { run(); }) {}

        ~TrainingLoop() { stop(); }

        void stop() {
            stop_.store(true, std::memory_order_release);
            if (thread_.joinable()) {
                thread_.join();
            }
        }

        [[nodiscard]] std::uint64_t steps() const { return steps_.load(std::memory_order_acquire); }

    private:
        void run() {
            EXPECT_EQ(cudaSetDevice(0), cudaSuccess);
            cudaStream_t stream = nullptr;
            ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
            while (!stop_.load(std::memory_order_acquire)) {
                const auto frame = arena_.begin_frame(stream, false);
                const auto step = steps_.fetch_add(1, std::memory_order_acq_rel) + 1;
                EXPECT_EQ(cudaLaunchHostFunc(stream, start_training_step_on_gpu, &tenants_), cudaSuccess);
                std::this_thread::sleep_for(std::chrono::microseconds(500));
                EXPECT_EQ(cudaLaunchHostFunc(stream, finish_training_step_on_gpu, &tenants_), cudaSuccess);
                arena_.end_frame(frame, stream, false);
                if (step % 4 == 0) {
                    EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
                }
            }
            EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
            EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
        }

        RasterizerMemoryArena& arena_;
        TenantCounter& tenants_;
        std::atomic<bool> stop_{false};
        std::atomic<std::uint64_t> steps_{0};
        std::thread thread_;
    };
} // namespace

class ArenaMetricsContentionTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    }
};

TEST_F(ArenaMetricsContentionTest, BoundedBeginFrameBailsWhileFrameHeld) {
    RasterizerMemoryArena arena;
    const uint64_t held = arena.begin_frame(nullptr, /*from_rendering=*/false);

    // A second thread with a bounded timeout must throw (not hang) because the
    // single arena frame is held.
    const bool finished = completes_within(std::chrono::milliseconds(2000), [&] {
        const RasterizerMemoryArena::ScopedBeginFrameTimeout timeout(50);
        bool threw = false;
        try {
            arena.begin_frame(nullptr, false);
        } catch (const std::exception&) {
            threw = true;
        }
        EXPECT_TRUE(threw) << "bounded begin_frame should bail while the frame is held";
    });
    EXPECT_TRUE(finished) << "bounded begin_frame hung instead of timing out";

    arena.end_frame(held, nullptr, false);
}

TEST_F(ArenaMetricsContentionTest, TrainerMetricsOppositeOrderNoDeadlock) {
    RasterizerMemoryArena arena;
    std::shared_mutex render_mutex;
    std::atomic<bool> trainer_has_frame{false};
    std::atomic<bool> metrics_has_render_lock{false};
    std::atomic<bool> metrics_timed_out{false};
    std::atomic<bool> trainer_acquired_render_lock{false};

    const bool finished = completes_within(std::chrono::milliseconds(2000), [&] {
        // Reproduce one deterministic lock inversion:
        // trainer: arena frame -> exclusive render lock
        // metrics: shared render lock -> bounded arena frame
        std::thread trainer([&] {
            cudaSetDevice(0);
            const uint64_t frame = arena.begin_frame(nullptr, false);
            trainer_has_frame.store(true, std::memory_order_release);
            while (!metrics_has_render_lock.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            {
                std::unique_lock<std::shared_mutex> exclusive(render_mutex);
                trainer_acquired_render_lock.store(true, std::memory_order_release);
            }
            arena.end_frame(frame, nullptr, false);
        });

        std::thread metrics([&] {
            cudaSetDevice(0);
            while (!trainer_has_frame.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            std::shared_lock<std::shared_mutex> shared(render_mutex);
            metrics_has_render_lock.store(true, std::memory_order_release);
            const RasterizerMemoryArena::ScopedBeginFrameTimeout timeout(20);
            try {
                const uint64_t frame = arena.begin_frame(nullptr, false);
                arena.end_frame(frame, nullptr, false);
            } catch (const std::exception&) {
                metrics_timed_out.store(true, std::memory_order_release);
            }
        });

        trainer.join();
        metrics.join();
    });

    EXPECT_TRUE(finished) << "trainer/metrics contention deadlocked";
    EXPECT_TRUE(metrics_timed_out.load());
    EXPECT_TRUE(trainer_acquired_render_lock.load());
    const auto next = arena.try_begin_frame(nullptr, false);
    ASSERT_TRUE(next.has_value());
    arena.end_frame(*next, nullptr, false);
}

TEST_F(ArenaMetricsContentionTest, RenderHandoffReservesNextIdleWindowAfterLockUnwind) {
    RasterizerMemoryArena arena;
    std::shared_mutex render_mutex;
    const auto held = arena.begin_frame(nullptr, false);

    const auto token = arena.request_render_handoff();
    ASSERT_NE(token, 0u);
    std::atomic<bool> trainer_waiting_for_exclusive{false};
    std::atomic<bool> trainer_finished{false};
    std::thread trainer;
    {
        std::shared_lock viewer_lock(render_mutex);
        trainer = std::thread([&] {
            EXPECT_EQ(cudaSetDevice(0), cudaSuccess);
            trainer_waiting_for_exclusive.store(true, std::memory_order_release);
            std::unique_lock trainer_lock(render_mutex);
            arena.end_frame(held, nullptr, false);
            trainer_finished.store(true, std::memory_order_release);
        });
        while (!trainer_waiting_for_exclusive.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        // Reproduce the production inversion window: the trainer owns the arena
        // while waiting for the renderer's shared model lock. The bounded arena
        // attempt must return so this scope can release that lock.
        EXPECT_FALSE(arena.try_begin_render_frame_for(15, token));
    }
    trainer.join();
    ASSERT_TRUE(trainer_finished.load(std::memory_order_acquire));

    // The trainer cannot immediately win another iteration while the renderer's
    // bounded request is live. The renderer consumes that exact reservation.
    EXPECT_FALSE(arena.try_begin_frame(nullptr, false));
    const auto rendered = arena.try_begin_render_frame_for(50, token);
    ASSERT_TRUE(rendered.has_value());
    arena.end_frame(*rendered, nullptr, true);
    EXPECT_FALSE(arena.has_render_handoff(token));

    const auto next_training = arena.try_begin_frame(nullptr, false);
    ASSERT_TRUE(next_training.has_value());
    arena.end_frame(*next_training, nullptr, false);
}

// Catches a render begin that host-waits for the previous training frame's GPU
// work (a device-wide or event sync on the UI thread) instead of declining and
// keeping its reservation.
TEST_F(ArenaMetricsContentionTest, RenderBeginDoesNotWaitForTrainingWorkStillOnTheGpu) {
    RasterizerMemoryArena arena;
    cudaStream_t training_stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&training_stream, cudaStreamNonBlocking), cudaSuccess);
    std::atomic<bool> released{false};
    const auto training = arena.begin_frame(training_stream, false);
    ASSERT_EQ(cudaLaunchHostFunc(training_stream, hold_stream_until_released, &released), cudaSuccess);
    arena.end_frame(training, training_stream, false);

    const auto token = arena.request_render_handoff();
    auto attempt = std::async(std::launch::async, [&] {
        EXPECT_EQ(cudaSetDevice(0), cudaSuccess);
        return arena.try_begin_render_frame_for(15, token);
    });
    const bool returned = attempt.wait_for(std::chrono::milliseconds(500)) == std::future_status::ready;
    const bool declined = returned && !attempt.get().has_value();
    EXPECT_TRUE(returned) << "render begin blocked on training work still running on the GPU";
    EXPECT_TRUE(declined) << "render claimed the arena before the training frame finished on the GPU";
    EXPECT_FALSE(arena.try_begin_frame(training_stream, false))
        << "the render reservation must keep the next training frame out";

    released.store(true, std::memory_order_release);
    if (!returned) {
        if (const auto blocked = attempt.get()) {
            arena.end_frame(*blocked, nullptr, true);
        }
    }
    ASSERT_EQ(cudaStreamSynchronize(training_stream), cudaSuccess);
    if (declined) {
        const auto frame = arena.try_begin_render_frame_for(15, arena.request_render_handoff(token));
        ASSERT_TRUE(frame.has_value());
        arena.end_frame(*frame, nullptr, true);
    }
    ASSERT_EQ(cudaStreamDestroy(training_stream), cudaSuccess);
}

// Catches a render begin that drops the frame although the training step
// holding the arena finishes on the GPU within the render's wait budget.
TEST_F(ArenaMetricsContentionTest, RenderBeginWaitsForATrainingStepAboutToFinish) {
    RasterizerMemoryArena arena;
    cudaStream_t training_stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&training_stream, cudaStreamNonBlocking), cudaSuccess);
    TenantCounter tenants;
    const auto training = arena.begin_frame(training_stream, false);
    ASSERT_EQ(cudaLaunchHostFunc(training_stream, start_training_step_on_gpu, &tenants), cudaSuccess);
    ASSERT_EQ(cudaLaunchHostFunc(training_stream, finish_training_step_on_gpu, &tenants), cudaSuccess);
    arena.end_frame(training, training_stream, false);

    const auto frame = arena.try_begin_render_frame_for(50);
    ASSERT_TRUE(frame.has_value()) << "render declined a training step that finished within its budget";
    EXPECT_EQ(tenants.inside.load(), 0) << "render began while the training step still ran on the GPU";
    arena.end_frame(*frame, nullptr, true);
    ASSERT_EQ(cudaStreamSynchronize(training_stream), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(training_stream), cudaSuccess);
}

// Catches a render begin that drains the whole device (for example an optimizer
// step queued after the rasterizer frame) instead of only the arena's last frame.
TEST_F(ArenaMetricsContentionTest, RenderBeginWaitsOnlyForTheArenaFrame) {
    RasterizerMemoryArena arena;
    cudaStream_t training_stream = nullptr;
    cudaStream_t other_stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&training_stream, cudaStreamNonBlocking), cudaSuccess);
    ASSERT_EQ(cudaStreamCreateWithFlags(&other_stream, cudaStreamNonBlocking), cudaSuccess);
    const auto training = arena.begin_frame(training_stream, false);
    arena.end_frame(training, training_stream, false);
    ASSERT_EQ(cudaStreamSynchronize(training_stream), cudaSuccess);
    std::atomic<bool> released{false};
    ASSERT_EQ(cudaLaunchHostFunc(other_stream, hold_stream_until_released, &released), cudaSuccess);

    auto attempt = std::async(std::launch::async, [&] {
        EXPECT_EQ(cudaSetDevice(0), cudaSuccess);
        return arena.try_begin_render_frame_for(15);
    });
    const bool returned = attempt.wait_for(std::chrono::milliseconds(500)) == std::future_status::ready;
    released.store(true, std::memory_order_release);
    const auto frame = attempt.get();
    EXPECT_TRUE(returned) << "render begin waited for CUDA work that never touched the arena";
    ASSERT_TRUE(frame.has_value());
    arena.end_frame(*frame, nullptr, true);
    ASSERT_EQ(cudaStreamSynchronize(other_stream), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(other_stream), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(training_stream), cudaSuccess);
}

// Catches a readiness poll that reports ready while training owns the arena or
// its last frame still runs, or that ignores another live reservation.
TEST_F(ArenaMetricsContentionTest, RenderFrameReadyOnlyWhenRenderCanBeginWithoutWaiting) {
    RasterizerMemoryArena arena;
    cudaStream_t training_stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&training_stream, cudaStreamNonBlocking), cudaSuccess);
    std::atomic<bool> released{false};
    const auto training = arena.begin_frame(training_stream, false);
    EXPECT_FALSE(arena.render_frame_ready());
    ASSERT_EQ(cudaLaunchHostFunc(training_stream, hold_stream_until_released, &released), cudaSuccess);
    arena.end_frame(training, training_stream, false);

    const auto token = arena.request_render_handoff();
    EXPECT_FALSE(arena.render_frame_ready(token));
    released.store(true, std::memory_order_release);
    ASSERT_EQ(cudaStreamSynchronize(training_stream), cudaSuccess);
    EXPECT_TRUE(arena.render_frame_ready(token));
    EXPECT_FALSE(arena.render_frame_ready());
    arena.cancel_render_handoff(token);
    EXPECT_TRUE(arena.render_frame_ready());
    ASSERT_EQ(cudaStreamDestroy(training_stream), cudaSuccess);
}

// Catches a reservation that ignores the training frames it was given (training
// stops while the camera moves), hands out more than that, or refills them on
// renewal so the viewer never gets its turn.
TEST_F(ArenaMetricsContentionTest, ReservationLetsExactlyItsTrainingFramesBeginFirst) {
    RasterizerMemoryArena arena;
    const auto token = arena.request_render_handoff(0, 1);
    ASSERT_NE(token, 0u);
    const auto step = arena.try_begin_frame(nullptr, false);
    ASSERT_TRUE(step.has_value()) << "the reservation kept out the training step it granted";
    arena.end_frame(*step, nullptr, false);
    EXPECT_FALSE(arena.try_begin_frame(nullptr, false)) << "training got a second step before the viewer's turn";
    EXPECT_EQ(arena.request_render_handoff(token, 1), token);
    EXPECT_FALSE(arena.try_begin_frame(nullptr, false)) << "renewing the reservation granted training another step";

    const auto render = arena.try_begin_render_frame_for(1, token);
    ASSERT_TRUE(render.has_value());
    arena.end_frame(*render, nullptr, true);
    const auto next = arena.try_begin_frame(nullptr, false);
    ASSERT_TRUE(next.has_value()) << "training stayed blocked after the viewer took its turn";
    arena.end_frame(*next, nullptr, false);
}

// Catches a navigation release that starves training, lets it run past its one
// step, or a kept window that still lets training in.
TEST_F(ArenaMetricsContentionTest, NavigationReleaseGivesTrainingOneStepThenTheViewer) {
    RasterizerMemoryArena arena;
    RasterizerMemoryArena::RenderHandoffToken token = 0;
    auto frame = arena.try_begin_render_frame_for(1, token);
    ASSERT_TRUE(frame.has_value());
    lfs::vis::releaseViewerArenaFrame(arena, *frame, &token, std::optional<std::uint32_t>(1));
    ASSERT_NE(token, 0u);
    const auto step = arena.try_begin_frame(nullptr, false);
    ASSERT_TRUE(step.has_value()) << "training got no step between two navigation frames";
    arena.end_frame(*step, nullptr, false);
    EXPECT_FALSE(arena.try_begin_frame(nullptr, false)) << "training ran past its single step";

    frame = arena.try_begin_render_frame_for(1, token);
    ASSERT_TRUE(frame.has_value());
    token = 0;
    lfs::vis::releaseViewerArenaFrame(arena, *frame, &token, std::optional<std::uint32_t>(0));
    EXPECT_FALSE(arena.try_begin_frame(nullptr, false)) << "a kept window let a training step in";

    frame = arena.try_begin_render_frame_for(1, token);
    ASSERT_TRUE(frame.has_value());
    token = 0;
    lfs::vis::releaseViewerArenaFrame(arena, *frame, &token, std::nullopt);
    const auto free_step = arena.try_begin_frame(nullptr, false);
    ASSERT_TRUE(free_step.has_value()) << "an ordinary release kept training out";
    arena.end_frame(*free_step, nullptr, false);
}

// Races a training loop against a navigating viewer on one arena. Catches
// overlapping tenants (including a render that starts while the last training
// step still runs on the GPU), a starved trainer or viewer, more than one
// training step between two viewer frames, a deadlock, and training that stays
// blocked after navigation stops.
TEST_F(ArenaMetricsContentionTest, NavigationTurnTakingUnderLoad) {
    using Clock = std::chrono::steady_clock;
    using namespace std::chrono_literals;
    RasterizerMemoryArena arena;
    TenantCounter tenants;
    std::uint64_t viewer_frames = 0;
    std::uint64_t most_steps_between_frames = 0;
    std::uint64_t steps_while_navigating = 0;
    std::uint64_t steps_after_navigation = 0;

    const bool finished = completes_within(20s, [&] {
        TrainingLoop training(arena, tenants);
        EXPECT_EQ(cudaSetDevice(0), cudaSuccess);
        RasterizerMemoryArena::RenderHandoffToken token = 0;
        std::uint64_t steps_at_last_frame = training.steps();
        auto last_release = Clock::now();
        const auto first_steps = training.steps();
        const auto navigation_end = Clock::now() + 1500ms;
        while (Clock::now() < navigation_end) {
            arena.set_rendering_active(true);
            const auto frame = arena.try_begin_render_frame_for(1, token);
            arena.set_rendering_active(false);
            if (!frame) {
                token = arena.request_render_handoff(token);
            } else {
                token = 0;
                tenants.enter();
                const std::uint64_t steps = training.steps();
                // A viewer paused past the lease legitimately lets training run on.
                if (Clock::now() - last_release < 50ms) {
                    most_steps_between_frames = std::max(most_steps_between_frames, steps - steps_at_last_frame);
                }
                steps_at_last_frame = steps;
                ++viewer_frames;
                std::this_thread::sleep_for(1ms);
                tenants.leave();
                lfs::vis::releaseViewerArenaFrame(arena, *frame, &token,
                                                  std::optional(lfs::vis::kTrainingFramesPerNavigationRender));
                last_release = Clock::now();
            }
            std::this_thread::sleep_for(3ms);
        }
        steps_while_navigating = training.steps() - first_steps;
        const auto steps_at_stop = training.steps();
        std::this_thread::sleep_for(RasterizerMemoryArena::kRenderHandoffLeaseMs * 1ms + 150ms);
        steps_after_navigation = training.steps() - steps_at_stop;
        training.stop();
    });

    ASSERT_TRUE(finished) << "viewer and training deadlocked on the arena";
    EXPECT_EQ(tenants.most.load(), 1) << "viewer and training used the arena at the same time";
    EXPECT_GE(viewer_frames, 60u) << "the viewer was starved while navigating";
    EXPECT_GE(steps_while_navigating, 60u) << "training was starved while the camera moved";
    EXPECT_LE(most_steps_between_frames, 1u) << "training ran more than one step between two viewer frames";
    EXPECT_GE(steps_after_navigation, 10u) << "training stayed blocked after navigation stopped";
}

// Races a training loop against a parked idle refresh that polls readiness.
// Catches a readiness report after which the render still declines, which would
// let the idle preview spin on retries while training keeps the arena.
TEST_F(ArenaMetricsContentionTest, ReadyParkedRefreshAlwaysBeginsUnderLoad) {
    using Clock = std::chrono::steady_clock;
    using namespace std::chrono_literals;
    RasterizerMemoryArena arena;
    TenantCounter tenants;
    std::uint64_t renders = 0;
    std::uint64_t declined_after_ready = 0;

    const bool finished = completes_within(20s, [&] {
        TrainingLoop training(arena, tenants);
        EXPECT_EQ(cudaSetDevice(0), cudaSuccess);
        const auto end = Clock::now() + 1s;
        while (Clock::now() < end) {
            auto token = arena.request_render_handoff(0);
            while (token != 0 && !arena.render_frame_ready(token)) {
                std::this_thread::sleep_for(1ms);
                token = arena.request_render_handoff(token);
            }
            if (token == 0) {
                continue;
            }
            arena.set_rendering_active(true);
            const auto frame = arena.try_begin_render_frame_for(1, token);
            arena.set_rendering_active(false);
            if (!frame) {
                ++declined_after_ready;
                arena.cancel_render_handoff(token);
                continue;
            }
            tenants.enter();
            ++renders;
            std::this_thread::sleep_for(1ms);
            tenants.leave();
            arena.end_frame(*frame, nullptr, true);
            std::this_thread::sleep_for(20ms);
        }
        training.stop();
    });

    ASSERT_TRUE(finished) << "parked refresh and training deadlocked on the arena";
    EXPECT_EQ(tenants.most.load(), 1) << "refresh and training used the arena at the same time";
    EXPECT_EQ(declined_after_ready, 0u) << "a render declined right after the arena reported it ready";
    EXPECT_GE(renders, 20u) << "the idle refresh was starved";
}

TEST_F(ArenaMetricsContentionTest, ArenaContentionNeverDropsValidCachedFrame) {
    lfs::vis::StaleFrameGuard guard;
    for (std::uint32_t attempt = 0;
         attempt < lfs::vis::StaleFrameGuard::kMaxCachedDeferrals * 3;
         ++attempt) {
        EXPECT_FALSE(guard.onDeferral(
            lfs::vis::StaleFrameGuard::DeferralKind::ArenaContention));
        EXPECT_TRUE(guard.canUseCachedFrame());
        EXPECT_FALSE(guard.takeRecoveryRequest());
    }

    // Non-contention failures keep the existing bounded recovery behavior.
    for (std::uint32_t attempt = 1;
         attempt <= lfs::vis::StaleFrameGuard::kMaxCachedDeferrals;
         ++attempt) {
        EXPECT_EQ(guard.onDeferral(),
                  attempt == lfs::vis::StaleFrameGuard::kMaxCachedDeferrals);
    }
    EXPECT_FALSE(guard.canUseCachedFrame());
    EXPECT_TRUE(guard.takeRecoveryRequest());
}

TEST_F(ArenaMetricsContentionTest, AbandonedRenderHandoffExpiresAndWakesTrainer) {
    RasterizerMemoryArena arena;
    const auto token = arena.request_render_handoff();
    ASSERT_NE(token, 0u);

    std::atomic<bool> trainer_finished{false};
    std::promise<void> trainer_done;
    auto trainer_done_future = trainer_done.get_future();
    const auto started = std::chrono::steady_clock::now();
    std::thread trainer([&] {
        EXPECT_EQ(cudaSetDevice(0), cudaSuccess);
        const auto frame = arena.begin_frame(nullptr, false);
        arena.end_frame(frame, nullptr, false);
        trainer_finished.store(true, std::memory_order_release);
        trainer_done.set_value();
    });
    const bool woke_on_expiry =
        trainer_done_future.wait_for(std::chrono::seconds(2)) == std::future_status::ready;
    if (!woke_on_expiry) {
        arena.cancel_render_handoff(token);
    }
    trainer.join();

    EXPECT_TRUE(woke_on_expiry) << "blocked trainer did not wake when the render lease expired";
    EXPECT_TRUE(trainer_finished.load(std::memory_order_acquire));
    const auto elapsed = std::chrono::steady_clock::now() - started;
    EXPECT_GE(elapsed, std::chrono::milliseconds(RasterizerMemoryArena::kRenderHandoffLeaseMs - 10));
    EXPECT_LT(elapsed, std::chrono::seconds(2));
    EXPECT_FALSE(arena.has_render_handoff(token));
}

TEST_F(ArenaMetricsContentionTest, AbandonedHandoffAlsoWakesTokenlessRenderer) {
    RasterizerMemoryArena arena;
    const auto token = arena.request_render_handoff();
    ASSERT_NE(token, 0u);

    std::atomic<bool> renderer_finished{false};
    std::promise<void> renderer_done;
    auto renderer_done_future = renderer_done.get_future();
    const auto started = std::chrono::steady_clock::now();
    std::thread renderer([&] {
        EXPECT_EQ(cudaSetDevice(0), cudaSuccess);
        const auto frame = arena.begin_frame(nullptr, true);
        arena.end_frame(frame, nullptr, true);
        renderer_finished.store(true, std::memory_order_release);
        renderer_done.set_value();
    });
    const bool woke_on_expiry =
        renderer_done_future.wait_for(std::chrono::seconds(2)) == std::future_status::ready;
    if (!woke_on_expiry) {
        arena.cancel_render_handoff(token);
    }
    renderer.join();

    EXPECT_TRUE(woke_on_expiry) << "tokenless renderer did not wake when the lease expired";
    EXPECT_TRUE(renderer_finished.load(std::memory_order_acquire));
    const auto elapsed = std::chrono::steady_clock::now() - started;
    EXPECT_GE(elapsed, std::chrono::milliseconds(RasterizerMemoryArena::kRenderHandoffLeaseMs - 10));
    EXPECT_LT(elapsed, std::chrono::seconds(2));
}

TEST_F(ArenaMetricsContentionTest, OldRenderTokenCannotCancelNewReservation) {
    RasterizerMemoryArena arena;
    const auto old_token = arena.request_render_handoff();
    ASSERT_NE(old_token, 0u);
    arena.cancel_render_handoff(old_token);

    const auto new_token = arena.request_render_handoff();
    ASSERT_NE(new_token, 0u);
    ASSERT_NE(new_token, old_token);
    arena.cancel_render_handoff(old_token);
    EXPECT_TRUE(arena.has_render_handoff(new_token));
    EXPECT_FALSE(arena.try_begin_frame(nullptr, false));

    arena.cancel_render_handoff(new_token);
    const auto training = arena.try_begin_frame(nullptr, false);
    ASSERT_TRUE(training.has_value());
    arena.end_frame(*training, nullptr, false);
}

TEST_F(ArenaMetricsContentionTest, ExternalGrowWaitsForPendingRender) {
    constexpr size_t MiB = 1024 * 1024;
    constexpr size_t physical_bytes = 8 * MiB;
    constexpr size_t initial_committed_bytes = 1 * MiB;

    void* device_ptr = nullptr;
    ASSERT_EQ(cudaMalloc(&device_ptr, physical_bytes), cudaSuccess);
    auto owner = std::shared_ptr<void>(device_ptr, [](void* ptr) {
        EXPECT_EQ(cudaFree(ptr), cudaSuccess);
    });

    RasterizerMemoryArena::Config config;
    config.max_physical = physical_bytes;
    config.enable_vmm = false;
    RasterizerMemoryArena arena(config);

    std::atomic<bool> render_pending{false};
    std::atomic<bool> callback_while_pending{false};
    std::atomic<int> grow_calls{0};
    RasterizerMemoryArena::ExternalBacking backing{
        .device_ptr = device_ptr,
        .size = initial_committed_bytes,
        .device = 0,
        .owner = owner,
        .label = "test.external",
        .grow = [&](const size_t need) -> size_t {
            grow_calls.fetch_add(1, std::memory_order_relaxed);
            if (render_pending.load(std::memory_order_acquire)) {
                callback_while_pending.store(true, std::memory_order_release);
            }
            return need <= physical_bytes ? physical_bytes : 0;
        },
    };
    ASSERT_TRUE(arena.install_external_backing(std::move(backing)));

    const uint64_t frame = arena.begin_frame(nullptr, /*from_rendering=*/false);
    auto allocate = arena.get_allocator(frame);

    // Reproduce the queued-render window while the trainer owns the only active
    // frame. The render's bounded acquisition eventually clears the pending bit;
    // external growth must retry until then and only invoke the mapping callback
    // after the frame gate is safe.
    render_pending.store(true, std::memory_order_release);
    arena.set_rendering_active(true);
    std::atomic<bool> allocation_started{false};
    std::thread clear_pending([&] {
        while (!allocation_started.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        render_pending.store(false, std::memory_order_release);
        arena.set_rendering_active(false);
    });

    allocation_started.store(true, std::memory_order_release);
    char* const allocation = allocate(2 * MiB);
    clear_pending.join();

    EXPECT_NE(allocation, nullptr);
    EXPECT_EQ(allocation, device_ptr);
    EXPECT_FALSE(callback_while_pending.load(std::memory_order_acquire));
    EXPECT_EQ(grow_calls.load(std::memory_order_relaxed), 1);
    EXPECT_EQ(arena.get_statistics().capacity, physical_bytes);

    arena.end_frame(frame, nullptr, /*from_rendering=*/false);
}

TEST_F(ArenaMetricsContentionTest, ExternalGrowFailureRetainsCommittedCapacity) {
    constexpr size_t MiB = 1024 * 1024;
    constexpr size_t physical_bytes = 4 * MiB;
    constexpr size_t initial_committed_bytes = 1 * MiB;

    void* device_ptr = nullptr;
    ASSERT_EQ(cudaMalloc(&device_ptr, physical_bytes), cudaSuccess);
    auto owner = std::shared_ptr<void>(device_ptr, [](void* ptr) {
        EXPECT_EQ(cudaFree(ptr), cudaSuccess);
    });

    RasterizerMemoryArena::Config config;
    config.max_physical = physical_bytes;
    config.enable_vmm = false;
    config.granularity = MiB;
    RasterizerMemoryArena arena(config);
    std::atomic<int> grow_calls{0};
    std::atomic<size_t> requested_bytes{0};
    RasterizerMemoryArena::ExternalBacking backing{
        .device_ptr = device_ptr,
        .size = initial_committed_bytes,
        .device = 0,
        .owner = owner,
        .label = "test.external.failure",
        .grow = [&](const size_t requested) {
            grow_calls.fetch_add(1, std::memory_order_relaxed);
            requested_bytes.store(requested, std::memory_order_release);
            return size_t{0};
        },
    };
    ASSERT_TRUE(arena.install_external_backing(std::move(backing)));

    std::mutex log_mutex;
    std::vector<std::string> error_logs;
    const auto log_handler = lfs::core::Logger::get().add_log_handler(
        [&](const lfs::core::LogLevel level, const lfs::core::SourceSite&, const std::string_view message) {
            if (level == lfs::core::LogLevel::Error) {
                std::scoped_lock lock(log_mutex);
                error_logs.emplace_back(message);
            }
        });

    const uint64_t frame = arena.begin_frame(nullptr, false);
    auto allocate = arena.get_allocator(frame, "arena.failure.identity");
    EXPECT_EQ(allocate(2 * MiB), nullptr);
    EXPECT_EQ(grow_calls.load(std::memory_order_relaxed), 1);
    EXPECT_EQ(requested_bytes.load(std::memory_order_acquire), 2 * MiB);
    EXPECT_EQ(arena.get_statistics().capacity, initial_committed_bytes);
    arena.end_frame(frame, nullptr, false);
    lfs::core::Logger::get().remove_log_handler(log_handler);

    std::string joined_logs;
    for (const auto& message : error_logs) {
        joined_logs += message;
        joined_logs += '\n';
    }
    EXPECT_NE(joined_logs.find("test.external.failure"), std::string::npos);
    EXPECT_NE(joined_logs.find("arena.failure.identity"), std::string::npos);
    EXPECT_NE(joined_logs.find("capacity=1 MiB"), std::string::npos);
    EXPECT_NE(joined_logs.find("Arena VRAM failure snapshot"), std::string::npos);
    EXPECT_NE(joined_logs.find("cuda_free="), std::string::npos);
}

TEST_F(ArenaMetricsContentionTest, ExternalGrowCommitsExactAlignedNeed) {
    constexpr size_t MiB = 1024 * 1024;
    constexpr size_t initial_committed_bytes = 1 * MiB;
    constexpr size_t aligned_need = 2 * MiB;

    void* device_ptr = nullptr;
    ASSERT_EQ(cudaMalloc(&device_ptr, 4 * MiB), cudaSuccess);
    auto owner = std::shared_ptr<void>(device_ptr, [](void* ptr) {
        EXPECT_EQ(cudaFree(ptr), cudaSuccess);
    });

    RasterizerMemoryArena::Config config;
    config.max_physical = 4 * MiB;
    config.enable_vmm = false;
    config.granularity = MiB;
    RasterizerMemoryArena arena(config);
    RasterizerMemoryArena::ExternalBacking backing{
        .device_ptr = device_ptr,
        .size = initial_committed_bytes,
        .device = 0,
        .owner = owner,
        .label = "test.external.exact_policy",
    };
    ASSERT_TRUE(arena.install_external_backing(std::move(backing)));

    std::vector<size_t> requests;
    const bool grew = arena.grow_external_backing(
        device_ptr, aligned_need,
        [&](const size_t requested) {
            requests.push_back(requested);
            return requested <= aligned_need;
        });
    ASSERT_TRUE(grew);
    ASSERT_EQ(requests.size(), 1u);
    EXPECT_EQ(requests[0], aligned_need);
    EXPECT_EQ(arena.get_statistics().capacity, aligned_need);
}

TEST_F(ArenaMetricsContentionTest, DetachedViewerBackingCanBeReinstalledAndGrown) {
    constexpr size_t MiB = 1024 * 1024;
    void* device_ptr = nullptr;
    ASSERT_EQ(cudaMalloc(&device_ptr, 2 * MiB), cudaSuccess);
    auto owner = std::shared_ptr<void>(device_ptr, [](void* ptr) {
        EXPECT_EQ(cudaFree(ptr), cudaSuccess);
    });
    RasterizerMemoryArena arena;
    const RasterizerMemoryArena::ExternalBacking backing{
        .device_ptr = device_ptr,
        .size = MiB,
        .device = 0,
        .owner = owner,
        .label = "test.external.pause_resume",
    };
    ASSERT_TRUE(arena.install_external_backing(backing));
    ASSERT_TRUE(arena.using_external_backing(device_ptr));
    EXPECT_FALSE(arena.using_external_backing(static_cast<char*>(device_ptr) + 1));

    // Trainer B3 detaches independently of the viewer's retained block/import.
    arena.clear_external_backing();
    EXPECT_FALSE(arena.using_external_backing(device_ptr));
    bool committed = false;
    const auto commit = [&](size_t) {
        committed = true;
        return true;
    };
    using GrowFailure = RasterizerMemoryArena::ExternalGrowFailure;
    GrowFailure failure = GrowFailure::None;
    EXPECT_FALSE(arena.grow_external_backing(device_ptr, 2 * MiB, commit, 0, &failure));
    EXPECT_EQ(failure, GrowFailure::BackingMissing);
    EXPECT_FALSE(committed);

    // Reinstallation must defer while resumed training owns an arena frame.
    const auto held = arena.begin_frame(nullptr, false);
    EXPECT_FALSE(arena.try_install_external_backing(backing));
    arena.end_frame(held, nullptr, false);
    ASSERT_TRUE(arena.try_install_external_backing(backing));
    EXPECT_TRUE(arena.using_external_backing(device_ptr));
    EXPECT_FALSE(arena.grow_external_backing(device_ptr, 2 * MiB, [](size_t) { return false; }, 0, &failure));
    EXPECT_EQ(failure, GrowFailure::CommitFailure);
    EXPECT_EQ(arena.get_statistics().capacity, MiB);
    ASSERT_TRUE(arena.grow_external_backing(device_ptr, 2 * MiB, commit, 0, &failure));
    EXPECT_EQ(failure, GrowFailure::None);
    EXPECT_TRUE(committed);
    EXPECT_EQ(arena.get_statistics().capacity, 2 * MiB);

    arena.clear_external_backing();
    EXPECT_FALSE(arena.using_external_backing(device_ptr));
}

class VkSplatSharedScratch : public ArenaMetricsContentionTest {};

TEST_F(VkSplatSharedScratch, PauseDetachThenLargerScratchRequestReinstallsBeforeGrow) {
    constexpr size_t MiB = 1024 * 1024;
    void* device_ptr = nullptr;
    ASSERT_EQ(cudaMalloc(&device_ptr, 2 * MiB), cudaSuccess);
    auto retained_block = std::shared_ptr<void>(device_ptr, [](void* ptr) {
        EXPECT_EQ(cudaFree(ptr), cudaSuccess);
    });
    // A local real arena keeps this test isolated from the global trainer.
    // B3's GlobalArenaManager::clear_external_backing forwards to this method.
    RasterizerMemoryArena arena;
    const RasterizerMemoryArena::ExternalBacking backing{
        .device_ptr = device_ptr,
        .size = MiB,
        .device = 0,
        .owner = retained_block,
        .label = "test.vksplat.retained_scratch",
    };
    ASSERT_TRUE(arena.install_external_backing(backing));
    bool renderer_installed = true;
    size_t renderer_capacity = MiB;
    arena.clear_external_backing();
    ASSERT_FALSE(arena.using_external_backing(device_ptr));
    ASSERT_TRUE(renderer_installed); // The viewer did not observe B3 detachment.

    constexpr size_t requested_capacity = 2 * MiB;
    ASSERT_GT(requested_capacity, renderer_capacity);
    int reinstall_count = 0;
    // This is the production renderer seam, called before its capacity/grow
    // branches. No Vulkan device or import is needed for the ownership decision.
    ASSERT_TRUE(lfs::vis::ensureRetainedSharedScratchInstalled(
        renderer_installed,
        [&] { return arena.using_external_backing(retained_block.get()); },
        [&] {
            ++reinstall_count;
            return arena.try_install_external_backing(backing);
        }));

    bool committed = false;
    using GrowFailure = RasterizerMemoryArena::ExternalGrowFailure;
    GrowFailure failure = GrowFailure::None;
    EXPECT_TRUE(arena.grow_external_backing(
        retained_block.get(), requested_capacity,
        [&](const size_t bytes) {
            committed = true;
            renderer_capacity = bytes;
            return true;
        },
        0, &failure));
    EXPECT_EQ(failure, GrowFailure::None); // Neither BackingMissing nor Busy.
    EXPECT_TRUE(committed);
    EXPECT_EQ(reinstall_count, 1);
    EXPECT_TRUE(renderer_installed);
    EXPECT_TRUE(arena.using_external_backing(retained_block.get()));
    EXPECT_EQ(renderer_capacity, requested_capacity);
    EXPECT_EQ(arena.get_statistics().capacity, requested_capacity);
    EXPECT_EQ(cudaMemset(retained_block.get(), 0, requested_capacity), cudaSuccess);
}

TEST_F(VkSplatSharedScratch, DetachedBackingDefersWhileTrainingOwnsArenaThenRetries) {
    constexpr size_t MiB = 1024 * 1024;
    void* device_ptr = nullptr;
    ASSERT_EQ(cudaMalloc(&device_ptr, MiB), cudaSuccess);
    auto retained_block = std::shared_ptr<void>(device_ptr, [](void* ptr) {
        EXPECT_EQ(cudaFree(ptr), cudaSuccess);
    });
    RasterizerMemoryArena arena;
    const RasterizerMemoryArena::ExternalBacking backing{
        .device_ptr = device_ptr,
        .size = MiB,
        .device = 0,
        .owner = retained_block,
        .label = "test.vksplat.busy_scratch",
    };
    ASSERT_TRUE(arena.install_external_backing(backing));
    bool renderer_installed = true;
    arena.clear_external_backing();
    const auto held = arena.begin_frame(nullptr, false);
    int reinstall_count = 0;
    const auto ensure_installed = [&] {
        return lfs::vis::ensureRetainedSharedScratchInstalled(
            renderer_installed,
            [&] { return arena.using_external_backing(device_ptr); },
            [&] {
                ++reinstall_count;
                return arena.try_install_external_backing(backing);
            });
    };
    const bool ready = ensure_installed();
    arena.end_frame(held, nullptr, false);
    EXPECT_FALSE(ready);
    EXPECT_FALSE(renderer_installed);
    EXPECT_EQ(reinstall_count, 1);
    EXPECT_TRUE(ensure_installed());
    EXPECT_TRUE(renderer_installed);
    EXPECT_TRUE(arena.using_external_backing(device_ptr));
    EXPECT_EQ(reinstall_count, 2);
    // The capacity-sufficient path must revalidate too, but an attached backing
    // needs no additional installation.
    EXPECT_TRUE(ensure_installed());
    EXPECT_EQ(reinstall_count, 2);
}

TEST_F(ArenaMetricsContentionTest, ViewerGrowTimeoutReleasesReservation) {
    RasterizerMemoryArena arena;
    const auto held = arena.begin_frame(nullptr, false);
    bool committed = false;
    using GrowFailure = RasterizerMemoryArena::ExternalGrowFailure;
    GrowFailure failure = GrowFailure::None;
    const auto started = std::chrono::steady_clock::now();
    EXPECT_FALSE(arena.grow_external_backing(nullptr, 1, [&](size_t) {
        committed = true;
        return true; }, 15, &failure));
    EXPECT_EQ(failure, GrowFailure::Busy);
    EXPECT_LT(std::chrono::steady_clock::now() - started, std::chrono::seconds(2));
    EXPECT_FALSE(committed);
    EXPECT_FALSE(arena.is_rendering_active());
    arena.end_frame(held, nullptr, false);
    // A timed-out viewer must not leave training gated behind a stale request.
    const auto next = arena.try_begin_frame(nullptr, false);
    ASSERT_TRUE(next.has_value());
    arena.end_frame(*next, nullptr, false);
}

TEST_F(ArenaMetricsContentionTest, ViewerGrowPreservesExistingRenderReservation) {
    RasterizerMemoryArena arena;
    arena.set_rendering_active(true);
    bool committed = false;
    EXPECT_FALSE(arena.grow_external_backing(nullptr, 1, [&](size_t) {
        committed = true;
        return true; }, 15));
    EXPECT_FALSE(committed);
    EXPECT_TRUE(arena.is_rendering_active());
    arena.set_rendering_active(false);
    const auto next = arena.try_begin_frame(nullptr, false);
    ASSERT_TRUE(next.has_value());
    arena.end_frame(*next, nullptr, false);
}

TEST_F(ArenaMetricsContentionTest, ViewerGrowReservesIdleWindowAfterTrainingFrame) {
    constexpr size_t MiB = 1024 * 1024;
    void* device_ptr = nullptr;
    ASSERT_EQ(cudaMalloc(&device_ptr, 2 * MiB), cudaSuccess);
    auto owner = std::shared_ptr<void>(device_ptr, [](void* ptr) {
        EXPECT_EQ(cudaFree(ptr), cudaSuccess);
    });
    RasterizerMemoryArena::Config config;
    config.max_physical = 2 * MiB;
    config.enable_vmm = false;
    config.granularity = MiB;
    RasterizerMemoryArena arena(config);
    ASSERT_TRUE(arena.install_external_backing({
        .device_ptr = device_ptr,
        .size = MiB,
        .device = 0,
        .owner = owner,
        .label = "test.external.viewer_growth",
    }));
    const auto held = arena.begin_frame(nullptr, false);
    std::atomic<bool> finished{false};
    bool grew = false;
    size_t committed = 0;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    std::thread viewer([&] {
        EXPECT_EQ(cudaSetDevice(0), cudaSuccess);
        // The observer below briefly owns sync_mutex_. A try-lock miss is a
        // legitimate Busy result, so retry it as the viewer does on later frames.
        // The reserved assertion still rejects an implementation that never queues.
        using Failure = RasterizerMemoryArena::ExternalGrowFailure;
        do {
            Failure failure = Failure::None;
            grew = arena.grow_external_backing(device_ptr, 2 * MiB, [&](size_t requested) {
                committed = requested;
                return true; }, 2000, &failure);
            if (grew || failure != Failure::Busy)
                break;
            std::this_thread::yield();
        } while (std::chrono::steady_clock::now() < deadline);
        finished.store(true, std::memory_order_release);
    });
    bool reserved = false;
    while (!(reserved = arena.is_rendering_active()) &&
           !finished.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    // Release only after growth has registered interest, avoiding a sleep-based
    // race that could let the old try-only implementation pass accidentally.
    arena.end_frame(held, nullptr, false);
    viewer.join();
    EXPECT_TRUE(reserved);
    EXPECT_TRUE(grew);
    EXPECT_EQ(committed, 2 * MiB);
    EXPECT_EQ(arena.get_statistics().capacity, 2 * MiB);
    EXPECT_FALSE(arena.is_rendering_active());
    const auto next = arena.try_begin_frame(nullptr, false);
    ASSERT_TRUE(next.has_value());
    arena.end_frame(*next, nullptr, false);
}

TEST_F(ArenaMetricsContentionTest, ViewerInstallReservesIdleWindowAfterTrainingFrame) {
    constexpr size_t MiB = 1024 * 1024;
    void* device_ptr = nullptr;
    ASSERT_EQ(cudaMalloc(&device_ptr, MiB), cudaSuccess);
    auto owner = std::shared_ptr<void>(device_ptr, [](void* ptr) {
        EXPECT_EQ(cudaFree(ptr), cudaSuccess);
    });
    RasterizerMemoryArena::Config config;
    config.max_physical = MiB;
    config.enable_vmm = false;
    config.granularity = MiB;
    RasterizerMemoryArena arena(config);
    const RasterizerMemoryArena::ExternalBacking backing{
        .device_ptr = device_ptr,
        .size = MiB,
        .device = 0,
        .owner = owner,
        .label = "test.external.viewer_reinstall",
    };

    // Reproduce the B3 detach while the viewer retains the physical block.
    ASSERT_TRUE(arena.install_external_backing(backing));
    arena.clear_external_backing(device_ptr);
    ASSERT_FALSE(arena.using_external_backing(device_ptr));

    const auto held = arena.begin_frame(nullptr, false);
    std::atomic<bool> finished{false};
    bool installed = false;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    std::thread viewer([&] {
        EXPECT_EQ(cudaSetDevice(0), cudaSuccess);
        // Observing the reservation also locks sync_mutex_; a failed initial
        // try-lock must not make this test depend on which thread runs first.
        do {
            installed = arena.try_install_external_backing(backing, 2000);
            if (installed)
                break;
            std::this_thread::yield();
        } while (std::chrono::steady_clock::now() < deadline);
        finished.store(true, std::memory_order_release);
    });
    bool reserved = false;
    while (!(reserved = arena.is_rendering_active()) &&
           !finished.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    // The retained block must register its request before training releases its
    // frame; this fails with the former try-only reinstall path.
    EXPECT_TRUE(reserved);
    arena.end_frame(held, nullptr, false);
    viewer.join();

    EXPECT_TRUE(installed);
    EXPECT_TRUE(arena.using_external_backing(device_ptr));
    EXPECT_FALSE(arena.is_rendering_active());
    arena.clear_external_backing(device_ptr);
}

TEST_F(ArenaMetricsContentionTest, ViewerInstallTimeoutReleasesOnlyItsOwnReservation) {
    constexpr size_t MiB = 1024 * 1024;
    void* device_ptr = nullptr;
    ASSERT_EQ(cudaMalloc(&device_ptr, MiB), cudaSuccess);
    auto owner = std::shared_ptr<void>(device_ptr, [](void* ptr) {
        EXPECT_EQ(cudaFree(ptr), cudaSuccess);
    });
    RasterizerMemoryArena arena;
    const RasterizerMemoryArena::ExternalBacking backing{
        .device_ptr = device_ptr,
        .size = MiB,
        .device = 0,
        .owner = owner,
        .label = "test.external.install_timeout",
    };

    const auto held = arena.begin_frame(nullptr, false);
    EXPECT_FALSE(arena.try_install_external_backing(backing, 15));
    EXPECT_FALSE(arena.is_rendering_active());
    EXPECT_FALSE(arena.using_external_backing(device_ptr));
    arena.end_frame(held, nullptr, false);

    // A timed-out installer must allow training to continue.
    const auto next = arena.try_begin_frame(nullptr, false);
    ASSERT_TRUE(next.has_value());
    arena.end_frame(*next, nullptr, false);

    // A different viewer's reservation must not be consumed by installation.
    arena.set_rendering_active(true);
    EXPECT_FALSE(arena.try_install_external_backing(backing, 15));
    EXPECT_TRUE(arena.is_rendering_active());
    EXPECT_FALSE(arena.using_external_backing(device_ptr));
    arena.set_rendering_active(false);
    ASSERT_TRUE(arena.try_install_external_backing(backing, 15));
    EXPECT_TRUE(arena.using_external_backing(device_ptr));
    arena.clear_external_backing(device_ptr);
}

TEST_F(ArenaMetricsContentionTest, FullResetRetainsExternalBackingAndCapacity) {
    constexpr size_t MiB = 1024 * 1024;
    constexpr size_t physical_bytes = 8 * MiB;
    constexpr size_t initial_committed_bytes = 1 * MiB;

    void* device_ptr = nullptr;
    ASSERT_EQ(cudaMalloc(&device_ptr, physical_bytes), cudaSuccess);
    auto owner = std::shared_ptr<void>(device_ptr, [](void* ptr) {
        EXPECT_EQ(cudaFree(ptr), cudaSuccess);
    });

    RasterizerMemoryArena::Config config;
    config.max_physical = physical_bytes;
    config.enable_vmm = false;
    config.granularity = MiB;
    RasterizerMemoryArena arena(config);
    RasterizerMemoryArena::ExternalBacking backing{
        .device_ptr = device_ptr,
        .size = initial_committed_bytes,
        .device = 0,
        .owner = owner,
        .label = "test.external.reset",
        .grow = [physical_bytes](const size_t requested) {
            return requested <= physical_bytes ? physical_bytes : size_t{0};
        },
    };
    ASSERT_TRUE(arena.install_external_backing(std::move(backing)));

    const uint64_t frame = arena.begin_frame(nullptr, false);
    auto allocate = arena.get_allocator(frame);
    ASSERT_NE(allocate(2 * MiB), nullptr);
    arena.end_frame(frame, nullptr, false);
    ASSERT_EQ(arena.get_statistics().capacity, physical_bytes);

    arena.full_reset();
    EXPECT_TRUE(arena.using_external_backing());
    EXPECT_EQ(arena.get_statistics().capacity, physical_bytes);

    const uint64_t post_reset_frame = arena.begin_frame(nullptr, false);
    auto post_reset_allocate = arena.get_allocator(post_reset_frame);
    EXPECT_NE(post_reset_allocate(3 * MiB), nullptr);
    arena.end_frame(post_reset_frame, nullptr, false);
}

TEST_F(ArenaMetricsContentionTest, FullResetDecommitsVmmHighWater) {
    constexpr size_t MiB = 1024 * 1024;
    RasterizerMemoryArena::Config config;
    config.virtual_size = 1ULL * 1024 * MiB;
    config.max_physical = 512 * MiB;
    config.granularity = 2 * MiB;
    RasterizerMemoryArena arena(config);

    const uint64_t frame = arena.begin_frame(nullptr, false);
    auto allocate = arena.get_allocator(frame);
    ASSERT_NE(allocate(192 * MiB), nullptr);
    arena.end_frame(frame, nullptr, false);

    const auto grown = arena.get_statistics();
    ASSERT_EQ(grown.capacity, 192 * MiB);
    const auto grown_info = arena.get_memory_info();
    EXPECT_EQ(grown_info.required_bytes, grown_info.peak_usage);
    EXPECT_EQ(grown_info.required_bytes, grown_info.arena_capacity);

    arena.full_reset();

    const auto reset = arena.get_statistics();
    EXPECT_EQ(reset.current_usage, 0u);
    EXPECT_EQ(reset.capacity, 0u);
    const auto reset_info = arena.get_memory_info();
    EXPECT_EQ(reset_info.peak_usage, 0u);
    EXPECT_EQ(reset_info.required_bytes, 0u);
    EXPECT_EQ(reset_info.arena_capacity, 0u);
}

TEST_F(ArenaMetricsContentionTest, FallbackGrowthUsesExactMeasuredRequirement) {
    constexpr size_t MiB = 1024 * 1024;
    RasterizerMemoryArena::Config config;
    config.max_physical = 512 * MiB;
    config.enable_vmm = false;
    RasterizerMemoryArena arena(config);

    const uint64_t frame = arena.begin_frame(nullptr, false);
    auto allocate = arena.get_allocator(frame);
    ASSERT_NE(allocate(128 * MiB), nullptr);

    const auto grown = arena.get_statistics();
    EXPECT_EQ(grown.capacity, 128 * MiB);
    const auto grown_info = arena.get_memory_info();
    EXPECT_EQ(grown_info.required_bytes, grown_info.peak_usage);
    EXPECT_EQ(grown_info.required_bytes, grown_info.arena_capacity);
    arena.end_frame(frame, nullptr, false);
}

TEST_F(ArenaMetricsContentionTest, RetainedFallbackPublishesLogicalRequiredAndClassCSlack) {
    constexpr size_t MiB = 1024 * 1024;
    RasterizerMemoryArena::Config config;
    config.max_physical = 512 * MiB;
    config.enable_vmm = false;
    RasterizerMemoryArena arena(config);

    const uint64_t large_frame = arena.begin_frame(nullptr, false);
    auto allocate_large = arena.get_allocator(large_frame);
    ASSERT_NE(allocate_large(64 * MiB), nullptr);
    arena.end_frame(large_frame, nullptr, false);

    const auto large = arena.get_memory_info();
    ASSERT_EQ(large.required_bytes, 64 * MiB);
    ASSERT_EQ(large.arena_capacity, 64 * MiB);

    const uint64_t small_frame = arena.begin_frame(nullptr, false);
    auto allocate_small = arena.get_allocator(small_frame);
    ASSERT_NE(allocate_small(10 * MiB), nullptr);
    arena.end_frame(small_frame, nullptr, false);
    ASSERT_TRUE(arena.shrink_to_current_at_boundary());

    const auto retained = arena.get_memory_info();
    EXPECT_EQ(retained.current_usage, 10 * MiB);
    EXPECT_EQ(retained.peak_usage, 10 * MiB);
    EXPECT_EQ(retained.required_bytes, 10 * MiB);
    EXPECT_EQ(retained.arena_capacity, 64 * MiB);
    EXPECT_EQ(retained.arena_capacity - retained.required_bytes, 54 * MiB);
}
