/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/event_bridge/event_bridge.hpp"
#include "core/scene.hpp"
#include "user_shader_linker.hpp"
#include "user_shader_measure.hpp"
#include <atomic>
#include <future>
#include <gtest/gtest.h>
#include <signal.h>
#include <sys/resource.h>
#include <thread>

namespace {
    using namespace lfs::test::measure;
    namespace shaders = lfs::vis::shaders;
    using namespace std::chrono_literals;
    std::unique_ptr<Options> options;
    constexpr const char* host = "projection_forward_custom";

    struct Linker : shaders::HostLinker {
        lfs::test::ScheduledLinker inner;
        bool throws;
        Linker(lfs::test::Schedule& gate, bool throwing) : inner(gate), throws(throwing) {}
        lfs::Result<shaders::LinkedHost> link(std::span<const shaders::EffectSource> effects,
                                              const std::string& label, shaders::CompileDiagnostics& diagnostics) override {
            if (throws)
                throw std::runtime_error("injected linker exception");
            auto linked = inner.link(effects, label, diagnostics);
            diagnostics.raw = "old completion marker";
            if (linked)
                linked->schemas.assign(effects.size(), {{"old_marker", 0, 1, 0, 1, 0}});
            return linked;
        }
    };
    template <class Predicate>
    bool until(Predicate done, std::chrono::milliseconds timeout = 2s) {
        const auto deadline = Clock::now() + timeout;
        while (!done()) {
            if (Clock::now() >= deadline)
                return false;
            std::this_thread::sleep_for(1ms);
        }
        return true;
    }
    struct State {
        lfs::core::Scene scene;
        lfs::test::Schedule gate;
        std::atomic<int> wakes = 0;
        std::unique_ptr<shaders::ShaderService> service;
        explicit State(bool throwing = false) : service(std::make_unique<shaders::ShaderService>(
                                                    scene, [this, throwing] { return std::make_unique<Linker>(gate, throwing); }, [this] { ++wakes; })) {}
        ~State() {
            gate.gate(false);
            service.reset();
        }
        // harness adapter: service mutation/demand surface shared by the fault cases.
        lfs::core::Uuid bind() {
            auto id = ServiceAccess::create(*service, "Fault fixture", "old source");
            if (!id || !ServiceAccess::bind(*service, {}, *id))
                throw std::runtime_error("fault fixture binding failed");
            return *id;
        }
    };
    std::string throwingLink() {
        State state(true);
        const auto id = state.bind();
        ServiceAccess::demand(*state.service, host);
        if (!until([&] { return ServiceAccess::status(*state.service, id)->state == shaders::ShaderState::Failed; }))
            return "asset did not reach Failed";
        const auto status = ServiceAccess::status(*state.service, id);
        return status->diagnostics.raw.empty() && status->diagnostics.entries.empty()
                   ? "Failed has no diagnostic"
                   : "";
    }
    std::string staleCompletion() {
        State state;
        std::mutex mutex;
        std::vector<shaders::ShaderEvent> events;
        lfs::event::ScopedHandler handler;
        handler.subscribe<shaders::ShaderEvent>([&](const auto& event) {
            const std::lock_guard lock(mutex);
            events.push_back(event);
        });
        state.gate.gate(true);
        const auto old = state.bind();
        ServiceAccess::demand(*state.service, host);
        if (!state.gate.waitStarted(1))
            return "old link never started";
        ServiceAccess::clear(*state.service);
        const auto fresh = ServiceAccess::create(*state.service, "Replacement fixture", "new source");
        if (!fresh)
            return "replacement creation failed";
        // harness adapter: no public restore API exists in BEFORE.
        shaders::UserShaderMeasureAccess::replaceId(*state.service, *fresh, old);
        if (!ServiceAccess::bind(*state.service, {}, old))
            return "replacement binding failed";
        {
            const std::lock_guard lock(mutex);
            events.clear();
        }
        state.gate.gate(false);
        // A dropped stale result need not wake anyone: let the released link
        // return and be processed or discarded, then look for leaks.
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        if (!until([&] { return !ServiceAccess::status(*state.service, old)->compiling; }))
            return "old completion never drained";
        const auto status = ServiceAccess::status(*state.service, old);
        std::string detail;
        if (status->links != 0)
            detail += "old link changed replacement link count; ";
        if (!status->parameters.empty() || status->schema_revision != 0)
            detail += "old schema reached replacement; ";
        if (!status->diagnostics.raw.empty() || !status->diagnostics.entries.empty() || status->diagnostics_revision != 0)
            detail += "old diagnostics reached replacement; ";
        const std::lock_guard lock(mutex);
        for (const auto& event : events)
            if (event.kind == shaders::ShaderEvent::Kind::Completed || event.kind == shaders::ShaderEvent::Kind::Failed)
                detail += "old terminal event escaped after clear; ";
        return detail;
    }
    std::string teardown() {
        State state;
        state.gate.gate(true);
        state.bind();
        auto pin = ServiceAccess::pin(*state.service);
        ServiceAccess::demand(pin, host);
        if (!state.gate.waitStarted(1))
            return "blocked link never started";
        std::promise<void> entering;
        auto entered = entering.get_future();
        auto waiter = std::async(std::launch::async, [pin = std::move(pin), &entering]() mutable {
            entering.set_value();
            const auto result = ServiceAccess::wait(pin);
            const bool cancelled = !result && result.error().code() == lfs::ErrorCode::Cancelled;
            pin = {}; // release the service lifetime hold before destruction joins us
            return std::pair{cancelled, Clock::now()};
        });
        entered.wait();
        if (waiter.wait_for(50ms) != std::future_status::timeout)
            return "export wait was not pending";
        auto released = Clock::now();
        std::thread release([&] {
            std::this_thread::sleep_for(50ms);
            released = Clock::now();
            state.gate.gate(false);
        });
        state.service.reset(); // destruction must cancel the waiter and join the blocked compiler
        release.join();
        if (waiter.wait_for(1s) != std::future_status::ready)
            return "wait did not finish within 1s";
        const auto [cancelled, finished] = waiter.get();
        if (!cancelled)
            return "wait did not return Cancelled";
        return finished - released > 1s || Clock::now() - released > 1s ? "teardown exceeded 1s after link release" : "";
    }
    // Fork before constructing the service; a worker exception cannot kill the gtest parent.
    void fault(const char* name, std::string (*body)()) {
        if (!options->matches(std::string("fault/") + name))
            GTEST_SKIP();
        Json samples = Json::array();
        for (int rep = 0; rep < options->reps; ++rep) {
            char path[] = "/tmp/user-shader-fault-XXXXXX";
            const int fd = mkstemp(path);
            ASSERT_GE(fd, 0);
            std::cout.flush();
            std::cerr.flush();
            const pid_t child = fork();
            ASSERT_GE(child, 0);
            if (child == 0) {
                rlimit limit{0, 0};
                setrlimit(RLIMIT_CORE, &limit);
                close(fd);
                Json result;
                try {
                    const auto detail = body();
                    result = {{"result", detail.empty() ? "pass" : "fail"}, {"detail", detail}};
                } catch (const std::exception& e) { result = {{"result", "fail"}, {"detail", e.what()}}; }
                std::ofstream(path) << result.dump();
                _exit(0);
            }
            close(fd);
            int status = 0;
            const auto start = Clock::now();
            pid_t done = 0;
            while ((done = waitpid(child, &status, WNOHANG)) == 0 && Clock::now() - start < 10s)
                std::this_thread::sleep_for(10ms);
            Json result;
            if (done == 0) {
                kill(child, SIGKILL);
                waitpid(child, &status, 0);
                result = {{"result", "hang"}, {"detail", "10s watchdog expired"}};
            } else if (done < 0) {
                result = {{"result", "fail"}, {"detail", "waitpid failed"}};
            } else if (WIFSIGNALED(status)) {
                result = {{"result", "crash"}, {"detail", "signal " + std::to_string(WTERMSIG(status))}};
            } else {
                std::ifstream file(path);
                result = Json::parse(file, nullptr, false);
                if (result.is_discarded())
                    result = {{"result", "fail"}, {"detail", "child produced no result"}};
            }
            unlink(path);
            samples.push_back(result);
        }
        const auto worst = std::ranges::max_element(samples, {}, [](const auto& s) {
            const std::string r = s.at("result");
            return r == "hang" ? 3 : r == "crash" ? 2
                                 : r == "fail"    ? 1
                                                  : 0;
        });
        auto result = *worst;
        result["metric"] = "fault";
        result["case"] = name;
        result["reps"] = options->reps;
        result["samples"] = samples;
        emit(result);
        EXPECT_EQ(result["result"], "pass") << result["detail"];
    }
    TEST(UserShaderFaults, ThrowingLinker) { fault("throwing_linker", throwingLink); }
    TEST(UserShaderFaults, StaleCompletion) { fault("stale_completion", staleCompletion); }
    TEST(UserShaderFaults, TeardownBlockedLink) { fault("teardown_blocked_link", teardown); }
} // namespace
int main(int argc, char** argv) {
    try {
        options = std::make_unique<Options>(argc, argv);
        ::testing::InitGoogleTest(&argc, argv);
        return RUN_ALL_TESTS();
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
