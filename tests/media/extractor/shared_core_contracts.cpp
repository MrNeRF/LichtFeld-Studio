// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "core/assert.hpp"
#include "core/error_bus.hpp"
#include "diagnostics/gpu_backend.hpp"
#include "media/media_backends.hpp"
#include "media/media_ingest.hpp"
#include <algorithm>
#include <atomic>
#include <stdexcept>
extern "C" void mediaTestRegisterWriter(lfs::core::FailureReportWriter);
extern "C" bool mediaTestDedup(std::uint64_t&);
extern "C" const void* mediaTestErrorBus();
namespace {
    std::atomic<int> writes{0};
    void writer(std::string_view) noexcept { ++writes; }
    void require(bool condition, const char* text) {
        if (!condition)
            throw std::runtime_error(text);
    }
    int event_records = 0;
    bool event_pending = true;
    const lfs::diagnostics::GpuDiagnosticsBackend diagnostic_backend{
        []() noexcept { return true; },
        [](std::size_t& used, std::size_t* total) noexcept { used = 128; if (total) *total = 512; return true; },
        [](lfs::diagnostics::VramProcessSnapshot& p) { p.cuda_used = 128; p.cuda_total = 512; p.cuda_memory_valid = true; },
        []() -> std::optional<std::size_t> { return 64; },
        []() noexcept -> void* { static int event; return &event; },
        [](void*, void*) noexcept { ++event_records; return true; },
        [](void*, void*, float& ms) noexcept {
            ms = 2.5f;
            return event_pending ? lfs::diagnostics::GpuEventStatus::Pending : lfs::diagnostics::GpuEventStatus::Ready;
        }};
    class RejectingHdr final : public lfs::io::HdrRenderer {
    public:
        bool isAvailable(std::string& error) override {
            error = "test backend rejected request";
            return false;
        }
        bool tonemapToSdr(const lfs::media::DecodedVideoFrame*, lfs::io::HdrFormat, int, int,
                          std::vector<unsigned char>&, std::string& error, lfs::io::HdrTonemapTiming*) override { return isAvailable(error); }
        bool tonemapToSdrRgba(const lfs::media::DecodedVideoFrame*, lfs::io::HdrFormat, int, int, int,
                              std::vector<unsigned char>&, std::string& error) override { return isAvailable(error); }
        void reset() override {}
    };
} // namespace
int runSharedCoreContracts() {
    using namespace lfs;
    require(mediaTestErrorBus() == &ErrorBus::instance(), "one ErrorBus across DLL consumers");
    core::reset_failure_report_dedup_for_testing();
    std::uint64_t count = 0;
    require(mediaTestDedup(count) && count == 1, "bridge emits first failure");
    require(!core::decide_failure_report_for_testing("media-shared-state", 1, "bridge", count) && count == 2,
            "dedup state is shared with executable");
    mediaTestRegisterWriter(writer);
    core::emit_failure_report({.family = "shared-writer", .location = LFS_SOURCE_SITE_CURRENT(), .capture_stack = false});
    require(writes == 1, "writer installed through bridge is used by lfs_error");
    bool threw = false;
    try {
        LFS_ASSERT_MSG(false, "shared contract");
    } catch (const std::runtime_error&) { threw = true; }
    require(threw && writes == 2, "production contract emits then throws through shared error library");
    core::register_failure_report_writer(nullptr);
    const auto base = media::MediaIngest::capabilities();
    require(!base.hdr_to_sdr, "headless host has no HDR backend");
    media::detail::registerHdrFactory([]() -> std::unique_ptr<io::HdrRenderer> { return std::make_unique<RejectingHdr>(); });
    require(media::MediaIngest::capabilities().hdr_to_sdr, "registered HDR capability is exposed");
    io::HdrLibplaceboRenderer renderer;
    std::string error;
    std::vector<unsigned char> pixels;
    require(!renderer.tonemapToSdr(nullptr, io::HdrFormat::HDR10, 1, 1, pixels, error),
            "renderer failure is propagated");
    require(error == "test backend rejected request" && pixels.empty(), "front facade delegates to host renderer");
    require(renderer.backendName() == "host", "backend identity delegates to the registered renderer");
    media::detail::registerHdrFactory(nullptr);
    require(!media::MediaIngest::capabilities().hdr_to_sdr, "unregister removes capability");
    auto& profiler = diagnostics::VramProfiler::instance();
    profiler.setEnabled(true);
    require(profiler.acquireGpuEventPair("media.contract", nullptr) == -1, "CPU diagnostics do not initialize CUDA");
    require(!diagnostics::process_device_memory_bytes(), "CPU diagnostics have no GPU sampler");
    require(diagnostics::register_gpu_diagnostics_backend(diagnostic_backend), "host registers native operations");
    require(diagnostics::register_gpu_diagnostics_backend(diagnostic_backend), "same backend registration is idempotent");
    const auto replacement = diagnostic_backend;
    require(!diagnostics::register_gpu_diagnostics_backend(replacement), "replacement cannot invalidate event handles");
    profiler.sampleCudaMemory();
    require(profiler.snapshot().process.cuda_used == 128, "host sampling updates shared profiler state");
    require(diagnostics::process_device_memory_bytes() == 64, "host process sampler is delegated");
    const auto pair = profiler.acquireGpuEventPair("media.contract", nullptr);
    require(pair >= 0, "host event is recorded");
    profiler.releaseGpuEventPair(pair, nullptr);
    require(event_records == 2, "start and stop use host event operations");
    profiler.drainGpuEvents();
    event_pending = false;
    profiler.drainGpuEvents();
    const auto snapshot = profiler.snapshot();
    const auto timer = std::find_if(snapshot.tree.begin(), snapshot.tree.end(), [](const auto& node) { return node.path == "media/contract"; });
    require(timer != snapshot.tree.end() && timer->gpu_call_count == 1 && timer->gpu_last_ms == 2.5,
            "pending event is retained and completed exactly once");
    profiler.setEnabled(false);
    core::reset_failure_report_dedup_for_testing();
    return 0;
}
