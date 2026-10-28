// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "core/crash_handler.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "diagnostics/vram_profiler.hpp"
#include "io/media_studio_backends.hpp"
#include "media/media_backends.hpp"
#include "media/media_ingest.hpp"
#include <algorithm>
#include <cstdlib>
#include <cuda_runtime.h>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <nlohmann/json.hpp>

namespace lfs::io {
    std::unique_ptr<media::detail::GpuJpegEncoder> createStudioJpegEncoder(const media::detail::JpegSettings&);
}
namespace {
    // Test-only replay of the former hybrid path: keep NVDEC and the real
    // CUDA JPEG encoder, but require host color conversion/readback.
    class HostColorEncoder final : public lfs::media::detail::GpuJpegEncoder {
        std::unique_ptr<lfs::media::detail::GpuJpegEncoder> encoder_;

    public:
        explicit HostColorEncoder(std::unique_ptr<lfs::media::detail::GpuJpegEncoder> encoder) : encoder_(std::move(encoder)) {}
        std::size_t capacity() const noexcept override { return encoder_->capacity(); }
        bool canConvertHardware() const noexcept override { return false; }
        void convertHardware(const lfs::media::detail::CudaVideoFrame&, std::uint8_t*) override { throw std::logic_error("host color test unexpectedly converted hardware"); }
        void finishHardware() override { encoder_->finishHardware(); }
        void* queueHardware(std::size_t) override { throw std::logic_error("host color test unexpectedly queued hardware"); }
        void* queueHost(std::size_t index, const std::uint8_t* bytes) override { return encoder_->queueHost(index, bytes); }
        std::vector<std::vector<std::uint8_t>> encode(const std::vector<void*>& frames, int width, int height, int quality) override { return encoder_->encode(frames, width, height, quality); }
    };
} // namespace

nlohmann::json runNativeVideoContracts(const nlohmann::json& request);

int main(int argc, char** argv) {
    using namespace lfs;
    using nlohmann::json;
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
            return 77;
        if (argc != 2)
            throw std::runtime_error("Expected JSON request path");
        if (std::string_view(argv[1]) == "--check-gpu")
            return 0;
        const bool handoff_tests = std::string_view(argv[1]) == "--handoff-unit";
        core::Logger::get().init(core::LogLevel::Warn, "", "", true,
                                 handoff_tests ? "" : core::path_to_utf8(core::utf8_to_path(argv[1]).parent_path()));
        if (cudaFree(nullptr) != cudaSuccess)
            throw std::runtime_error("CUDA context unavailable");
        struct GpuShutdown {
            ~GpuShutdown() { core::teardown_gpu_before_exit(); }
        } shutdown;
        if (handoff_tests) {
            int test_argc = 1;
            char* test_argv[] = {argv[0], nullptr};
            ::testing::InitGoogleTest(&test_argc, test_argv);
            return RUN_ALL_TESTS();
        }
        io::registerStudioMediaBackends();
        const auto caps = media::MediaIngest::capabilities();
        std::ifstream stream(core::utf8_to_path(argv[1]));
        const auto request = json::parse(stream);
        if (request.value("verbose", false))
            core::Logger::get().set_level(core::LogLevel::Info);
        if (request.value("disable_jpeg_provider", false))
            // Keep the registered host's decode capabilities unchanged while
            // simulating an encoder that cannot be allocated.
            media::detail::registerGpuJpegFactory(+[](const media::detail::JpegSettings&) -> std::unique_ptr<media::detail::GpuJpegEncoder> { return nullptr; });
        const auto operation = request.value("operation", "extract");
        if (operation == "native-preview" || operation == "native-encode" || operation == "native-encode-session" || operation == "native-conversion" || operation == "native-yuv-color" || operation == "native-quality") {
            std::cout << runNativeVideoContracts(request).dump() << '\n';
            return 0;
        }
        if (request.value("cpu_color", false))
            media::detail::registerGpuJpegFactory(+[](const media::detail::JpegSettings& settings) -> std::unique_ptr<media::detail::GpuJpegEncoder> {
                auto encoder = io::createStudioJpegEncoder(settings);
                return encoder ? std::make_unique<HostColorEncoder>(std::move(encoder)) : nullptr;
            });
        media::IngestRequest input;
        input.input = core::utf8_to_path(request.at("input").get<std::string>());
        input.selection.mode = media::SelectionMode::Interval;
        input.selection.interval = 1;
        if (request.contains("fps")) {
            input.selection.mode = media::SelectionMode::FPS;
            input.selection.fps = request.at("fps").get<double>();
        }
        input.end_seconds = request.value("end_seconds", .35);
        input.allow_hardware_decode = request.value("hardware", true);
        input.convert_hdr_to_sdr = request.value("hdr", false);
        input.geometry.clockwise_rotation = request.value("rotation", 0);
        if (request.contains("scale")) {
            input.geometry.mode = media::ResizeMode::Scale;
            input.geometry.scale = request.at("scale").get<float>();
        }
        input.sharpness.enabled = request.value("sharpness", false);
        input.sharpness.window = request.value("window", false);
        input.sharpness.window_candidates = 2;
        media::FileExtraction files;
        files.files.output_directory = core::utf8_to_path(request.at("output").get<std::string>());
        files.files.format = request.value("jpeg", false) ? media::FrameFileFormat::JPEG : media::FrameFileFormat::PNG;
        files.files.embed_source_metadata = request.value("embed_source_metadata", false);
        files.files.image_output.overwrite = request.value("overwrite", true);
        files.write_metadata = true;
        files.resume = request.value("resume", false);
        const int exit_after = request.value("exit_after", 0);
        if (exit_after > 0)
            input.progress = [exit_after](const media::IngestProgress& progress) {
                if (progress.processed >= exit_after)
                    std::_Exit(73);
            };
        media::MemoryFrameSink memory;
        const bool to_memory = request.value("memory", false);
        auto result = to_memory ? media::MediaIngest::extract(input, memory) : media::MediaIngest::extractFiles(input, files);
        json output{{"success", result.has_value()}, {"error", result ? "" : result.error().detail()}, {"hardware", caps.hardware_decode}, {"hdr", caps.hdr_to_sdr}};
        if (result)
            output["accepted"] = result->frames_accepted;
        if (to_memory && result) {
            output["frames"] = json::array();
            for (const auto& frame : memory.frames()) {
                const auto view = frame.view();
                output["frames"].push_back(std::vector<std::uint8_t>(view.pixels.begin(), view.pixels.end()));
            }
        }
        auto& profiler = diagnostics::VramProfiler::instance();
        profiler.setEnabled(true);
        profiler.sampleCudaMemory();
        if (!profiler.snapshot().process.cuda_memory_valid)
            throw std::runtime_error("Studio GPU diagnostics sampler not registered");
        const auto event = profiler.acquireGpuEventPair("media/studio", nullptr);
        if (event < 0)
            throw std::runtime_error("Studio GPU timer not registered");
        profiler.releaseGpuEventPair(event, nullptr);
        if (cudaDeviceSynchronize() != cudaSuccess)
            throw std::runtime_error("CUDA event synchronization failed");
        profiler.drainGpuEvents();
        const auto snapshot = profiler.snapshot();
        const auto timer = std::find_if(snapshot.tree.begin(), snapshot.tree.end(), [](const auto& node) { return node.path == "media/studio"; });
        if (timer == snapshot.tree.end() || timer->gpu_call_count != 1)
            throw std::runtime_error("Studio GPU event not collected");
        output["cuda_diagnostics"] = true;
        profiler.setEnabled(false);
        std::cout << output.dump() << std::endl;
        return result ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
