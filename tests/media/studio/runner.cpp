// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "core/crash_handler.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "diagnostics/vram_profiler.hpp"
#include "io/media_studio_backends.hpp"
#include "media/media_ingest.hpp"
#include <algorithm>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

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
        core::Logger::get().init(core::LogLevel::Warn, "", "", true,
                                 core::path_to_utf8(core::utf8_to_path(argv[1]).parent_path()));
        if (cudaFree(nullptr) != cudaSuccess)
            throw std::runtime_error("CUDA context unavailable");
        struct GpuShutdown {
            ~GpuShutdown() { core::teardown_gpu_before_exit(); }
        } shutdown;
        io::registerStudioMediaBackends();
        const auto caps = media::MediaIngest::capabilities();
        std::ifstream stream(core::utf8_to_path(argv[1]));
        const auto request = json::parse(stream);
        media::IngestRequest input;
        input.input = core::utf8_to_path(request.at("input").get<std::string>());
        input.selection.mode = media::SelectionMode::Interval;
        input.selection.interval = 1;
        input.end_seconds = .35;
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
        files.write_metadata = true;
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
