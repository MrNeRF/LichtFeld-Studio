// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "core/path_utils.hpp"
#include "io/video_frame_extractor.hpp"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>

// Test adapter only: decoding, selection, geometry, codecs and metadata execute
// the production sources directly. The JSON protocol is not a public CLI.
int main(int argc, char** argv) {
    using namespace lfs::io;
    using nlohmann::json;
    try {
        if (argc != 2)
            throw std::runtime_error("Expected one JSON request path");
        std::ifstream input(lfs::core::utf8_to_path(argv[1]));
        const auto request = json::parse(input);
        VideoFrameExtractor::Params params;
        params.video_path = lfs::core::utf8_to_path(request.at("input").get<std::string>());
        params.output_dir = lfs::core::utf8_to_path(request.at("output").get<std::string>());
        params.mode = request.value("mode", "interval") == "fps" ? ExtractionMode::FPS : ExtractionMode::INTERVAL;
        params.fps = request.value("fps", 10.0);
        params.frame_interval = request.value("interval", 1);
        params.start_time = request.value("start", 0.0);
        params.end_time = request.value("end", -1.0);
        params.rotation = request.value("rotation", 0);
        params.filename_pattern = request.value("naming", "frame_%03d");
        params.generate_metadata = request.value("metadata", true);
        const auto format = request.value("format", "png");
        if (format != "png" && format != "jpg")
            throw std::runtime_error("SDR test runner supports PNG/JPG only");
        params.format = format == "png" ? ImageFormat::PNG : ImageFormat::JPG;
        params.jpg_quality = request.value("quality", 95);
        if (request.contains("scale")) {
            params.resolution_mode = ResolutionMode::Scale;
            params.scale = request.at("scale").get<float>();
        }
        if (request.contains("width")) {
            params.resolution_mode = ResolutionMode::Custom;
            params.custom_width = request.at("width").get<int>();
            params.custom_height = request.at("height").get<int>();
        }
        params.sharpness.enabled = request.value("sharpness", false);
        params.sharpness.threshold = request.value("threshold", 0.0);
        int progressed = 0;
        json progress = json::array();
        params.progress_callback = [&](int current, int total, int discarded) {
            progressed = current;
            progress.push_back({current, total, discarded});
        };
        const int cancel_after = request.value("cancel_after", -1);
        params.cancel_requested = [&]() { return cancel_after >= 0 && progressed >= cancel_after; };
        VideoFrameExtractor extractor;
        std::string error;
        const bool success = extractor.extract(params, error);
        const auto outcome = extractor.lastOutcome();
        const char* state = outcome == ExtractionOutcome::Completed ? "completed" : outcome == ExtractionOutcome::Cancelled ? "cancelled"
                                                                                                                            : "failed";
        std::cout << json{{"success", success}, {"outcome", state}, {"error", error}, {"progress", progress}}.dump() << '\n';
        // A failed extraction is a result to assert, not a harness crash.
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Test adapter failed: " << error.what() << '\n';
        return 2;
    }
}
