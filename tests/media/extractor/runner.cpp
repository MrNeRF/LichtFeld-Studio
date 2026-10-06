// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "core/path_utils.hpp"
#include "io/media/media_probe.hpp"
#include "io/video_frame_extractor.hpp"
#include "io/video_player.hpp"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>

int runProbeUnitContracts();

// Test adapter only: decoding, selection, geometry, codecs and metadata execute
// the production sources directly. The JSON protocol is not a public CLI.
int main(int argc, char** argv) {
    using namespace lfs::io;
    using nlohmann::json;
    try {
        if (argc != 2)
            throw std::runtime_error("Expected one JSON request path");
        if (std::string_view(argv[1]) == "--probe-unit")
            return runProbeUnitContracts();
        std::ifstream input(lfs::core::utf8_to_path(argv[1]));
        const auto request = json::parse(input);
        if (request.value("operation", "extract") == "preview") {
            VideoPlayer player;
            const bool success = player.open(lfs::core::utf8_to_path(request.at("input").get<std::string>()));
            json output{{"success", success}, {"error", player.takeError()}};
            if (success) {
                output["rotation"] = player.rotation();
                output["gpu_rotation"] = player.currentFrameHasGpuRotation();
                output["size"] = json::array({player.width(), player.height()});
                const auto* pixels = player.currentFrameData();
                const size_t size = static_cast<size_t>(player.width()) * player.height() * player.currentFrameChannels();
                output["pixels"] = std::vector<unsigned char>(pixels, pixels + size);
            }
            std::cout << output.dump() << '\n';
            return 0;
        }
        if (request.value("operation", "extract") == "probe") {
            using namespace lfs::media;
            ProbeOptions options;
            options.depth = request.value("headers_only", false) ? ProbeDepth::Headers : ProbeDepth::StreamInfo;
            options.timeout = std::chrono::milliseconds(request.value("timeout_ms", 10000));
            const auto result = MediaProbe::inspect(lfs::core::utf8_to_path(request.at("input").get<std::string>()), options);
            auto optional = []<typename T>(const std::optional<T>& value) -> json { return value ? json(*value) : json(nullptr); };
            auto rational = [](const std::optional<Rational>& value) -> json {
                return value ? json::array({value->numerator, value->denominator}) : json(nullptr);
            };
            auto timestamp = [&](const std::optional<Timestamp>& value) -> json {
                return value ? json{{"ticks", value->ticks}, {"time_base", json::array({value->time_base.numerator, value->time_base.denominator})}} : json(nullptr);
            };
            json output{{"success", result.media.has_value()}, {"error_code", static_cast<int>(result.error.code)}, {"ffmpeg_code", result.error.ffmpeg_code}, {"error", result.error.message}};
            if (result.media) {
                const auto& source = *result.media;
                output["container"] = source.container;
                output["stream_info_probed"] = source.stream_info_probed;
                output["duration"] = timestamp(source.duration);
                output["selected_video"] = optional(source.selected_video_stream);
                output["streams"] = json::array();
                for (const auto& stream : source.streams) {
                    output["streams"].push_back({{"index", stream.index}, {"kind", static_cast<int>(stream.kind)}, {"codec", stream.codec}, {"width", optional(stream.width)}, {"height", optional(stream.height)}, {"pixel_format", optional(stream.pixel_format)}, {"time_base", rational(stream.time_base)}, {"nominal_fps", rational(stream.nominal_frame_rate)}, {"average_fps", rational(stream.average_frame_rate)}, {"start", timestamp(stream.start)}, {"duration", timestamp(stream.duration)}, {"depth", optional(stream.color.component_depth)}, {"matrix", optional(stream.color.matrix)}, {"transfer", optional(stream.color.transfer)}, {"primaries", optional(stream.color.primaries)}, {"range", optional(stream.color.range)}, {"rotation", optional(stream.orientation.clockwise_degrees)}, {"rotation_source", static_cast<int>(stream.orientation.source)}, {"display_matrix", optional(stream.orientation.display_matrix)}, {"legacy_rotation", legacyQuarterTurn(stream.orientation)}});
                }
            }
            std::cout << output.dump() << '\n';
            return 0;
        }
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
