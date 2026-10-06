/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "mcp_media_tools.hpp"
#include "core/error_envelope.hpp"
#include "core/guarded_task.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "media/media_ingest.hpp"
#include "media/media_json.hpp"
#include <algorithm>
#include <climits>
#include <cmath>
#include <mutex>
#include <thread>

namespace lfs::app {
    namespace {
        using nlohmann::json;
        using namespace media;

        json failure_json(const Error& error) {
            const auto envelope = core::to_wire_envelope(error);
            return {{"success", false}, {"error", envelope}, {"error_info", envelope}, {"error_message", envelope.at("message")}};
        }
        Error job_error(ErrorCode code, std::string message) {
            return make_error({.code = code, .domain = ErrorDomain::IO, .user_message = message, .detail = std::move(message), .detection = LFS_SOURCE_SITE_CURRENT()});
        }
        void publish(std::string phase, json data) noexcept {
            try {
                MediaExtractEvent{std::move(phase), std::move(data)}.emit();
            } catch (const std::exception& error) {
                LOG_ERROR("Media runtime event subscriber failed: {}", error.what());
            } catch (...) {
                LOG_ERROR("Media runtime event subscriber failed with an unknown exception");
            }
        }

        class ExtractionJob {
            std::mutex lifecycle_, state_mutex_;
            std::jthread worker_;
            bool closing_ = false;
            json state_{{"id", "media.extract"}, {"label", "Media Extraction"}, {"kind", "media"}, {"active", false}, {"status", "idle"}, {"stage", "idle"}, {"progress", 0.0}, {"cancel_supported", false}, {"dismiss_supported", false}, {"details", json::object()}};
            std::uint64_t generation_ = 0;

        public:
            ~ExtractionJob() { shutdown(); }
            json snapshot() {
                std::lock_guard lock(state_mutex_);
                auto result = state_;
                result["actions"] = {{"start", false}, {"pause", false}, {"resume", false}, {"cancel", state_.at("active")}, {"dismiss", false}};
                return result;
            }
            json start(IngestRequest request, FileExtraction files) {
                std::lock_guard lifecycle(lifecycle_);
                if (closing_)
                    return failure_json(job_error(ErrorCode::Unavailable, "Media extraction is shutting down"));
                if (snapshot().at("active").get<bool>())
                    return failure_json(job_error(ErrorCode::FailedPrecondition, "Media extraction is already running"));
                if (worker_.joinable())
                    worker_.join();
                {
                    std::lock_guard lock(state_mutex_);
                    state_["active"] = true;
                    state_["status"] = "running";
                    state_["stage"] = "extracting";
                    state_["outcome"] = "";
                    state_["progress"] = 0.0;
                    state_["cancel_supported"] = true;
                    state_.erase("error");
                    state_.erase("error_info");
                    state_["details"] = {{"input", core::path_to_utf8(request.input)},
                                         {"output_directory", core::path_to_utf8(files.files.output_directory)},
                                         {"generation", ++generation_},
                                         {"processed", 0},
                                         {"estimated", 0},
                                         {"discarded", 0},
                                         {"frames_accepted", 0}};
                }
                try {
                    worker_ = std::jthread([this, request = std::move(request), files = std::move(files)](std::stop_token stop) mutable {
                        bool settled = false;
                        struct ResetActive {
                            ExtractionJob& job;
                            bool& settled;
                            ~ResetActive() {
                                if (!settled) {
                                    std::lock_guard lock(job.state_mutex_);
                                    job.state_["active"] = false;
                                    job.state_["cancel_supported"] = false;
                                    job.state_["status"] = "failed";
                                }
                            }
                        } reset{*this, settled};
                        core::run_guarded<IngestReport>(
                            core::TaskContext{.name = "media.extract", .domain = ErrorDomain::IO, .operation_id = OperationId::generate(), .site = LFS_SOURCE_SITE_CURRENT()},
                            [this, &request, &files, stop]() -> Result<IngestReport> {
                                auto initial = snapshot();
                                initial["job_id"] = "media.extract";
                                publish("started", std::move(initial));
                                request.cancelled = [stop] { return stop.stop_requested(); };
                                request.progress = [this, last_event = std::chrono::steady_clock::time_point{}](const IngestProgress& progress) mutable {
                                    const auto now = std::chrono::steady_clock::now();
                                    const bool emit = now - last_event >= std::chrono::milliseconds(100) || (progress.estimated > 0 && progress.processed >= progress.estimated);
                                    json event;
                                    {
                                        std::lock_guard lock(state_mutex_);
                                        state_["progress"] = progress.estimated > 0
                                                                 ? std::clamp(static_cast<double>(progress.processed) / progress.estimated, 0.0, 1.0)
                                                                 : 0.0;
                                        state_["details"]["processed"] = progress.processed;
                                        state_["details"]["estimated"] = progress.estimated;
                                        state_["details"]["discarded"] = progress.discarded;
                                        if (emit)
                                            event = state_;
                                    }
                                    if (!emit)
                                        return;
                                    last_event = now;
                                    event["job_id"] = "media.extract";
                                    publish("progress", std::move(event));
                                };
                                return MediaIngest::extractFiles(request, files);
                            },
                            [this, &settled](Result<IngestReport>&& result) {
                                json completed;
                                std::string phase;
                                {
                                    std::lock_guard lock(state_mutex_);
                                    phase = result ? "completed" : result.error().code() == ErrorCode::Cancelled ? "cancelled"
                                                                                                                 : "failed";
                                    state_["status"] = result ? "finished" : phase;
                                    state_["stage"] = phase;
                                    state_["outcome"] = phase;
                                    state_["cancel_supported"] = false;
                                    if (result) {
                                        state_["progress"] = 1.0;
                                        state_["details"]["frames_accepted"] = result->frames_accepted;
                                        state_["details"]["discarded"] = result->discarded;
                                    } else {
                                        const auto failure = failure_json(result.error());
                                        state_["error"] = failure.at("error");
                                        state_["error_info"] = failure.at("error_info");
                                        for (const auto& frame : result.error().frames())
                                            for (const auto& field : frame.fields.entries())
                                                if (field.key == "frames_accepted")
                                                    if (const auto count = std::get_if<std::uint64_t>(&field.value))
                                                        state_["details"]["frames_accepted"] = *count;
                                    }
                                    completed = state_;
                                }
                                completed["active"] = false;
                                completed["job_id"] = "media.extract";
                                publish(phase, std::move(completed));
                                // Publish the terminal event before allowing the next generation.
                                std::lock_guard lock(state_mutex_);
                                state_["active"] = false;
                                settled = true;
                            });
                    });
                } catch (const std::exception& error) {
                    // LFS-CENSUS-OK(empty-catch): thread creation failure is retained in the job and returned as a typed wire error.
                    std::lock_guard lock(state_mutex_);
                    state_["active"] = false;
                    state_["cancel_supported"] = false;
                    state_["status"] = "failed";
                    state_["stage"] = "failed";
                    state_["outcome"] = "failed";
                    const auto failure = failure_json(job_error(ErrorCode::Unavailable, error.what()));
                    state_["error"] = failure.at("error");
                    state_["error_info"] = failure.at("error_info");
                    return failure;
                }
                return json{{"success", true}, {"job_id", "media.extract"}, {"resource_uri", "lichtfeld://runtime/jobs/media.extract"}, {"job", snapshot()}};
            }
            Result<void> cancel() {
                std::lock_guard lifecycle(lifecycle_);
                if (!snapshot().at("active").get<bool>())
                    return Result<void>::failure(job_error(ErrorCode::FailedPrecondition, "Media extraction is not running"));
                worker_.request_stop();
                return {};
            }
            void shutdown() {
                std::jthread finishing;
                {
                    std::lock_guard lifecycle(lifecycle_);
                    if (closing_)
                        return;
                    worker_.request_stop();
                    closing_ = true;
                    finishing = std::move(worker_);
                }
                // Event subscribers can request cancellation; do not join while
                // holding the lifecycle mutex they acquire.
                if (finishing.joinable())
                    finishing.join();
                std::lock_guard lifecycle(lifecycle_);
                closing_ = false;
            }
        };
        ExtractionJob& job() {
            static ExtractionJob value;
            return value;
        }

        json extraction_properties() {
            const auto number = json{{"type", "number"}};
            const auto integer = json{{"type", "integer"}, {"minimum", 0}, {"maximum", INT_MAX}};
            const auto boolean = json{{"type", "boolean"}};
            return {{"input", {{"type", "string"}}},
                    {"output_directory", {{"type", "string"}}},
                    {"selection", {{"type", "object"}, {"properties", {{"mode", {{"type", "string"}, {"enum", {"fps", "interval"}}}}, {"fps", number}, {"interval", integer}}}}},
                    {"geometry", {{"type", "object"}, {"properties", {{"mode", {{"type", "string"}, {"enum", {"original", "scale", "custom"}}}}, {"scale", number}, {"width", integer}, {"height", integer}, {"clockwise_rotation", {{"type", "integer"}, {"enum", {0, 90, 180, 270}}}}}}}},
                    {"sharpness", {{"type", "object"}, {"properties", {{"enabled", boolean}, {"threshold", number}, {"window", boolean}, {"window_candidates", integer}, {"method", {{"type", "string"}, {"enum", {"laplacian", "tenengrad", "combined"}}}}}}}},
                    {"start_seconds", number},
                    {"end_seconds", number},
                    {"convert_hdr_to_sdr", boolean},
                    {"allow_hardware_decode", boolean},
                    {"format", {{"type", "string"}, {"enum", {"png", "jpeg"}}}},
                    {"jpeg_quality", integer},
                    {"filename_pattern", {{"type", "string"}}},
                    {"write_metadata", boolean}};
        }
        std::optional<std::string> invalid_field(const json& args, const json& properties, std::string prefix = "") {
            for (const auto& [key, value] : args.items()) {
                const auto path = prefix + key;
                if (!properties.contains(key))
                    return "Unknown media parameter: " + path;
                const auto& rule = properties.at(key);
                const auto type = rule.at("type").get<std::string>();
                if ((type == "object" && !value.is_object()) || (type == "string" && !value.is_string()) ||
                    (type == "boolean" && !value.is_boolean()) || (type == "number" && !value.is_number()) ||
                    (type == "integer" && !value.is_number_integer()))
                    return "Invalid type for media parameter: " + path;
                if (rule.contains("enum") && std::find(rule.at("enum").begin(), rule.at("enum").end(), value) == rule.at("enum").end())
                    return "Invalid value for media parameter: " + path;
                if (value.is_number()) {
                    const auto number = value.get<double>();
                    if (!std::isfinite(number) || (rule.contains("minimum") && number < rule.at("minimum").get<double>()) ||
                        (rule.contains("maximum") && number > rule.at("maximum").get<double>()))
                        return "Out-of-range media parameter: " + path;
                }
                if (type == "object")
                    if (auto error = invalid_field(value, rule.at("properties"), path + "."))
                        return error;
            }
            return std::nullopt;
        }
        IngestRequest request_from_json(const json& args) {
            IngestRequest request;
            request.input = core::utf8_to_path(args.at("input").get<std::string>());
            const auto selection = args.value("selection", json::object());
            request.selection.mode = selection.value("mode", "fps") == "interval" ? SelectionMode::Interval : SelectionMode::FPS;
            request.selection.fps = selection.value("fps", 1.0);
            request.selection.interval = selection.value("interval", 1);
            const auto geometry = args.value("geometry", json::object());
            const auto mode = geometry.value("mode", "original");
            request.geometry.mode = mode == "custom" ? ResizeMode::Custom : mode == "scale" ? ResizeMode::Scale
                                                                                            : ResizeMode::Original;
            request.geometry.scale = geometry.value("scale", 1.f);
            request.geometry.width = geometry.value("width", 0);
            request.geometry.height = geometry.value("height", 0);
            request.geometry.clockwise_rotation = geometry.value("clockwise_rotation", 0);
            const auto sharpness = args.value("sharpness", json::object());
            request.sharpness.enabled = sharpness.value("enabled", false);
            request.sharpness.threshold = sharpness.value("threshold", 0.0);
            request.sharpness.window = sharpness.value("window", false);
            request.sharpness.window_candidates = sharpness.value("window_candidates", 10);
            const auto method = sharpness.value("method", "combined");
            request.sharpness.method = method == "laplacian" ? SharpnessMethod::Laplacian : method == "tenengrad" ? SharpnessMethod::Tenengrad
                                                                                                                  : SharpnessMethod::Combined;
            request.start_seconds = args.value("start_seconds", 0.0);
            request.end_seconds = args.value("end_seconds", -1.0);
            request.convert_hdr_to_sdr = args.value("convert_hdr_to_sdr", false);
            request.allow_hardware_decode = args.value("allow_hardware_decode", true);
            return request;
        }
    } // namespace

    json media_extract_job_snapshot() { return job().snapshot(); }
    Result<void> cancel_media_extract_job() { return job().cancel(); }
    void shutdown_media_extract_job() { job().shutdown(); }

    void register_media_tools(mcp::ToolRegistry& registry) {
        registry.register_tool(mcp::McpTool{
                                   .name = "media.capabilities",
                                   .description = "Inspect the registered Media Ingest capabilities and codec build",
                                   .input_schema = {.type = "object", .properties = json::object(), .required = {}},
                                   .metadata = {.category = "media", .kind = "query", .runtime = "process"}},
                               [](const json&) {
                                   const auto caps = MediaIngest::capabilities();
                                   const auto codecs = MediaIngest::codecBuildInfo();
                                   return json{{"success", true}, {"software_decode", caps.software_decode}, {"rgb8", caps.rgb8}, {"png", caps.png}, {"jpeg", caps.jpeg}, {"hardware_decode", caps.hardware_decode}, {"hdr_to_sdr", caps.hdr_to_sdr}, {"ffmpeg_version", codecs.ffmpeg_version}, {"ffmpeg_license", codecs.ffmpeg_license}, {"ffmpeg_configuration", codecs.ffmpeg_configuration}};
                               });
        registry.register_tool(mcp::McpTool{
                                   .name = "media.probe",
                                   .description = "Inspect media streams without creating output files",
                                   .input_schema = {.type = "object", .properties = {{"input", {{"type", "string"}}}, {"headers_only", {{"type", "boolean"}}}, {"timeout_ms", {{"type", "integer"}, {"minimum", 0}, {"maximum", INT_MAX}}}}, .required = {"input"}},
                                   .metadata = {.category = "media", .kind = "query", .runtime = "process"}},
                               [](const json& args) {
                                   const json properties{{"input", {{"type", "string"}}}, {"headers_only", {{"type", "boolean"}}}, {"timeout_ms", {{"type", "integer"}, {"minimum", 0}, {"maximum", INT_MAX}}}};
                                   if (const auto error = invalid_field(args, properties))
                                       return mcp::invalid_argument_result(*error, "media.probe");
                                   ProbeOptions options;
                                   options.depth = args.value("headers_only", false) ? ProbeDepth::Headers : ProbeDepth::StreamInfo;
                                   options.timeout = std::chrono::milliseconds(args.value("timeout_ms", 10000));
                                   const auto result = MediaIngest::probe(core::utf8_to_path(args.at("input").get<std::string>()), options);
                                   return result ? json{{"success", true}, {"media", json_detail::description(*result)}} : failure_json(result.error());
                               });
        registry.register_tool(mcp::McpTool{
                                   .name = "media.extract",
                                   .description = "Start media extraction; inspect, wait or cancel the media.extract runtime job",
                                   .input_schema = {.type = "object", .properties = extraction_properties(), .required = {"input", "output_directory"}},
                                   .metadata = {.category = "media", .kind = "command", .runtime = "process", .destructive = true, .long_running = true}},
                               [](const json& args) {
                                   if (const auto error = invalid_field(args, extraction_properties()))
                                       return mcp::invalid_argument_result(*error, "media.extract");
                                   FileExtraction files;
                                   files.files.output_directory = core::utf8_to_path(args.at("output_directory").get<std::string>());
                                   files.files.format = args.value("format", "png") == "jpeg" ? FrameFileFormat::JPEG : FrameFileFormat::PNG;
                                   files.files.jpeg_quality = args.value("jpeg_quality", 95);
                                   files.files.filename_pattern = args.value("filename_pattern", files.files.filename_pattern);
                                   files.write_metadata = args.value("write_metadata", false);
                                   return job().start(request_from_json(args), std::move(files));
                               });
    }
} // namespace lfs::app
