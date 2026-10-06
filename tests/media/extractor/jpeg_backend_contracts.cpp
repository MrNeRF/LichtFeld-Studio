// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "core/path_utils.hpp"
#include "io/media/media_ingest.hpp"
#include "media/media_backends.hpp"
#include <array>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace {
    std::string mode;
    std::vector<std::uint8_t> jpeg;
    int factory_calls = 0, batches = 0;
    class FaultEncoder final : public lfs::media::detail::GpuJpegEncoder {
        std::array<int, 2> slots{};

    public:
        std::size_t capacity() const noexcept override { return slots.size(); }
        bool canConvertHardware() const noexcept override { return false; }
        void convertHardware(const AVFrame*, std::uint8_t*) override { throw std::runtime_error("unexpected hardware frame"); }
        void finishHardware() override {}
        void* queueHardware(std::size_t) override { throw std::runtime_error("unexpected hardware queue"); }
        void* queueHost(std::size_t i, const std::uint8_t*) override { return &slots.at(i); }
        std::vector<std::vector<std::uint8_t>> encode(const std::vector<void*>& frames, int, int, int) override {
            ++batches;
            if (mode == "throw-later" && batches == 2)
                throw std::runtime_error("injected second batch failure");
            std::vector<std::vector<std::uint8_t>> result(frames.size(), jpeg);
            if (mode == "short")
                result.pop_back();
            if (mode == "empty")
                result.back().clear();
            return result;
        }
    };
    std::unique_ptr<lfs::media::detail::GpuJpegEncoder> factory(const lfs::media::detail::JpegSettings&) {
        ++factory_calls;
        return mode == "unavailable" ? nullptr : std::make_unique<FaultEncoder>();
    }
} // namespace

nlohmann::json runJpegBackendContracts(const nlohmann::json& request) {
    using namespace lfs;
    mode = request.at("backend_mode").get<std::string>();
    std::ifstream reference(core::utf8_to_path(request.at("encoded_reference").get<std::string>()), std::ios::binary);
    jpeg.assign(std::istreambuf_iterator<char>(reference), {});
    if (jpeg.empty())
        throw std::runtime_error("missing reference JPEG");
    media::detail::registerGpuJpegFactory(factory);
    struct Reset {
        ~Reset() { media::detail::registerGpuJpegFactory(nullptr); }
    } reset;
    media::IngestRequest input;
    input.input = core::utf8_to_path(request.at("input").get<std::string>());
    input.selection.mode = media::SelectionMode::Interval;
    input.selection.interval = 1;
    input.end_seconds = .35;
    input.allow_hardware_decode = request.value("allow_hardware", true);
    media::FileExtraction files;
    files.files.output_directory = core::utf8_to_path(request.at("output").get<std::string>());
    files.files.filename_pattern = "frame_%03d";
    files.files.format = media::FrameFileFormat::JPEG;
    media::MemoryFrameSink memory;
    auto result = mode == "memory" ? media::MediaIngest::extract(input, memory) : media::MediaIngest::extractFiles(input, files);
    std::uint64_t accepted = result ? result->frames_accepted : 0;
    if (!result) {
        for (const auto& frame : result.error().frames())
            for (const auto& field : frame.fields.entries())
                if (field.key == "frames_accepted")
                    accepted = std::get<std::uint64_t>(field.value);
    }
    return {{"success", result.has_value()}, {"accepted", accepted}, {"factory_calls", factory_calls}, {"batches", batches}, {"error", result ? "" : result.error().detail()}};
}
