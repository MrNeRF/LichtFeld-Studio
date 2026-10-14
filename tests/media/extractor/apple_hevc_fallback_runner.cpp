// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
// Compile the production session with a local codec-open seam. Only the
// hardware HEVC open fails; the Apple software encoder is the real provider.
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/opt.h>
}
#include <cstring>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
namespace {
    bool reject_hardware = true;
    int rejected_opens = 0;
    int fallbackCodecOpen(AVCodecContext* context, const AVCodec* codec, AVDictionary** options) {
        int64_t software = 0;
        if (reject_hardware && codec && std::strcmp(codec->name, "hevc_videotoolbox") == 0 &&
            av_opt_get_int(context->priv_data, "require_sw", 0, &software) >= 0 && !software) {
            ++rejected_opens;
            return AVERROR_EXTERNAL;
        }
        return avcodec_open2(context, codec, options);
    }
} // namespace
#define avcodec_open2 fallbackCodecOpen
#include "../../../src/media/video_encode_session.cpp"
#undef avcodec_open2
nlohmann::json runEncodeSessionContracts(const nlohmann::json& request);
int main(int argc, char** argv) {
    try {
        if (argc != 2)
            throw std::runtime_error("JSON request required");
        std::ifstream input(argv[1]);
        auto request = nlohmann::json::parse(input);
        request["hevc"] = true;
        request["videotoolbox"] = true;
        auto report = runEncodeSessionContracts(request);
        if (report.at("backend") != static_cast<int>(lfs::media::VideoEncodeBackend::VideoToolboxSoftware) || rejected_opens == 0)
            throw std::runtime_error("real Apple software fallback was not exercised");
        reject_hardware = false;
        request["output"] = request.at("output").get<std::string>() + ".recovery.mp4";
        auto recovery = runEncodeSessionContracts(request);
        if (recovery.at("backend") != static_cast<int>(lfs::media::VideoEncodeBackend::VideoToolbox))
            throw std::runtime_error("hardware provider did not recover after fallback");
        report["recovery_output"] = request["output"];
        report["recovery_backend"] = recovery["backend"];
        std::cout << report.dump() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
