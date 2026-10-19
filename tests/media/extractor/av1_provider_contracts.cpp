// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
// Production session, deterministic provider absence/option rejection seams;
// actual libaom performs the encoding. No dummy encoder or fake encoded file.
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/opt.h>
}
#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <stdexcept>
namespace {
    bool hide_software = false;
    bool reject_option = false;
    bool reject_trailer = false;
    int native_lookups = 0;
    const AVCodec* findEncoder(const char* name) {
        if (std::strcmp(name, "av1_nvenc") == 0) {
            ++native_lookups;
            return nullptr;
        }
        if (hide_software && std::strcmp(name, "libaom-av1") == 0)
            return nullptr;
        return avcodec_find_encoder_by_name(name);
    }
    int writeTrailer(AVFormatContext* format) {
        return reject_trailer ? AVERROR(EIO) : av_write_trailer(format);
    }
    int setEncoderOption(void* object, const char* name, int64_t value, int flags) {
        if (reject_option && std::strcmp(name, "row-mt") == 0)
            return AVERROR_OPTION_NOT_FOUND;
        return av_opt_set_int(object, name, value, flags);
    }
} // namespace
#define avcodec_find_encoder_by_name findEncoder
#define av_opt_set_int               setEncoderOption
#define av_write_trailer             writeTrailer
#include "../../../src/media/video_encode_session.cpp"
#undef av_write_trailer
#undef av_opt_set_int
#undef avcodec_find_encoder_by_name
nlohmann::json runEncodeSessionContracts(const nlohmann::json& request);
int main() {
    using namespace lfs::media;
    const auto root = std::filesystem::temp_directory_path() /
                      ("lfs-av1-provider-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup() {
            std::error_code error;
            std::filesystem::remove_all(path, error);
        }
    } cleanup{root};
    try {
        std::filesystem::create_directory(root);
        const auto output = root / "fallback.mp4";
        const auto report = runEncodeSessionContracts({{"output", lfs::core::path_to_utf8(output)}, {"av1", true}, {"cuda", true}});
        if (report.at("backend") != static_cast<int>(VideoEncodeBackend::Software) || native_lookups == 0)
            throw std::runtime_error("real same-codec software fallback not exercised");
        const auto bytes = [&] {
            std::ifstream file(output, std::ios::binary);
            return std::string(std::istreambuf_iterator<char>(file), {});
        };
        const auto original = bytes();
        VideoEncodeSession session;
        VideoEncodeOptions options{.width = 64, .height = 48, .framerate = 10, .preferred_backend = VideoEncodeBackend::Cuda};
        options.codec = VideoEncodeCodec::AV1;
        hide_software = true;
        auto result = session.open(output, options);
        if (result || result.error().code() != lfs::ErrorCode::Unavailable || session.isOpen() ||
            result.error().detail().find("AV1") == std::string_view::npos || bytes() != original)
            throw std::runtime_error("unavailable AV1 did not preserve file and codec diagnostic");
        hide_software = false;
        reject_option = true;
        result = session.open(output, options);
        if (result || result.error().code() != lfs::ErrorCode::Unavailable || session.isOpen() ||
            result.error().detail().find("row-mt") == std::string_view::npos || bytes() != original)
            throw std::runtime_error("rejected AV1 provider option did not preserve output");
        reject_option = false;
        result = session.open(root / "recovery.mp4", options);
        if (!result || session.backend() != VideoEncodeBackend::Software)
            throw std::runtime_error("same session cannot recover after unavailable provider/options");
        struct Writer : VideoEncodeWriter {
            lfs::Result<void> write(const VideoEncodeTarget& target) override {
                for (const auto& plane : target.planes)
                    for (int row = 0; row < plane.height; ++row)
                        std::fill_n(plane.data + row * plane.row_stride, plane.width, uint8_t{128});
                return {};
            }
        } writer;
        if (!session.writeFrame(writer) || !session.close())
            throw std::runtime_error("recovered AV1 session cannot finalize");
        if (!session.open(root / "trailer-error.mp4", options) || !session.writeFrame(writer))
            throw std::runtime_error("trailer-error setup failed");
        reject_trailer = true;
        result = session.close();
        if (result || session.isOpen() || result.error().detail().find("Trailer") == std::string_view::npos || !session.close())
            throw std::runtime_error("trailer failure did not close and release the session");
        reject_trailer = false;
        if (!session.open(root / "after-trailer.mp4", options) || !session.writeFrame(writer) || !session.close())
            throw std::runtime_error("AV1 session cannot recover after trailer failure");
        std::cout << "PASS AV1 native absence, real AOM fallback, missing provider, checked options, file preservation, trailer error and recovery\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
