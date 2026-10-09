// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "core/path_utils.hpp"
#include "media/video_encode_session.hpp"
#include <algorithm>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <utility>

namespace {
    void require(bool value, const char* message) {
        if (!value)
            throw std::runtime_error(message);
    }
    struct Writer final : lfs::media::VideoEncodeWriter {
        lfs::media::VideoEncodeSession* reenter = nullptr;
        const lfs::media::VideoEncodeOptions* options = nullptr;
        std::filesystem::path path;
        bool reentrant_checked = false;
        int calls = 0;
        int luma = 32;
        bool reject = false;
        bool throwing = false;
        bool typed_exception = false;
        bool nonstandard_exception = false;
        lfs::media::VideoEncodeBackend expected_backend = lfs::media::VideoEncodeBackend::Software;
        lfs::Result<void> write(const lfs::media::VideoEncodeTarget& target) override {
            ++calls;
            if (reenter) {
                require(reenter->close().error().code() == lfs::ErrorCode::FailedPrecondition, "reentrant close rejected");
                require(reenter->open(path, *options).error().code() == lfs::ErrorCode::FailedPrecondition, "reentrant open rejected");
                require(reenter->writeFrame(*this).error().code() == lfs::ErrorCode::FailedPrecondition, "reentrant write rejected without another callback");
                bool move_rejected = false;
                try {
                    lfs::media::VideoEncodeSession moved(std::move(*reenter));
                } catch (const lfs::Exception& error) { move_rejected = error.error().code() == lfs::ErrorCode::FailedPrecondition; }
                require(move_rejected, "reentrant move rejected without transferring ownership");
                lfs::media::VideoEncodeSession destination;
                move_rejected = false;
                try {
                    destination = std::move(*reenter);
                } catch (const lfs::Exception& error) { move_rejected = error.error().code() == lfs::ErrorCode::FailedPrecondition; }
                require(move_rejected && !destination.isOpen(), "reentrant move assignment preserves ownership");
                move_rejected = false;
                try {
                    *reenter = std::move(destination);
                } catch (const lfs::Exception& error) { move_rejected = error.error().code() == lfs::ErrorCode::FailedPrecondition; }
                require(move_rejected, "reentrant destination assignment preserves active state");
                require(reenter->isOpen(), "reentrant calls preserve active session");
                reentrant_checked = true;
            }
            require(target.backend == expected_backend &&
                        target.layout == lfs::media::VideoEncodeLayout::YUV420P,
                    "CPU producer must receive YUV420P planes with the actual encoder backend");
            if (throwing)
                throw std::runtime_error("writer failure");
            if (typed_exception)
                throw lfs::Exception(lfs::make_error({.code = lfs::ErrorCode::InvalidArgument,
                                                      .domain = lfs::ErrorDomain::IO,
                                                      .detail = "typed writer failure",
                                                      .detection = LFS_SOURCE_SITE_CURRENT()}));
            if (nonstandard_exception)
                throw 42;
            if (reject)
                return lfs::Result<void>::failure(lfs::make_error({.code = lfs::ErrorCode::Cancelled,
                                                                   .domain = lfs::ErrorDomain::IO,
                                                                   .detail = "writer cancelled",
                                                                   .detection = LFS_SOURCE_SITE_CURRENT()}));
            for (int plane = 0; plane < 3; ++plane) {
                const auto& pixels = target.planes[plane];
                require(pixels.data && pixels.row_stride >= static_cast<std::size_t>(pixels.width),
                        "writable plane extent");
                for (int row = 0; row < pixels.height; ++row)
                    std::fill_n(pixels.data + row * pixels.row_stride, pixels.width,
                                static_cast<std::uint8_t>(plane == 0 ? luma : 128));
            }
            return {};
        }
    };
} // namespace

nlohmann::json runEncodeSessionContracts(const nlohmann::json& request) {
    using namespace lfs::media;
    const auto path = lfs::core::utf8_to_path(request.at("output").get<std::string>());
    VideoEncodeSession session;
    VideoEncodeOptions options{.width = 64, .height = 48, .framerate = 10, .crf = 18, .comment = "media session é 日本語"};
    options.width = request.value("width", options.width);
    options.height = request.value("height", options.height);
    if (request.value("videotoolbox", false)) {
        options.preferred_backend = VideoEncodeBackend::VideoToolbox;
        options.matrix = ColorMatrix::Smpte170M;
        options.range = ColorRange::Limited;
    }
    Writer writer;
    require(!session.isOpen(), "new session closed");
    require(session.writeFrame(writer).error().code() == lfs::ErrorCode::FailedPrecondition && writer.calls == 0,
            "inactive write never calls producer");
    require(session.close().has_value(), "inactive close is idempotent");
    for (int variant = 0; variant < 10; ++variant) {
        auto invalid = options;
        if (variant == 0)
            invalid.width = 0;
        if (variant == 1)
            invalid.width = 63;
        if (variant == 2)
            invalid.height = 47;
        if (variant == 3)
            invalid.framerate = 1001;
        if (variant == 4)
            invalid.crf = 52;
        if (variant == 5)
            invalid.preferred_backend = static_cast<VideoEncodeBackend>(99);
        if (variant == 6)
            invalid.matrix = static_cast<ColorMatrix>(999);
        if (variant == 7)
            invalid.range = static_cast<ColorRange>(999);
        if (variant == 8)
            invalid.matrix = static_cast<ColorMatrix>(3);
        if (variant == 9)
            invalid.matrix = static_cast<ColorMatrix>(15);
        auto result = session.open(path, invalid);
        require(!result && result.error().code() == lfs::ErrorCode::InvalidArgument,
                "invalid options rejected before opening output");
        const auto detail = result.error().detail();
        if (variant == 3)
            require(detail.find("1001") != std::string_view::npos, "fps diagnostic includes rejected value");
        if (variant == 4)
            require(detail.find("52") != std::string_view::npos, "CRF diagnostic includes rejected value");
        if (variant == 5)
            require(detail.find("99") != std::string_view::npos, "backend diagnostic includes rejected value");
        require(!std::filesystem::exists(path), "invalid options do not create output");
    }
    const auto unavailable_path = path.parent_path() / "missing-parent" / "video.mp4";
    auto unavailable = session.open(unavailable_path, options);
    require(!unavailable && unavailable.error().code() == lfs::ErrorCode::Unavailable && !session.isOpen(),
            "output failure cleans up the session before retry");
    auto opened = session.open(path, options);
    if (!opened)
        throw std::runtime_error(std::string(opened.error().detail()));
    require(session.isOpen(), "session open");
    const auto backend = session.backend();
    if (!request.value("videotoolbox", false))
        require(backend == VideoEncodeBackend::Software, "software session open");
    writer.expected_backend = backend;
    require(session.open(path, options).error().code() == lfs::ErrorCode::FailedPrecondition, "overlapping open rejected");
    writer.reenter = &session;
    writer.options = &options;
    writer.path = path;
    writer.reject = true;
    require(session.writeFrame(writer).error().code() == lfs::ErrorCode::Cancelled, "producer error preserved");
    writer.reject = false;
    writer.throwing = true;
    require(session.writeFrame(writer).error().code() == lfs::ErrorCode::Internal, "producer exception is structured");
    writer.throwing = false;
    writer.typed_exception = true;
    const auto typed = session.writeFrame(writer);
    require(!typed && typed.error().code() == lfs::ErrorCode::InvalidArgument &&
                typed.error().detail() == "typed writer failure",
            "typed producer exception is preserved");
    writer.typed_exception = false;
    writer.nonstandard_exception = true;
    require(session.writeFrame(writer).error().code() == lfs::ErrorCode::Internal, "nonstandard producer exception is structured");
    writer.nonstandard_exception = false;
    auto moved = std::move(session);
    require(!session.isOpen() && moved.isOpen() && session.close().has_value(), "session ownership transfers safely");
    VideoEncodeSession destination;
    destination = std::move(moved);
    require(!moved.isOpen() && destination.isOpen(), "move assignment transfers session ownership");
    writer.reenter = &destination;
    writer.options = &options;
    writer.path = path;
    for (int frame = 0; frame < 4; ++frame) {
        writer.luma = 32 + frame * 32;
        auto result = destination.writeFrame(writer);
        if (!result)
            throw std::runtime_error(std::string(result.error().detail()));
    }
    require(writer.reentrant_checked, "callback reentrancy exercised");
    require(destination.close().has_value() && !destination.isOpen() && destination.close().has_value(), "flush and repeat close");
    require(destination.writeFrame(writer).error().code() == lfs::ErrorCode::FailedPrecondition, "write after close rejected");
    nlohmann::json report{{"success", true}, {"frames", 4}, {"writer_calls", writer.calls}, {"comment", options.comment}, {"backend", static_cast<int>(backend)}};
    if (request.value("verify_apple_recovery", false)) {
        require(backend == VideoEncodeBackend::VideoToolboxSoftware, "8K must use Apple's software encoder");
        auto small = options;
        small.width = 64;
        small.height = 48;
        const auto recovery_path = path.parent_path() / "apple-recovery.mp4";
        require(destination.open(recovery_path, small).has_value(), "same session reopens at a hardware-supported extent");
        require(destination.backend() == VideoEncodeBackend::VideoToolbox, "same session returns to Apple hardware after fallback");
        Writer recovery_writer;
        recovery_writer.expected_backend = VideoEncodeBackend::VideoToolbox;
        require(destination.writeFrame(recovery_writer).has_value(), "recovered hardware session accepts a frame");
        require(destination.close().has_value(), "recovered hardware session flushes");
        report["recovery_backend"] = static_cast<int>(VideoEncodeBackend::VideoToolbox);
        report["recovery_output"] = lfs::core::path_to_utf8(recovery_path);
    }
    return report;
}
