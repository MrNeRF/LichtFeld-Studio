// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "core/image_codecs.hpp"
#include "media/image_sequence.hpp"
#include "media/media_ingest.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
    void require(bool value, const char* message) {
        if (!value)
            throw std::runtime_error(message);
    }
    template <class T>
    void good(const T& value) {
        if (!value)
            throw std::runtime_error(std::string(value.error().detail()));
    }
} // namespace

int main() {
    using namespace lfs;
    namespace fs = std::filesystem;
    const auto root = fs::temp_directory_path() / ("lfs-render-sequence-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    struct Cleanup {
        fs::path path;
        ~Cleanup() {
            std::error_code error;
            fs::remove_all(path, error);
        }
    } cleanup{root};
    try {
        media::ImageSequence sequence;
        media::FileExtraction extraction;
        extraction.files.numbering = media::FrameFileNumbering::DeliveryIndex;
        extraction.files.output_directory = root / "unused-extraction";
        media::IngestRequest request;
        request.input = root / "absent-video.mp4";
        const auto rejected_extraction = media::MediaIngest::extractFiles(request, extraction);
        require(!rejected_extraction && rejected_extraction.error().code() == ErrorCode::InvalidArgument,
                "extraction accepted renderer numbering without checkpoint identity");
        require(!fs::exists(extraction.files.output_directory), "invalid extraction numbering touched output");
        media::ImageSequenceOptions options;
        options.files.output_directory = root / "rgb";
        options.files.filename_pattern = "render_%04d";
        options.files.first_source_frame_number = 0;
        options.files.image_output.overwrite = false;
        options.files.image_output.durable = false;
        options.frame_count = 3;
        options.framerate = 24;
        // Odd extents and padded rows are valid for image sequences.
        options.layout = {3, 5, 12, media::FramePixelFormat::RGB8, {media::ColorTransfer::Srgb, media::ColorPrimaries::Bt709}};
        std::vector<uint8_t> pixels(57, 99);
        media::FrameView frame{options.layout, {}, pixels};
        good(sequence.open(options));
        require(!sequence.open(options), "duplicate open succeeded");
        require(sequence.isOpen(), "duplicate open destroyed active sequence");
        for (int index = 0; index < 3; ++index) {
            std::fill(pixels.begin(), pixels.end(), uint8_t(30 + index * 50));
            good(sequence.write(frame));
        }
        good(sequence.close());
        good(sequence.close());
        require(sequence.framesAccepted() == 3, "accepted count changed after close");
        for (int index = 0; index < 3; ++index) {
            const auto file = root / "rgb" / ("render_000" + std::to_string(index) + ".png");
            require(fs::file_size(file) > 0, "numbered image absent");
            core::image_codecs::Image decoded;
            std::string error;
            require(core::image_codecs::decode(file, decoded, error), "sequence image did not decode");
            require(decoded.width == 3 && decoded.height == 5 && decoded.channels == 3, "sequence dimensions/channels changed");
            require(std::all_of(decoded.data.begin(), decoded.data.end(), [index](uint8_t value) {
                        return value == 30 + index * 50;
                    }),
                    "sequence frame order or samples changed");
        }
        good(sequence.open(options));
        require(!sequence.write(frame), "collision replaced an existing image");
        require(!sequence.close(), "failed sequence reported successful close");
        options.files.output_directory = root / "incomplete";
        good(sequence.open(options));
        good(sequence.write(frame));
        sequence.abort();
        require(!sequence.close(), "explicit abort reported completion");
        options.files.output_directory = root / "incomplete-again";
        good(sequence.open(options));
        good(sequence.write(frame));
        require(!sequence.close(), "incomplete sequence reported success");
        require(fs::exists(root / "incomplete-again/render_0000.png"), "incomplete sequence lost its completed frame");

        // A failed frame does not advance the index or permit later completion.
        options.files.output_directory = root / "short";
        good(sequence.open(options));
        frame.pixels = std::span(pixels).first(8);
        require(!sequence.write(frame), "short input accepted");
        require(sequence.framesAccepted() == 0 && !sequence.close(), "short input advanced/completed sequence");
        frame.pixels = pixels;
        bool cancelled = false;
        options.cancelled = [&] { return cancelled; };
        options.files.output_directory = root / "cancel";
        good(sequence.open(options));
        good(sequence.write(frame));
        cancelled = true;
        require(!sequence.write(frame), "cancelled frame accepted");
        require(!sequence.close(), "cancelled sequence reported successful completion");
        require(!fs::exists(root / "cancel/render_0001.png"), "cancelled frame was published");
        cancelled = false;
        options.cancelled = {};

        options.files.output_directory = root / "reentrant-abort";
        bool abort_from_callback = false;
        options.cancelled = [&] {
            if (abort_from_callback)
                sequence.abort();
            return false;
        };
        good(sequence.open(options));
        abort_from_callback = true;
        require(!sequence.write(frame), "reentrant abort was ignored");
        require(!sequence.close(), "reentrant abort reported completion");
        require(!fs::exists(root / "reentrant-abort/render_0000.png"), "reentrant abort published a frame");
        options.cancelled = {};

        options.files.output_directory = root / "writer-reentrant-abort";
        bool abort_from_writer = false;
        options.files.image_output.cancelled = [&] {
            if (abort_from_writer)
                sequence.abort();
            return false;
        };
        good(sequence.open(options));
        abort_from_writer = true;
        require(!sequence.write(frame), "writer callback abort was ignored");
        require(!sequence.close(), "writer callback abort reported completion");
        require(!fs::exists(root / "writer-reentrant-abort/render_0000.png"), "writer callback abort published a frame");
        options.files.image_output.cancelled = {};

        for (const auto pattern : {"../frame_%d", "folder/frame_%d", "C:frame_%d", "folder\\frame_%d"}) {
            options.files.filename_pattern = pattern;
            require(!sequence.open(options), "unsafe or unnumbered pattern accepted");
        }
        options.files.filename_pattern = "render_%04d";
        options.files.remove_stale_frames = true;
        require(!sequence.open(options), "extraction cleanup enabled for renderer");
        options.files.remove_stale_frames = false;
        options.files.first_source_frame_number = std::numeric_limits<int>::max();
        require(!sequence.open(options), "numbering overflow accepted");
        options.files.first_source_frame_number = 0;

        options.files.output_directory = root / "unicode";
        options.files.filename_pattern = "rendu_\xe6\x97\xa5_%04d";
        options.frame_count = 1;
        good(sequence.open(options));
        good(sequence.write(frame));
        good(sequence.close());
        require(fs::exists(root / "unicode" / fs::path(std::u8string(u8"rendu_\u65e5_0000.png"))), "UTF8 filename pattern changed on publication");
        options.files.filename_pattern = "render_%04d";

        // Reuse after failure, with explicit high precision and straight alpha.
        std::vector<float> linear(3 * 5 * 4, .375f);
        for (const auto format : {media::FrameFileFormat::EXR, media::FrameFileFormat::TIFF16,
                                  media::FrameFileFormat::JXL, media::FrameFileFormat::AVIF}) {
            options.files.format = format;
            options.files.output_directory = root / ("float-" + std::to_string(int(format)));
            options.frame_count = 1;
            options.layout = {3, 5, 3 * 4 * sizeof(float), media::FramePixelFormat::RGBAFloat32, {media::ColorTransfer::Linear, media::ColorPrimaries::Bt709, media::AlphaMode::Independent}};
            frame = {options.layout, {}, {reinterpret_cast<const uint8_t*>(linear.data()), linear.size() * sizeof(float)}};
            good(sequence.open(options));
            good(sequence.write(frame));
            good(sequence.close());
        }
        options.files.format = media::FrameFileFormat::EXR;
        for (const auto units : {media::ColorUnits::Relative, media::ColorUnits::Nits, media::ColorUnits::SceneRelative}) {
            options.files.output_directory = root / ("exr-units-" + std::to_string(int(units)));
            options.layout.color.units = units;
            options.layout.color.alpha = media::AlphaMode::Premultiplied;
            frame.layout = options.layout;
            good(sequence.open(options));
            good(sequence.write(frame));
            good(sequence.close());
        }
        std::cout << "PASS rendered sequence ordering, padded/odd frames, formats, collision, failure, cancellation and reuse\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
