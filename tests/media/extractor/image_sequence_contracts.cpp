// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "core/image_codecs.hpp"
#include "media/image_sequence.hpp"
#include "media/media_ingest.hpp"
#include "media/metadata_observation.hpp"
#include <algorithm>
#include <chrono>
#include <fstream>
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
    std::string read(const std::filesystem::path& path) {
        std::ifstream file(path, std::ios::binary);
        require(file.good(), "Cannot read sequence output");
        return {std::istreambuf_iterator<char>(file), {}};
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
        options.write_metadata_sidecars = true;
        good(sequence.open(options));
        good(sequence.write(frame));
        good(sequence.close());
        require(fs::exists(root / "unicode" / fs::path(std::u8string(u8"rendu_\u65e5_0000.png"))), "UTF8 filename pattern changed on publication");
        require(fs::exists(root / "unicode" / fs::path(std::u8string(u8"rendu_\u65e5_0000.png.xmp"))), "UTF8 companion filename changed on publication");
        options.write_metadata_sidecars = false;
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
        // Render metadata uses output time and never trusts source-camera fields
        // supplied by a caller in FrameView::info.
        for (const auto format : {media::FrameFileFormat::PNG, media::FrameFileFormat::JPEG,
                                  media::FrameFileFormat::EXR, media::FrameFileFormat::TIFF16,
                                  media::FrameFileFormat::JXL, media::FrameFileFormat::AVIF}) {
            media::ImageSequenceOptions metadata;
            metadata.files.format = format;
            metadata.files.filename_pattern = "render_%04d";
            metadata.files.first_source_frame_number = 7;
            metadata.files.output_directory = root / ("metadata-" + std::to_string(int(format)));
            metadata.frame_count = 2;
            metadata.framerate = 30000;
            metadata.framerate_denominator = 1001;
            metadata.write_metadata_sidecars = true;
            metadata.provenance = R"({"app_version":"test<&>","build_commit":"abc","node":"\u65e5"})";
            const bool bytes = format == media::FrameFileFormat::PNG || format == media::FrameFileFormat::JPEG;
            metadata.layout = bytes ? media::FrameLayout{3, 5, 12, media::FramePixelFormat::RGB8, {media::ColorTransfer::Srgb, media::ColorPrimaries::Bt709}}
                                    : media::FrameLayout{3, 5, 3 * 4 * sizeof(float), media::FramePixelFormat::RGBAFloat32, {media::ColorTransfer::Linear, media::ColorPrimaries::Bt709, media::AlphaMode::Independent}};
            media::FrameView render{metadata.layout, {}, bytes ? std::span<const uint8_t>(pixels) : std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(linear.data()), linear.size() * sizeof(float))};
            render.info.legacy_source_frame = 999;
            render.info.source_timestamp = media::Timestamp{999, {1, 1}};
            good(sequence.open(metadata));
            good(sequence.write(render));
            good(sequence.write(render));
            good(sequence.close());
            const std::string extension = format == media::FrameFileFormat::PNG ? ".png" : format == media::FrameFileFormat::JPEG ? ".jpg"
                                                                                       : format == media::FrameFileFormat::EXR    ? ".exr"
                                                                                       : format == media::FrameFileFormat::TIFF16 ? ".tif"
                                                                                       : format == media::FrameFileFormat::JXL    ? ".jxl"
                                                                                                                                  : ".avif";
            const auto image = metadata.files.output_directory / ("render_0008" + extension);
            auto companion = image;
            companion += ".xmp";
            const auto packet = read(companion);
            require(packet.find("<lfs:MetadataPolicy>image-render-xmp-v1</lfs:MetadataPolicy>") != std::string::npos, "render policy absent");
            require(packet.find("<lfs:DeliveryIndex>1</lfs:DeliveryIndex>") != std::string::npos, "render ordinal uses filename numbering");
            require(packet.find("<lfs:OutputTimestampTicks>1</lfs:OutputTimestampTicks>") != std::string::npos &&
                        packet.find("<lfs:OutputTimeBaseNumerator>1001</lfs:OutputTimeBaseNumerator>") != std::string::npos &&
                        packet.find("<lfs:OutputTimeBaseDenominator>30000</lfs:OutputTimeBaseDenominator>") != std::string::npos,
                    "fractional output clock changed");
            require(packet.find("SourceFrame") == std::string::npos && packet.find("SourceTimestamp") == std::string::npos && packet.find("SourceMetadata") == std::string::npos,
                    "render invents source metadata");
            require(packet.find("test&lt;&amp;&gt;") != std::string::npos, "provenance XML escaping changed");
            const auto collected = media::MediaIngest::sourceSidecarMetadata(image, companion);
            good(collected);
            const auto interpreted = media::summarizeImageMetadata(collected->catalog);
            good(interpreted);
            require(interpreted->xmp_packets.size() == 1 && interpreted->xmp_packets[0].state == media::XmpPacketState::Parsed,
                    "render sidecar is not well-formed supported RDF/XML");
            require(interpreted->facts.empty(), "render sidecar was interpreted as camera metadata");
            core::image_codecs::Image decoded;
            std::string error;
            require(core::image_codecs::decode(image, decoded, error), "metadata render image did not decode");
            require(decoded.width == 3 && decoded.height == 5, "metadata changed rendered dimensions");
            const auto original_image = read(image);
            good(sequence.open(metadata));
            require(!sequence.write(render) && !sequence.close(), "render metadata ignored no-overwrite");
            require(read(image) == original_image && read(companion) == packet, "refused write changed existing pair");
            metadata.files.image_output.overwrite = metadata.files.exr.overwrite = metadata.files.tiff16.overwrite = metadata.files.jxl.overwrite = metadata.files.avif.overwrite = true;
            metadata.write_metadata_sidecars = false;
            good(sequence.open(metadata));
            good(sequence.write(render));
            good(sequence.write(render));
            good(sequence.close());
            require(!fs::exists(companion), "sidecars disabled left obsolete render metadata");
            require(read(image) == original_image, "render sidecars changed encoded image bytes");
            auto altered_policy = packet;
            const auto policy_offset = altered_policy.find("image-render-xmp-v1");
            require(policy_offset != std::string::npos, "render producer packet declares its policy");
            altered_policy[policy_offset] = 'X';
            const std::string foreign_xml = "<rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\"><rdf:Description rdf:about=\"\"><!--" +
                                            std::string(packet.size(), 'x') + "--></rdf:Description></rdf:RDF>";
            for (const auto& foreign : {std::string("foreign XMP"), std::string(packet.size(), 'x'), altered_policy, foreign_xml}) {
                std::ofstream(companion, std::ios::binary) << foreign;
                good(sequence.open(metadata));
                good(sequence.write(render));
                good(sequence.write(render));
                good(sequence.close());
                require(fs::is_regular_file(companion) && read(companion) == foreign, "render cleanup removed or changed foreign metadata, including full-length packets");
            }

            // A failed second publication retains the image but cannot accept
            // or complete the frame. No pair-atomicity is promised.
            metadata.files.output_directory /= "partial";
            metadata.write_metadata_sidecars = true;
            fs::create_directories(metadata.files.output_directory / ("render_0007" + extension + ".xmp"));
            good(sequence.open(metadata));
            require(!sequence.write(render) && !sequence.close() && sequence.framesAccepted() == 0,
                    "partial image/XMP publication reported success");
            require(fs::is_regular_file(metadata.files.output_directory / ("render_0007" + extension)), "partial publication lost the completed image");

            // Derive the cancellation boundary from the same successful image
            // writer, rather than fixing its internal callback count.
            metadata.files.output_directory /= "cancellation-control";
            metadata.write_metadata_sidecars = false;
            metadata.frame_count = 1;
            int polls = 0;
            const auto count_polls = [&] { ++polls; return false; };
            metadata.files.image_output.cancelled = count_polls;
            metadata.files.exr.cancelled = count_polls;
            metadata.files.tiff16.cancelled = count_polls;
            metadata.files.jxl.cancelled = count_polls;
            metadata.files.avif.cancelled = count_polls;
            good(sequence.open(metadata));
            good(sequence.write(render));
            good(sequence.close());
            const auto image_polls = polls;
            require(image_polls > 0, "image writer did not poll cancellation");
            metadata.files.output_directory /= "cancel-sidecar";
            metadata.write_metadata_sidecars = true;
            polls = 0;
            const auto cancel_sidecar = [&] { return ++polls > image_polls; };
            metadata.files.image_output.cancelled = cancel_sidecar;
            metadata.files.exr.cancelled = cancel_sidecar;
            metadata.files.tiff16.cancelled = cancel_sidecar;
            metadata.files.jxl.cancelled = cancel_sidecar;
            metadata.files.avif.cancelled = cancel_sidecar;
            good(sequence.open(metadata));
            const auto interrupted = sequence.write(render);
            require(!interrupted && interrupted.error().code() == ErrorCode::Cancelled && sequence.framesAccepted() == 0 && !sequence.close(),
                    "sidecar cancellation accepted/completed a frame");
            const auto partial_image = metadata.files.output_directory / ("render_0007" + extension);
            require(fs::is_regular_file(partial_image) && !fs::exists(metadata.files.output_directory / ("render_0007" + extension + ".xmp")),
                    "sidecar cancellation did not preserve only the completed image");

            // Cancellation must not leave the previous render's provenance on
            // a newly published image. Use the visible image commit as the
            // boundary, independent of internal cancellation poll counts.
            for (const bool replace_companion : {false, true}) {
                metadata.files.output_directory = root / ("existing-pair-" + std::to_string(int(format)) + (replace_companion ? "-foreign" : "-owned"));
                metadata.files.image_output.cancelled = {};
                metadata.files.exr.cancelled = {};
                metadata.files.tiff16.cancelled = {};
                metadata.files.jxl.cancelled = {};
                metadata.files.avif.cancelled = {};
                metadata.provenance = R"({"export_id":"previous-render"})";
                std::fill(pixels.begin(), pixels.end(), uint8_t(40));
                std::fill(linear.begin(), linear.end(), .25f);
                good(sequence.open(metadata));
                good(sequence.write(render));
                good(sequence.close());
                const auto replaced_image = metadata.files.output_directory / ("render_0007" + extension);
                const auto replaced_companion = metadata.files.output_directory / ("render_0007" + extension + ".xmp");
                const auto previous_pixels = read(replaced_image);
                const auto previous_packet = read(replaced_companion);
                metadata.provenance = R"({"export_id":"current-render"})";
                std::fill(pixels.begin(), pixels.end(), uint8_t(200));
                std::fill(linear.begin(), linear.end(), .75f);
                bool replaced = false;
                const auto interrupt_replacement = [&] {
                    if (read(replaced_image) == previous_pixels)
                        return false;
                    if (replace_companion && !replaced) {
                        std::ofstream foreign(replaced_companion, std::ios::binary | std::ios::trunc);
                        foreign << "foreign replacement";
                        foreign.close();
                        require(!foreign.fail(), "foreign companion replacement failed");
                        replaced = true;
                    }
                    return true;
                };
                metadata.files.image_output.cancelled = interrupt_replacement;
                metadata.files.exr.cancelled = interrupt_replacement;
                metadata.files.tiff16.cancelled = interrupt_replacement;
                metadata.files.jxl.cancelled = interrupt_replacement;
                metadata.files.avif.cancelled = interrupt_replacement;
                good(sequence.open(metadata));
                const auto result = sequence.write(render);
                require(!result && result.error().code() == ErrorCode::Cancelled && !sequence.close() && sequence.framesAccepted() == 0,
                        "interrupted pair replacement was accepted");
                require(read(replaced_image) != previous_pixels, "replacement image was not committed");
                if (replace_companion)
                    require(read(replaced_companion) == "foreign replacement", "failure cleanup removed another writer's companion");
                else
                    require(!fs::exists(replaced_companion), "failure cleanup left the previous render XMP on new pixels");
            }
            metadata.files.output_directory = root / ("orphan-no-replace-" + std::to_string(int(format)));
            metadata.files.image_output.cancelled = {};
            metadata.files.exr.cancelled = {};
            metadata.files.tiff16.cancelled = {};
            metadata.files.jxl.cancelled = {};
            metadata.files.avif.cancelled = {};
            good(sequence.open(metadata));
            good(sequence.write(render));
            good(sequence.close());
            const auto orphan_image = metadata.files.output_directory / ("render_0007" + extension);
            const auto orphan_companion = metadata.files.output_directory / ("render_0007" + extension + ".xmp");
            const auto orphan_packet = read(orphan_companion);
            fs::remove(orphan_image);
            metadata.files.image_output.overwrite = metadata.files.exr.overwrite = metadata.files.tiff16.overwrite = metadata.files.jxl.overwrite = metadata.files.avif.overwrite = false;
            good(sequence.open(metadata));
            require(!sequence.write(render) && !sequence.close() && sequence.framesAccepted() == 0,
                    "orphan no-replace companion did not refuse output");
            require(fs::is_regular_file(orphan_image) && read(orphan_companion) == orphan_packet,
                    "failure cleanup bypassed no-replace for an existing generated companion");
        }
        media::ImageSequenceOptions invalid_metadata;
        invalid_metadata.files.output_directory = root / "invalid-metadata";
        invalid_metadata.layout = {3, 5, 12, media::FramePixelFormat::RGB8, {media::ColorTransfer::Srgb, media::ColorPrimaries::Bt709}};
        invalid_metadata.frame_count = 1;
        invalid_metadata.write_metadata_sidecars = true;
        for (const std::string& invalid : std::vector<std::string>{"not JSON", "[]", "{\"a\":[[[[[[0]]]]]]}", std::string("{\"a\":\"\xff\"}")}) {
            invalid_metadata.provenance = invalid;
            const auto preflight = media::validateImageSequenceOptions(invalid_metadata);
            require(!preflight && preflight.error().code() == ErrorCode::InvalidArgument, "preflight accepted invalid render provenance");
            const auto result = sequence.open(invalid_metadata);
            require(!result && result.error().code() == ErrorCode::InvalidArgument, "invalid render provenance accepted");
            const auto diagnostic = std::string(result.error().detail());
            if (invalid == "[]")
                require(diagnostic.find("type=array") != std::string::npos, "wrong JSON type diagnostic omits observed type");
            else if (invalid.find("[[[[") != std::string::npos)
                require(diagnostic.find("depth=5, limit=4") != std::string::npos, "nesting diagnostic omits observed depth and limit");
            else
                require(diagnostic.find("byte=") != std::string::npos && diagnostic.find("size=" + std::to_string(invalid.size())) != std::string::npos,
                        "JSON parse diagnostic omits byte position and input size");
            require(preflight.error().detail() == result.error().detail(), "preflight and open report different deterministic metadata errors");
            require(!fs::exists(invalid_metadata.files.output_directory), "invalid provenance created output");
        }
        invalid_metadata.provenance.assign(32 * 1024 * 1024, 'x');
        const auto limited = sequence.open(invalid_metadata);
        require(!limited && limited.error().code() == ErrorCode::ResourceExhausted, "oversized provenance accepted");
        require(std::string(limited.error().detail()).find("size=33554432, limit=65536") != std::string::npos,
                "provenance budget diagnostic omits actual size and limit");
        require(!fs::exists(invalid_metadata.files.output_directory), "oversized provenance created output");
        invalid_metadata.provenance = "{\"x\":\"" + std::string(64 * 1024 - 8, '&') + "\"}";
        good(sequence.open(invalid_metadata));
        good(sequence.write({invalid_metadata.layout, {}, pixels}));
        good(sequence.close());
        const auto large_packet = read(invalid_metadata.files.output_directory / "frame_1.png.xmp");
        require(std::count(large_packet.begin(), large_packet.end(), '&') == 64 * 1024 - 8,
                "large provenance escaping lost data");
        const auto large_catalog = media::MediaIngest::sourceSidecarMetadata(invalid_metadata.files.output_directory / "frame_1.png",
                                                                             invalid_metadata.files.output_directory / "frame_1.png.xmp");
        good(large_catalog);
        const auto large_summary = media::summarizeImageMetadata(large_catalog->catalog, 4096, 1024 * 1024);
        good(large_summary);
        require(large_summary->xmp_packets.size() == 1 && large_summary->xmp_packets[0].state == media::XmpPacketState::Parsed,
                "maximum render provenance produced invalid XML");
        invalid_metadata.files.output_directory = root / "over-provenance-limit";
        invalid_metadata.provenance += ' ';
        require(!sequence.open(invalid_metadata) && !fs::exists(invalid_metadata.files.output_directory), "provenance limit plus one was accepted");
        invalid_metadata.files.output_directory = root / "metadata-disabled";
        invalid_metadata.write_metadata_sidecars = false;
        invalid_metadata.provenance = "not parsed when metadata is disabled";
        good(sequence.open(invalid_metadata));
        good(sequence.write({invalid_metadata.layout, {}, pixels}));
        good(sequence.close());
        require(!fs::exists(invalid_metadata.files.output_directory / "frame_1.png.xmp"), "disabled render metadata wrote a sidecar");
        std::cout << "PASS rendered sequence ordering, metadata, formats, collision, failure, cancellation and reuse\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
