// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "core/image_codecs.hpp"
#include "media/file_frame_sink.hpp"
#include "media/frame_sink.hpp"
#include "media/source_metadata.hpp"
#include "media/video_frame_extractor.hpp"
#include <chrono>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>

namespace {
    void require(bool ok, const char* message) {
        if (!ok)
            throw std::runtime_error(message);
    }
} // namespace
int runFrameSinkUnitContracts() {
    using namespace lfs::media;
    {
        lfs::io::VideoFrameExtractor engine;
        lfs::io::VideoFrameExtractor::Params params;
        params.preserve_source_metadata = true;
        std::string error;
        require(!engine.extract(params, error) && engine.lastError() && engine.lastError()->code() == lfs::ErrorCode::InvalidArgument,
                "metadata preservation without a manifest is rejected before input I/O");
        params.generate_metadata = true;
        MemoryFrameSink sink(1024, 4);
        require(!engine.extractToSink(params, sink, error) && engine.lastError() && engine.lastError()->code() == lfs::ErrorCode::InvalidArgument,
                "memory delivery cannot silently discard requested metadata preservation");
    }
    std::vector<uint8_t> pixels{1, 2, 3, 99, 99, 4, 5, 6};
    FrameView frame{{1, 2, 5, FramePixelFormat::RGB8}, {}, pixels};
    frame.info.source_timestamp = Timestamp{-42, {1, 1000}};
    auto snapshot = FrameSurface::copyOf(frame);
    require(snapshot.has_value(), "valid snapshot accepted");
    auto surface = std::move(*snapshot);
    auto copied = surface;
    pixels.assign(8, 0);
    surface = {};
    const auto view = copied.view();
    require(view.pixels[0] == 1 && view.pixels[5] == 4 && view.pixels[3] == 0 && view.pixels[4] == 0,
            "snapshot owns visible rows and zeroes padding");
    require(view.info.source_timestamp->ticks == -42, "snapshot retains signed source timestamp");
    for (int variant = 0; variant < 5; ++variant) {
        auto invalid = frame;
        if (variant == 0)
            invalid.layout.width = 0;
        if (variant == 1)
            invalid.layout.row_stride = 2;
        if (variant == 2)
            invalid.layout.row_stride = std::numeric_limits<size_t>::max();
        if (variant == 3)
            invalid.pixels = invalid.pixels.first(2);
        if (variant == 4)
            invalid.layout.format = static_cast<FramePixelFormat>(-1);
        const auto required = invalid.requiredBytes();
        require(!required && required.error().code() == lfs::ErrorCode::InvalidArgument,
                "invalid layout returns structured error without throwing");
        const auto rejected = FrameSurface::copyOf(invalid);
        require(!rejected && rejected.error().code() == lfs::ErrorCode::InvalidArgument,
                "invalid snapshot rejected before allocation/copy");
    }
    MemoryFrameSink memory(8, 2);
    require(!memory.write(view).has_value(), "inactive write rejected");
    require(memory.begin({}).has_value(), "begin accepted");
    require(!memory.begin({}).has_value(), "overlapping begin rejected");
    auto short_frame = view;
    short_frame.pixels = short_frame.pixels.first(1);
    require(memory.write(short_frame).error().code() == lfs::ErrorCode::InvalidArgument, "invalid sink layout is structured error");
    require(memory.write(view).has_value(), "within budget accepted");
    require(!memory.write(view).has_value() && memory.payloadBytes() == 8, "budget preserves accepted data");
    require(memory.write(view).error().code() == lfs::ErrorCode::ResourceExhausted, "budget uses core error taxonomy");
    memory.abort({SinkOutcome::Cancelled, 1, {}});
    require(memory.frames().size() == 1 && memory.frames()[0].view().pixels[5] == 4, "abort retains partial snapshots");
    require(memory.begin({}).has_value() && memory.frames().empty(), "reuse resets results");
    require(memory.complete({SinkOutcome::Completed, 0, {}}).has_value(), "complete accepted");
    MemoryFrameSink limited(100, 0);
    (void)limited.begin({});
    require(!limited.write(view).has_value(), "frame count bounds allocation overhead");
    const auto directory = std::filesystem::temp_directory_path() /
                           ("lfs-padded-sink-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    FileFrameSink files({directory});
    auto file_view = view;
    file_view.info.legacy_source_frame = 1;
    require(files.begin({}).has_value(), "file sink begins");
    require(files.write(file_view).has_value(), "padded RGB file write");
    lfs::core::image_codecs::Image decoded;
    std::string error;
    require(lfs::core::image_codecs::decode(directory / "frame_1.png", decoded, error), "decode padded sink PNG");
    require(decoded.width == 1 && decoded.height == 2 && decoded.data == std::vector<uint8_t>({1, 2, 3, 4, 5, 6}),
            "file sink omits padding and preserves visible rows");
    require(!files.write(file_view).has_value(), "duplicate filenames rejected");
    files.abort({SinkOutcome::Failed, 1, {}});
    std::filesystem::remove(directory / "frame_1.png");
    for (const auto format : {FrameFileFormat::JPEG, FrameFileFormat::PNG, FrameFileFormat::TIFF16, FrameFileFormat::EXR, FrameFileFormat::JXL, FrameFileFormat::AVIF}) {
        const bool tiff = format == FrameFileFormat::TIFF16;
        const bool exr = format == FrameFileFormat::EXR;
        const bool jxl = format == FrameFileFormat::JXL;
        const bool avif = format == FrameFileFormat::AVIF;
        const std::string name = avif ? "avif-metadata" : jxl                           ? "jxl-metadata"
                                                      : exr                             ? "exr-metadata"
                                                      : tiff                            ? "tiff-metadata"
                                                      : format == FrameFileFormat::JPEG ? "jpeg-metadata"
                                                                                        : "png-metadata";
        const std::array<float, 6> linear{0.f, .1f, .2f, .3f, .4f, 1.f};
        const FrameView float_view{{1, 2, 3 * sizeof(float), FramePixelFormat::RGBFloat32, {ColorTransfer::Linear, ColorPrimaries::Bt709, AlphaMode::None, ColorUnits::Relative}},
                                   file_view.info,
                                   {reinterpret_cast<const uint8_t*>(linear.data()), sizeof(linear)}};
        const auto& metadata_view = (tiff || exr || jxl || avif) ? float_view : file_view;
        FileFrameSinkOptions options;
        options.output_directory = directory / name;
        options.format = format;
        options.embed_source_metadata = true;
        FileFrameSink missing(options);
        require(!missing.begin({}) && !std::filesystem::exists(options.output_directory),
                "embedded image rejects an absent source observation before output");
        MediaDescription observed;
        observed.source_metadata.emplace();
        observed.source_metadata->tags.push_back({"comment", "owned source & <metadata>" + std::string(40 * 1024, '&')});
        auto catalog = catalogContainerMetadata(std::move(observed));
        require(catalog.has_value(), "construct explicit static source observation");
        SinkSession session;
        session.source_metadata = std::make_shared<const SourceMetadataCatalog>(std::move(*catalog));
        FileFrameSink writer(options);
        require(writer.begin(session).has_value(), "embedded image session begins");
        session = {};
        require(writer.write(metadata_view).has_value(), "embedded image owns source snapshot after begin");
        require(writer.complete({SinkOutcome::Completed, 1, {}}).has_value(), "embedded image session completes");
        const auto path = options.output_directory / (avif ? "frame_1.avif" : jxl                           ? "frame_1.jxl"
                                                                          : exr                             ? "frame_1.exr"
                                                                          : tiff                            ? "frame_1.tif"
                                                                          : format == FrameFileFormat::JPEG ? "frame_1.jpg"
                                                                                                            : "frame_1.png");
        require(lfs::core::image_codecs::decode(path, decoded, error) && decoded.width == 1 && decoded.height == 2,
                "embedded metadata leaves the image independently decodable");
        const auto original_size = std::filesystem::file_size(path);
        require(original_size > 65535, "owned provenance exceeds a JPEG APP1 payload");
        options.image_output.overwrite = false;
        FileFrameSink no_replace(options);
        MediaDescription next;
        next.source_metadata.emplace();
        auto next_catalog = catalogContainerMetadata(std::move(next));
        require(next_catalog.has_value(), "construct second static source observation");
        session.source_metadata = std::make_shared<const SourceMetadataCatalog>(*next_catalog);
        require(no_replace.begin(session).has_value() && !no_replace.write(metadata_view),
                "embedded image honors atomic no-replace");
        no_replace.abort({SinkOutcome::Failed, 0, {}});
        require(std::filesystem::file_size(path) == original_size, "failed embedded write preserves destination");
        options.image_output.overwrite = true;
        options.tiff16.overwrite = true;
        options.exr.overwrite = true;
        options.jxl.overwrite = true;
        options.avif.overwrite = true;
        options.image_output.cancelled = [] { return true; };
        options.tiff16.cancelled = options.image_output.cancelled;
        options.exr.cancelled = options.image_output.cancelled;
        options.jxl.cancelled = options.image_output.cancelled;
        options.avif.cancelled = options.image_output.cancelled;
        FileFrameSink cancelled(options);
        require(cancelled.begin(session).has_value() && !cancelled.write(metadata_view),
                "embedded image observes cancellation at atomic publication");
        cancelled.abort({SinkOutcome::Cancelled, 0, {}});
        require(std::filesystem::file_size(path) == original_size, "cancelled embedded write preserves destination");
        for (const auto& item : std::filesystem::directory_iterator(options.output_directory))
            require(item.path() == path, "embedded write failures clean temporary files");
        MediaDescription oversized;
        oversized.source_metadata.emplace();
        oversized.source_metadata->tags.push_back({"comment", std::string(6 * 1024 * 1024, '&')});
        auto oversized_catalog = catalogContainerMetadata(std::move(oversized));
        require(oversized_catalog.has_value(), "construct bounded but XML-expanding source observation");
        session.source_metadata = std::make_shared<const SourceMetadataCatalog>(std::move(*oversized_catalog));
        options.output_directory = directory / (name + "-over-budget");
        options.image_output.cancelled = {};
        options.tiff16.cancelled = {};
        options.exr.cancelled = {};
        options.jxl.cancelled = {};
        options.avif.cancelled = {};
        FileFrameSink bounded(options);
        require(bounded.begin(session).has_value(),
                "CPU metadata preparation waits for a CPU frame");
        const auto refused = bounded.write(metadata_view);
        require(!refused && refused.error().code() == lfs::ErrorCode::ResourceExhausted &&
                    std::filesystem::is_empty(options.output_directory),
                "XML-expanded provenance is bounded before output publication");
        bounded.abort({SinkOutcome::Failed, 0, {}});
        session.source_metadata =
            std::make_shared<const SourceMetadataCatalog>(*next_catalog);
        require(bounded.begin(session).has_value() &&
                    bounded.write(metadata_view).has_value() &&
                    bounded.complete({SinkOutcome::Completed, 1, {}}).has_value(),
                "aborted lazy preparation does not retain the previous catalog");
    }
    // Isolate JXL from legacy assertions and stale-file fixtures below.
    {
        FileFrameSinkOptions jxl_options;
        jxl_options.output_directory = directory / "jxl";
        jxl_options.format = FrameFileFormat::JXL;
        FileFrameSink jxl(jxl_options);
        auto declared = file_view;
        declared.layout.color = {ColorTransfer::Srgb, ColorPrimaries::Bt709, AlphaMode::None};
        require(jxl.begin({}).has_value() && jxl.write(declared).has_value(), "JXL sink accepts declared padded sRGB bytes");
        require(jxl.complete({SinkOutcome::Completed, 1, {}}).has_value(), "JXL sink completes");
        require(lfs::core::image_codecs::decode(jxl_options.output_directory / "frame_1.jxl", decoded, error), "JXL sink output is discoverable and decodable");
        require(decoded.data == std::vector<uint8_t>({1, 2, 3, 4, 5, 6}), "JXL sink preserves packed source bytes");
    }
    for (const int quality : {0, -5, 101}) {
        const auto suffix = std::to_string(quality);
        FileFrameSink png({directory, "png_" + suffix + "_%d", FrameFileFormat::PNG, quality});
        require(png.begin({}).has_value() && png.write(file_view).has_value(), "PNG ignores JPEG quality");
        (void)png.complete({SinkOutcome::Completed, 1, {}});
        FileFrameSink jpeg({directory, "jpeg_" + suffix + "_%d", FrameFileFormat::JPEG, quality});
        require(jpeg.begin({}).has_value() && jpeg.write(file_view).has_value(), "JPEG preserves default and clamping");
        (void)jpeg.complete({SinkOutcome::Completed, 1, {}});
        const auto expected_quality = quality == 0 ? 90 : quality < 1 ? 1
                                                                      : 100;
        const auto expected = directory / ("expected_" + suffix + ".jpg");
        const std::vector<uint8_t> packed{1, 2, 3, 4, 5, 6};
        require(lfs::core::image_codecs::write_jpeg(expected, packed.data(), 1, 2, 3,
                                                    expected_quality, std::nullopt, error, expected_quality > 90),
                "reference JPEG written");
        auto read_bytes = [](const auto& path) {
            std::ifstream stream(path, std::ios::binary);
            return std::vector<char>(std::istreambuf_iterator<char>(stream), {});
        };
        require(read_bytes(directory / ("jpeg_" + suffix + "_1.jpg")) == read_bytes(expected), "JPEG normalization matches writer bytes");
        std::filesystem::remove(directory / ("png_" + suffix + "_1.png"));
        std::filesystem::remove(directory / ("jpeg_" + suffix + "_1.jpg"));
        std::filesystem::remove(expected);
    }
    const auto seed = [&](const char* name) { std::ofstream(directory / name) << "existing"; };
    seed("frame_5.png");
    seed("frame_77.exr");
    seed("photo.png");
    seed("other_9.exr");
    FileFrameSinkOptions replacement;
    replacement.output_directory = directory;
    replacement.remove_stale_frames = true;
    FileFrameSink failed_replacement(replacement);
    require(failed_replacement.begin({}).has_value(), "replacement snapshot begins");
    failed_replacement.abort({SinkOutcome::Cancelled, 0, {}});
    require(std::filesystem::exists(directory / "frame_5.png") && std::filesystem::exists(directory / "frame_77.exr"), "cancel preserves obsolete frames");
    FileFrameSink successful_replacement(replacement);
    require(successful_replacement.begin({}).has_value() && successful_replacement.write(file_view).has_value(), "replacement writes frame");
    require(std::filesystem::exists(directory / "frame_77.exr"), "stale files survive until completion");
    require(successful_replacement.complete({SinkOutcome::Completed, 1, {}}).has_value(), "replacement complete cleans stale frames");
    require(!std::filesystem::exists(directory / "frame_5.png") && !std::filesystem::exists(directory / "frame_77.exr"), "obsolete generated PNG and EXR removed");
    require(std::filesystem::exists(directory / "frame_1.png") && std::filesystem::exists(directory / "photo.png") && std::filesystem::exists(directory / "other_9.exr"), "PNG replacement preserves unrelated images");
    {
        MediaDescription observed;
        observed.source_metadata.emplace();
        auto catalog = catalogContainerMetadata(std::move(observed));
        require(catalog.has_value(), "sidecar cleanup source catalog");
        SinkSession session;
        session.source_metadata = std::make_shared<const SourceMetadataCatalog>(std::move(*catalog));
        replacement.write_source_metadata_sidecars = true;
        auto seed_options = replacement;
        seed_options.remove_stale_frames = false;
        FileFrameSink seed_pair(seed_options);
        require(seed_pair.begin(session).has_value() && seed_pair.write(file_view).has_value() &&
                    seed_pair.complete({SinkOutcome::Completed, 1, {}}).has_value(),
                "produce a real source XMP packet for companion tests");
        std::ifstream packet_input(directory / "frame_1.png.xmp", std::ios::binary);
        const std::string packet(std::istreambuf_iterator<char>(packet_input), {});
        packet_input.close();
        seed("frame_5.png");
        std::ofstream(directory / "frame_5.png.xmp", std::ios::binary) << packet;
        seed("frame_6.png");
        seed("frame_6.png.xmp");
        seed("photo.png.xmp");
        FileFrameSink cancelled_pair(replacement);
        require(cancelled_pair.begin(session).has_value(), "sidecar cleanup snapshot begins");
        cancelled_pair.abort({SinkOutcome::Cancelled, 0, {}});
        require(std::filesystem::exists(directory / "frame_5.png.xmp"), "abort retains stale companion");
        FileFrameSink replaced_pair(replacement);
        require(replaced_pair.begin(session).has_value() && replaced_pair.write(file_view).has_value(), "plain sidecar replacement writes");
        require(std::filesystem::exists(directory / "frame_5.png.xmp"), "stale companion survives before complete");
        require(replaced_pair.complete({SinkOutcome::Completed, 1, {}}).has_value(), "sidecar cleanup completes");
        require(!std::filesystem::exists(directory / "frame_5.png") && !std::filesystem::exists(directory / "frame_5.png.xmp"), "stale pair removed");
        require(!std::filesystem::exists(directory / "frame_6.png") && std::filesystem::exists(directory / "frame_6.png.xmp"),
                "stale image cleanup preserves a foreign companion");
        require(std::filesystem::exists(directory / "frame_1.png.xmp") && std::filesystem::exists(directory / "photo.png.xmp"), "current and unrelated companions retained");
        {
            auto source_options = seed_options;
            source_options.output_directory = directory / "source-sidecar-failure";
            source_options.image_output.overwrite = true;
            const auto image = source_options.output_directory / "frame_1.png";
            const auto companion = source_options.output_directory / "frame_1.png.xmp";
            const auto read = [](const std::filesystem::path& path) {
                std::ifstream input(path, std::ios::binary);
                return std::string(std::istreambuf_iterator<char>(input), {});
            };
            FileFrameSink seeded(source_options);
            require(seeded.begin(session).has_value() && seeded.write(file_view).has_value() && seeded.complete({SinkOutcome::Completed, 1, {}}).has_value(),
                    "source replacement failure seed writes a pair");
            const auto original = read(image);
            std::vector<uint8_t> changed_pixels(6, 220);
            FrameView changed_frame{{1, 2, 3, FramePixelFormat::RGB8}, file_view.info, changed_pixels};
            source_options.image_output.cancelled = [&] { return read(image) != original; };
            FileFrameSink interrupted_source(source_options);
            require(interrupted_source.begin(session).has_value(), "source replacement failure begins");
            const auto interrupted = interrupted_source.write(changed_frame);
            require(!interrupted && interrupted.error().code() == lfs::ErrorCode::Cancelled && read(image) != original && !std::filesystem::exists(companion),
                    "failed source sidecar replacement removes only its superseded generated packet");
            interrupted_source.abort({SinkOutcome::Cancelled, 0, {}});
#ifdef _WIN32
            source_options.image_output.cancelled = {};
            FileFrameSink restored(source_options);
            require(restored.begin(session).has_value() && restored.write(file_view).has_value() && restored.complete({SinkOutcome::Completed, 1, {}}).has_value(),
                    "source cleanup failure seed restores its pair");
            const auto before_locked = read(image);
            std::ifstream locked;
            source_options.image_output.cancelled = [&] {
                if (read(image) == before_locked)
                    return false;
                if (!locked.is_open())
                    locked.open(companion, std::ios::binary);
                return true;
            };
            FileFrameSink blocked_cleanup(source_options);
            require(blocked_cleanup.begin(session).has_value(), "source cleanup failure begins");
            const auto blocked = blocked_cleanup.write(changed_frame);
            require(locked.is_open(), "source companion held during failure cleanup");
            locked.close();
            require(!blocked && blocked.error().code() != lfs::ErrorCode::Cancelled && std::filesystem::exists(companion),
                    "source companion cleanup failure must be reported rather than hidden by cancellation");
            blocked_cleanup.abort({SinkOutcome::Failed, 0, {}});
#endif
        }
        replacement.write_source_metadata_sidecars = false;
        seed_options.write_source_metadata_sidecars = false;
        seed_options.image_output.overwrite = false;
        FileFrameSink no_replace(seed_options);
        require(no_replace.begin({}).has_value() && !no_replace.write(file_view), "failed image replacement refuses before companion cleanup");
        no_replace.abort({SinkOutcome::Failed, 0, {}});
        require(std::filesystem::exists(directory / "frame_1.png.xmp"), "failed image replacement retains its source XMP");
        seed_options.image_output.overwrite = true;
        auto other_policy = packet;
        const auto policy = other_policy.find("image-static-source-xmp-v1");
        require(policy != std::string::npos, "producer packet declares its policy");
        other_policy[policy] = 'X';
        const std::string foreign_xml = "<rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\"><rdf:Description rdf:about=\"\"><!--" +
                                        std::string(packet.size(), 'x') + "--></rdf:Description></rdf:RDF>";
        for (const auto& foreign : {std::string(packet.size(), 'x'), other_policy, foreign_xml}) {
            std::ofstream(directory / "frame_1.png.xmp", std::ios::binary) << foreign;
            FileFrameSink preserve_foreign(seed_options);
            require(preserve_foreign.begin({}).has_value() && preserve_foreign.write(file_view).has_value(),
                    "plain replacement accepts an unrelated companion longer than the recognition prefix");
            preserve_foreign.abort({SinkOutcome::Cancelled, 1, {}});
            std::ifstream retained(directory / "frame_1.png.xmp", std::ios::binary);
            require(std::string(std::istreambuf_iterator<char>(retained), {}) == foreign,
                    "full-length foreign XML, equal-size foreign bytes and changed policy survive replacement");
        }
        // Edits made before begin retain generated ownership if the canonical
        // prefix and policy remain intact; they are not edits during output.
        std::ofstream(directory / "frame_1.png.xmp", std::ios::binary) << packet << "\n";
        FileFrameSink edited_before(seed_options);
        require(edited_before.begin({}).has_value() && edited_before.write(file_view).has_value() &&
                    !std::filesystem::exists(directory / "frame_1.png.xmp"),
                "pre-existing edits with an intact generated prefix follow generated cleanup policy");
        edited_before.abort({SinkOutcome::Cancelled, 1, {}});
        std::ofstream(directory / "frame_1.png.xmp", std::ios::binary) << packet;
        seed_options.image_output.cancelled = [] { return true; };
        FileFrameSink cancel_replace(seed_options);
        require(cancel_replace.begin({}).has_value() && !cancel_replace.write(file_view), "cancelled image replacement refuses");
        cancel_replace.abort({SinkOutcome::Cancelled, 0, {}});
        require(std::filesystem::exists(directory / "frame_1.png.xmp"), "cancelled image replacement retains its source XMP");
        seed_options.image_output.cancelled = {};
        FileFrameSink plain(seed_options);
        require(plain.begin({}).has_value() && plain.write(file_view).has_value(), "plain image replacement succeeds");
        require(!std::filesystem::exists(directory / "frame_1.png.xmp"), "superseded source XMP removed immediately after image publication");
        plain.abort({SinkOutcome::Cancelled, 1, {}});
        require(!std::filesystem::exists(directory / "frame_1.png.xmp"), "later abort cannot leave stale XMP attached to the new image");
#ifdef _WIN32
        std::ofstream(directory / "frame_1.png.xmp", std::ios::binary) << packet;
        std::ifstream companion_reader;
        seed_options.image_output.cancelled = [&] {
            if (!companion_reader.is_open())
                companion_reader.open(directory / "frame_1.png.xmp", std::ios::binary);
            return false;
        };
        FileFrameSink denied_cleanup(seed_options);
        require(denied_cleanup.begin({}).has_value(), "reader-held companion output begins");
        const auto denied = denied_cleanup.write(file_view);
        require(companion_reader.is_open(), "a Windows reader holds the companion during cleanup");
        companion_reader.close();
        require(!denied && std::filesystem::exists(directory / "frame_1.png.xmp"),
                "failed companion deletion returns an error instead of claiming complete output");
        denied_cleanup.abort({SinkOutcome::Failed, 0, {}});
        seed_options.image_output.cancelled = {};
#endif
        std::ofstream(directory / "frame_1.png.xmp", std::ios::binary) << packet;
        const auto original_time = std::filesystem::last_write_time(directory / "frame_1.png.xmp");
        std::string changed(packet.size(), 'x');
        bool exchanged = false;
        seed_options.image_output.cancelled = [&] {
            if (!exchanged) {
                exchanged = true;
                const auto changed_file = lfs::core::writeTextFileAtomically(directory / "frame_1.png.xmp", changed);
                if (!changed_file)
                    throw std::runtime_error(std::string(changed_file.error().detail()));
                std::filesystem::last_write_time(directory / "frame_1.png.xmp", original_time);
            }
            return false;
        };
        FileFrameSink exchanged_pair(seed_options);
        require(exchanged_pair.begin({}).has_value(), "image output with companion replacement begins");
        const auto exchanged_write = exchanged_pair.write(file_view);
        if (!exchanged_write)
            throw std::runtime_error(std::string(exchanged_write.error().detail()));
        exchanged_pair.abort({SinkOutcome::Cancelled, 1, {}});
        std::ifstream preserved(directory / "frame_1.png.xmp", std::ios::binary);
        require(std::string(std::istreambuf_iterator<char>(preserved), {}) == changed,
                "a replacement companion with identical size and mtime survives by file identity");
        preserved.close();
        seed_options.image_output.cancelled = {};
        std::ofstream(directory / "frame_8.png") << "old pixels";
        std::ofstream(directory / "frame_8.png.xmp", std::ios::binary) << packet;
        std::ofstream(directory / "frame_9.png") << "old pixels";
        std::ofstream(directory / "frame_9.png.xmp", std::ios::binary) << packet;
        FileFrameSink plain_cleanup(replacement);
        require(plain_cleanup.begin({}).has_value(), "snapshot stale image and unchanged companion");
        std::ofstream(directory / "frame_9.png", std::ios::app) << " changed after begin";
        require(plain_cleanup.write(file_view).has_value() &&
                    plain_cleanup.complete({SinkOutcome::Completed, 1, {}}).has_value(),
                "plain stale cleanup completes with the sidecar option disabled");
        require(!std::filesystem::exists(directory / "frame_8.png") && !std::filesystem::exists(directory / "frame_8.png.xmp"),
                "plain stale cleanup removes obsolete images and their generated XMP");
        require(std::filesystem::exists(directory / "frame_9.png"), "changed stale image survives completion");
        std::ifstream retained_stale(directory / "frame_9.png.xmp", std::ios::binary);
        require(std::string(std::istreambuf_iterator<char>(retained_stale), {}) == packet,
                "unchanged generated companion survives when its changed stale image was not removed");
    }
    {
        MediaDescription observed;
        observed.source_metadata.emplace();
        observed.source_metadata->tags.push_back({"comment", std::string(5 * 1024 * 1024, '&')});
        auto catalog = catalogContainerMetadata(std::move(observed));
        require(catalog.has_value(), "large sidecar source catalog");
        SinkSession session;
        session.source_metadata = std::make_shared<const SourceMetadataCatalog>(std::move(*catalog));
        auto oversized = replacement;
        oversized.output_directory = directory / "oversized-sidecar";
        oversized.remove_stale_frames = false;
        oversized.write_source_metadata_sidecars = true;
        FileFrameSink sink(oversized);
        require(sink.begin(session).has_value(), "large source observation remains lazy until a frame arrives");
        const auto written = sink.write(file_view);
        require(!written && written.error().code() == lfs::ErrorCode::ResourceExhausted,
                "XML escaping expansion is refused at the export budget");
        require(std::filesystem::is_empty(oversized.output_directory),
                "oversized source metadata cannot publish either pair member");
        sink.abort({SinkOutcome::Failed, 0, {}});
    }
    seed("img_1.png");
    seed("img_31.jpg");
    std::ofstream(directory / "extraction_metadata.json") << R"({"frames":[{"file":"img_1.png"},{"file":"img_31.jpg"}]})";
    FileFrameSink changed_pattern(replacement);
    require(changed_pattern.begin({}).has_value() && changed_pattern.write(file_view).has_value(), "changed pattern extraction begins");
    require(std::filesystem::exists(directory / "img_1.png"), "old pattern survives until success");
    require(changed_pattern.complete({SinkOutcome::Completed, 1, {}}).has_value(), "changed pattern extraction completes");
    require(!std::filesystem::exists(directory / "img_1.png") && !std::filesystem::exists(directory / "img_31.jpg") &&
                !std::filesystem::exists(directory / "extraction_metadata.json"),
            "old RGB pattern and stale metadata removed when metadata is off");
    seed("old_3.exr");
    seed("photo.png");
    std::ofstream(directory / "extraction_metadata.json") << R"({"frames":[{"file":"old_3.exr"},{"file":"../outside.exr"}]})";
    replacement.format = FrameFileFormat::EXR;
    replacement.preserve_metadata = true;
    std::vector<uint8_t> linear_pixels(12);
    FrameView linear_view{{1, 1, 12, FramePixelFormat::RGBFloat32}, {}, linear_pixels};
    linear_view.info.legacy_source_frame = 2;
    linear_view.layout.color.transfer = ColorTransfer::Linear;
    linear_view.layout.color.primaries = ColorPrimaries::Bt709;
    FileFrameSink exr_replacement(replacement);
    require(exr_replacement.begin({}).has_value() && exr_replacement.write(linear_view).has_value(), "EXR prior-manifest replacement begins");
    // A regenerated manifest may have the same size and coarse filesystem
    // timestamp. Its ownership must not depend on either changing.
    require(exr_replacement.complete({SinkOutcome::Completed, 1, {}}).has_value(), "EXR prior-manifest replacement completes");
    require(!std::filesystem::exists(directory / "old_3.exr") && std::filesystem::exists(directory / "photo.png") &&
                std::filesystem::exists(directory / "other_9.exr") && std::filesystem::exists(directory / "extraction_metadata.json"),
            "EXR cleans prior listed frames, keeps unrelated files and new metadata");
    {
        AvifOutputOptions options;
        size_t conversions = 0;
        options.cancelled = [&] { ++conversions; return false; };
        const auto path = directory / "invalid-xmp.avif";
        const std::array<float, 3> pixel{0.1f, 0.2f, 0.3f};
        const FrameView frame{{1, 1, sizeof(pixel), FramePixelFormat::RGBFloat32, {ColorTransfer::Linear, ColorPrimaries::Bt709, AlphaMode::None, ColorUnits::Relative}},
                              {},
                              {reinterpret_cast<const uint8_t*>(pixel.data()), sizeof(pixel)}};
        for (const auto& packet : {std::string("x\0mp", 4), std::string(32 * 1024 * 1024 + 1, 'a')}) {
            const auto result = ImageOutput::writeAvif(path, frame, options, packet);
            require(!result && result.error().code() == lfs::ErrorCode::InvalidArgument,
                    "invalid AVIF metadata rejected before conversion");
            require(conversions == 0 && !std::filesystem::exists(path),
                    "invalid AVIF metadata neither converts nor publishes");
        }
    }
    std::filesystem::remove_all(directory);
    return 0;
}
