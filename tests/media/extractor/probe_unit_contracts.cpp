// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "core/image_codecs.hpp"
#include "core/path_utils.hpp"
#include "core/source_metadata.hpp"
#include "media/decoded_video_frame_ffmpeg.hpp"
#include "media/dji_timed_metadata.hpp"
#include "media/media_ingest.hpp"
#include "media/media_json.hpp"
#include "media/media_probe_ffmpeg.hpp"
#include "media/video_player.hpp"
extern "C" {
#include <libavformat/avformat.h>
#include <libavutil/display.h>
}
#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string_view>

namespace {
    void require(bool condition, std::string_view message) {
        if (!condition)
            throw std::runtime_error(std::string(message));
    }

    void sourceExifContracts() {
        using namespace lfs::core::image_codecs;
        struct Directory {
            std::filesystem::path path;
            ~Directory() {
                std::error_code ec;
                std::filesystem::remove_all(path, ec);
            }
        } directory{std::filesystem::temp_directory_path() /
                    ("lfs-source-exif-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()))};
        require(std::filesystem::create_directory(directory.path), "create metadata fixture directory");

        {
            // Independent classic TIFF fixture: RGB framing plus an opaque unknown
            // tag. Pixels deliberately live beyond the metadata range.
            std::vector<uint8_t> tiff_bytes{'I', 'I', 42, 0, 8, 0, 0, 0};
            auto put = [&](uint32_t value, unsigned width) {
                for (unsigned i = 0; i < width; ++i)
                    tiff_bytes.push_back(static_cast<uint8_t>(value >> (8 * i)));
            };
            put(10, 2);
            auto tag = [&](uint16_t id, uint16_t type, uint32_t count, uint32_t value) {
                put(id, 2);
                put(type, 2);
                put(count, 4);
                put(value, 4);
            };
            tag(256, 4, 1, 1);
            tag(257, 4, 1, 1);
            tag(258, 3, 1, 8);
            tag(259, 3, 1, 1);
            tag(262, 3, 1, 2);
            tag(273, 4, 1, 1024);
            tag(274, 3, 1, 6);
            tag(277, 3, 1, 3);
            tag(279, 4, 1, 3);
            tag(65000, 7, 4, 0xff004100);
            put(0, 4);
            const auto tiff_path = directory.path / "source-metadata.bin";
            auto write_tiff_fixture = [&](size_t count) {
                std::ofstream file(tiff_path, std::ios::binary | std::ios::trunc);
                file.write(reinterpret_cast<const char*>(tiff_bytes.data()), static_cast<std::streamsize>(count));
            };
            write_tiff_fixture(tiff_bytes.size());
            auto tiff = read_source_exif(tiff_path);
            require(tiff.status == MetadataStatus::Collected && tiff.tiff && tiff.blocks.empty() &&
                        tiff.tiff->directories.size() == 1,
                    "TIFF metadata does not require available pixel strips");
            const auto archived = tiff;
            // Unconfigured pixel compression is not damaged metadata and must
            // not require installing a decoder just to return the read prefix.
            tiff_bytes[8 + 2 + 3 * 12 + 8] = 0x51;
            tiff_bytes[8 + 2 + 3 * 12 + 9] = 0xc3; // WebP compression, 50001.
            write_tiff_fixture(tiff_bytes.size());
            const auto unconfigured = read_source_exif(tiff_path);
            require(unconfigured.status == MetadataStatus::Collected || unconfigured.status == MetadataStatus::Unsupported,
                    "unconfigured TIFF pixel compression is unsupported, not malformed");
            tiff_bytes[8 + 2 + 3 * 12 + 8] = 1;
            tiff_bytes[8 + 2 + 3 * 12 + 9] = 0;
            write_tiff_fixture(tiff_bytes.size());
            for (const auto& range : tiff.tiff->source_ranges)
                require(range.offset <= tiff_bytes.size() && range.bytes.size() <= tiff_bytes.size() - range.offset &&
                            std::equal(range.bytes.begin(), range.bytes.end(), tiff_bytes.begin() + range.offset),
                        "TIFF source spans preserve exact bytes and original offsets");
            require(std::any_of(tiff.tiff->directories[0].tags.begin(), tiff.tiff->directories[0].tags.end(), [](const auto& entry) {
                        return entry.id == 65000 && entry.bytes == std::vector<uint8_t>{0, 0x41, 0, 0xff};
                    }),
                    "TIFF unknown byte tags retain NUL and invalid UTF8 through libtiff");
            MetadataLimits tiff_limits;
            tiff_limits.jpeg_header_bytes = 0;
            tiff_limits.png_chunks = 0;
            tiff_limits.blocks = 0;
            require(read_source_exif(tiff_path, tiff_limits).status == MetadataStatus::Collected,
                    "TIFF ignores unrelated JPEG and PNG budgets");
            tiff_limits = {};
            tiff_limits.tiff_reads = 1;
            const auto limited_tiff = read_source_exif(tiff_path, tiff_limits);
            require(limited_tiff.status == MetadataStatus::Limited && limited_tiff.tiff &&
                        !limited_tiff.tiff->source_ranges.empty(),
                    "TIFF source budget retains complete header spans");
            tiff_limits = {};
            tiff_limits.tiff_entries = 1;
            tiff = read_source_exif(tiff_path, tiff_limits);
            require(tiff.status == MetadataStatus::Limited && tiff.tiff->directories.size() == 1 &&
                        tiff.tiff->directories[0].tags.size() == 1,
                    "TIFF entry budget retains decoded prefix");
            tiff_limits = {};
            tiff_limits.tiff_value_bytes = 7;
            require(read_source_exif(tiff_path, tiff_limits).status == MetadataStatus::Limited,
                    "TIFF checks decoded value storage before copying");
            tiff_limits = {};
            size_t stored_values = 0;
            for (const auto& entry : archived.tiff->directories[0].tags)
                stored_values += entry.bytes.size() + 8 * (entry.unsigned_values.size() + entry.signed_values.size() + entry.floating_values.size());
            tiff_limits.tiff_value_bytes = stored_values;
            require(read_source_exif(tiff_path, tiff_limits).status == MetadataStatus::Collected,
                    "TIFF decoded-value accounting accepts the exact owned-storage budget");
            --tiff_limits.tiff_value_bytes;
            const auto value_prefix = read_source_exif(tiff_path, tiff_limits);
            require(value_prefix.status == MetadataStatus::Limited && !value_prefix.tiff->directories[0].tags.empty(),
                    "TIFF decoded-value accounting rejects one byte below the owned-storage budget and retains its prefix");
            tiff_limits = {};
            tiff_limits.total_bytes = 8;
            require(read_source_exif(tiff_path, tiff_limits).status == MetadataStatus::Limited,
                    "TIFF original range copies obey aggregate byte budget");
            for (size_t count = 4; count < tiff_bytes.size(); ++count) {
                write_tiff_fixture(count);
                const auto truncated = read_source_exif(tiff_path);
                require(truncated.status != MetadataStatus::Collected && truncated.status != MetadataStatus::Absent,
                        "every incomplete TIFF directory framing is diagnosed");
            }
            write_tiff_fixture(tiff_bytes.size());
            // A libtiff-reported cycle is bounded and never silently treated as EOF.
            tiff_bytes[tiff_bytes.size() - 4] = 8;
            write_tiff_fixture(tiff_bytes.size());
            tiff = read_source_exif(tiff_path);
            require(tiff.status == MetadataStatus::Malformed,
                    "TIFF directory cycles are malformed");
            require(!tiff.tiff->directories.empty(), "TIFF cycle retains the first complete directory");
            std::filesystem::remove(tiff_path);
            require(!archived.tiff->source_ranges.empty() && archived.tiff->directories.size() == 1,
                    "TIFF originals and decoded tags own their storage after source removal");
        }
        const auto path = directory.path / "source.jpg";
        const uint8_t pixel[]{20, 80, 160};
        std::string error;
        require(write_jpeg(path, pixel, 1, 1, 3, 95, {}, error), error);
        require(read_source_exif(path).status == MetadataStatus::Absent, "JPEG without EXIF is absent");
        MetadataLimits jpeg_limits;
        jpeg_limits.jpeg_header_bytes = 0;
        require(read_source_exif(path, jpeg_limits).status == MetadataStatus::Limited,
                "JPEG still rejects an invalid JPEG header budget");
        jpeg_limits = {};
        jpeg_limits.png_scan_bytes = 0;
        jpeg_limits.png_chunks = 0;
        require(read_source_exif(path, jpeg_limits).status == MetadataStatus::Absent,
                "JPEG collection does not depend on unrelated PNG budgets");
        std::ifstream input(path, std::ios::binary);
        const std::vector<uint8_t> jpeg{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
        input.close();
        const std::vector<uint8_t> exif{'E', 'x', 'i', 'f', 0, 0, 'I', 'I', 42, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        auto write = [&](const std::vector<uint8_t>& bytes) {
            std::ofstream output(path, std::ios::binary | std::ios::trunc);
            output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
            require(static_cast<bool>(output), "write metadata fixture");
        };
        std::vector<uint8_t> enriched{0xff, 0xd8};
        for (int duplicate = 0; duplicate < 2; ++duplicate) {
            enriched.insert(enriched.end(), {0xff, 0xe1, 0, static_cast<uint8_t>(exif.size() + 2)});
            enriched.insert(enriched.end(), exif.begin(), exif.end());
        }
        enriched.insert(enriched.end(), jpeg.begin() + 2, jpeg.end());
        write(enriched);
        const auto metadata = read_source_exif(path);
        require(metadata.status == MetadataStatus::Collected && metadata.blocks.size() == 2,
                "ordered duplicate APP1 blocks retained");
        require(metadata.scope == "jpeg-before-sos" && metadata.blocks[0].bytes == exif && metadata.blocks[1].bytes == exif,
                "exact original envelope retained");
        require(!metadata.blocks[0].image_item && !metadata.blocks[0].metadata_item,
                "JPEG has no fabricated HEIF item association");
        MetadataLimits limits;
        limits.blocks = 1;
        const auto limited = read_source_exif(path, limits);
        require(limited.status == MetadataStatus::Limited && limited.blocks.size() == 1,
                "block count limit preserves collected prefix without completeness claim");
        limits = {};
        limits.total_bytes = exif.size() - 1;
        require(read_source_exif(path, limits).status == MetadataStatus::Limited,
                "aggregate budget checked before retained payload copy");
        limits = {};
        limits.jpeg_header_bytes = 16;
        require(read_source_exif(path, limits).status == MetadataStatus::Limited, "header window exhaustion explicit");
        Image decoded;
        require(decode(path, decoded, error) && decoded.width == 1 && decoded.height == 1,
                "metadata inspection does not alter JPEG decode");
        enriched.insert(enriched.end(), {0xff, 0xe1, 0, static_cast<uint8_t>(exif.size() + 2)});
        enriched.insert(enriched.end(), exif.begin(), exif.end());
        write(enriched);
        require(read_source_exif(path).blocks.size() == 2, "JPEG metadata scope does not claim markers after SOS");
        write({0xff, 0xd8, 0xff, 0xe1, 0, 100, 'E'});
        require(read_source_exif(path).status == MetadataStatus::Malformed, "truncated header is malformed");
        require(metadata.blocks[0].bytes == exif, "owned bytes survive source replacement");
        write({1, 2, 3});
        require(read_source_exif(path).status == MetadataStatus::Unsupported, "unsupported source is not absent");
        require(read_source_exif(directory.path / "missing.jpg").status == MetadataStatus::IoError,
                "missing file is not absent metadata");
        require(read_source_exif(directory.path).status == MetadataStatus::IoError,
                "directory input is an I/O failure rather than absent or unsupported metadata");
        // Camera-sized files must not turn an unrelated header warning into
        // window exhaustion, or discard completed APP1 blocks on any failure.
        const size_t prefix_size = 2 + 2 * (exif.size() + 4);
        auto check_prefix = [&](const std::vector<uint8_t>& bytes, MetadataStatus expected, size_t blocks, const MetadataLimits& bounds = {}) {
            write(bytes);
            const auto collected = read_source_exif(path, bounds);
            require(collected.status == expected, "JPEG header outcome matches the actual failure");
            require(collected.blocks.size() == blocks, "JPEG failure preserves only completed EXIF blocks");
            for (const auto& block : collected.blocks)
                require(block.bytes == exif, "JPEG collected prefix remains byte-exact");
        };
        std::vector<uint8_t> prefix(enriched.begin(), enriched.begin() + prefix_size);
        auto warned = prefix;
        warned.insert(warned.end(), {1, 2, 3});
        warned.insert(warned.end(), jpeg.begin() + 2, jpeg.end());
        warned.resize(9 * 1024 * 1024);
        check_prefix(warned, MetadataStatus::Malformed, 2);
        auto long_header = prefix;
        for (int i = 0; i < 21; ++i) {
            long_header.insert(long_header.end(), {0xff, 0xe2, 0xff, 0xff});
            long_header.resize(long_header.size() + 65533);
        }
        long_header.insert(long_header.end(), jpeg.begin() + 2, jpeg.end());
        long_header.resize(9 * 1024 * 1024);
        check_prefix(long_header, MetadataStatus::Limited, 2);
        auto fatal = prefix;
        fatal.insert(fatal.end(), {0xff, 0xc0, 0, 2});
        fatal.resize(9 * 1024 * 1024);
        check_prefix(fatal, MetadataStatus::Malformed, 2);
        std::vector<uint8_t> truncated(prefix.begin(), prefix.begin() + 2 + exif.size() + 4);
        truncated.insert(truncated.end(), {0xff, 0xe1, 0, 100, 'E', 'x', 'i', 'f', 0, 0});
        check_prefix(truncated, MetadataStatus::Malformed, 1);
        limits = {};
        limits.jpeg_header_bytes = truncated.size();
        check_prefix(truncated, MetadataStatus::Malformed, 1, limits);
        truncated.resize(2 * 1024 * 1024);
        limits.jpeg_header_bytes -= 1;
        check_prefix(truncated, MetadataStatus::Limited, 1, limits);

        // Independent PNG chunk construction: bitwise reference CRC rather
        // than the production zlib implementation.
        auto chunk = [](const char* type, const std::vector<uint8_t>& payload) {
            std::vector<uint8_t> bytes;
            auto put = [&](uint32_t value) {
                for (int shift = 24; shift >= 0; shift -= 8)
                    bytes.push_back(static_cast<uint8_t>(value >> shift));
            };
            put(static_cast<uint32_t>(payload.size()));
            bytes.insert(bytes.end(), type, type + 4);
            bytes.insert(bytes.end(), payload.begin(), payload.end());
            uint32_t crc = 0xffffffff;
            for (size_t i = 4; i < bytes.size(); ++i) {
                crc ^= bytes[i];
                for (int bit = 0; bit < 8; ++bit)
                    crc = (crc >> 1) ^ ((crc & 1) ? 0xedb88320 : 0);
            }
            put(~crc);
            return bytes;
        };
        const std::vector<uint8_t> tiff(exif.begin() + 6, exif.end());
        std::vector<uint8_t> png{137, 'P', 'N', 'G', 13, 10, 26, 10};
        auto append = [&](const char* type, const std::vector<uint8_t>& payload) {
            const auto bytes = chunk(type, payload);
            png.insert(png.end(), bytes.begin(), bytes.end());
        };
        append("IHDR", {0, 0, 0, 1, 0, 0, 0, 1, 8, 2, 0, 0, 0});
        append("eXIf", tiff);
        const auto png_prefix = png;
        append("IDAT", {1, 2, 3}); // Intentionally not valid deflate: no pixel decoding.
        append("IEND", {});
        write(png);
        const auto png_metadata = read_source_exif(path);
        require(png_metadata.status == MetadataStatus::Collected && png_metadata.blocks.size() == 1 &&
                    png_metadata.scope == "png-chunks-through-iend" && png_metadata.blocks[0].bytes == tiff &&
                    png_metadata.blocks[0].envelope == ExifEnvelope::PngExif,
                "PNG is signature detected and retains bare TIFF without decoding pixels");
        limits = {};
        limits.jpeg_header_bytes = 0;
        require(read_source_exif(path, limits).status == MetadataStatus::Collected,
                "PNG collection does not depend on the unrelated JPEG header budget");
        limits = {};
        limits.png_scan_bytes = png.size() - 3;
        require(read_source_exif(path, limits).status == MetadataStatus::Collected,
                "exact PNG scan budget succeeds without charging skipped pixel payload");
        --limits.png_scan_bytes;
        require(read_source_exif(path, limits).status == MetadataStatus::Limited, "PNG scan exhaustion is explicit");
        limits = {};
        limits.png_chunks = 2;
        require(read_source_exif(path, limits).status == MetadataStatus::Limited && read_source_exif(path, limits).blocks.size() == 1,
                "PNG chunk limit preserves complete EXIF prefix");
        limits = {};
        limits.block_bytes = tiff.size() - 1;
        require(read_source_exif(path, limits).status == MetadataStatus::Limited && read_source_exif(path, limits).blocks.empty(),
                "PNG oversized EXIF is rejected before copying");
        const auto valid_png = png;
        auto oversized_declaration = png_prefix;
        oversized_declaration.insert(oversized_declaration.end(), {0x10, 0, 0, 0, 'e', 'X', 'I', 'f', 0, 0, 0, 0});
        write(oversized_declaration);
        require(read_source_exif(path).status == MetadataStatus::Malformed && read_source_exif(path).blocks.size() == 1,
                "bogus PNG chunk length is truncation, not a budget failure; completed prefix survives");
        for (size_t size = 8; size < valid_png.size(); ++size) {
            write(std::vector<uint8_t>(valid_png.begin(), valid_png.begin() + size));
            require(read_source_exif(path).status == MetadataStatus::Malformed, "PNG truncation is never absent or collected");
        }
        png = valid_png;
        png.back() ^= 1;
        write(png);
        require(read_source_exif(path).status == MetadataStatus::Malformed && read_source_exif(path).blocks.size() == 1,
                "late CRC failure preserves complete metadata prefix");
        png = png_prefix;
        append("eXIf", tiff);
        append("IDAT", {});
        append("IEND", {});
        write(png);
        require(read_source_exif(path).status == MetadataStatus::Malformed && read_source_exif(path).blocks.size() == 2,
                "duplicate PNG EXIF is preserved without claiming a conforming container");
        png.assign(valid_png.begin(), valid_png.begin() + 33);
        append("IDAT", {});
        append("eXIf", tiff);
        append("IEND", {});
        write(png);
        require(read_source_exif(path).status == MetadataStatus::Malformed && read_source_exif(path).blocks.size() == 1,
                "post-IDAT EXIF is preserved with explicit nonconformance");
        png.assign(valid_png.begin(), valid_png.begin() + 33);
        append("IDAT", {});
        append("IEND", {});
        write(png);
        require(read_source_exif(path).status == MetadataStatus::Absent, "complete PNG without EXIF is absent");
        require(png_metadata.blocks[0].bytes == tiff, "PNG original bytes outlive the source file");
        {
            std::ofstream output(path, std::ios::binary | std::ios::trunc);
            output.write(reinterpret_cast<const char*>(png_prefix.data()), static_cast<std::streamsize>(png_prefix.size()));
            const uint8_t header[]{0, 32, 0, 0, 'I', 'D', 'A', 'T'};
            output.write(reinterpret_cast<const char*>(header), sizeof(header));
            output.seekp(2 * 1024 * 1024, std::ios::cur);
            const uint8_t ignored_crc[4]{};
            output.write(reinterpret_cast<const char*>(ignored_crc), sizeof(ignored_crc));
            const auto end = chunk("IEND", {});
            output.write(reinterpret_cast<const char*>(end.data()), static_cast<std::streamsize>(end.size()));
            require(static_cast<bool>(output), "write large skipped PNG pixel payload");
        }
        limits = {};
        limits.png_scan_bytes = png_prefix.size() + 24;
        const auto skipped = read_source_exif(path, limits);
        require(skipped.status == MetadataStatus::Collected && skipped.blocks.size() == 1 && skipped.blocks[0].bytes == tiff,
                "large PNG pixel payload does not consume the metadata examination budget or require a valid pixel CRC");
        for (const size_t size : {0u, 1u, 8191u, 8192u, 8193u, 65536u}) {
            png = png_prefix;
            append("IDAT", std::vector<uint8_t>(size, 7));
            append("IEND", {});
            write(png);
            limits = {};
            limits.png_scan_bytes = png_prefix.size() + 24;
            const auto boundary = read_source_exif(path, limits);
            require(boundary.status == MetadataStatus::Collected && boundary.blocks.size() == 1 && boundary.blocks[0].bytes == tiff,
                    "buffered discard and seeking preserve originals and identical metadata byte charges at the threshold");
            --limits.png_scan_bytes;
            require(read_source_exif(path, limits).status == MetadataStatus::Limited,
                    "both pixel skip paths still charge the CRC and next chunk framing");
            png.resize(png_prefix.size() + 8 + size / 2);
            write(png);
            const auto truncated_skip = read_source_exif(path);
            require(truncated_skip.status == MetadataStatus::Malformed && truncated_skip.blocks.size() == 1 && truncated_skip.blocks[0].bytes == tiff,
                    "truncated pixel chunks preserve the complete EXIF prefix on either skip path");
        }
        for (const bool with_exif : {false, true}) {
            png.assign(valid_png.begin(), valid_png.begin() + 33);
            if (with_exif)
                append("eXIf", tiff);
            for (int i = 0; i < 5000; ++i)
                append("IDAT", {});
            append("IEND", {});
            write(png);
            limits = {};
            limits.png_chunks = with_exif ? 3 : 2;
            limits.png_scan_bytes = png.size();
            const auto segmented = read_source_exif(path, limits);
            require(segmented.status == (with_exif ? MetadataStatus::Collected : MetadataStatus::Absent) &&
                        segmented.blocks.size() == (with_exif ? 1 : 0),
                    "pixel segmentation does not consume the non-IDAT chunk budget");
            --limits.png_scan_bytes;
            require(read_source_exif(path, limits).status == MetadataStatus::Limited,
                    "even empty IDAT headers and CRC bytes consume the scan budget");
            limits.png_scan_bytes = png.size();
            --limits.png_chunks;
            require(read_source_exif(path, limits).status == MetadataStatus::Limited,
                    "non-IDAT chunks still enforce the independent count budget");
        }
        auto damaged_text = chunk("tEXt", {'k', 0, 'v'});
        damaged_text.back() ^= 1;
        png.assign(valid_png.begin(), valid_png.begin() + 33);
        for (int i = 0; i < 16; ++i)
            png.insert(png.end(), damaged_text.begin(), damaged_text.end());
        append("eXIf", tiff);
        append("IDAT", {});
        append("IEND", {});
        write(png);
        const auto recovered = read_source_exif(path);
        require(recovered.status == MetadataStatus::Malformed && recovered.blocks.size() == 1 && recovered.blocks[0].bytes == tiff,
                "damaged ancillary CRCs do not hide later intact EXIF");
        require(recovered.diagnostic == "PNG chunk CRC mismatch", "repeated ancillary CRC failures do not grow diagnostics");
        png[32] ^= 1;
        write(png);
        require(read_source_exif(path).status == MetadataStatus::Malformed && read_source_exif(path).blocks.empty(),
                "damaged critical CRC remains terminal");
        png.assign(valid_png.begin(), valid_png.begin() + 33);
        png.insert(png.end(), damaged_text.begin(), damaged_text.end());
        png.insert(png.end(), {0, 0, 0, 0, '!', 'X', 'I', 'f', 0, 0, 0, 0});
        append("eXIf", tiff);
        append("IDAT", {});
        append("IEND", {});
        write(png);
        require(read_source_exif(path).status == MetadataStatus::Malformed && read_source_exif(path).blocks.empty(),
                "ancillary recovery never searches past invalid next-chunk framing");
    }
    void djiProfileContracts() {
        using namespace lfs::media;
        TimedMetadataObservation broken;
        broken.codec_tag = MKTAG('d', 'j', 'm', 'd');
        broken.packets.resize(2);
        broken.packets[0].bytes = {10, 27, 10, 25, 10, 16, 'd', 'v', 't', 'm', '_', 'a', 'c', '2', '0', '4', '.', 'p', 'r', 'o', 't', 'o', 26, 5, '2', '.', '0', '.', '1'};
        const auto reject = [&](const std::vector<std::uint8_t>& bytes, lfs::ErrorCode code, std::string_view detail) {
            broken.packets[1].bytes = bytes;
            const auto refused = interpretDjiMetadata(broken);
            require(!refused && refused.error().code() == code && refused.error().detail().find(detail) != std::string_view::npos,
                    "malformed packet reaches the intended diagnostic after a qualified first packet");
        };
        // A complete nested timestamp is valid. Every nonempty proper prefix
        // cuts a tag, length or payload inside its enclosing length-delimited field.
        const std::vector<std::uint8_t> timestamp{0x1a, 4, 0x0a, 2, 0x10, 1};
        broken.packets[1].bytes = timestamp;
        const auto complete = interpretDjiMetadata(broken);
        require(complete && (*complete)[1].camera_timestamp == 1, "complete timestamp control is accepted");
        reject({}, lfs::ErrorCode::DataLoss, "Empty DJI packet");
        for (std::size_t size = 1; size < timestamp.size(); ++size)
            reject(std::vector<std::uint8_t>(timestamp.begin(), timestamp.begin() + size), lfs::ErrorCode::DataLoss, "Truncated");
        reject({0}, lfs::ErrorCode::DataLoss, "Invalid DJI protobuf field number");
        reject({0x1a, 13, 0x0a, 11, 0x10, 255, 255, 255, 255, 255, 255, 255, 255, 255, 2},
               lfs::ErrorCode::DataLoss, "Overflowing DJI protobuf varint");
        reject({10, 255, 255, 255, 255, 127}, lfs::ErrorCode::DataLoss, "Truncated DJI protobuf field");
        reject({0x1a, 2, 0x12, 4}, lfs::ErrorCode::DataLoss, "Truncated DJI protobuf field");
        reject({0x1a, 9, 0x12, 7, 0x1a, 5, 0x0d, 0, 0, 0x80, 0x7f}, lfs::ErrorCode::DataLoss, "Nonfinite DJI scalar");
        reject({0x1a, 8, 0x12, 6, 0x22, 4, 0x0a, 2, 1, 0}, lfs::ErrorCode::DataLoss, "nonzero denominator");
        reject({0x1a, 9, 0x12, 7, 0x22, 5, 0x0a, 3, 1, 2, 3}, lfs::ErrorCode::DataLoss, "Too many DJI exposure operands");
        reject({0x1a, 6, 0x12, 4, 0x1a, 2, 8, 1}, lfs::ErrorCode::DataLoss, "Wrong wire type");
        reject({0x0b}, lfs::ErrorCode::Unsupported, "groups are unsupported");
        broken.packets[1].bytes = {0x1a, 13, 0x0a, 11, 0x10, 255, 255, 255, 255, 255, 255, 255, 255, 255, 1};
        const auto maximum = interpretDjiMetadata(broken);
        require(maximum && (*maximum)[1].camera_timestamp == std::numeric_limits<std::uint64_t>::max(),
                "ten-byte uint64 maximum is accepted without losing bits");
        broken.packets[1].bytes = {0x1a, 6, 0x0a, 4, 0x10, 1, 0x10, 2};
        const auto repeated = interpretDjiMetadata(broken);
        require(repeated && (*repeated)[1].camera_timestamp == 2, "last singular scalar wins");
        broken.packets[1].bytes = {0x1a, 4, 0x12, 2, 0x4a, 0, 0x7a, 3, 0xff, 0xff, 0xff};
        const auto unknown = interpretDjiMetadata(broken);
        require(unknown && (*unknown)[1].quaternion_wxyz == std::array<float, 4>{} && !(*unknown)[1].iso,
                "present messages use proto3 zero defaults while absent messages stay absent");
    }
    void timedPacketContracts() {
        using namespace lfs::media;
        const auto path = std::filesystem::temp_directory_path() / ("lfs-timed-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".mp4");
        struct Fixture {
            std::filesystem::path path;
            AVFormatContext* context = nullptr;
            ~Fixture() {
                if (context) {
                    if (context->pb)
                        avio_closep(&context->pb);
                    avformat_free_context(context);
                }
                try {
                    std::error_code ignored;
                    std::filesystem::remove(path, ignored);
                } catch (const std::exception&) {}
            }
        } fixture{path};
        const auto native = lfs::core::path_to_native_utf8(path);
        require(avformat_alloc_output_context2(&fixture.context, nullptr, "mp4", native.c_str()) >= 0 && fixture.context, "allocate timed metadata fixture muxer");
        for (int i = 0; i < 3; ++i) {
            auto* stream = avformat_new_stream(fixture.context, nullptr);
            require(stream != nullptr, "allocate explicit subtitle stream");
            stream->codecpar->codec_type = i == 2 ? AVMEDIA_TYPE_DATA : AVMEDIA_TYPE_SUBTITLE;
            stream->codecpar->codec_id = i == 2 ? AV_CODEC_ID_BIN_DATA : AV_CODEC_ID_MOV_TEXT;
            if (i == 2)
                stream->codecpar->codec_tag = MKTAG('g', 'p', 'm', 'd');
            stream->time_base = {1, 1000};
        }
        require(avio_open(&fixture.context->pb, native.c_str(), AVIO_FLAG_WRITE) >= 0 && avformat_write_header(fixture.context, nullptr) >= 0, "write metadata-only fixture header");
        const std::array<std::uint8_t, 7> payload{0, 5, 'h', 'e', 0, 255, 'o'};
        const auto append = [&](int stream, std::int64_t pts) {
            AVPacket packet{};
            packet.data = const_cast<std::uint8_t*>(payload.data());
            packet.size = static_cast<int>(payload.size());
            packet.stream_index = stream;
            packet.pts = packet.dts = pts;
            packet.duration = 500;
            av_packet_rescale_ts(&packet, {1, 1000}, fixture.context->streams[stream]->time_base);
            std::array<std::uint8_t, 30> configuration{};
            configuration.back() = 255;
            if (stream == 0 && pts == 500) {
                auto* side = av_packet_new_side_data(&packet, AV_PKT_DATA_NEW_EXTRADATA, configuration.size());
                require(side != nullptr, "allocate changed subtitle configuration");
                std::memcpy(side, configuration.data(), configuration.size());
            }
            const auto written = av_write_frame(fixture.context, &packet);
            av_packet_free_side_data(&packet);
            require(written >= 0, "mux raw subtitle packet without an encoder");
        };
        append(0, 0);
        append(1, 0);
        append(0, 500);
        append(2, 0);
        require(av_write_trailer(fixture.context) >= 0 && avio_closep(&fixture.context->pb) >= 0, "finish timed metadata fixture");
        ProbeOptions options;
        options.depth = ProbeDepth::Headers;
        const auto ordinary = MediaProbe::inspect(path, options);
        require(ordinary && !ordinary->timed_metadata && !json_detail::description(*ordinary).contains("timed_metadata"), "ordinary probe remains free of timed packet collection");
        options.timed_metadata_stream = 0;
        options.collect_source_metadata = true;
        const auto observed = MediaProbe::inspect(path, options);
        require(observed && observed->timed_metadata && observed->timed_metadata->packets.size() == 2, "explicit stream selection retains only its packets");
        const auto& timed = *observed->timed_metadata;
        require(timed.provider && timed.provider->name == "ffmpeg", "packet observation identifies its provider");
        for (const auto& packet : timed.packets)
            require(packet.bytes == std::vector<std::uint8_t>(payload.begin(), payload.end()) && packet.pts && packet.dts, "raw subtitle bytes retain NUL/nonUTF8 and original clocks");
        require(timed.packets[0].pts->ticks == 0 && timed.packets[1].pts->ticks * timed.packets[1].pts->time_base.numerator * 1000 / timed.packets[1].pts->time_base.denominator == 500, "packet clocks are not replaced by a frame-rate estimate");
        options.timed_metadata_stream = 1;
        const auto alternate = MediaProbe::inspect(path, options);
        require(alternate && alternate->timed_metadata && alternate->timed_metadata->packets.size() == 1, "second track is independently selectable");
        options.timed_metadata_stream = 2;
        const auto binary = MediaProbe::inspect(path, options);
        require(binary && binary->timed_metadata && binary->timed_metadata->codec_tag == MKTAG('g', 'p', 'm', 'd') && binary->timed_metadata->packets.size() == 1 && binary->timed_metadata->packets[0].bytes == std::vector<std::uint8_t>(payload.begin(), payload.end()), "binary data tracks are observed without interpreting a vendor schema");
        options.timed_metadata_stream = 3;
        require(!MediaProbe::inspect(path, options), "missing track is refused without guessing another stream");
        options.timed_metadata_stream = 0;
        options.depth = ProbeDepth::StreamInfo;
        require(!MediaProbe::inspect(path, options), "packet scan refuses codec-probing mode");
        options.depth = ProbeDepth::Headers;
        options.timed_metadata_limits.records = 1;
        const auto limited = MediaProbe::inspect(path, options);
        require(!limited && limited.error().code() == lfs::ErrorCode::ResourceExhausted, "packet record exhaustion does not return a truncated successful scan");
        std::size_t retained = timed.extradata.size(), records = 0, side_blocks = 0;
        for (const auto& packet : timed.packets) {
            retained += packet.bytes.size();
            ++records;
            for (const auto& side : packet.side_data) {
                retained += side.bytes.size();
                ++records;
                ++side_blocks;
                require(side.type == AV_PKT_DATA_NEW_EXTRADATA && !side.bytes.empty(), "changed subtitle configuration is preserved as owned packet side data");
            }
        }
        require(side_blocks > 0, "fixture must actually exercise packet side data");
        options.timed_metadata_limits.records = records;
        options.timed_metadata_limits.bytes = retained;
        require(MediaProbe::inspect(path, options).has_value(), "exact byte and record limits accept the complete observation");
        --options.timed_metadata_limits.bytes;
        require(!MediaProbe::inspect(path, options), "one byte below aggregate metadata retention is refused");
        options.timed_metadata_limits.bytes = retained;
        --options.timed_metadata_limits.records;
        require(!MediaProbe::inspect(path, options), "side data shares the selected packet record budget");
        options.timed_metadata_limits = {};
        for (auto limits : {TimedMetadataLimits{.bytes = 0}, TimedMetadataLimits{.records = timed_metadata_maxima.records + 1}, TimedMetadataLimits{.records = 0}}) {
            options.timed_metadata_limits = limits;
            require(!MediaProbe::inspect(path, options), "invalid timed packet limits are refused");
        }
        const auto refused = catalogContainerMetadata(*observed);
        require(!refused && refused.error().detail().find("timed packet") != std::string_view::npos, "static catalogs refuse timed observations even alongside valid static metadata");
        require(!MediaIngest::sourceContainerMetadata(path, options), "static collection refuses timed options before reading packets");
        options.timed_metadata_limits = {};
        auto sidecar = path;
        sidecar.replace_extension(".srt");
        Fixture sidecar_fixture{sidecar};
        std::ofstream(sidecar, std::ios::binary) << "00:00:00,000 --> 00:00:01,000\nFrameCnt: 1 [latitude: 1]\n";
        require(!MediaProbe::inspect(sidecar, options), "embedded collection never invokes a standalone subtitle demuxer");
        require(MediaIngest::sourceSrtMetadata(sidecar).has_value(), "standalone subtitles remain supported by their single existing parser");
        const auto serialized = json_detail::dump(json_detail::description(*observed));
        require(nlohmann::json::parse(serialized)["timed_metadata"]["packets"][0]["bytes"] == "0005686500ff6f", "shared JSON retains opaque binary payload exactly");
        // A truncated container must not be reported as an empty successful scan.
        std::filesystem::resize_file(path, 16);
        const auto truncated = MediaProbe::inspect(path, options);
        require(!truncated && truncated.error().code() == lfs::ErrorCode::DataLoss, "truncated MOV headers produce a typed failure without a partial observation");
        std::filesystem::remove(path);
        require(observed->timed_metadata->packets[0].bytes.size() == payload.size(), "packet observation owns bytes after input removal");
    }
} // namespace

int runProbeUnitContracts() {
    sourceExifContracts();
    timedPacketContracts();
    djiProfileContracts();
    using namespace lfs::media;
    const auto* dav1d = avcodec_find_decoder_by_name("libdav1d");
    require(detail::softwareVideoDecoder(AV_CODEC_ID_AV1) ==
                (dav1d ? dav1d : avcodec_find_decoder(AV_CODEC_ID_AV1)),
            "AV1 prefers dav1d and retains the registered fallback when absent");
    require(detail::softwareVideoDecoder(AV_CODEC_ID_H264) == avcodec_find_decoder(AV_CODEC_ID_H264),
            "AV1 selection does not change other codec choices");
    require(MediaProbe::inspect({}).error().code() == lfs::ErrorCode::InvalidArgument, "empty path rejected");
    const auto optionFailure = MediaProbe::inspect({});
    require(!optionFailure && !optionFailure.error().native(), "option failure has no native FFmpeg status");
    const auto copied = optionFailure;
    require(copied.error().domain() == lfs::ErrorDomain::IO &&
                !copied.error().frames().empty() && !copied.error().detail().empty(),
            "structured error owns diagnostics and detection site after copy");
    auto nulPath = std::filesystem::path("bad").native();
    nulPath.push_back(0);
    nulPath += std::filesystem::path("suffix").native();
    require(MediaProbe::inspect(std::filesystem::path(nulPath)).error().code() == lfs::ErrorCode::InvalidArgument, "embedded NUL rejected before path conversion");
    ProbeOptions invalidOptions;
    invalidOptions.depth = static_cast<ProbeDepth>(-1);
    require(MediaProbe::inspect("unused", invalidOptions).error().code() == lfs::ErrorCode::InvalidArgument, "invalid depth rejected");
    auto release = [](AVFormatContext* context) { avformat_free_context(context); };
    std::unique_ptr<AVFormatContext, decltype(release)> context(avformat_alloc_context(), release);
    require(context != nullptr, "allocate test context");
    auto* audio = avformat_new_stream(context.get(), nullptr);
    auto* video = avformat_new_stream(context.get(), nullptr);
    auto* second = avformat_new_stream(context.get(), nullptr);
    require(audio && video && second, "allocate test streams");
    audio->codecpar->codec_type = AVMEDIA_TYPE_AUDIO;
    audio->codecpar->codec_id = AV_CODEC_ID_PCM_S16LE;
    for (auto* stream : {video, second}) {
        stream->codecpar->codec_type = AVMEDIA_TYPE_VIDEO;
        stream->codecpar->codec_id = AV_CODEC_ID_FFV1;
        stream->codecpar->width = 64;
        stream->codecpar->height = 32;
        stream->time_base = {1, 1000};
        stream->start_time = AV_NOPTS_VALUE;
        stream->duration = AV_NOPTS_VALUE;
        stream->avg_frame_rate = {0, 0};
        stream->r_frame_rate = {0, 1};
    }
    context->duration = AV_NOPTS_VALUE;
    auto description = detail::describeContext(context.get());
    require(description.streams.size() == 3 && description.selected_video_stream == 1, "all streams and first usable video");
    require(!description.duration && !description.streams[1].duration && !description.streams[1].start, "unknown timestamps stay unknown");
    require(!description.streams[1].average_frame_rate && !description.streams[1].nominal_frame_rate, "no invented FPS");
    require(!description.streams[1].color.component_depth && !description.streams[1].color.transfer, "no invented pixel depth or transfer");
    require(!description.streams[1].orientation.clockwise_degrees, "absent rotation is unknown");
    require(audio->discard == AVDISCARD_DEFAULT && second->discard == AVDISCARD_DEFAULT, "description has no discard side effects");
    video->start_time = -40;
    video->duration = 0;
    auto stream = detail::describeStream(video);
    require(stream.start && stream.start->ticks == -40 && stream.start->time_base.denominator == 1000, "signed raw start preserved");
    require(stream.duration && stream.duration->ticks == 0, "zero duration is known");
    video->time_base = {1, 0};
    require(!detail::describeStream(video).start, "invalid timebase is unknown");
    video->time_base = {1, 1000};

    int32_t matrix[9];
    av_display_rotation_set(matrix, 90);
    auto* side = av_packet_side_data_new(&video->codecpar->coded_side_data,
                                         &video->codecpar->nb_coded_side_data, AV_PKT_DATA_DISPLAYMATRIX, sizeof(matrix), 0);
    require(side && side->data, "allocate display matrix");
    std::memcpy(side->data, matrix, sizeof(matrix));
    stream = detail::describeStream(video);
    require(stream.orientation.display_matrix && legacyQuarterTurn(stream.orientation) == 90, "clockwise display matrix rotation");
    require(stream.orientation.reflected == false, "ordinary rotation is not reflected");
    av_display_matrix_flip(matrix, 1, 0);
    auto reflected = detail::describeDisplayMatrix(reinterpret_cast<unsigned char*>(matrix), sizeof(matrix));
    require(reflected.reflected == true && reflected.display_matrix, "mirror and raw matrix preserved");
    auto truncated = detail::describeDisplayMatrix(reinterpret_cast<unsigned char*>(matrix), sizeof(matrix) - 1);
    require(!truncated.display_matrix && !truncated.clockwise_degrees, "truncated matrix never read");
    std::memset(matrix, 0, sizeof(matrix));
    auto degenerate = detail::describeDisplayMatrix(reinterpret_cast<unsigned char*>(matrix), sizeof(matrix));
    require(!degenerate.clockwise_degrees && !degenerate.reflected, "degenerate matrix produces no angle");
    av_dict_set(&video->metadata, "rotate", "-90", 0);
    stream = detail::describeStream(video);
    require(stream.orientation.source == OrientationSource::RotateTag && stream.orientation.rotate_tag == "-90" &&
                stream.orientation.display_matrix && legacyQuarterTurn(stream.orientation) == 270,
            "tag precedence preserves matrix and negative angle");
    for (const std::string invalid : {std::string("90junk"), std::string("nan"), std::string("inf"), std::string(400, '9')}) {
        av_dict_set(&video->metadata, "rotate", invalid.c_str(), 0);
        require(!detail::describeStream(video).orientation.clockwise_degrees, "invalid tag is unknown");
    }
    av_dict_set(&video->metadata, "rotate", " +90.9 ", 0);
    stream = detail::describeStream(video);
    require(stream.orientation.clockwise_degrees == 90.9 && legacyQuarterTurn(stream.orientation) == 90,
            "valid signed/spaced fractional tags preserve legacy truncation");
    av_dict_set(&video->metadata, "rotate", "45", 0);
    stream = detail::describeStream(video);
    require(stream.orientation.clockwise_degrees == 45 && legacyQuarterTurn(stream.orientation) == 0, "raw non-quarter angle retained without unsupported preview rotation");
    for (const auto bytes : {Orientation::max_rotate_tag_bytes - 1, Orientation::max_rotate_tag_bytes,
                             Orientation::max_rotate_tag_bytes + 1}) {
        const auto value = std::string(bytes - 2, ' ') + "90";
        require(av_dict_set(&video->metadata, "rotate", value.c_str(), 0) >= 0, "set bounded rotation fixture");
        stream = detail::describeStream(video);
        const bool limited = bytes > Orientation::max_rotate_tag_bytes;
        require(stream.orientation.rotate_tag_limited == limited &&
                    stream.orientation.rotate_tag.has_value() == !limited,
                "rotation text retention has an exact inclusive byte bound");
        require(stream.orientation.clockwise_degrees == 90 && legacyQuarterTurn(stream.orientation) == 90,
                "omitting oversized text does not change numeric or legacy rotation");
        auto observed = detail::describeContext(context.get());
        detail::collectContextMetadata(context.get(), observed);
        const auto catalog = catalogContainerMetadata(std::move(observed));
        require(bool(catalog), "bounded rotation catalog");
        const auto summary = summarizeSourceMetadata(*catalog);
        require(summary && summary->limited == limited, "orientation text omission remains visible in the summary");
        require(catalog->container()->source_metadata->streams[1].tags.back().value == value,
                "raw tag collection retains its independent byte budget and exact original");
        const auto json = json_detail::sourceMetadata(*catalog);
        require(json["description"]["streams"][1]["orientation"].contains("rotate_tag_limited") == limited,
                "normal orientation JSON is unchanged and omitted text is explicit");
    }
    const auto invalid_large = std::string(Orientation::max_rotate_tag_bytes + 1, '9');
    require(av_dict_set(&video->metadata, "rotate", invalid_large.c_str(), 0) >= 0, "set oversized invalid rotation");
    stream = detail::describeStream(video);
    require(stream.orientation.rotate_tag_limited && !stream.orientation.clockwise_degrees &&
                legacyQuarterTurn(stream.orientation) == 0,
            "oversized malformed rotation stays unknown without matrix fallback");
    detail::keepOnlyStream(context.get(), 0);
    require(audio->discard == AVDISCARD_DEFAULT && video->discard == AVDISCARD_ALL && second->discard == AVDISCARD_ALL,
            "existing discard helper keeps an explicitly selected non-video stream");
    detail::keepOnlyStream(context.get(), 1);
    require(audio->discard == AVDISCARD_ALL && second->discard == AVDISCARD_ALL && video->discard == AVDISCARD_DEFAULT, "legacy discard policy");
    require(detail::findUsableHeaderVideoStream(nullptr) == -1, "null context");
    require(detail::describeContext(nullptr).streams.empty(), "null description");
    require(!detail::describeDecodedVideoFrame(nullptr), "missing decoded frame rejected");
    auto releaseFrame = [](AVFrame* f) { av_frame_free(&f); };
    std::unique_ptr<AVFrame, decltype(releaseFrame)> decoded(av_frame_alloc(), releaseFrame);
    require(decoded != nullptr, "allocate descriptor frame");
    decoded->format = AV_PIX_FMT_YUV420P10LE;
    decoded->width = 65;
    decoded->height = 33;
    decoded->color_primaries = AVCOL_PRI_BT2020;
    decoded->color_trc = AVCOL_TRC_SMPTE2084;
    decoded->colorspace = AVCOL_SPC_BT2020_NCL;
    decoded->color_range = AVCOL_RANGE_MPEG;
    require(av_frame_get_buffer(decoded.get(), 64) >= 0, "allocate padded descriptor planes");
    auto* master = av_mastering_display_metadata_create_side_data(decoded.get());
    require(master != nullptr, "allocate mastering metadata");
    master->has_luminance = 1;
    master->min_luminance = {1, 1000};
    master->max_luminance = {1000, 1};
    auto* plus = av_dynamic_hdr_plus_create_side_data(decoded.get());
    require(plus != nullptr, "allocate HDR10+ metadata");
    plus->num_windows = 1;
    plus->params[0].maxscl[0] = {1, 10};
    plus->params[0].maxscl[1] = {1, 5};
    plus->params[0].maxscl[2] = {3, 10};
    plus->params[0].average_maxrgb = {1, 100};
    auto view = detail::describeDecodedVideoFrame(decoded.get(), video);
    require(view.has_value(), "describe padded HDR frame");
    require(view->plane_count == 3 && view->component_count == 3 && view->components[0].depth == 10, "component description");
    require(view->planes[1].width == 33 && view->planes[1].height == 17 && view->planes[0].pitch == decoded->linesize[0], "odd chroma extent and byte pitches preserved");
    require(view->planes[0].data == decoded->data[0] && view->color_trc == ColorTransfer::Pq && view->color_primaries == ColorPrimaries::Bt2020, "borrowed storage and independent colour enums");
    require(view->frame_hdr.mastering && view->frame_hdr.mastering->max_luma == 1000.f && view->frame_hdr.mastering->min_luma == .001f, "mastering luminance precision");
    require(view->frame_hdr.hdr10_plus && view->frame_hdr.hdr10_plus->maxscl == std::array{.1, .2, .3} && view->frame_hdr.hdr10_plus->average_maxrgb == .01, "HDR10+ normalized values");
    auto invalid_view = *view;
    invalid_view.planes[1].pitch = 1;
    auto invalid_storage = validateDecodedVideoFrame(invalid_view);
    require(!invalid_storage && invalid_storage.error().detail().find("plane=1") != std::string_view::npos && invalid_storage.error().detail().find("pitch=1") != std::string_view::npos && invalid_storage.error().detail().find("row_bytes=66") != std::string_view::npos, "stride diagnostic includes plane, observed pitch and row bytes");
    invalid_view = *view;
    invalid_view.components[1].plane = 4;
    require(!validateDecodedVideoFrame(invalid_view), "invalid component indices rejected before indexing");
    invalid_view = *view;
    invalid_view.frame_hdr.hdr10_plus->num_anchors = 16;
    require(!validateDecodedVideoFrame(invalid_view), "HDR10+ count cannot exceed metadata storage");
    invalid_view = *view;
    invalid_view.planes[0].data += (invalid_view.planes[0].height - 1) * invalid_view.planes[0].pitch;
    invalid_view.planes[0].pitch *= -1;
    require(validateDecodedVideoFrame(invalid_view).has_value(), "negative software pitch remains representable");
    const auto invalid_format = decoded->format;
    decoded->format = -100;
    require(!detail::describeDecodedVideoFrame(decoded.get()), "invalid pixel format rejected");
    decoded->format = invalid_format;
    // Metadata-only hardware fixtures: no device or decoder is initialized.
    std::unique_ptr<AVFrame, decltype(releaseFrame)> hardware(av_frame_alloc(), releaseFrame);
    require(hardware != nullptr, "allocate hardware descriptor fixture");
    hardware->format = AV_PIX_FMT_VIDEOTOOLBOX;
    hardware->width = 64;
    hardware->height = 32;
    hardware->hw_frames_ctx = av_buffer_allocz(sizeof(AVHWFramesContext));
    require(hardware->hw_frames_ctx != nullptr, "allocate hardware metadata fixture");
    auto* hardware_context = reinterpret_cast<AVHWFramesContext*>(hardware->hw_frames_ctx->data);
    hardware_context->sw_format = AV_PIX_FMT_NV12;
    std::uint8_t first_slot = 0, pixel_buffer = 0;
    hardware->data[0] = &first_slot;
    hardware->data[3] = &pixel_buffer;
    auto hw_view = detail::describeDecodedVideoFrame(hardware.get());
    require(hw_view && hw_view->hardware && hw_view->hardware_handle == &pixel_buffer,
            "VideoToolbox identity comes from data[3], even when data[0] is present");
    require(hw_view->plane_count == 2 && hw_view->format_name == "nv12",
            "hardware view retains logical software layout");
    require(hw_view->planes[0].data == nullptr && hw_view->planes[1].data == nullptr &&
                hw_view->planes[0].pitch == 0 && hw_view->planes[1].pitch == 0,
            "hardware handles never masquerade as CPU plane storage");
    hardware->data[3] = nullptr;
    require(!detail::describeDecodedVideoFrame(hardware.get()),
            "VideoToolbox without pixel buffer is rejected instead of falling back to data[0]");
    hardware->format = AV_PIX_FMT_CUDA;
    hw_view = detail::describeDecodedVideoFrame(hardware.get());
    require(hw_view && hw_view->hardware_handle == &first_slot,
            "CUDA identity retains its data[0] location");
    hardware->data[0] = nullptr;
    require(!detail::describeDecodedVideoFrame(hardware.get()), "missing CUDA handle rejected");
    return 0;
}
