// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "media/media_json.hpp"
#include "media/source_exif.hpp"
#include <future>
#include <stdexcept>

namespace {
    using namespace lfs::media;
    using namespace lfs::core::image_codecs;
    void require(bool condition, const char* text) {
        if (!condition)
            throw std::runtime_error(text);
    }
    struct Fixture {
        bool little;
        std::vector<uint8_t> bytes;
        explicit Fixture(bool le) : little(le), bytes{uint8_t(le ? 'I' : 'M'), uint8_t(le ? 'I' : 'M')} {
            put(42, 2);
            put(8, 4);
        }
        void put(uint32_t value, unsigned count) {
            for (unsigned i = 0; i < count; ++i)
                bytes.push_back(static_cast<uint8_t>(value >> (8 * (little ? i : count - 1 - i))));
        }
        void tag(uint16_t id, uint16_t type, uint32_t count, uint32_t payload) {
            put(id, 2);
            put(type, 2);
            put(count, 4);
            put(payload, 4);
        }
        ExifBlock block() const {
            ExifBlock result;
            result.bytes = {'E', 'x', 'i', 'f', 0, 0};
            result.bytes.insert(result.bytes.end(), bytes.begin(), bytes.end());
            return result;
        }
    };
    ExifBlock values(bool le) {
        Fixture f(le);
        f.put(6, 2);
        f.tag(0x112, 3, 1, le ? 6 : 6u << 16);
        f.tag(0xc7fe, 4, 1, 123);
        f.tag(0xc7fe, 4, 1, 456);
        f.tag(0x829a, 5, 1, 86);
        f.tag(0x927c, 7, 8, 94);
        f.tag(0x10f, 2, 4, le ? 0x004100ffu : 0xff004100u);
        f.put(0, 4);
        f.put(0xffffffffu, 4);
        f.put(0, 4);
        f.bytes.insert(f.bytes.end(), {'I', 'I', 42, 0, 8, 0, 0, 0});
        return f.block();
    }
} // namespace

int runSourceExifContracts() {
    {
        // Compare against the previous complete JSON -> XML representation,
        // including invalid UTF-8 replacement, controls and numeric formatting.
        const std::vector<nlohmann::json> values{
            nullptr, true, -42, UINT64_MAX, -0.0, 1.25,
            nlohmann::json::array(), nlohmann::json::object(),
            nlohmann::json{{"<&>\"", {std::string("\0\n\t", 3), "\xc3\xa9", "\xf0\x9f\x98\x80", "bad\xff\xfe", "a<&>b"}}, {"array", {nullptr, false, 42, 1.5}}},
            std::string(65536, '&')};
        for (const auto& value : values) {
            const auto json = value.dump(-1, ' ', true, nlohmann::json::error_handler_t::replace);
            std::string escaped;
            for (const auto c : json) {
                if (c == '&')
                    escaped += "&amp;";
                else if (c == '<')
                    escaped += "&lt;";
                else if (c == '>')
                    escaped += "&gt;";
                else
                    escaped += c;
            }
            const auto exact = json_detail::sourceMetadataXmlProperty(value, escaped.size());
            require(exact && *exact == "<lfs:SourceMetadata>" + escaped + "</lfs:SourceMetadata>",
                    "bounded source JSON preserves exact previous XML bytes");
            require(!json_detail::sourceMetadataXmlProperty(value, escaped.size() - 1),
                    "one-byte-short XML budget refuses before property allocation");
            require(json_detail::sourceMetadataXmlProperty(value, escaped.size() + 1) == exact,
                    "larger XML budget does not change serialized bytes");
        }
        require(!json_detail::sourceMetadataXmlProperty(nlohmann::json::array({"&", "tail"}), 0),
                "zero XML budget cannot underflow or accept a partial value");
    }
    auto unknown = values(true);
    unknown.envelope = static_cast<ExifEnvelope>(-1);
    ImageExifDescription unknown_description;
    unknown_description.original.blocks.push_back(unknown);
    unknown_description.interpretations.push_back(interpretExif(unknown));
    require(unknown_description.interpretations[0].status == ExifParseStatus::Unsupported &&
                json_detail::sourceExif(unknown_description).at("blocks")[0].at("envelope") == "unknown",
            "unknown EXIF envelopes are never serialized as HEIF");
    ImageExifDescription diagnostics;
    diagnostics.original.diagnostic = "bad\xff box";
    diagnostics.original.blocks.emplace_back();
    diagnostics.interpretations.emplace_back();
    diagnostics.interpretations[0].diagnostic = "bad\xfe tag";
    const auto diagnostic_json = nlohmann::json::parse(json_detail::dump(json_detail::sourceExif(diagnostics)));
    require(diagnostic_json.at("diagnostic") == "bad\xef\xbf\xbd box" &&
                diagnostic_json.at("blocks")[0].at("interpretation").at("diagnostic") == "bad\xef\xbf\xbd tag",
            "shared JSON transport replaces invalid diagnostic UTF8 without failing");
    for (const bool le : {false, true}) {
        const auto original = values(le);
        const auto snapshot = original.bytes;
        const auto result = interpretExif(original);
        require(result.status == ExifParseStatus::Parsed && result.entries.size() == 6, "LE/BE EXIF entries decoded");
        require(result.entries[0].unsigned_values == std::vector<uint64_t>{6}, "orientation retained as a value only");
        require(result.entries[1].id == result.entries[2].id && result.entries[1].unsigned_values[0] == 123 && result.entries[2].unsigned_values[0] == 456,
                "unknown duplicate tags stay ordered");
        require(result.entries[3].rational_values[0].numerator == 4294967295LL && result.entries[3].rational_values[0].denominator == 0,
                "unsigned rational and zero denominator preserved without division");
        require(result.entries[4].id == 0x927c && result.entries[4].bytes == std::vector<uint8_t>({'I', 'I', 42, 0, 8, 0, 0, 0}),
                "MakerNote that resembles an IFD remains opaque");
        require(result.entries[5].bytes == std::vector<uint8_t>({255, 0, 65, 0}), "ASCII byte count, invalid UTF8 and NUL preserved");
        require(original.bytes == snapshot, "private parsing never mutates originals");
        auto png = original;
        png.envelope = ExifEnvelope::PngExif;
        png.bytes.erase(png.bytes.begin(), png.bytes.begin() + 6);
        const auto png_result = interpretExif(png);
        require(png_result.status == ExifParseStatus::Parsed && png_result.entries.size() == result.entries.size(),
                "PNG eXIf uses its original bare TIFF payload without an invented envelope");
        ImageExifDescription png_description;
        png_description.original.blocks.push_back(png);
        png_description.interpretations.push_back(png_result);
        require(json_detail::sourceExif(png_description).at("blocks")[0].at("envelope") == "png_exif",
                "PNG envelope is explicit in the shared serializer");
        ImageExifDescription description;
        description.original.blocks.push_back(original);
        description.interpretations.push_back(result);
        const auto json = json_detail::sourceExif(description);
        require(json.at("blocks")[0].at("bytes") == json_detail::hexBytes(snapshot), "JSON preserves entire envelope reversibly");
        require(json.at("blocks")[0].at("interpretation").at("entries")[3].at("rational_values")[0].at("numerator") == 4294967295LL,
                "JSON unsigned rational is not negative");
        ExifParseLimits limits;
        limits.value_bytes = 16;
        require(interpretExif(original, limits).status == ExifParseStatus::Limited, "value budget applied before FFmpeg");
        limits = {};
        limits.entries = 5;
        require(interpretExif(original, limits).status == ExifParseStatus::Limited, "aggregate entry budget enforced");
        auto heif = original;
        heif.envelope = ExifEnvelope::HeifOffset;
        heif.bytes = {0, 0, 0, 3, 9, 8, 7};
        heif.bytes.insert(heif.bytes.end(), snapshot.begin() + 6, snapshot.end());
        require(interpretExif(heif).status == ExifParseStatus::Parsed, "nonzero HEIF offset respected");
        auto jxl = heif;
        jxl.envelope = ExifEnvelope::JxlOffset;
        jxl.jxl_box = JxlBox{7, true};
        require(interpretExif(jxl).status == ExifParseStatus::Parsed, "JXL nonzero TIFF offset supports either byte order");
        ImageExifDescription jxl_description;
        jxl_description.original.blocks.push_back(jxl);
        jxl_description.interpretations.push_back(interpretExif(jxl));
        const auto jxl_json = json_detail::sourceExif(jxl_description).at("blocks")[0];
        require(jxl_json.at("envelope") == "jxl_offset" && jxl_json.at("representation") == "libjxl_decoded_bytes" &&
                    jxl_json.at("jxl_box").at("ordinal") == 7 && jxl_json.at("jxl_box").at("brotli_compressed") == true &&
                    jxl_json.at("bytes") == json_detail::hexBytes(jxl.bytes),
                "JXL EXIF transport preserves decoded envelope bytes and provider box provenance");
        for (size_t size = 0; size < jxl.bytes.size(); ++size) {
            auto short_jxl = jxl;
            short_jxl.bytes.resize(size);
            require(interpretExif(short_jxl).status != ExifParseStatus::Parsed, "truncated JXL TIFF envelope cannot be fully interpreted");
        }
        jxl.bytes[0] = 255;
        require(interpretExif(jxl).status == ExifParseStatus::Malformed, "JXL out-of-range TIFF offset is rejected");
        heif.bytes[0] = 255;
        require(interpretExif(heif).status == ExifParseStatus::Malformed, "oversized HEIF offset rejected");
        for (size_t size = 0; size < snapshot.size(); ++size) {
            auto truncated = original;
            truncated.bytes.resize(size);
            require(interpretExif(truncated).status != ExifParseStatus::Parsed, "truncated payload cannot be fully interpreted");
        }
    }
    Fixture gps(true);
    gps.put(1, 2);
    gps.tag(0x8825, 4, 1, 26);
    gps.put(0, 4);
    gps.put(1, 2);
    gps.tag(2, 5, 3, 44);
    gps.put(0, 4);
    for (const auto value : {10, 20, 30}) {
        gps.put(value, 4);
        gps.put(1, 4);
    }
    auto parsed = interpretExif(gps.block());
    require(parsed.status == ExifParseStatus::Parsed && parsed.entries.size() == 2, "GPS IFD parsed");
    require(parsed.entries[0].directory && parsed.entries[1].directory_path == std::vector<uint32_t>({0, 0}) &&
                parsed.entries[1].rational_values.size() == 3,
            "GPS directory provenance retained");
    ExifParseLimits limits;
    limits.directories = 1;
    require(interpretExif(gps.block(), limits).status == ExifParseStatus::Limited, "directory count bounded");
    Fixture roots(true);
    roots.put(1, 2);
    roots.tag(0xfffc, 4, 1, 7);
    roots.put(26, 4);
    roots.put(1, 2);
    roots.tag(0x112, 3, 1, 6);
    roots.put(0, 4);
    parsed = interpretExif(roots.block());
    require(parsed.status == ExifParseStatus::Parsed && parsed.entries.size() == 2 && parsed.entries[0].id == 0xfffc &&
                parsed.entries[1].directory_path == std::vector<uint32_t>{1},
            "next IFD cannot collide with original synthetic-range tag");
    auto cycle = roots.block();
    cycle.bytes[6 + 40] = 8;
    require(interpretExif(cycle).status != ExifParseStatus::Parsed, "cyclic root IFD rejected");
    auto overlap = gps.block();
    overlap.bytes[6 + 36] = 8;
    require(interpretExif(overlap).status == ExifParseStatus::Malformed, "value overlapping IFD rejected");
    Fixture signed_value(false);
    signed_value.put(1, 2);
    signed_value.tag(0x9201, 10, 1, 26);
    signed_value.put(0, 4);
    signed_value.put(0xffffffffu, 4);
    signed_value.put(2, 4);
    parsed = interpretExif(signed_value.block());
    require(parsed.status == ExifParseStatus::Parsed && parsed.entries[0].rational_values[0].numerator == -1,
            "SRATIONAL keeps signed interpretation");
    Fixture nan(true);
    nan.put(1, 2);
    nan.tag(0xc7ff, 12, 1, 26);
    nan.put(0, 4);
    nan.put(1, 4);
    nan.put(0x7ff80000, 4);
    parsed = interpretExif(nan.block());
    ImageExifDescription description;
    description.original.blocks.push_back(nan.block());
    description.interpretations.push_back(parsed);
    require(json_detail::sourceExif(description).at("blocks")[0].at("interpretation").at("entries")[0].at("floating_values")[0] == "nan",
            "nonfinite value is explicit rather than silently serialized as null");
    auto first = std::async(std::launch::async, [] { return interpretExif(values(true)); });
    auto second = std::async(std::launch::async, [] { return interpretExif(values(false)); });
    require(first.get().status == ExifParseStatus::Parsed && second.get().status == ExifParseStatus::Parsed, "concurrent independent readers");
    for (const auto type : {1, 6, 8, 9, 11}) {
        Fixture scalar(true);
        scalar.put(1, 2);
        scalar.tag(0xc700, static_cast<uint16_t>(type), 1, type == 11 ? 0x3f000000 : 0xffffffff);
        scalar.put(0, 4);
        const auto value = interpretExif(scalar.block());
        require(value.status == ExifParseStatus::Parsed, "scalar TIFF type accepted");
        const auto& entry = value.entries.front();
        if (type == 1)
            require(entry.bytes == std::vector<uint8_t>{255}, "BYTE preserved");
        else if (type == 11)
            require(entry.floating_values == std::vector<double>{0.5}, "FLOAT decoded by FFmpeg");
        else
            require(entry.signed_values == std::vector<int64_t>{-1}, "signed integer type decoded by FFmpeg");
    }
    Fixture shared(true);
    shared.put(3, 2);
    for (int i = 0; i < 3; ++i)
        shared.tag(0x829a, 5, 1, 50);
    shared.put(0, 4);
    shared.put(2, 4);
    shared.put(3, 4);
    require(interpretExif(shared.block()).entries.size() == 3, "shared value payloads preserve duplicate entries");
    limits = {};
    limits.value_bytes = 64;
    require(interpretExif(shared.block(), limits).status == ExifParseStatus::Limited,
            "repeated payload allocations charged per reference");
    auto unsupported = values(true);
    unsupported.bytes[8] = 43;
    require(interpretExif(unsupported).status == ExifParseStatus::Unsupported, "BigTIFF is not parsed as classic EXIF");
    return 0;
}
