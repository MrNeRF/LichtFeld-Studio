// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "core/logger.hpp"
#include "io/hdr_libplacebo.hpp"
#include <iostream>
#include <stdexcept>

// Test-only log sink: no user log files, singleton app or diagnostics runtime.
// No decoder, writer, selection, geometry or metadata implementation is mocked.
namespace lfs::core {
    struct Logger::Impl {};
    Logger::Logger() = default;
    Logger::~Logger() = default;
    Logger& Logger::get() {
        static Logger logger;
        return logger;
    }
    void Logger::log(LogLevel, const SourceSite&, std::string_view message) {
        std::cerr << message << '\n';
    }
} // namespace lfs::core

// This target is deliberately SDR-only. Reaching the HDR renderer is a failure,
// never a successful fake tone map. Production HDR gets separate backend tests.
namespace lfs::io {
    class HdrLibplaceboRenderer::Impl {};
    HdrLibplaceboRenderer::HdrLibplaceboRenderer() {
        throw std::runtime_error("HDR renderer reached in SDR extraction contract tests");
    }
    HdrLibplaceboRenderer::~HdrLibplaceboRenderer() = default;
    bool HdrLibplaceboRenderer::isAvailable(std::string&) { throw std::runtime_error("Unexpected HDR call"); }
    bool HdrLibplaceboRenderer::tonemapToSdr(const AVFrame*, const AVStream*, HdrFormat, int, int,
                                             std::vector<unsigned char>&, std::string&, HdrTonemapTiming*) {
        throw std::runtime_error("Unexpected HDR call");
    }
    bool HdrLibplaceboRenderer::tonemapToSdrRgba(const AVFrame*, const AVStream*, HdrFormat, int, int, int,
                                                 std::vector<unsigned char>&, std::string&) {
        throw std::runtime_error("Unexpected HDR call");
    }
    void HdrLibplaceboRenderer::reset() { throw std::runtime_error("Unexpected HDR call"); }
} // namespace lfs::io
