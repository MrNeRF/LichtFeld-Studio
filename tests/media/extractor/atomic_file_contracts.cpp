// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "core/atomic_file.hpp"
#include "core/path_utils.hpp"
#include <atomic>
#include <barrier>
#include <chrono>
#include <fstream>
#include <stdexcept>
#include <thread>
#ifndef _WIN32
#include <sys/stat.h>
#endif
namespace {
    void require(bool value, const char* message) {
        if (!value)
            throw std::runtime_error(message);
    }
    std::string read(const std::filesystem::path& path) {
        std::ifstream stream(path, std::ios::binary);
        return {std::istreambuf_iterator<char>(stream), {}};
    }
} // namespace
int runAtomicFileContracts() {
    using namespace lfs;
    using namespace lfs::core;
    const auto directory = std::filesystem::temp_directory_path() / utf8_to_path("atomic-é-日本語-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup() {
            std::error_code ignored;
            std::filesystem::remove_all(path, ignored);
        }
    } cleanup{directory};
    const auto file = directory / "settings.json";
    const std::string binary("a\n\r\n\0z", 6);
    require(writeTextFileAtomically(file, binary).has_value() && read(file) == binary, "atomic settings bytes preserved exactly");
    auto collision = writeTextFileAtomically(file, "replacement", {.overwrite = false});
    require(!collision && collision.error().code() == ErrorCode::AlreadyExists && read(file) == binary, "no-replace preserves existing bytes");
    require(collision.error().native() && collision.error().detail().find("settings.json") != std::string_view::npos, "atomic native/path diagnostic");
    bool cancel = false;
    auto cancelled = writeFileAtomically(file, [&](FILE* stream) -> Status {
        std::fputs("discard", stream);
        cancel = true;
        return {};
    },
                                         {.cancelled = [&] { return cancel; }});
    require(!cancelled && cancelled.error().code() == ErrorCode::Cancelled && read(file) == binary, "cancellation before commit preserves destination");
    auto failed = writeFileAtomically(file, [](FILE* stream) -> Status {
        std::fputs("discard", stream);
        return Status::failure(make_error({.code = ErrorCode::DataLoss, .domain = ErrorDomain::IO, .detail = "fixture write failure", .detection = LFS_SOURCE_SITE_CURRENT()}));
    });
    require(!failed && failed.error().code() == ErrorCode::DataLoss && read(file) == binary, "writer error preserves destination");
    auto thrown = writeFileAtomically(file, [](FILE*) -> Status { throw std::runtime_error("fixture exception"); });
    require(!thrown && thrown.error().detail().find("fixture exception") != std::string_view::npos && read(file) == binary, "throwing writer cleaned");
    const auto racing = directory / "race.bin";
    std::barrier gate(2);
    std::atomic<int> winners{0}, collisions{0};
    auto compete = [&](const char* contents) {
        const auto result = writeFileAtomically(racing, [&](FILE* stream) -> Status {
            std::fputs(contents, stream);
            gate.arrive_and_wait();
            return {};
        },
                                                {.overwrite = false, .durable = false});
        if (result)
            ++winners;
        else if (result.error().code() == ErrorCode::AlreadyExists)
            ++collisions;
    };
    std::thread first(compete, "first"), second(compete, "second");
    first.join();
    second.join();
    require(winners == 1 && collisions == 1 && (read(racing) == "first" || read(racing) == "second"), "racing no-replace has exactly one complete winner");
#ifndef _WIN32
    const auto mask = ::umask(0022);
    const auto permission_result = writeTextFileAtomically(directory / "mode.json", "permissions");
    ::umask(mask);
    struct stat state {};
    require(permission_result.has_value() && ::stat((directory / "mode.json").c_str(), &state) == 0 && (state.st_mode & 0777) == 0644, "published permissions honor umask");
#endif
    for (const auto& entry : std::filesystem::directory_iterator(directory))
        require(entry.path().filename().string().find(".tmp-") == std::string::npos, "no owned temporary remains");
    return 0;
}
