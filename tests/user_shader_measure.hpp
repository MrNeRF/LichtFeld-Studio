/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/export.hpp"
#include "core/uuid.hpp"
#include "shaders/shader_service.hpp"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

namespace lfs::vis::shaders {
    class ShaderService;
    // Only defined in test builds; exercises project restoration of an identical UUID.
    struct UserShaderMeasureAccess {
        static LFS_VIS_API void replaceId(ShaderService&, const core::Uuid&, const core::Uuid&);
    };
} // namespace lfs::vis::shaders

namespace lfs::test::measure {
    using Json = nlohmann::json;
    using Clock = std::chrono::steady_clock;
    inline double ms(Clock::duration d) { return std::chrono::duration<double, std::milli>(d).count(); }
    inline double median(std::vector<double> v) {
        if (v.empty())
            throw std::runtime_error("no measurement samples");
        std::ranges::sort(v);
        return (v[(v.size() - 1) / 2] + v[v.size() / 2]) * 0.5;
    }
    inline void emit(const Json& row) { std::cout << "BENCHJSON " << row.dump() << std::endl; }
    struct Options {
        int reps = 5, sample = -1;
        std::string filter;
        bool matches(const std::string& key) const { return key.contains(filter); }
        Options(int& argc, char** argv) {
            int keep = 1;
            for (int i = 1; i < argc; ++i) {
                const std::string arg = argv[i];
                if (arg == "--reps" || arg == "--filter" || arg == "--sample") {
                    if (++i == argc)
                        throw std::runtime_error("missing option value");
                    if (arg == "--filter")
                        filter = argv[i];
                    else if (arg == "--reps")
                        reps = std::stoi(argv[i]);
                    else
                        sample = std::stoi(argv[i]);
                } else
                    argv[keep++] = argv[i];
            }
            argc = keep;
            argv[keep] = nullptr;
            if (reps < 2)
                throw std::runtime_error("--reps must be at least 2");
        }
    };
    // Each repetition is exec'd before any CUDA/Vulkan use, with an empty driver cache.
    inline int repetitions(const Options& options) {
        const char* cache = std::getenv("__GL_SHADER_DISK_CACHE_PATH");
        auto root = cache ? std::filesystem::path(cache) : std::filesystem::temp_directory_path();
        std::string pattern = (root / "shader-measure-XXXXXX").string();
        std::filesystem::create_directories(root);
        if (!mkdtemp(pattern.data()))
            throw std::runtime_error("cannot create private measurement directory");
        const auto executable = std::filesystem::read_symlink("/proc/self/exe").string();
        std::map<std::string, std::vector<Json>> rows;
        for (int rep = 0; rep < options.reps; ++rep) {
            const auto directory = std::filesystem::path(pattern) / std::to_string(rep);
            std::filesystem::create_directories(directory / "cache");
            const auto output = directory / "sample.log";
            const std::string index = std::to_string(rep);
            std::cout.flush();
            std::cerr.flush();
            const pid_t pid = fork();
            if (pid < 0)
                throw std::runtime_error("fork failed");
            if (pid == 0) {
                if (!freopen(output.c_str(), "w", stdout))
                    _exit(125);
                if (dup2(STDOUT_FILENO, STDERR_FILENO) < 0)
                    _exit(125);
                unsetenv("LFS_HOME");
                setenv("__GL_SHADER_DISK_CACHE", "1", 1);
                setenv("__GL_SHADER_DISK_CACHE_PATH", (directory / "cache").c_str(), 1);
                setenv("__GL_SHADER_DISK_CACHE_SIZE", "4000000000", 1);
                execl(executable.c_str(), executable.c_str(), "--sample", index.c_str(),
                      "--filter", options.filter.c_str(), static_cast<char*>(nullptr));
                _exit(126);
            }
            int status = 0;
            if (waitpid(pid, &status, 0) != pid || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
                std::cerr << "measurement child failed; log: " << output << '\n';
                return 1;
            }
            std::ifstream stream(output);
            for (std::string line; std::getline(stream, line);) {
                if (!line.starts_with("BENCHJSON "))
                    continue;
                auto row = Json::parse(line.substr(10));
                Json key;
                for (const char* field : {"metric", "scene", "mode", "state", "nodes", "case"})
                    if (row.contains(field))
                        key[field] = row[field];
                rows[key.dump()].push_back(std::move(row));
            }
        }
        if (rows.empty())
            throw std::runtime_error("filter matched no measurements");
        for (const auto& [key, samples] : rows) {
            if (samples.size() != static_cast<std::size_t>(options.reps))
                throw std::runtime_error("incomplete repetition set");
            auto result = samples.front();
            const auto dimensions = Json::parse(key);
            for (const auto& [field, value] : samples.front().items()) {
                if (dimensions.contains(field))
                    continue;
                if (value.is_number()) {
                    std::vector<double> values;
                    for (const auto& row : samples)
                        values.push_back(row.at(field).get<double>());
                    result[field] = median(values);
                    result["min"][field] = *std::ranges::min_element(values);
                    result["max"][field] = *std::ranges::max_element(values);
                } else if (value.is_boolean()) {
                    result[field] = std::ranges::all_of(samples, [&](const auto& row) { return row.at(field) == true; });
                }
            }
            result["reps"] = options.reps;
            result["samples"] = samples;
            emit(result);
        }
        return 0;
    }
    // All service APIs scheduled for replacement live in this adapter, including the export lease type.
    struct ServiceAccess {
        using Service = vis::shaders::ShaderService;
        using Lease = Service::ExportLease;                                    // harness adapter
        static auto create(Service& s, std::string name, std::string source) { // harness adapter
            return s.command(vis::shaders::SetShaderSource{{}, std::move(source), std::move(name), {}});
        }
        static auto bind(Service& s, const vis::shaders::ShaderSlot& slot, const core::Uuid& id) { // harness adapter
            return s.command(vis::shaders::BindShader{slot, id});
        }
        static void clear(Service& s) { s.clear(); }                                    // harness adapter
        static auto status(Service& s, const core::Uuid& id) { return s.status(id); }   // harness adapter
        static auto edit(Service& s, const core::Uuid& id, const std::string& source) { // harness adapter
            return s.command(vis::shaders::SetShaderSource{id, source, {}, {}});
        }
        static auto snapshot(Service& s) { return s.candidate(); }                                  // harness adapter
        static void demand(Service& s, const std::string& host) { s.demandLive(std::array{host}); } // harness adapter
        static bool adoptReady(Service& s) {                                                        // harness adapter: CPU fixture only, no Vulkan publication.
            const auto snapshot = s.candidate();
            if (!snapshot)
                return false;
            // CPU packing measurements do not claim on-screen adoption.
            return true;
        }
        static auto pin(Service& s) { return s.prepareExport(std::array{std::string("projection_forward_custom")}); } // harness adapter
        template <class Manager, class Model, class Settings>
        static auto pin(Manager& m, const Model& model, const Settings& settings, const glm::ivec2 frame) { return m.prepareShaderExport(Service::ExportFrame{.width = frame.x, .height = frame.y}, &model, &settings); } // harness adapter
        static auto frozen(const Lease& lease) { return lease.snapshot; }                                                                                                                                                 // harness adapter
        static void demand(const Lease&, const std::string& host) {
            if (host != "projection_forward_custom")
                throw std::runtime_error("fault probe requires a host outside its closed plan");
        } // harness adapter: pin above enqueues its closed plan before the fault gate.
        static auto wait(const Lease& lease) { // harness adapter: cancellation probe owns only the shared signal.
            while (!lease.ended()) {
                const auto generation = lease.changes->generation.load();
                if (!lease.ended())
                    lease.changes->wait(generation, Clock::time_point::max());
            }
            return Status::failure(make_error(ErrorInit{.code = ErrorCode::Cancelled, .domain = ErrorDomain::Rendering, .user_message = "The export was cancelled.", .detection = LFS_SOURCE_SITE_CURRENT()}));
        }
        template <class Manager, class Post, class Draw>
        static auto draw(Manager& manager, const Lease& lease, Post post, Draw draw) { // harness adapter
            return manager.drawWithShaders(&lease, std::move(post), std::move(draw));
        }
    };
    // harness adapter: idle fallback; no product mutex instrumentation is installed.
    template <class Manager>
    double idleSync(Manager& manager) {
        manager.syncCustomPrograms();
        const auto start = Clock::now();
        for (int i = 0; i < 100000; ++i)
            manager.syncCustomPrograms();
        return ms(Clock::now() - start) * 1e6 / 100000;
    }
} // namespace lfs::test::measure
