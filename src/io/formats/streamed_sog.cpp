/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "streamed_sog.hpp"
#include "core/logger.hpp"
#include "sogs.hpp"
#include "streamed_sog_morton.hpp"
#if __has_include("io/splat_decimate.hpp")
#include "io/splat_decimate.hpp"
#else
#include "cuda/splat_decimate.hpp"
#endif
#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <deque>
#include <fstream>
#include <map>
#include <mutex>
#include <nlohmann/json.hpp>
#include <numeric>
#include <regex>
#include <set>
#include <tbb/parallel_for.h>
#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace lfs::io {
    namespace {
        namespace fs = std::filesystem;
        using Json = nlohmann::json;
        using core::Device;
        using core::Tensor;
        using Clock = std::chrono::steady_clock;
        struct Cancelled {};
        void progress(const StreamedSogSaveOptions& o, float p, const std::string& stage) {
            if (o.progress_callback && !o.progress_callback(p, stage))
                throw Cancelled{};
        }
        fs::path manifest_path(const fs::path& p) {
            return fs::is_directory(p) ? p / "lod-meta.json" : p;
        }
        Json read_json(const fs::path& p) {
            const auto size = fs::file_size(p);
            if (size == 0 || size > 64 * 1024 * 1024)
                throw std::runtime_error("Invalid metadata size: " + p.string());
            std::ifstream f(p, std::ios::binary);
            if (!f)
                throw std::runtime_error("Cannot open " + p.string());
            return Json::parse(f);
        }
        size_t integer(const Json& j, const char* name) {
            if (!j.is_number_integer() || (j.is_number_integer() && j.get<double>() < 0) ||
                j.get<double>() > static_cast<double>(INT_MAX))
                throw std::runtime_error(std::string("Invalid lod-meta.json ") + name);
            return j.get<size_t>();
        }
        fs::path related(const fs::path& base, const Json& name) {
            if (!name.is_string())
                throw std::runtime_error("Invalid lod-meta.json filename");
            const fs::path p = core::utf8_to_path(name.get<std::string>());
            if (p.empty() || p.is_absolute() || p.has_root_name())
                throw std::runtime_error("Invalid unit path");
            for (const auto& part : p)
                if (part == "..")
                    throw std::runtime_error("Unit path escapes streamed SOG directory");
            const auto root = fs::weakly_canonical(base);
            const auto resolved = fs::weakly_canonical(base / p);
            const auto rel = resolved.lexically_relative(root);
            if (rel.empty() || *rel.begin() == "..")
                throw std::runtime_error("Unit path escapes streamed SOG directory");
            return base / p;
        }
        struct Manifest {
            Json json;
            fs::path base;
            std::vector<std::map<size_t, size_t>> files;
            std::vector<size_t> counts;
        };
        Manifest parse_manifest(const fs::path& path) {
            const auto p = fs::absolute(manifest_path(path));
            Manifest m{read_json(p), p.parent_path(), {}, {}};
            const auto& j = m.json;
            if (!j.is_object())
                throw std::runtime_error("Invalid lod-meta.json object");
            if (j.contains("version") && j.at("version") != 1)
                throw std::runtime_error("Unsupported lod-meta.json version");
            const auto levels = integer(j.at("lodLevels"), "lodLevels");
            if (!levels || levels > 1024)
                throw std::runtime_error("Invalid lod-meta.json lodLevels");
            if (!j.at("filenames").is_array())
                throw std::runtime_error("Invalid lod-meta.json filenames");
            std::vector<size_t> unit_counts;
            std::set<fs::path> unique;
            for (const auto& name : j.at("filenames")) {
                const auto unit = related(m.base, name);
                if (!unique.insert(fs::weakly_canonical(unit)).second)
                    throw std::runtime_error("Duplicate SOG unit filename");
                unit_counts.push_back(integer(read_json(unit).at("count"), "unit count"));
            }
            m.files.resize(levels);
            m.counts.resize(levels);
            using Range = std::pair<size_t, size_t>;
            std::vector<std::vector<Range>> ranges(unit_counts.size());
            std::vector<int> owner(unit_counts.size(), -1);
            const auto visit = [&](auto&& self, const Json& node, int depth) -> void {
                if (depth > 64 || !node.is_object())
                    throw std::runtime_error("Invalid lod-meta.json tree");
                const auto& bound = node.at("bound");
                for (const char* side : {"min", "max"}) {
                    if (!bound.at(side).is_array() || bound.at(side).size() != 3)
                        throw std::runtime_error("Invalid tree bound");
                    for (const auto& v : bound.at(side))
                        if (!v.is_number() || !std::isfinite(v.get<double>()))
                            throw std::runtime_error("Invalid tree bound");
                }
                for (int a = 0; a < 3; ++a)
                    if (bound["min"][a].get<double>() > bound["max"][a].get<double>())
                        throw std::runtime_error("Inverted tree bound");
                if (node.contains("errors")) {
                    const auto& errors = node["errors"];
                    if (!errors.is_array() || errors.size() != levels)
                        throw std::runtime_error("Invalid LOD errors");
                    for (const auto& e : errors)
                        if (!e.is_number() || !std::isfinite(e.get<double>()) || e.get<double>() < 0)
                            throw std::runtime_error("Invalid LOD errors");
                }
                if (node.contains("children")) {
                    if (node.contains("lods") || !node["children"].is_array() || node["children"].size() != 2)
                        throw std::runtime_error("Invalid tree children");
                    for (const auto& child : node["children"])
                        self(self, child, depth + 1);
                } else {
                    if (!node.at("lods").is_object())
                        throw std::runtime_error("Invalid tree lods");
                    for (const auto& [key, ref] : node["lods"].items()) {
                        size_t l = 0;
                        const auto [end, ec] = std::from_chars(key.data(), key.data() + key.size(), l);
                        if (ec != std::errc{} || end != key.data() + key.size() || l >= levels)
                            throw std::runtime_error("Invalid LOD level reference");
                        const size_t f = integer(ref.at("file"), "file index");
                        const size_t off = integer(ref.at("offset"), "offset");
                        const size_t n = integer(ref.at("count"), "count");
                        if (f >= unit_counts.size())
                            throw std::runtime_error("LOD file index out of range");
                        if (off > unit_counts[f] || n > unit_counts[f] - off)
                            throw std::runtime_error("LOD offset/count beyond unit count");
                        if (owner[f] != -1 && owner[f] != static_cast<int>(l))
                            throw std::runtime_error("SOG unit shared by different LODs");
                        owner[f] = static_cast<int>(l);
                        ranges[f].emplace_back(off, n);
                        m.files[l][f] += n;
                        m.counts[l] += n;
                    }
                }
            };
            visit(visit, j.at("tree"), 0);
            for (size_t f = 0; f < ranges.size(); ++f) {
                if (ranges[f].empty())
                    continue;
                std::sort(ranges[f].begin(), ranges[f].end());
                size_t end = 0;
                for (auto [off, n] : ranges[f]) {
                    if (off != end)
                        throw std::runtime_error("LOD unit ranges overlap or contain gaps");
                    end += n;
                }
                if (end != unit_counts[f])
                    throw std::runtime_error("LOD unit count mismatch");
            }
            if (j.contains("counts")) {
                if (!j["counts"].is_array() || j["counts"].size() != levels)
                    throw std::runtime_error("Invalid lod-meta.json counts");
                for (size_t l = 0; l < levels; ++l)
                    if (integer(j["counts"][l], "counts") != m.counts[l])
                        throw std::runtime_error("LOD count mismatch");
            }
            if (j.contains("count") && integer(j["count"], "count") != std::accumulate(m.counts.begin(), m.counts.end(), size_t{0}))
                throw std::runtime_error("Total LOD count mismatch");
            if (j.contains("environment"))
                (void)read_json(related(m.base, j["environment"]));
            return m;
        }

        // Pageable levels bound large-scene VRAM. Small exports may additionally
        // retain CUDA attributes for device row gathers.
        struct HostSplats {
            int degree = 0;
            float scale = 1;
            Tensor means, sh0, shN, scaling, rotation, opacity;
            Tensor device_means, device_sh0, device_shN, device_scaling, device_rotation, device_opacity;
            size_t size() const { return means.is_valid() ? means.size(0) : 0; }
            explicit HostSplats(const SplatData& s, bool resident = false)
                : degree(s.get_max_sh_degree()), scale(s.get_scene_scale()), means(s.means().to_pageable_host()), sh0(resident ? Tensor{} : s.sh0().to_pageable_host()), shN(resident ? Tensor{} : s.shN_canonical_cpu()), scaling(s.scaling_raw().to_pageable_host()), rotation(s.rotation_raw().to_pageable_host()), opacity(resident ? Tensor{} : s.opacity_raw().to_pageable_host()) {
                if (resident) {
                    device_means = s.means().cuda();
                    device_sh0 = s.sh0().cuda();
                    device_shN = s.shN_canonical().cuda();
                    device_scaling = s.scaling_raw().cuda();
                    device_rotation = s.rotation_raw().cuda();
                    device_opacity = s.opacity_raw().cuda();
                }
            }
            HostSplats() = default;
            SplatData materialize(const std::vector<int>& rows) const {
                const bool resident = device_means.is_valid();
                const auto indices = Tensor::from_vector(rows, {rows.size()}, resident ? Device::CUDA : Device::CPU);
                const auto gather = [&](const Tensor& host, const Tensor& device) {
                    return (resident ? device : host).index_select(0, indices).cuda();
                };
                auto rest = degree ? gather(shN, device_shN) : Tensor::empty({rows.size(), 0, 3}, Device::CUDA);
                return SplatData(degree, gather(means, device_means), gather(sh0, device_sh0), std::move(rest), gather(scaling, device_scaling), gather(rotation, device_rotation), gather(opacity, device_opacity), scale);
            }
            void filter(const Tensor& keep) {
                for (auto* t : {&means, &sh0, &scaling, &rotation, &opacity})
                    if (t->is_valid())
                        *t = t->index_select(0, keep);
                if (degree && shN.is_valid())
                    shN = shN.index_select(0, keep);
                if (device_means.is_valid()) {
                    const auto ids = keep.cuda();
                    for (auto* t : {&device_means, &device_sh0, &device_scaling, &device_rotation, &device_opacity})
                        *t = t->index_select(0, ids);
                    if (degree)
                        device_shN = device_shN.index_select(0, ids);
                }
            }
        };
        SplatData concatenate(std::vector<HostSplats>& parts) {
            int degree = 0;
            size_t count = 0;
            for (const auto& p : parts) {
                degree = std::max(degree, p.degree);
                count += p.size();
            }
            const size_t k = (degree + 1) * (degree + 1) - 1;
            std::vector<Tensor> means, sh0, scaling, rotation, opacity;
            auto shN = Tensor::zeros({count, k, 3}, Device::CPU);
            size_t offset = 0;
            auto* rest = shN.ptr<float>();
            for (const auto& p : parts) {
                means.push_back(p.means);
                sh0.push_back(p.sh0.reshape({static_cast<int>(p.size()), 1, 3}));
                scaling.push_back(p.scaling);
                rotation.push_back(p.rotation);
                opacity.push_back(p.opacity.reshape({static_cast<int>(p.size()), 1}));
                const size_t pk = (p.degree + 1) * (p.degree + 1) - 1;
                if (pk) {
                    const auto* source = p.shN.ptr<float>();
                    for (size_t i = 0; i < p.size(); ++i)
                        std::copy_n(source + i * pk * 3, pk * 3, rest + (offset + i) * k * 3);
                }
                offset += p.size();
            }
            if (!count)
                return SplatData(0, Tensor::empty({0, 3}, Device::CUDA), Tensor::empty({0, 1, 3}, Device::CUDA), Tensor::empty({0, 0, 3}, Device::CUDA), Tensor::empty({0, 3}, Device::CUDA), Tensor::empty({0, 4}, Device::CUDA), Tensor::empty({0, 1}, Device::CUDA), 1);
            return SplatData(degree, Tensor::cat(means).cuda(), Tensor::cat(sh0).cuda(), shN.cuda(), Tensor::cat(scaling).cuda(), Tensor::cat(rotation).cuda(), Tensor::cat(opacity).cuda(), 1);
        }
        struct Bound {
            std::array<double, 3> min{INFINITY, INFINITY, INFINITY}, max{-INFINITY, -INFINITY, -INFINITY};
            void add(const Bound& b) {
                for (int a = 0; a < 3; ++a) {
                    min[a] = std::min(min[a], b.min[a]);
                    max[a] = std::max(max[a], b.max[a]);
                }
            }
            Json json() const { return {{"min", min}, {"max", max}}; }
            int axis() const {
                int a = 0;
                for (int b = 1; b < 3; ++b)
                    if (max[b] - min[b] > max[a] - min[a])
                        a = b;
                return a;
            }
        };
        Bound splat_bound(const float* means, const float* rotation, const float* scaling, size_t i) {
            const float* q = rotation + i * 4;
            const double len = std::sqrt(double(q[0]) * q[0] + double(q[1]) * q[1] + double(q[2]) * q[2] + double(q[3]) * q[3]);
            if (!std::isfinite(len) || len == 0)
                throw std::runtime_error("Invalid quaternion in streamed SOG input");
            const double w = q[0] / len, x = q[1] / len, y = q[2] / len, z = q[3] / len;
            const double r[3][3] = {{1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)}, {2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)}, {2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)}};
            std::array<double, 3> scales;
            for (int c = 0; c < 3; ++c)
                scales[c] = std::exp(double(scaling[i * 3 + c]));
            Bound b;
            for (int a = 0; a < 3; ++a) {
                double extent = 0;
                for (int c = 0; c < 3; ++c)
                    extent += std::abs(r[a][c]) * scales[c];
                const double p = means[i * 3 + a];
                b.min[a] = p - extent;
                b.max[a] = p + extent;
                if (!std::isfinite(b.min[a]) || !std::isfinite(b.max[a]))
                    throw std::runtime_error("Non-finite geometry in streamed SOG input");
            }
            return b;
        }
        struct TreeNode {
            size_t begin, end;
            Bound centroids;
            int left = -1, right = -1;
        };
        struct Unit {
            int level, index;
            std::vector<int> rows;
            std::vector<size_t> bins;
        };
        struct TempDirectory {
            fs::path path;
            ~TempDirectory() {
                std::error_code ec;
                fs::remove_all(path, ec);
            }
        };
        // Match the reference's JSON number rounding without altering provenance strings.
        void round_numbers(Json& j) {
            if (j.is_number_float()) {
                const double v = j.get<double>();
                if (std::trunc(v) == v && std::abs(v) < 9e18) {
                    j = static_cast<int64_t>(v);
                    return;
                }
                char buf[64];
                const auto [end, ec] = std::to_chars(buf, buf + sizeof(buf), v, std::chars_format::general, 7);
                if (ec != std::errc{})
                    throw std::runtime_error("Cannot encode manifest number");
                double rounded;
                std::from_chars(buf, end, rounded);
                if (std::trunc(rounded) == rounded && std::abs(rounded) < 9e18)
                    j = static_cast<int64_t>(rounded);
                else
                    j = rounded;
            } else if (j.is_structured())
                for (auto& value : j)
                    round_numbers(value);
        }
    } // namespace

    bool is_streamed_sog_path(const std::filesystem::path& p) {
        std::error_code ec;
        return (p.filename() == "lod-meta.json" && fs::is_regular_file(p, ec)) ||
               (fs::is_directory(p, ec) && fs::is_regular_file(p / "lod-meta.json", ec));
    }
    std::expected<void, std::string> validate_streamed_sog(const std::filesystem::path& p) {
        try {
            (void)parse_manifest(p);
            return {};
        } catch (const std::exception& e) { return std::unexpected(std::string("Invalid streamed SOG: ") + e.what()); }
    }
    std::expected<SplatData, std::string> load_streamed_sog(const std::filesystem::path& p, const StreamedSogLoadOptions& options) {
        try {
            const auto m = parse_manifest(p);
            const int level = options.lod_level < 0 ? static_cast<int>(m.files.size()) + options.lod_level : options.lod_level;
            if (level < 0 || level >= static_cast<int>(m.files.size()))
                throw std::runtime_error("Requested LOD level out of range");
            std::vector<HostSplats> parts;
            size_t count = 0;
            // Read and WebP-decode one unit ahead, but reconstruct CUDA tensors only
            // on this thread. The queued work owns CPU buffers and uses no CUDA state.
            std::vector<std::pair<fs::path, size_t>> files;
            for (auto [file, expected] : m.files[level])
                files.emplace_back(related(m.base, m.json["filenames"][file]).parent_path(), expected);
            const auto prepare = [&](size_t i) {
                return std::async(std::launch::async, [path = files[i].first] { return prepare_sog_directory(path); });
            };
            std::future<std::expected<SogDirectoryReconstruct, std::string>> next;
            if (!files.empty())
                next = prepare(0);
            for (size_t i = 0; i < files.size(); ++i) {
                auto ready = next.get();
                if (!ready)
                    throw std::runtime_error(ready.error());
                if (i + 1 < files.size())
                    next = prepare(i + 1);
                auto unit = (*ready)();
                if (!unit)
                    throw std::runtime_error(unit.error());
                if (unit->size() != files[i].second)
                    throw std::runtime_error("Decoded SOG unit count mismatch");
                count += unit->size();
                parts.emplace_back(*unit);
            }
            if (count != m.counts[level])
                throw std::runtime_error("Decoded LOD count mismatch");
            if (m.json.contains("environment")) {
                auto env = load_sog(related(m.base, m.json["environment"]));
                if (!env)
                    throw std::runtime_error(env.error());
                parts.emplace_back(*env);
            }
            return concatenate(parts);
        } catch (const std::exception& e) { return std::unexpected(std::string("Failed to load streamed SOG: ") + e.what()); }
    }

    Result<void> save_streamed_sog(const SplatData& input, const StreamedSogSaveOptions& o) {
        try {
            const auto started = Clock::now();
            if (o.output_path.empty() || o.lod_levels < 1 || o.lod_levels > 1024 || !std::isfinite(o.lod_ratio) || o.lod_ratio <= 0 || o.lod_ratio >= 1 || o.chunk_count_k <= 0 || o.chunk_min_k < 0 || !std::isfinite(o.chunk_extent) || o.chunk_extent <= 0 || o.kmeans_iterations < 1)
                return make_error(ErrorCode::INVALID_DATASET, "Invalid streamed SOG export options", o.output_path);
            if (!input.size())
                return make_error(ErrorCode::EMPTY_DATASET, "No splats to write", o.output_path);
            auto requested = o.output_path.filename() == "lod-meta.json" ? o.output_path.parent_path() : o.output_path;
            auto out = fs::absolute(requested.empty() ? fs::path(".") : requested).lexically_normal();
            if (out != out.root_path() && out.filename().empty())
                out = out.parent_path();
            if (out == out.root_path())
                return make_error(ErrorCode::PATH_NOT_WRITABLE, "Cannot export to filesystem root", out);
            fs::create_directories(out.parent_path());
            static std::atomic_uint64_t sequence{0};
#ifdef _WIN32
            const auto pid = _getpid();
#else
            const auto pid = getpid();
#endif
            TempDirectory temp{fs::path(out.string() + std::format(".tmp-{}-{}", pid, sequence++))};
            if (!fs::create_directory(temp.path))
                throw std::runtime_error("Temporary export directory already exists");
            progress(o, 0, "Preparing streamed SOG");
            std::vector<HostSplats> levels;
            size_t free_cuda = 0, total_cuda = 0;
            const bool memory_known = cudaMemGetInfo(&free_cuda, &total_cuda) == cudaSuccess;
            const double level_rows = input.size() * (1.0 - std::pow(double(o.lod_ratio), o.lod_levels)) / (1.0 - o.lod_ratio);
            const double resident_bytes = level_rows * (14 + 3 * input.max_sh_coeffs_rest()) * sizeof(float);
            // Reserve most available VRAM for decimation, original storage, and
            // two encoder workspaces. Large scenes retain pageable level staging.
            const bool resident = memory_known && resident_bytes < std::min<double>(1024.0 * 1024 * 1024, free_cuda * 0.25);
            LOG_DEBUG("Streamed SOG level storage: resident={} estimated_bytes={:.0f} free_cuda={}", resident, resident_bytes, free_cuda);
            levels.emplace_back(input, resident);
            if (input.has_deleted_mask())
                levels[0].filter(input.deleted().logical_not().to_pageable_host());
            const auto n0 = levels[0].size();
            if (!n0)
                return make_error(ErrorCode::EMPTY_DATASET, "No visible splats to write", out);
            if (n0 > INT_MAX)
                return make_error(ErrorCode::INVALID_DATASET, "Too many splats", out);
            const auto prepared = Clock::now();
            std::optional<SplatData> previous_level;
            for (int l = 1; l < o.lod_levels; ++l) {
                const size_t target = static_cast<size_t>(std::round(n0 * std::pow(double(o.lod_ratio), l)));
                progress(o, 0.05f + 0.25f * (l - 1) / o.lod_levels, "Decimating LOD " + std::to_string(l));
                if (!target) {
                    levels.emplace_back();
                    continue;
                }
                const auto& previous = previous_level ? *previous_level : input;
                DecimateOptions decimate;
                decimate.target_count = target;
                decimate.use_gpu = o.use_gpu;
                bool cancelled = false;
                decimate.progress = [&](float p, const std::string& stage) {
                    if (o.progress_callback && !o.progress_callback(0.05f + 0.25f * (l - 1 + p) / o.lod_levels, stage))
                        cancelled = true;
                    return !cancelled;
                };
                auto result = decimate_splats(previous, decimate);
                if (cancelled)
                    throw Cancelled{};
                if (!result)
                    return make_error(ErrorCode::ENCODING_FAILED, result.error(), out);
                previous_level.emplace(std::move(*result));
                levels.emplace_back(*previous_level, resident);
            }
            previous_level.reset();
            const auto decimated = Clock::now();
            progress(o, 0.30f, "Partitioning streamed SOG");
            std::vector<size_t> cum{0}, counts;
            std::vector<std::array<float, 3>> positions;
            std::vector<Bound> bounds;
            const auto total = std::accumulate(levels.begin(), levels.end(), size_t{0}, [](size_t n, const auto& l) { return n + l.size(); });
            if (total > INT_MAX)
                throw std::runtime_error("Too many total LOD rows");
            positions.resize(total);
            bounds.resize(total);
            for (const auto& l : levels) {
                const auto base = cum.back();
                counts.push_back(l.size());
                cum.push_back(base + l.size());
                if (!l.size())
                    continue;
                const auto* means = l.means.ptr<float>();
                const auto* rotation = l.rotation.ptr<float>();
                const auto* scaling = l.scaling.ptr<float>();
                tbb::parallel_for(size_t{0}, l.size(), [&](size_t i) {
                    const auto* p = means + i * 3;
                    positions[base + i] = {p[0], p[1], p[2]};
                    bounds[base + i] = splat_bound(means, rotation, scaling, i);
                });
            }
            if (cum.back() > INT_MAX)
                throw std::runtime_error("Too many total LOD rows");
            std::vector<int> indices(cum.back());
            std::iota(indices.begin(), indices.end(), 0);
            std::vector<TreeNode> tree;
            const auto split = [&](auto&& self, size_t begin, size_t end) -> int {
                Bound box;
                for (size_t i = begin; i < end; ++i)
                    for (int a = 0; a < 3; ++a) {
                        box.min[a] = std::min(box.min[a], double(positions[indices[i]][a]));
                        box.max[a] = std::max(box.max[a], double(positions[indices[i]][a]));
                    }
                const int node = static_cast<int>(tree.size());
                tree.push_back({begin, end, box});
                if (end - begin > 256) {
                    const auto mid = begin + (end - begin) / 2;
                    const int a = box.axis();
                    std::nth_element(indices.begin() + begin, indices.begin() + mid, indices.begin() + end, [&](int i, int j) { return positions[i][a] < positions[j][a]; });
                    const int left = self(self, begin, mid), right = self(self, mid, end);
                    tree[node].left = left;
                    tree[node].right = right;
                }
                return node;
            };
            split(split, 0, indices.size());
            const size_t bin_size = size_t(o.chunk_count_k) * 1024, bin_min = size_t(o.chunk_min_k) * 1024;
            std::vector<Unit> units;
            std::vector<int> current(o.lod_levels, -1), next(o.lod_levels, 0);
            std::vector<std::string> filenames;
            const auto build = [&](auto&& self, int id) -> std::pair<Json, Bound> {
                const auto& node = tree[id];
                const int a = node.centroids.axis();
                if (node.left >= 0 && (node.end - node.begin > bin_size || (node.centroids.max[a] - node.centroids.min[a] > o.chunk_extent && node.end - node.begin > bin_min))) {
                    auto [left, lb] = self(self, node.left);
                    auto [right, rb] = self(self, node.right);
                    lb.add(rb);
                    return {Json{{"bound", lb.json()}, {"children", Json::array({std::move(left), std::move(right)})}}, lb};
                }
                std::map<int, std::vector<int>> bins;
                Bound bound;
                for (size_t i = node.begin; i < node.end; ++i) {
                    const int flat = indices[i];
                    const int l = static_cast<int>(std::upper_bound(cum.begin(), cum.end(), flat) - cum.begin() - 1);
                    bins[l].push_back(static_cast<int>(flat - cum[l]));
                    bound.add(bounds[flat]);
                }
                Json lods = Json::object();
                for (auto& [l, rows] : bins) {
                    if (current[l] < 0) {
                        current[l] = static_cast<int>(units.size());
                        const int index = next[l]++;
                        units.push_back({l, index, {}, {}});
                        filenames.push_back(std::format("{}_{}/meta.json", l, index));
                    }
                    const int f = current[l];
                    auto& unit = units[f];
                    lods[std::to_string(l)] = {{"file", f}, {"offset", unit.rows.size()}, {"count", rows.size()}};
                    unit.rows.insert(unit.rows.end(), rows.begin(), rows.end());
                    unit.bins.push_back(rows.size());
                    if (unit.rows.size() > bin_size)
                        current[l] = -1;
                }
                return {Json{{"bound", bound.json()}, {"lods", std::move(lods)}}, bound};
            };
            auto [root, bound] = build(build, 0);
            tree.clear();
            tree.shrink_to_fit();
            positions.clear();
            positions.shrink_to_fit();
            bounds.clear();
            bounds.shrink_to_fit();
            indices.clear();
            indices.shrink_to_fit();
            const auto partitioned = Clock::now();
            double morton_ms = 0, encode_ms = 0;
            auto stamp = o.provenance.value_or(core::make_minimal_provenance_stamp());
            // Two units reduced garden unit-write wall time by about a third, with
            // under 2 GiB process RSS. Keep the pool bounded and allow a lower-memory
            // single-worker override. CUDA scratch is per-call; host levels are immutable.
            size_t workers = 2;
            if (const auto* value = std::getenv("LFS_SSOG_UNIT_WORKERS")) {
                const std::string_view text(value);
                size_t requested = 0;
                const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), requested);
                if (ec != std::errc{} || end != text.data() + text.size() || requested < 1 || requested > 3)
                    throw std::runtime_error("LFS_SSOG_UNIT_WORKERS must be 1, 2 or 3");
                workers = requested;
            }
            struct UnitTiming {
                double morton, encode;
            };
            std::mutex progress_mutex;
            std::vector<float> unit_progress(units.size());
            bool cancelled = false;
            const auto encode_unit = [&](size_t f) -> Result<UnitTiming> {
                auto& u = units[f];
                const auto& l = levels[u.level];
                const auto sort_start = Clock::now();
                std::vector<size_t> offsets{0};
                for (const auto n : u.bins)
                    offsets.push_back(offsets.back() + n);
                const auto* positions = l.means.ptr<float>();
                tbb::parallel_for(size_t{0}, u.bins.size(), [&](size_t bin) {
                    sort_streamed_sog_leaf(positions, std::span<int>(u.rows).subspan(offsets[bin], u.bins[bin]));
                });
                const double sort_ms = std::chrono::duration<double, std::milli>(Clock::now() - sort_start).count();
                const auto encode_start = Clock::now();
                auto data = l.materialize(u.rows);
                const auto gathered = Clock::now();
                SogEncodeOptions encode;
                encode.output_path = temp.path / fs::path(filenames[f]).parent_path();
                encode.kmeans_iterations = o.kmeans_iterations;
                encode.use_gpu = o.use_gpu;
                encode.provenance = stamp;
                encode.presorted = true;
                encode.fast_webp = true;
                encode.progress_callback = [&](float p, const std::string& stage) {
                    std::lock_guard lock(progress_mutex);
                    if (cancelled)
                        return false;
                    unit_progress[f] = std::max(unit_progress[f], p);
                    const float total = std::accumulate(unit_progress.begin(), unit_progress.end(), 0.0f);
                    if (o.progress_callback && !o.progress_callback(0.4f + 0.59f * total / units.size(), stage))
                        cancelled = true;
                    return !cancelled;
                };
                auto result = encode_sog_directory(data, encode);
                if (!result)
                    return std::unexpected(result.error());
                LOG_DEBUG("Streamed SOG unit: file={} level={} rows={} leaves={} morton_ms={:.3f} gather_ms={:.3f} encode_ms={:.3f}", filenames[f], u.level, u.rows.size(), u.bins.size(), sort_ms, std::chrono::duration<double, std::milli>(gathered - encode_start).count(), std::chrono::duration<double, std::milli>(Clock::now() - gathered).count());
                return UnitTiming{sort_ms, std::chrono::duration<double, std::milli>(Clock::now() - encode_start).count()};
            };
            if (workers == 1) {
                for (size_t f = 0; f < units.size(); ++f) {
                    auto result = encode_unit(f);
                    if (!result)
                        return std::unexpected(result.error());
                    morton_ms += result->morton;
                    encode_ms += result->encode;
                }
            } else {
                std::deque<std::future<Result<UnitTiming>>> pending;
                size_t next_unit = 0;
                const auto launch = [&] {
                    const size_t f = next_unit++;
                    pending.push_back(std::async(std::launch::async, [&, f] { return encode_unit(f); }));
                };
                while (next_unit < std::min(workers, units.size()))
                    launch();
                while (!pending.empty()) {
                    auto result = pending.front().get();
                    pending.pop_front();
                    if (!result)
                        return std::unexpected(result.error());
                    morton_ms += result->morton;
                    encode_ms += result->encode;
                    if (next_unit < units.size())
                        launch();
                }
            }
            Json manifest{{"version", 1}, {"asset", {{"generator", "LichtFeld Studio"}, {"chunkGaussians", bin_size}, {"chunkExtent", o.chunk_extent}, {"chunkMinGaussians", bin_min}}}, {"count", cum.back()}, {"counts", counts}, {"lodLevels", o.lod_levels}, {"lodErrors", false}, {"filenames", filenames}, {"tree", std::move(root)}};
            round_numbers(manifest);
            manifest["asset"]["lichtfeld_provenance"] = Json::parse(core::provenance_to_json(stamp));
            {
                std::ofstream file(temp.path / "lod-meta.json", std::ios::binary);
                file << manifest.dump();
                file.close();
                if (!file)
                    throw std::runtime_error("Failed to write lod-meta.json");
            }
            progress(o, 1, "Complete");
            const auto encoded = Clock::now();
            // Preserve unrelated user files. Move all format-owned old paths to a rollback
            // directory before installing units, then publish the completed manifest last.
            fs::create_directories(out);
            TempDirectory backup{fs::path(temp.path.string() + ".backup")};
            fs::create_directory(backup.path);
            std::vector<fs::path> old_paths, installed;
            const bool replacing = fs::exists(out / "lod-meta.json");
            const std::regex unit_name("[0-9]+_[0-9]+");
            for (const auto& e : fs::directory_iterator(out)) {
                const auto name = e.path().filename();
                const bool owned = name == "lod-meta.json" || name == "env" || std::regex_match(name.string(), unit_name);
                if (owned && replacing)
                    old_paths.push_back(name);
            }
            for (const auto& u : units) {
                const auto name = fs::path(std::format("{}_{}", u.level, u.index));
                if (fs::exists(out / name) && !replacing)
                    throw std::runtime_error("Output unit already exists in unrelated directory");
            }
            std::vector<fs::path> moved;
            try {
                for (const auto& name : old_paths) {
                    fs::rename(out / name, backup.path / name);
                    moved.push_back(name);
                }
                for (const auto& u : units) {
                    const auto name = fs::path(std::format("{}_{}", u.level, u.index));
                    fs::rename(temp.path / name, out / name);
                    installed.push_back(name);
                }
                fs::rename(temp.path / "lod-meta.json", out / "lod-meta.json");
            } catch (...) {
                std::error_code ec;
                for (const auto& name : installed)
                    fs::remove_all(out / name, ec);
                for (const auto& name : moved)
                    fs::rename(backup.path / name, out / name, ec);
                // Retain a backup if rollback itself fails, rather than deleting recoverable data.
                if (!fs::is_empty(backup.path))
                    backup.path.clear();
                throw;
            }
            const auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
            LOG_DEBUG("Streamed SOG export stages: prepare_ms={:.3f} decimate_ms={:.3f} partition_ms={:.3f} morton_ms={:.3f} encode_ms={:.3f} unit_wall_ms={:.3f} commit_ms={:.3f} total_ms={:.3f} units={} workers={}", ms(started, prepared), ms(prepared, decimated), ms(decimated, partitioned), morton_ms, encode_ms, ms(partitioned, encoded), ms(encoded, Clock::now()), ms(started, Clock::now()), units.size(), workers);
            return {};
        } catch (const Cancelled&) { return make_error(ErrorCode::CANCELLED, "Export cancelled by user", o.output_path); } catch (const std::exception& e) {
            return make_error(ErrorCode::ENCODING_FAILED, std::string("Failed to save streamed SOG: ") + e.what(), o.output_path);
        }
    }
} // namespace lfs::io
