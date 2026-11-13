/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tile_exposure.hpp"

#include "core/camera.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace lfs::core {
namespace {
    struct PointKey {
        std::array<std::uint32_t, 3> bits;
        bool operator==(const PointKey&) const = default;
        bool operator<(const PointKey& other) const { return bits < other.bits; }
    };

    PointKey point_key(const glm::vec3& point) {
        const auto bits = [](const float value) {
            return std::bit_cast<std::uint32_t>(value == 0.0f ? 0.0f : value);
        };
        return {{bits(point.x), bits(point.y), bits(point.z)}};
    }

    std::uint64_t point_hash(const PointKey& key) {
        std::uint64_t hash = 14695981039346656037ull;
        for (const auto value : key.bits) {
            hash ^= value;
            hash *= 1099511628211ull;
        }
        return hash;
    }

    double median(std::vector<double>& values) {
        const size_t mid = values.size() / 2;
        std::nth_element(values.begin(), values.begin() + mid, values.end());
        const double upper = values[mid];
        if (values.size() % 2 != 0) return upper;
        return 0.5 * (upper + *std::max_element(values.begin(), values.begin() + mid));
    }

    float patch_luminance(const unsigned char* pixels, const int width,
                          const int height, const int channels, const float u,
                          const float v) {
        const int x = static_cast<int>(std::lround(u));
        const int y = static_cast<int>(std::lround(v));
        if (x < 1 || y < 1 || x >= width - 1 || y >= height - 1)
            return std::numeric_limits<float>::quiet_NaN();
        std::array<float, 9> values{};
        int count = 0;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                const size_t offset = (static_cast<size_t>(y + dy) * width + x + dx) * channels;
                const float r = pixels[offset] / 255.0f;
                const float g = channels >= 3 ? pixels[offset + 1] / 255.0f : r;
                const float b = channels >= 3 ? pixels[offset + 2] / 255.0f : r;
                values[count++] = 0.2126f * r + 0.7152f * g + 0.0722f * b;
            }
        }
        std::nth_element(values.begin(), values.begin() + 4, values.end());
        return values[4];
    }

    struct Record {
        PointKey point;
        int camera = -1;
        double log_luminance = 0.0;
    };

    struct Edge {
        int first = -1;
        int second = -1;
        double delta = 0.0;
        double weight = 0.0;
    };

    std::vector<double> solve_exposures(const size_t camera_count,
                                        const std::vector<Edge>& edges,
                                        const std::vector<bool>& active,
                                        const std::vector<double>& weights) {
        std::vector<double> rhs(camera_count), diagonal(camera_count, 1.0);
        double total_weight = 0.0;
        int active_count = 0;
        for (const bool enabled : active) active_count += enabled;
        for (size_t i = 0; i < edges.size(); ++i)
            total_weight += weights[i];
        const double ridge = std::max(1e-6, total_weight /
            std::max(1, active_count) * 1e-4);
        for (size_t i = 0; i < camera_count; ++i)
            if (active[i]) diagonal[i] = ridge;
        for (size_t i = 0; i < edges.size(); ++i) {
            const auto& edge = edges[i];
            const double weight = weights[i];
            rhs[edge.first] -= weight * edge.delta;
            rhs[edge.second] += weight * edge.delta;
            diagonal[edge.first] += weight;
            diagonal[edge.second] += weight;
        }
        const auto multiply = [&](const std::vector<double>& input) {
            std::vector<double> output(camera_count);
            for (size_t i = 0; i < camera_count; ++i)
                output[i] = active[i] ? ridge * input[i] : input[i];
            for (size_t i = 0; i < edges.size(); ++i) {
                const auto& edge = edges[i];
                const double difference = weights[i] * (input[edge.first] - input[edge.second]);
                output[edge.first] += difference;
                output[edge.second] -= difference;
            }
            return output;
        };
        std::vector<double> solution(camera_count), residual = rhs;
        std::vector<double> direction(camera_count);
        double rz = 0.0;
        for (size_t i = 0; i < camera_count; ++i) {
            direction[i] = residual[i] / diagonal[i];
            rz += residual[i] * direction[i];
        }
        const double initial_rz = rz;
        for (int iteration = 0; iteration < 300 && rz > initial_rz * 1e-12; ++iteration) {
            const auto product = multiply(direction);
            double denominator = 0.0;
            for (size_t i = 0; i < camera_count; ++i)
                denominator += direction[i] * product[i];
            if (denominator <= 0.0 || !std::isfinite(denominator)) break;
            const double step = rz / denominator;
            double next_rz = 0.0;
            std::vector<double> preconditioned(camera_count);
            for (size_t i = 0; i < camera_count; ++i) {
                solution[i] += step * direction[i];
                residual[i] -= step * product[i];
                preconditioned[i] = residual[i] / diagonal[i];
                next_rz += residual[i] * preconditioned[i];
            }
            const double beta = next_rz / rz;
            for (size_t i = 0; i < camera_count; ++i)
                direction[i] = preconditioned[i] + beta * direction[i];
            rz = next_rz;
        }
        return solution;
    }
}

std::vector<TileExposureSample> sample_tile_exposure(
    const std::vector<TileCameraEvidence>& cameras,
    const size_t max_tracks_per_camera,
    const std::function<bool()>& stop_requested) {
    std::vector<TileExposureSample> samples;
    if (max_tracks_per_camera == 0) return samples;
    for (const auto& view : cameras) {
        if (stop_requested && stop_requested()) break;
        if (!view.source || view.width <= 0 || view.height <= 0) continue;
        try {
            // The gain is fitted in encoded [0,1] RGB. Float HDR images do not
            // share that transfer function and should keep their original values.
            if (image_quantization_step(view.source->image_path()) == 0.0f) continue;
            const auto& tracks = view.source->sfm_observations();
            if (tracks.empty()) continue;
            std::vector<std::pair<std::uint64_t, size_t>> candidates;
            candidates.reserve(std::min(tracks.size(), max_tracks_per_camera));
            for (size_t i = 0; i < tracks.size(); ++i) {
                const auto& track = tracks[i];
                if (!std::isfinite(track.u) || !std::isfinite(track.v) ||
                    !std::isfinite(track.x) || !std::isfinite(track.y) ||
                    !std::isfinite(track.z)) continue;
                const auto candidate = std::pair{point_hash(point_key({track.x, track.y, track.z})), i};
                if (candidates.size() < max_tracks_per_camera) {
                    candidates.push_back(candidate);
                    std::push_heap(candidates.begin(), candidates.end());
                } else if (candidate < candidates.front()) {
                    std::pop_heap(candidates.begin(), candidates.end());
                    candidates.back() = candidate;
                    std::push_heap(candidates.begin(), candidates.end());
                }
            }
            if (candidates.empty()) continue;
            auto [raw, width, height, channels] = load_image_thumbnail_no_exif(
                view.source->image_path(), 1024);
            std::unique_ptr<unsigned char, decltype(&free_image)> pixels(raw, &free_image);
            if (!pixels || width < 3 || height < 3 || channels < 1) continue;
            const double source_aspect = static_cast<double>(view.width) / view.height;
            const double decoded_aspect = static_cast<double>(width) / height;
            if (std::abs(source_aspect / decoded_aspect - 1.0) > 0.01) continue;
            const float sx = static_cast<float>(width) / view.width;
            const float sy = static_cast<float>(height) / view.height;
            for (const auto& [_, index] : candidates) {
                const auto& track = tracks[index];
                const float luminance = patch_luminance(
                    pixels.get(), width, height, channels,
                    (track.u + 0.5f) * sx - 0.5f,
                    (track.v + 0.5f) * sy - 0.5f);
                if (std::isfinite(luminance) && luminance >= 0.04f && luminance <= 0.96f)
                    samples.push_back({view.uid, {track.x, track.y, track.z}, luminance});
            }
        } catch (const std::exception& error) {
            LOG_WARN("Skipping exposure samples for '{}': {}", view.image_name, error.what());
        }
    }
    return samples;
}

TileExposureCalibration estimate_tile_exposure(
    const std::vector<TileExposureSample>& samples,
    const std::vector<int>& camera_uids) {
    TileExposureCalibration result;
    result.cameras.reserve(camera_uids.size());
    std::unordered_map<int, int> index_by_uid;
    for (const int uid : camera_uids) {
        if (!index_by_uid.emplace(uid, static_cast<int>(result.cameras.size())).second)
            throw std::invalid_argument("Exposure calibration needs unique camera UIDs");
        result.cameras.push_back({uid, 1.0f, 0});
    }
    std::vector<Record> records;
    records.reserve(samples.size());
    for (const auto& sample : samples) {
        const auto it = index_by_uid.find(sample.camera_uid);
        if (it == index_by_uid.end() || !std::isfinite(sample.luminance) ||
            sample.luminance < 0.04f || sample.luminance > 0.96f ||
            !std::isfinite(sample.point.x) || !std::isfinite(sample.point.y) ||
            !std::isfinite(sample.point.z)) continue;
        records.push_back({point_key(sample.point), it->second, std::log(sample.luminance)});
    }
    std::sort(records.begin(), records.end(), [](const Record& a, const Record& b) {
        if (a.point == b.point) return a.camera < b.camera;
        return a.point < b.point;
    });
    std::unordered_map<std::uint64_t, std::vector<double>> pair_differences;
    for (size_t begin = 0; begin < records.size();) {
        size_t end = begin + 1;
        while (end < records.size() && records[end].point == records[begin].point) ++end;
        std::vector<std::pair<int, double>> views;
        for (size_t i = begin; i < end;) {
            size_t next = i + 1;
            std::vector<double> values{records[i].log_luminance};
            while (next < end && records[next].camera == records[i].camera)
                values.push_back(records[next++].log_luminance);
            views.emplace_back(records[i].camera, median(values));
            i = next;
        }
        if (views.size() >= 2) {
            ++result.shared_points;
            for (const auto& [camera, _] : views)
                ++result.cameras[camera].shared_points;
            const auto [reference, reference_value] = views.front();
            for (size_t i = 1; i < views.size(); ++i) {
                const auto [camera, value] = views[i];
                const std::uint64_t key = (static_cast<std::uint64_t>(reference) << 32) |
                                          static_cast<std::uint32_t>(camera);
                pair_differences[key].push_back(value - reference_value);
            }
        }
        begin = end;
    }

    std::vector<Edge> edges;
    std::vector<std::vector<int>> adjacency(camera_uids.size());
    for (auto& [key, differences] : pair_differences) {
        if (differences.size() < 8) continue;
        const double delta = median(differences);
        for (auto& difference : differences)
            difference = std::abs(difference - delta);
        const double mad = median(differences);
        if (mad > 0.35) continue;
        const int first = static_cast<int>(key >> 32);
        const int second = static_cast<int>(key & 0xffffffffu);
        const double weight = std::min<size_t>(differences.size(), 256) /
                              (1.0 + 10.0 * mad);
        const int edge_index = static_cast<int>(edges.size());
        edges.push_back({first, second, delta, weight});
        adjacency[first].push_back(edge_index);
        adjacency[second].push_back(edge_index);
    }
    result.reliable_edges = edges.size();
    std::vector<bool> seen(camera_uids.size(), false), active(camera_uids.size(), false);
    std::vector<int> best_component;
    for (size_t start = 0; start < camera_uids.size(); ++start) {
        if (seen[start] || adjacency[start].empty()) continue;
        std::vector<int> component{static_cast<int>(start)};
        seen[start] = true;
        for (size_t head = 0; head < component.size(); ++head) {
            const int camera = component[head];
            for (const int edge_index : adjacency[camera]) {
                const auto& edge = edges[edge_index];
                const int neighbor = edge.first == camera ? edge.second : edge.first;
                if (!seen[neighbor]) {
                    seen[neighbor] = true;
                    component.push_back(neighbor);
                }
            }
        }
        if (component.size() > best_component.size())
            best_component = std::move(component);
    }
    if (best_component.size() < 2) return result;
    result.connected_cameras = static_cast<int>(best_component.size());
    for (const int index : best_component) active[index] = true;
    std::erase_if(edges, [&](const Edge& edge) {
        return !active[edge.first] || !active[edge.second];
    });
    std::vector<double> weights(edges.size());
    std::transform(edges.begin(), edges.end(), weights.begin(),
                   [](const Edge& edge) { return edge.weight; });
    std::vector<double> exposures;
    for (int pass = 0; pass < 3; ++pass) {
        exposures = solve_exposures(camera_uids.size(), edges, active, weights);
        for (size_t i = 0; i < edges.size(); ++i) {
            const auto& edge = edges[i];
            const double residual = exposures[edge.second] - exposures[edge.first] - edge.delta;
            weights[i] = edge.weight * std::min(1.0, 0.1 / std::max(std::abs(residual), 1e-9));
        }
    }
    std::vector<double> centered;
    centered.reserve(best_component.size());
    for (const int index : best_component) centered.push_back(exposures[index]);
    const double reference = median(centered);
    constexpr double max_log_gain = 0.6931471805599453; // one stop in either direction
    for (const int index : best_component) {
        const double correction = std::clamp(reference - exposures[index],
                                             -max_log_gain, max_log_gain);
        result.cameras[index].gain = std::clamp(
            static_cast<float>(std::exp(correction)), 0.5f, 2.0f);
    }
    return result;
}

} // namespace lfs::core
