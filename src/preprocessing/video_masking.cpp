/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "preprocessing/video_masking.hpp"
#include "preprocessing/operator_view.hpp"
#include "video_mask_cells.hpp"
#include "video_mask_recovery.hpp"

#include "core/cuda_error.hpp"
#include "core/cuda_error_typed.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "core/nn/models/sam2.hpp"
#include "core/path_utils.hpp"
#include "core/source_site.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_cuda_interop.hpp"
#include "io/pipelined_image_loader.hpp"
#include "preprocessing/preprocess.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <deque>
#include <future>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <queue>
#include <set>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace lfs::preprocessing {
    namespace {
        using Clock = std::chrono::steady_clock;
        using core::DataType;
        using core::Device;
        using core::Tensor;
        using core::nn::models::kSam2MaxArea;
        using core::nn::models::kSam2MinIou;
        using core::nn::models::Sam2;
        using core::nn::models::Sam2PointPrompt;

        using detail::CellMotion;
        constexpr int kGridColumns = detail::kCellGrid, kGridRows = detail::kCellGrid, kGridCells = kGridColumns * kGridRows;
        constexpr std::array kPairSeconds{1.0, 2.0, 3.0};
        // Share of fixed observations from which a region moves with the camera. Scene regions stay near zero; a whole
        // operator, whose limbs move, reaches about 0.1 to 0.2 and its torso about 0.4.
        constexpr double kCameraFixedShare = 0.1;
        constexpr size_t kCachedFrames = 128;
        // Bounded asynchronous mask outputs, including downloads in flight.
        constexpr size_t kPendingMaskWrites = 4;
        struct MaskDecodeSettings {
            size_t window = 2, slots = 4, batch = 1, decoders = 2;
        };

        MaskDecodeSettings decode_settings(const size_t budget) {
            if (budget <= (size_t(4) << 30))
                return {1, 4, 1, 1};
            if (budget >= (size_t(20) << 30))
                return {4, 6, 4, 2};
            return {};
        }

        size_t reusable_mask_memory(const size_t free_bytes, const size_t total_bytes) {
            // Completed runs can leave reusable allocations in our own pools.
            // Counting only device-free bytes would reject a subsequent run on a
            // small GPU. Neither query synchronizes or trims active allocations.
            int device = 0;
            cudaMemPool_t pool = nullptr;
            uint64_t reserved = 0, used = 0;
            LFS_CUDA_CHECK(cudaGetDevice(&device));
            LFS_CUDA_CHECK(cudaDeviceGetMemPool(&pool, device));
            LFS_CUDA_CHECK(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &reserved));
            LFS_CUDA_CHECK(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &used));
            const size_t unused = reserved > used ? size_t(reserved - used) : 0;
            size_t available = free_bytes + std::min(total_bytes - free_bytes, unused);
            // Bucket-cache blocks remain CUDA-pool "used", so this is disjoint
            // from reserved-minus-used. Live model and decode storage is excluded.
            const auto cached = core::cuda_bucket_cached_bytes();
            available += std::min(total_bytes - available, size_t(cached));
            return available;
        }

        struct ScheduledPair {
            int first = 0, second = 0;
        };

        // How often each grid cell of a frame kept or changed its content against the frame's scheduled partners.
        struct CellTally {
            std::array<std::uint8_t, kGridCells> fixed{}, moving{};
        };

        double seconds(const Clock::time_point start) {
            return std::chrono::duration<double>(Clock::now() - start).count();
        }

        Error failure(const ErrorCode code, std::string message, std::string detail = {}) {
            return make_error({.code = code,
                               .domain = ErrorDomain::Core,
                               .user_message = std::move(message),
                               .detail = std::move(detail),
                               .detection = LFS_SOURCE_SITE_CURRENT()});
        }

        struct MaskingCancelled {};

        struct MaskCompletion {
            cudaEvent_t ready = nullptr;
            MaskCompletion(cudaStream_t stream) {
                LFS_CUDA_CHECK(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
                LFS_CUDA_CHECK(cudaEventRecord(ready, stream));
            }
            ~MaskCompletion() { LFS_CUDA_LOG_TEARDOWN(cudaEventDestroy(ready), nullptr, "video mask completion event teardown"); }
            MaskCompletion(const MaskCompletion&) = delete;
            MaskCompletion& operator=(const MaskCompletion&) = delete;
        };

        struct MaskSource {
            io::ReadyImage image;
            cudaStream_t stream;
            std::optional<MaskCompletion> completion;
            explicit MaskSource(cudaStream_t value) : stream(value) {}
            ~MaskSource() {
                // Normally an output worker has already passed this event. On an
                // early error, drain input consumers before returning decoder leases.
                if (completion)
                    LFS_CUDA_LOG_TEARDOWN(cudaEventSynchronize(completion->ready), stream, "video mask source drain");
                else
                    LFS_CUDA_LOG_TEARDOWN(cudaStreamSynchronize(stream), stream, "video mask source drain");
            }
        };

        // One decoder queue covers both propagation directions.
        // Requests have unique sequence ids, including the second visit to a seed.
        class MaskFrameLoader {
        public:
            MaskFrameLoader(std::span<const VideoMaskFrame> frames, std::vector<int> order,
                            const MaskDecodeSettings settings)
                : frames_(frames), order_(std::move(order)), loader_(configuration(settings)) {
                for (; submitted_ < std::min(settings.window, order_.size()); ++submitted_)
                    prefetch(submitted_);
            }

            io::ReadyImage get(int image) {
                if (next_ >= order_.size() || order_[next_] != image)
                    throw std::logic_error("Video mask decode schedule diverged from tracking");
                while (!ready_.contains(next_)) {
                    auto decoded = loader_.get();
                    ready_.emplace(decoded.sequence_id, std::move(decoded));
                }
                auto result = std::move(ready_.extract(next_++).mapped());
                if (submitted_ < order_.size())
                    prefetch(submitted_++);
                return result;
            }

        private:
            static io::PipelinedLoaderConfig configuration(const MaskDecodeSettings settings) {
                io::PipelinedLoaderConfig config;
                // Completed CUDA readers release leases independently of PNG
                // completion. Even the throughput mode has a bounded six-slot ring.
                config.jpeg_batch_size = settings.batch;
                config.output_queue_size = config.prefetch_count = settings.window;
                // Use two decoders when memory allows, while bounding the frame ring.
                config.decoder_pool_size = settings.decoders;
                config.decode_frame_ring_capacity = settings.slots;
                return config;
            }
            void prefetch(size_t request) {
                loader_.prefetch(request, frames_[size_t(order_[request])].image_path, {.skip_blob_cache = true});
            }
            std::span<const VideoMaskFrame> frames_;
            std::vector<int> order_;
            io::PipelinedImageLoader loader_;
            std::map<size_t, io::ReadyImage> ready_;
            size_t submitted_ = 0, next_ = 0;
        };

        io::PipelinedLoaderConfig single_frame_loader(const size_t ring) {
            io::PipelinedLoaderConfig config;
            config.jpeg_batch_size = 1;
            config.prefetch_count = 1;
            config.output_queue_size = 1;
            config.decoder_pool_size = 1;
            config.decode_frame_ring_capacity = ring;
            return config;
        }

        // Cell patches of decoded frames, computed on demand and kept for the most recently used frames.
        class FrameCellSource {
        public:
            FrameCellSource(std::span<const VideoMaskFrame> frames, const VideoMaskOptions& options,
                            cudaStream_t stream)
                : frames_(frames), options_(options), stream_(stream) {}

            int image_count() const { return int(frames_.size()); }
            int width(int image) const { return frames_[size_t(image)].width; }
            int height(int image) const { return frames_[size_t(image)].height; }

            void notify(VideoMaskStage stage, size_t completed, size_t total) const {
                if (options_.cancel_requested && options_.cancel_requested())
                    throw MaskingCancelled{};
                if (options_.progress)
                    options_.progress({stage, completed, total});
            }

            detail::CellComparison compare(const int first, const int second) const {
                const auto& a = load(first);
                const auto& b = load(second);
                return detail::compare_cells(a.fine, b.fine, a.scene, b.scene);
            }

            void release() {
                patches_.clear();
                loader_.reset();
            }

        private:
            struct FramePatches {
                detail::CellPatches fine, scene;
                uint64_t last_used = 0;
            };

            const FramePatches& load(const int image) const {
                if (const auto cached = patches_.find(image); cached != patches_.end()) {
                    cached->second.last_used = ++access_;
                    return cached->second;
                }
                notify(VideoMaskStage::Patches, std::min(completed_++, frames_.size()), frames_.size());
                if (!loader_)
                    loader_ = std::make_unique<io::PipelinedImageLoader>(single_frame_loader(2));
                loader_->prefetch(size_t(image), frames_[size_t(image)].image_path, {.skip_blob_cache = true});
                auto ready = loader_->get();
                if (!ready.error.empty())
                    throw std::runtime_error(ready.error);
                const auto& rgb = ready.tensor;
                if (rgb.ndim() != 3 || rgb.size(0) != 3 || rgb.size(1) != size_t(height(image)) ||
                    rgb.size(2) != size_t(width(image)))
                    throw std::invalid_argument("Decoded video frame dimensions or channels changed");
                if (ready.image_ready)
                    ready.image_ready->wait_on(static_cast<void*>(stream_));
                ready.tensor.set_stream(stream_);
                const auto gray = rgb.slice(0, 0, 1)
                                      .mul(0.299f)
                                      .add(rgb.slice(0, 1, 2).mul(0.587f))
                                      .add(rgb.slice(0, 2, 3).mul(0.114f))
                                      .unsqueeze(0);
                FramePatches value{detail::cell_patches(gray, detail::kCellGrid), detail::cell_patches(gray, detail::kSceneGrid),
                                   ++access_};
                LFS_CUDA_CHECK(cudaStreamSynchronize(stream_));
                if (patches_.size() >= kCachedFrames)
                    patches_.erase(std::ranges::min_element(patches_, {}, [](const auto& entry) {
                        return entry.second.last_used;
                    }));
                return patches_.emplace(image, std::move(value)).first->second;
            }

            std::span<const VideoMaskFrame> frames_;
            const VideoMaskOptions& options_;
            cudaStream_t stream_;
            mutable std::map<int, FramePatches> patches_;
            mutable std::unique_ptr<io::PipelinedImageLoader> loader_;
            mutable size_t completed_ = 0;
            mutable uint64_t access_ = 0;
        };

        bool inside_circle(const ImageCircle& circle, const double normalized_x, const double normalized_y) {
            if (!(circle.radius_x > 0 && circle.radius_y > 0))
                return normalized_x >= 0 && normalized_x <= 1 && normalized_y >= 0 && normalized_y <= 1;
            const double x = (normalized_x - circle.center_x) / circle.radius_x;
            const double y = (normalized_y - circle.center_y) / circle.radius_y;
            return x * x + y * y <= 1;
        }

        std::vector<std::uint8_t> circle_domain(const ImageCircle& circle) {
            std::vector<std::uint8_t> domain(kGridCells);
            for (int row = 0; row < kGridRows; ++row)
                for (int column = 0; column < kGridColumns; ++column)
                    domain[size_t(row * kGridColumns + column)] =
                        inside_circle(circle, (column + 0.5) / kGridColumns, (row + 0.5) / kGridRows);
            return domain;
        }

        std::vector<int> boundary_distance(const std::span<const std::uint8_t> domain) {
            std::vector<int> distance(kGridCells, std::numeric_limits<int>::max());
            std::queue<int> pending;
            for (int cell = 0; cell < kGridCells; ++cell) {
                if (!domain[size_t(cell)])
                    continue;
                const int x = cell % kGridColumns, y = cell / kGridColumns;
                bool boundary = x == 0 || x + 1 == kGridColumns || y == 0 || y + 1 == kGridRows;
                for (int dy = -1; dy <= 1 && !boundary; ++dy)
                    for (int dx = -1; dx <= 1; ++dx) {
                        const int nx = x + dx, ny = y + dy;
                        boundary |= nx < 0 || nx >= kGridColumns || ny < 0 || ny >= kGridRows ||
                                    !domain[size_t(ny * kGridColumns + nx)];
                    }
                if (boundary) {
                    distance[size_t(cell)] = 0;
                    pending.push(cell);
                }
            }
            while (!pending.empty()) {
                const int cell = pending.front();
                pending.pop();
                const int x = cell % kGridColumns, y = cell / kGridColumns;
                constexpr std::array<std::array<int, 2>, 4> steps{{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}};
                for (const auto [dx, dy] : steps) {
                    const int nx = x + dx, ny = y + dy;
                    if (nx < 0 || nx >= kGridColumns || ny < 0 || ny >= kGridRows)
                        continue;
                    const int next = ny * kGridColumns + nx;
                    if (domain[size_t(next)] && distance[size_t(next)] > distance[size_t(cell)] + 1) {
                        distance[size_t(next)] = distance[size_t(cell)] + 1;
                        pending.push(next);
                    }
                }
            }
            return distance;
        }

        std::vector<ScheduledPair> temporal_pairs(const std::span<const VideoMaskFrame> frames) {
            std::vector<int> members(frames.size());
            std::iota(members.begin(), members.end(), 0);
            std::set<std::pair<int, int>> unique;
            std::vector<ScheduledPair> pairs;

            std::sort(members.begin(), members.end(), [&](const int a, const int b) {
                return std::tie(frames[size_t(a)].time_seconds, a) <
                       std::tie(frames[size_t(b)].time_seconds, b);
            });
            for (size_t i = 0; i < members.size(); ++i) {
                for (const double separation : kPairSeconds) {
                    const double target = frames[size_t(members[i])].time_seconds + separation;
                    const auto after =
                        std::lower_bound(members.begin() + std::ptrdiff_t(i + 1), members.end(), target,
                                         [&](const int image, const double time) {
                                             return frames[size_t(image)].time_seconds < time;
                                         });
                    if (after == members.end())
                        continue;
                    int partner = *after;
                    if (after != members.begin() + std::ptrdiff_t(i + 1)) {
                        const int before = *std::prev(after);
                        if (std::abs(frames[size_t(before)].time_seconds - target) <
                            std::abs(frames[size_t(partner)].time_seconds - target))
                            partner = before;
                    }
                    const double actual = frames[size_t(partner)].time_seconds -
                                          frames[size_t(members[i])].time_seconds;
                    if (actual < 0.5 * separation || actual > 1.5 * separation)
                        continue;
                    const auto key = std::minmax(members[i], partner);
                    if (unique.insert(key).second)
                        pairs.push_back({members[i], partner});
                }
            }

            return pairs;
        }

        size_t bit_count(const std::span<const std::uint64_t> words) {
            return std::accumulate(words.begin(), words.end(), size_t(0), [](const size_t sum, const auto word) {
                return sum + std::popcount(word);
            });
        }

        std::vector<std::vector<int>> components_of(const std::span<const std::uint8_t> cells) {
            std::vector<std::vector<int>> components;
            std::vector<std::uint8_t> visited(kGridCells);
            for (int origin = 0; origin < kGridCells; ++origin) {
                if (!cells[size_t(origin)] || visited[size_t(origin)])
                    continue;
                std::vector<int> component;
                std::queue<int> pending;
                pending.push(origin);
                visited[size_t(origin)] = 1;
                while (!pending.empty()) {
                    const int cell = pending.front();
                    pending.pop();
                    component.push_back(cell);
                    const int x = cell % kGridColumns, y = cell / kGridColumns;
                    for (int dy = -1; dy <= 1; ++dy)
                        for (int dx = -1; dx <= 1; ++dx) {
                            const int nx = x + dx, ny = y + dy;
                            if (nx < 0 || nx >= kGridColumns || ny < 0 || ny >= kGridRows)
                                continue;
                            const int next = ny * kGridColumns + nx;
                            if (cells[size_t(next)] && !visited[size_t(next)]) {
                                visited[size_t(next)] = 1;
                                pending.push(next);
                            }
                        }
                }
                components.push_back(std::move(component));
            }
            return components;
        }

        std::vector<int> select_positive_cells(const VideoMaskDetectionReport& report) {
            std::vector<int> candidates;
            for (int cell = 0; cell < kGridCells; ++cell)
                if (report.prompt_cells[size_t(cell)])
                    candidates.push_back(cell);
            if (candidates.empty())
                return {};
            std::vector<int> selected;
            selected.push_back(*std::max_element(candidates.begin(), candidates.end(), [&](const int a, const int b) {
                return report.evidence[size_t(a)].fixed_pairs < report.evidence[size_t(b)].fixed_pairs;
            }));
            while (selected.size() < 6) {
                int best = -1;
                double best_score = 0;
                for (const int cell : candidates) {
                    if (std::ranges::find(selected, cell) != selected.end())
                        continue;
                    const int x = cell % kGridColumns, y = cell / kGridColumns;
                    int minimum = std::numeric_limits<int>::max();
                    for (const int chosen : selected) {
                        const int dx = x - chosen % kGridColumns, dy = y - chosen / kGridColumns;
                        minimum = std::min(minimum, dx * dx + dy * dy);
                    }
                    if (minimum < 9)
                        continue;
                    const double score = report.evidence[size_t(cell)].fixed_pairs * std::sqrt(double(minimum));
                    if (score > best_score) {
                        best = cell;
                        best_score = score;
                    }
                }
                if (best < 0)
                    break;
                selected.push_back(best);
            }
            if (selected.size() == 1) {
                const auto second = std::ranges::max_element(candidates, [&](const int a, const int b) {
                    const size_t a_pairs = a == selected.front() ? 0 : report.evidence[size_t(a)].fixed_pairs;
                    const size_t b_pairs = b == selected.front() ? 0 : report.evidence[size_t(b)].fixed_pairs;
                    return a_pairs < b_pairs;
                });
                if (second != candidates.end() && *second != selected.front())
                    selected.push_back(*second);
            }
            return selected;
        }

        std::vector<int> select_negative_cells(const VideoMaskDetectionReport& report,
                                               const std::span<const std::uint8_t> domain) {
            std::vector<int> ranked(kGridCells);
            std::iota(ranked.begin(), ranked.end(), 0);
            std::sort(ranked.begin(), ranked.end(), [&](const int a, const int b) {
                return report.evidence[size_t(a)].moving_pairs > report.evidence[size_t(b)].moving_pairs;
            });
            std::vector<int> selected;
            const auto far_from_prompts = [&](const int cell) {
                const int x = cell % kGridColumns, y = cell / kGridColumns;
                for (int prompt = 0; prompt < kGridCells; ++prompt) {
                    if (!report.prompt_cells[size_t(prompt)])
                        continue;
                    const int dx = x - prompt % kGridColumns, dy = y - prompt / kGridColumns;
                    if (dx * dx + dy * dy < 36)
                        return false;
                }
                return true;
            };
            for (const int cell : ranked) {
                const auto& evidence = report.evidence[size_t(cell)];
                if (!domain[size_t(cell)] || evidence.moving_pairs < 6 ||
                    std::popcount(evidence.moving_time_bins) < 2 || !far_from_prompts(cell))
                    continue;
                const int x = cell % kGridColumns, y = cell / kGridColumns;
                if (std::ranges::any_of(selected, [&](const int chosen) {
                        const int dx = x - chosen % kGridColumns, dy = y - chosen / kGridColumns;
                        return dx * dx + dy * dy < 36;
                    }))
                    continue;
                selected.push_back(cell);
                if (selected.size() == 8)
                    break;
            }
            if (selected.size() < 4) {
                std::sort(ranked.begin(), ranked.end(), [&](const int a, const int b) {
                    const int ax = a % kGridColumns, ay = a / kGridColumns;
                    const int bx = b % kGridColumns, by = b / kGridColumns;
                    const auto nearest = [&](const int x, const int y) {
                        int distance = std::numeric_limits<int>::max();
                        for (int prompt = 0; prompt < kGridCells; ++prompt) {
                            if (!report.prompt_cells[size_t(prompt)])
                                continue;
                            const int dx = x - prompt % kGridColumns, dy = y - prompt / kGridColumns;
                            distance = std::min(distance, dx * dx + dy * dy);
                        }
                        return distance;
                    };
                    return nearest(ax, ay) > nearest(bx, by);
                });
                for (const int cell : ranked) {
                    if (!domain[size_t(cell)] || !far_from_prompts(cell) ||
                        std::ranges::find(selected, cell) != selected.end())
                        continue;
                    const int x = cell % kGridColumns, y = cell / kGridColumns;
                    if (std::ranges::any_of(selected, [&](const int chosen) {
                            const int dx = x - chosen % kGridColumns, dy = y - chosen / kGridColumns;
                            return dx * dx + dy * dy < 36;
                        }))
                        continue;
                    selected.push_back(cell);
                    if (selected.size() == 4)
                        break;
                }
            }
            return selected;
        }

        std::vector<CellTally> detect_prompts(const FrameCellSource& input,
                                              const std::span<const VideoMaskFrame> frames,
                                              const ImageCircle& circle,
                                              VideoMaskDetectionReport& report, const VideoMaskOptions& options) {
            std::vector<CellTally> tallies(frames.size());
            report.images = frames.size();
            report.evidence.resize(kGridCells);
            report.prompt_cells.assign(kGridCells, 0);
            if (!options.enabled) {
                report.status = "disabled";
                return tallies;
            }
            if (options.prompt.has_value()) {
                report.status = "accepted";
                return tallies;
            }
            const auto domain = circle_domain(circle);
            const auto distance = boundary_distance(domain);
            const size_t frame_words = (report.images + 63) / 64;
            std::vector<std::uint64_t> support_frames(size_t(kGridCells) * frame_words);

            const auto pairs = temporal_pairs(frames);
            report.pairs_tested = pairs.size();
            const auto [first, last] = std::ranges::minmax_element(frames, {}, &VideoMaskFrame::time_seconds);
            const std::array time_range{first->time_seconds, last->time_seconds};
            for (size_t pair_index = 0; pair_index < pairs.size(); ++pair_index) {
                const auto& pair = pairs[pair_index];
                input.notify(VideoMaskStage::Motion, pair_index, pairs.size());
                if (input.width(pair.first) != input.width(pair.second) ||
                    input.height(pair.first) != input.height(pair.second))
                    continue;
                const auto comparison = input.compare(pair.first, pair.second);
                if (comparison.scene_correlation > detail::kMaximumSceneCorrelation)
                    continue;
                ++report.motion_pairs;
                const double midpoint = 0.5 * (frames[size_t(pair.first)].time_seconds +
                                               frames[size_t(pair.second)].time_seconds);
                const double time_fraction =
                    (midpoint - time_range[0]) / std::max(1e-9, time_range[1] - time_range[0]);
                const auto time_bin = std::uint8_t(1 << std::clamp(int(3 * time_fraction), 0, 2));
                for (int cell = 0; cell < kGridCells; ++cell) {
                    const auto motion = comparison.cells[size_t(cell)];
                    if (!domain[size_t(cell)] || motion == CellMotion::Unknown)
                        continue;
                    auto& evidence = report.evidence[size_t(cell)];
                    for (const int image : {pair.first, pair.second}) {
                        auto& count = (motion == CellMotion::Fixed ? tallies[size_t(image)].fixed
                                                                   : tallies[size_t(image)].moving)[size_t(cell)];
                        count = std::uint8_t(std::min(255, count + 1));
                    }
                    if (motion == CellMotion::Moving) {
                        ++evidence.moving_pairs;
                        evidence.moving_time_bins |= time_bin;
                        continue;
                    }
                    ++evidence.fixed_pairs;
                    evidence.fixed_time_bins |= time_bin;
                    for (const int local : {pair.first, pair.second})
                        support_frames[size_t(cell) * frame_words + size_t(local) / 64] |=
                            std::uint64_t{1} << (size_t(local) % 64);
                }
            }

            if (report.motion_pairs < 24) {
                report.status = "insufficient_motion_pairs";
                return tallies;
            }
            const size_t minimum_frames = std::max<size_t>(10, (report.images + 11) / 12);
            std::vector<std::uint8_t> candidate(kGridCells), strong(kGridCells);
            for (int cell = 0; cell < kGridCells; ++cell) {
                auto& evidence = report.evidence[size_t(cell)];
                evidence.support_frames = bit_count(std::span(
                    support_frames.data() + size_t(cell) * frame_words, frame_words));
                candidate[size_t(cell)] = domain[size_t(cell)] && distance[size_t(cell)] <= 5 &&
                                          evidence.fixed_pairs >= 3 &&
                                          std::popcount(evidence.fixed_time_bins) >= 2 &&
                                          evidence.support_frames >= minimum_frames;
                strong[size_t(cell)] = candidate[size_t(cell)] && evidence.fixed_pairs >= 8;
            }
            for (const auto& component : components_of(candidate)) {
                const bool has_strong = std::ranges::any_of(component, [&](const int cell) {
                    return strong[size_t(cell)] != 0;
                });
                size_t maximum_pairs = 0;
                for (const int cell : component)
                    maximum_pairs = std::max(maximum_pairs, report.evidence[size_t(cell)].fixed_pairs);
                if (!has_strong || (component.size() < 2 && maximum_pairs < 16))
                    continue;
                for (const int cell : component)
                    report.prompt_cells[size_t(cell)] = 1;
            }
            std::vector<std::uint64_t> prompt_frames(frame_words);
            std::uint8_t prompt_bins = 0;
            for (int cell = 0; cell < kGridCells; ++cell) {
                if (!report.prompt_cells[size_t(cell)])
                    continue;
                prompt_bins |= report.evidence[size_t(cell)].fixed_time_bins;
                for (size_t word = 0; word < frame_words; ++word)
                    prompt_frames[word] |= support_frames[size_t(cell) * frame_words + word];
            }
            if (std::popcount(prompt_bins) != 3 ||
                bit_count(prompt_frames) < std::max<size_t>(12, (report.images + 5) / 6)) {
                report.status = "insufficient_prompt_persistence";
                std::fill(report.prompt_cells.begin(), report.prompt_cells.end(), 0);
                return tallies;
            }
            const auto positive = select_positive_cells(report);
            if (positive.size() < 2) {
                report.status = "insufficient_prompt_anchors";
                std::fill(report.prompt_cells.begin(), report.prompt_cells.end(), 0);
                return tallies;
            }
            for (const int cell : positive)
                report.positive_cells.push_back({cell % kGridColumns, cell / kGridColumns});
            for (const int cell : select_negative_cells(report, domain))
                report.negative_points.push_back(
                    {(cell % kGridColumns + 0.5) / kGridColumns,
                     (cell / kGridColumns + 0.5) / kGridRows});
            report.status = "accepted";
            return tallies;
        }

        std::vector<std::uint8_t> circle_keep_mask(const int width, const int height,
                                                   const ImageCircle& circle) {
            std::vector<std::uint8_t> keep(size_t(width) * height, 255);
            if (!(circle.radius_x > 0 && circle.radius_y > 0))
                return keep;
            for (int y = 0; y < height; ++y)
                for (int x = 0; x < width; ++x)
                    if (!inside_circle(circle, (x + 0.5) / width, (y + 0.5) / height))
                        keep[size_t(y) * width + x] = 0;
            return keep;
        }

        Tensor byte_tensor(const std::span<const std::uint8_t> bytes, const int height, const int width) {
            auto tensor = Tensor::empty({size_t(height), size_t(width)}, Device::CPU, DataType::UInt8);
            std::memcpy(tensor.data_ptr(), bytes.data(), bytes.size());
            return tensor;
        }

        std::vector<Sam2PointPrompt> prompts_for_image(const VideoMaskDetectionReport& detection, const int width,
                                                       const int height) {
            std::vector<Sam2PointPrompt> prompts;
            prompts.reserve(detection.positive_cells.size() + detection.negative_points.size());
            for (const auto& [column, row] : detection.positive_cells)
                prompts.push_back({(column + 0.5f) / kGridColumns * width, (row + 0.5f) / kGridRows * height, 1});
            // Moving limbs also occupy the image rim; use the central scene for background prompts.
            for (const auto& point : detection.negative_points)
                if (std::hypot(point[0] - 0.5, point[1] - 0.5) < 0.32)
                    prompts.push_back({float(point[0] * width), float(point[1] * height), 0});
            return prompts;
        }

        constexpr int kMaskSide = 512;
        using Mask = std::vector<std::uint8_t>;

        Mask morphology(const Mask& input, const int radius, const bool dilate) {
            constexpr int stride = kMaskSide + 1;
            std::vector<int> integral(size_t(stride) * stride);
            for (int y = 0; y < kMaskSide; ++y) {
                int row = 0;
                for (int x = 0; x < kMaskSide; ++x) {
                    row += input[size_t(y * kMaskSide + x)];
                    integral[size_t((y + 1) * stride + x + 1)] =
                        row + integral[size_t(y * stride + x + 1)];
                }
            }
            Mask output(input.size());
            for (int y = 0; y < kMaskSide; ++y)
                for (int x = 0; x < kMaskSide; ++x) {
                    const int left = std::max(0, x - radius), right = std::min(kMaskSide, x + radius + 1);
                    const int top = std::max(0, y - radius), bottom = std::min(kMaskSide, y + radius + 1);
                    const int sum = integral[size_t(bottom * stride + right)] -
                                    integral[size_t(top * stride + right)] -
                                    integral[size_t(bottom * stride + left)] + integral[size_t(top * stride + left)];
                    output[size_t(y * kMaskSide + x)] = dilate ? sum > 0 : sum == (right - left) * (bottom - top);
                }
            return output;
        }

        int mask_pixel(const float x, const float y, const int width, const int height) {
            return std::clamp(int(y / height * kMaskSide), 0, kMaskSide - 1) * kMaskSide +
                   std::clamp(int(x / width * kMaskSide), 0, kMaskSide - 1);
        }

        int cell_mask_pixel(const int cell) {
            return mask_pixel(float(cell % kGridColumns) + 0.5f, float(cell / kGridColumns) + 0.5f, kGridColumns, kGridRows);
        }

        // Share of fixed observations among an image's cells whose centres lie in `region` and outside `excluded`.
        std::pair<size_t, double> fixed_share(const CellTally& tally, const Mask& region, const Mask* excluded = nullptr) {
            size_t fixed = 0, moving = 0;
            for (int cell = 0; cell < kGridCells; ++cell) {
                const auto pixel = size_t(cell_mask_pixel(cell));
                if (!region[pixel] || (excluded && (*excluded)[pixel]))
                    continue;
                fixed += tally.fixed[size_t(cell)];
                moving += tally.moving[size_t(cell)];
            }
            return {fixed, fixed + moving ? double(fixed) / double(fixed + moving) : 0.0};
        }

        Mask seeded_components(const Mask& input, const std::span<const Sam2PointPrompt> prompts,
                               const int width, const int height) {
            Mask seeds(input.size()), visited(input.size()), result(input.size());
            for (const auto& prompt : prompts)
                if (prompt.label == 1)
                    seeds[size_t(mask_pixel(prompt.x, prompt.y, width, height))] = 1;
            std::vector<int> component;
            for (int origin = 0; origin < int(input.size()); ++origin) {
                if (!input[size_t(origin)] || visited[size_t(origin)])
                    continue;
                component.clear();
                component.push_back(origin);
                visited[size_t(origin)] = 1;
                size_t seed_hits = 0;
                for (size_t next = 0; next < component.size(); ++next) {
                    const int cell = component[next], x = cell % kMaskSide, y = cell / kMaskSide;
                    seed_hits += seeds[size_t(cell)];
                    for (int dy = -1; dy <= 1; ++dy)
                        for (int dx = -1; dx <= 1; ++dx) {
                            const int nx = x + dx, ny = y + dy;
                            if (nx < 0 || nx >= kMaskSide || ny < 0 || ny >= kMaskSide)
                                continue;
                            const int neighbor = ny * kMaskSide + nx;
                            if (input[size_t(neighbor)] && !visited[size_t(neighbor)]) {
                                visited[size_t(neighbor)] = 1;
                                component.push_back(neighbor);
                            }
                        }
                }
                if (component.size() >= 8 && seed_hits)
                    for (const int cell : component)
                        result[size_t(cell)] = 1;
            }
            return result;
        }

        void fill_holes(Mask& mask) {
            Mask exterior(mask.size());
            std::vector<int> pending;
            const auto append = [&](const int cell) {
                if (!mask[size_t(cell)] && !exterior[size_t(cell)]) {
                    exterior[size_t(cell)] = 1;
                    pending.push_back(cell);
                }
            };
            for (int i = 0; i < kMaskSide; ++i) {
                append(i);
                append((kMaskSide - 1) * kMaskSide + i);
                append(i * kMaskSide);
                append(i * kMaskSide + kMaskSide - 1);
            }
            for (size_t next = 0; next < pending.size(); ++next) {
                const int cell = pending[next], x = cell % kMaskSide, y = cell / kMaskSide;
                if (x)
                    append(cell - 1);
                if (x + 1 < kMaskSide)
                    append(cell + 1);
                if (y)
                    append(cell - kMaskSide);
                if (y + 1 < kMaskSide)
                    append(cell + kMaskSide);
            }
            for (size_t cell = 0; cell < mask.size(); ++cell)
                mask[cell] = !exterior[cell];
        }

        std::optional<std::array<float, 4>> seed_box(const std::span<const Sam2PointPrompt> prompts,
                                                     const int width, const int height) {
            float left = float(width), top = float(height), right = 0, bottom = 0;
            for (const auto& point : prompts)
                if (point.label == 1) {
                    left = std::min(left, point.x);
                    top = std::min(top, point.y);
                    right = std::max(right, point.x);
                    bottom = std::max(bottom, point.y);
                }
            if (left > right || top > bottom)
                return std::nullopt;
            return std::array<float, 4>{std::max(0.0f, left - 0.3f * width),
                                        std::max(0.0f, top - 0.3f * height),
                                        std::min(float(width), right + 0.3f * width),
                                        std::min(float(height), bottom + 0.3f * height)};
        }

        std::vector<Sam2PointPrompt> refinement_points(const Mask& mask, const CellTally& tally, const int width,
                                                       const int height, const ImageCircle& circle) {
            if (mask.empty())
                return {};
            const auto covered = morphology(mask, 3, true);
            const auto nearby = morphology(mask, 64, true);
            std::vector<int> ranked(kGridCells);
            std::iota(ranked.begin(), ranked.end(), 0);
            std::sort(ranked.begin(), ranked.end(), [&](const int a, const int b) {
                return tally.fixed[size_t(a)] > tally.fixed[size_t(b)];
            });
            std::vector<Sam2PointPrompt> points;
            for (const int cell : ranked) {
                const int fixed = tally.fixed[size_t(cell)], moving = tally.moving[size_t(cell)];
                if (fixed < 2 || fixed < kCameraFixedShare * (fixed + moving))
                    continue;
                const float x = (float(cell % kGridColumns) + 0.5f) / kGridColumns;
                const float y = (float(cell / kGridColumns) + 0.5f) / kGridRows;
                if (circle.radius_x > 0 && circle.radius_y > 0 &&
                    std::hypot((x - circle.center_x) / circle.radius_x, (y - circle.center_y) / circle.radius_y) < 0.8)
                    continue;
                const auto pixel = size_t(cell_mask_pixel(cell));
                if (covered[pixel] || !nearby[pixel] ||
                    std::ranges::any_of(points, [&](const auto& chosen) {
                        return std::hypot(x - chosen.x / width, y - chosen.y / height) < 0.06;
                    }))
                    continue;
                points.push_back({x * width, y * height, 1});
                if (points.size() == 6)
                    break;
            }
            return points;
        }

        Mask choose_seed_candidate(const core::nn::models::Sam2Prediction& prediction,
                                   const std::span<const Sam2PointPrompt> prompts, const ImageCircle& circle,
                                   const int image_width, const int image_height,
                                   const CellTally& tally, const bool primary,
                                   cudaStream_t stream) {
            const int count = int(prediction.low_res_logits.size(0));
            const auto low = core::nn::resize2d(
                                 prediction.low_res_logits.reshape({count, 1, 256, 256}),
                                 kMaskSide, kMaskSide, core::nn::ResizeMode::Bilinear,
                                 core::nn::CoordTransform::HalfPixel)
                                 .to(Device::CPU, stream)
                                 .contiguous();
            const auto quality = prediction.iou.to(Device::CPU, stream).contiguous();
            LFS_CUDA_CHECK(cudaStreamSynchronize(stream));
            size_t circle_pixels = 0;
            Mask domain(size_t(kMaskSide) * kMaskSide);
            for (int y = 0; y < kMaskSide; ++y)
                for (int x = 0; x < kMaskSide; ++x) {
                    const bool inside = inside_circle(circle, (x + 0.5) / kMaskSide, (y + 0.5) / kMaskSide);
                    domain[size_t(y * kMaskSide + x)] = inside;
                    circle_pixels += inside;
                }
            const int positives = std::ranges::count_if(prompts, [](const auto& point) { return point.label == 1; });
            Mask best;
            double best_score = -std::numeric_limits<double>::infinity();
            for (int candidate = 0; candidate < count; ++candidate) {
                const float* logits = low.ptr<float>() + size_t(candidate) * domain.size();
                Mask object(domain.size());
                for (size_t pixel = 0; pixel < object.size(); ++pixel)
                    object[pixel] = domain[pixel] && logits[pixel] > 0;
                object = morphology(morphology(object, 3, true), 3, false);
                object = seeded_components(object, prompts, image_width, image_height);
                fill_holes(object);
                int positive_hits = 0, negative_hits = 0;
                size_t area = 0;
                for (size_t pixel = 0; pixel < object.size(); ++pixel) {
                    object[pixel] &= domain[pixel];
                    area += object[pixel];
                }
                for (const auto& prompt : prompts) {
                    const bool hit = object[size_t(mask_pixel(prompt.x, prompt.y, image_width, image_height))];
                    positive_hits += prompt.label == 1 && hit;
                    negative_hits += prompt.label == 0 && hit;
                }
                const double fraction = circle_pixels ? double(area) / circle_pixels : 0;
                if (positive_hits < std::max(1, (2 * positives + 2) / 3) ||
                    negative_hits > 1 || fraction < (primary ? 0.002 : 0.0001) || fraction > 0.35)
                    continue;
                const double predicted_iou = quality.ptr<float>()[candidate];
                const auto [fixed, share] = fixed_share(tally, object);
                if (fixed < 3 || share < kCameraFixedShare)
                    continue;
                // The quality head favors small parts at the fisheye edge even when the full silhouette is usable.
                const double score = 0.05 * predicted_iou + std::sqrt(fraction) +
                                     0.5 * positive_hits / std::max(1, positives) - 0.3 * negative_hits;
                if (score > best_score) {
                    best = std::move(object);
                    best_score = score;
                }
            }
            return best;
        }

        void write_mask(const std::filesystem::path& path, const std::span<const std::uint8_t> keep,
                        const int width, const int height) {
            core::native_fs::create_directories(path.parent_path());
            if (!core::save_png(path, keep.data(), width, height, 1, 8, 1))
                throw std::runtime_error("Cannot write the SAM2 mask " + core::path_to_utf8(path));
        }

    } // namespace

    std::expected<VideoMaskResourcePlan, Error> plan_video_mask_resources(
        const std::span<const VideoMaskFrame> frames, const VideoMaskOptions& options,
        const size_t available_bytes) {
        if (frames.empty())
            return std::unexpected(failure(ErrorCode::InvalidArgument, "Invalid video masking resource request"));
        // Model weights, arena, temporal state, two attention score tiles and
        // allocator slack; full-resolution decode, remap and output storage follow.
        constexpr size_t model_bytes = size_t(2) << 30;
        const size_t budget = options.gpu_memory_budget_bytes ? std::min(available_bytes, options.gpu_memory_budget_bytes) : available_bytes;
        const auto decode = decode_settings(budget);
        const size_t pixel_bytes = (decode.slots + decode.batch - 1) * 3 * sizeof(float) + 40;
        size_t pixels = 0;
        for (const auto& frame : frames) {
            if (frame.width <= 0 || frame.height <= 0 ||
                size_t(frame.width) > (std::numeric_limits<size_t>::max() - model_bytes) / pixel_bytes / size_t(frame.height))
                return std::unexpected(failure(ErrorCode::InvalidArgument, "Invalid video masking dimensions"));
            pixels = std::max(pixels, size_t(frame.width) * size_t(frame.height));
        }
        VideoMaskResourcePlan plan{
            .budget_bytes = budget,
            .reserve_bytes = size_t(512) << 20,
            .worker_bytes = model_bytes + pixels * pixel_bytes};
        if (plan.budget_bytes <= plan.reserve_bytes || plan.budget_bytes - plan.reserve_bytes < plan.worker_bytes)
            return std::unexpected(failure(ErrorCode::ResourceExhausted, "Insufficient GPU memory for one video masking worker",
                                           "Estimated worker bytes: " + std::to_string(plan.worker_bytes) +
                                               ", budget bytes: " + std::to_string(plan.budget_bytes)));
        return plan;
    }

    std::expected<VideoMaskReport, Error> mask_video_sequence(
        const std::span<const VideoMaskFrame> frames,
        const ImageCircle& circle,
        const VideoMaskOptions& options, cudaStream_t stream) {
        if (frames.empty() || options.dilation_radius < 0 || options.dilation_radius > 128 ||
            (options.require_operator && !options.enabled) ||
            (options.recenter_fisheye && (!(options.fisheye_fov_degrees > 0 && options.fisheye_fov_degrees < 360) ||
                                          !(options.view_fov_degrees > 0 && options.view_fov_degrees < 360))))
            return std::unexpected(failure(ErrorCode::InvalidArgument, "Invalid video masking inputs"));
        for (const auto& frame : frames)
            if (frame.width <= 0 || frame.height <= 0 ||
                frame.mask_path.empty() || frame.image_path.empty() || !std::isfinite(frame.time_seconds))
                return std::unexpected(failure(ErrorCode::InvalidArgument, "Invalid video masking frame"));
        std::set<std::filesystem::path> destinations;
        for (const auto& frame : frames) {
            const std::array dimensions{frame.width, frame.height};
            if (dimensions != std::array{frames.front().width, frames.front().height} ||
                frame.image_path.lexically_normal() == frame.mask_path.lexically_normal() ||
                !destinations.insert(frame.mask_path.lexically_normal()).second)
                return std::unexpected(failure(ErrorCode::InvalidArgument, "Video masks need distinct destinations and consistent dimensions"));
        }
        if (!std::isfinite(circle.center_x) || !std::isfinite(circle.center_y) ||
            !std::isfinite(circle.radius_x) || !std::isfinite(circle.radius_y) ||
            circle.radius_x < 0 || circle.radius_y < 0 || (circle.radius_x == 0) != (circle.radius_y == 0))
            return std::unexpected(failure(ErrorCode::InvalidArgument, "Invalid video image circle"));
        if (options.prompt) {
            const auto& prompt = *options.prompt;
            if (prompt.frame < 0 || prompt.frame >= int(frames.size()) || prompt.points.empty() ||
                !std::ranges::any_of(prompt.points, [](const auto& point) { return point[2] == 1; }))
                return std::unexpected(failure(ErrorCode::InvalidArgument, "Invalid operator seed prompt"));
            for (const auto& point : prompt.points)
                if (!std::isfinite(point[0]) || !std::isfinite(point[1]) ||
                    point[0] < 0 || point[0] > 1 || point[1] < 0 || point[1] > 1 ||
                    (point[2] != 0 && point[2] != 1))
                    return std::unexpected(failure(ErrorCode::InvalidArgument, "Invalid normalized operator prompt"));
            if (prompt.box) {
                const auto& box = *prompt.box;
                if (std::ranges::any_of(box, [](float value) { return !std::isfinite(value) || value < 0 || value > 1; }) ||
                    box[0] >= box[2] || box[1] >= box[3])
                    return std::unexpected(failure(ErrorCode::InvalidArgument, "Invalid normalized operator prompt box"));
            }
        }
        // SAM2 and the cell kernels run on CUDA streams; other tensor backends have no port yet.
        if (core::default_gpu_backend() != core::GpuBackend::CUDA)
            return std::unexpected(failure(ErrorCode::Unsupported, "Video masking requires the CUDA tensor backend"));
        const core::CUDAStreamGuard guard(stream);
        FrameCellSource input(frames, options, stream);
        const auto wall_start = Clock::now();
        VideoMaskReport result;
        size_t sam_frames = 0;
        try {
            const auto tallies = detect_prompts(input, frames, circle, result.detection, options);
            input.release();
            auto& detection = result.detection;
            const auto warn = [&] {
                std::string reason;
                if (detection.status == "insufficient_motion_pairs")
                    reason = "not enough camera motion (" + std::to_string(detection.motion_pairs) + " usable pairs, at least 24 needed)";
                else if (detection.status == "insufficient_prompt_persistence")
                    reason = "camera-fixed cells did not persist throughout the sequence";
                else if (detection.status == "insufficient_prompt_anchors")
                    reason = "too few camera-fixed cells to seed the operator";
                else
                    reason = "SAM2 could not confirm an operator seed";
                result.warning = "Operator was not localized: " + reason + ". Review the masks or use a manual seed.";
                LOG_WARN("Video operator detection: {}", result.warning);
            };
            if (options.enabled && detection.status != "accepted") {
                warn();
                if (options.require_operator)
                    return std::unexpected(failure(ErrorCode::FailedPrecondition, result.warning));
            }
            const auto circle_keep = circle_keep_mask(frames.front().width, frames.front().height, circle);
            std::vector<int> sequence;
            for (int image = 0; image < input.image_count(); ++image) {
                if (options.enabled && detection.status == "accepted") {
                    sequence.push_back(image);
                } else {
                    input.notify(VideoMaskStage::Writing, size_t(image), frames.size());
                    write_mask(frames[size_t(image)].mask_path, circle_keep, input.width(image), input.height(image));
                }
            }
            std::sort(sequence.begin(), sequence.end(), [&](const int a, const int b) {
                return std::tie(frames[size_t(a)].time_seconds, a) < std::tie(frames[size_t(b)].time_seconds, b);
            });
            int seed_index = 0;
            if (options.prompt) {
                seed_index = options.prompt->frame;
            } else if (!sequence.empty()) {
                const auto support = [&](const int image) {
                    size_t fixed = 0;
                    for (int cell = 0; cell < kGridCells; ++cell)
                        fixed += detection.prompt_cells[size_t(cell)] ? tallies[size_t(image)].fixed[size_t(cell)] : 0;
                    return fixed;
                };
                const auto seed = std::max_element(sequence.begin(), sequence.end(), [&](const int a, const int b) {
                    return support(a) < support(b);
                });
                seed_index = int(seed - sequence.begin());
            }
            std::optional<Sam2> model;
            std::filesystem::path model_path = options.weights;
            if (!sequence.empty() && model_path.empty()) {
                auto ensured = ensure_sam2_video_weights(options.allow_download);
                if (!ensured)
                    return std::unexpected(ensured.error());
                model_path = std::move(*ensured);
            }
            MaskDecodeSettings decode;
            Tensor seed_mask;
            if (!sequence.empty()) {
                size_t free_bytes = 0, total_bytes = 0;
                LFS_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
                const auto plan = plan_video_mask_resources(frames, options, reusable_mask_memory(free_bytes, total_bytes));
                if (!plan)
                    return std::unexpected(plan.error());
                decode = decode_settings(plan->budget_bytes);
                auto loaded = Sam2::load(model_path, Device::GPU, DataType::Float32);
                if (!loaded)
                    return std::unexpected(loaded.error());
                model.emplace(std::move(*loaded));
                model->set_encoder_dtype(DataType::Float16);
                io::PipelinedImageLoader loader(single_frame_loader(4));
                const io::LoadParams once{.skip_blob_cache = true};

                const int image = sequence[size_t(seed_index)];
                input.notify(VideoMaskStage::Seed, 0, 1);
                loader.prefetch(size_t(image), frames[size_t(image)].image_path, once);
                auto ready = loader.get();
                if (!ready.error.empty())
                    throw std::runtime_error(ready.error);
                if (ready.image_ready)
                    ready.image_ready->wait_on(static_cast<void*>(stream));
                ready.tensor.set_stream(stream);
                const int width = input.width(image), height = input.height(image);
                const auto& tally = tallies[size_t(image)];
                const auto& manual = options.prompt;
                auto anchors = !manual
                                   ? prompts_for_image(detection, width, height)
                                   : std::vector<Sam2PointPrompt>{};
                if (manual.has_value())
                    for (const auto& point : manual->points)
                        anchors.push_back({std::min(point[0] * width, float(width - 1)),
                                           std::min(point[1] * height, float(height - 1)), int(point[2])});
                const auto encoded = model->set_image(ready.tensor.unsqueeze(0));
                if (!encoded)
                    return std::unexpected(encoded.error());
                Mask selected;
                const auto propose = [&](const std::vector<Sam2PointPrompt>& points,
                                         const std::optional<std::array<float, 4>>& box,
                                         const bool primary) -> std::expected<void, Error> {
                    auto prediction = model->predict(points, box, true);
                    if (!prediction)
                        return std::unexpected(prediction.error());
                    auto candidate = choose_seed_candidate(*prediction, points, circle, width, height, tally,
                                                           primary, stream);
                    if (!candidate.empty() && !selected.empty()) {
                        const auto interior = morphology(selected, 4, true);
                        if (fixed_share(tally, candidate, &interior).second < kCameraFixedShare)
                            return {};
                    }
                    if (selected.empty())
                        selected = std::move(candidate);
                    else if (!candidate.empty())
                        for (size_t pixel = 0; pixel < selected.size(); ++pixel)
                            selected[pixel] |= candidate[pixel];
                    return {};
                };
                if (manual.has_value()) {
                    auto box = manual->box;
                    if (box) {
                        (*box)[0] *= width;
                        (*box)[1] *= height;
                        (*box)[2] *= width;
                        (*box)[3] *= height;
                    }
                    auto prediction = model->predict(anchors, box, false);
                    if (!prediction)
                        return std::unexpected(prediction.error());
                    const auto low = core::nn::resize2d(prediction->low_res_logits.reshape({1, 1, 256, 256}),
                                                        kMaskSide, kMaskSide, core::nn::ResizeMode::Bilinear,
                                                        core::nn::CoordTransform::HalfPixel)
                                         .to(Device::CPU, stream)
                                         .contiguous();
                    LFS_CUDA_CHECK(cudaStreamSynchronize(stream));
                    selected.resize(size_t(kMaskSide) * kMaskSide);
                    for (size_t pixel = 0; pixel < selected.size(); ++pixel)
                        selected[pixel] = low.ptr<float>()[pixel] > 0;
                } else {
                    std::vector<Sam2PointPrompt> background;
                    for (const auto& point : anchors)
                        if (point.label == 0)
                            background.push_back(point);
                    auto strongest = background;
                    strongest.push_back(*std::ranges::find_if(anchors, [](const auto& point) { return point.label == 1; }));
                    if (const auto proposed = propose(strongest, std::nullopt, true); !proposed)
                        return std::unexpected(proposed.error());
                    auto supported = strongest;
                    if (!selected.empty()) {
                        const auto vicinity = morphology(selected, 12, true);
                        for (const auto& point : anchors)
                            if (point.label == 1 && vicinity[size_t(mask_pixel(point.x, point.y, width, height))])
                                supported.push_back(point);
                    }
                    if (const auto proposed = propose(supported, std::nullopt, true); !proposed)
                        return std::unexpected(proposed.error());
                    if (const auto proposed = propose(supported, seed_box(supported, width, height), true); !proposed)
                        return std::unexpected(proposed.error());
                    for (const auto& point : refinement_points(selected, tally, width, height, circle)) {
                        auto part = background;
                        part.push_back(point);
                        if (const auto proposed = propose(part, std::nullopt, false); !proposed)
                            return std::unexpected(proposed.error());
                    }
                }
                if (std::ranges::any_of(selected, [](uint8_t value) { return value != 0; })) {
                    selected = morphology(morphology(selected, 3, true), 3, false);
                    fill_holes(selected);
                    const int radius = std::max(1, int(std::ceil(12.0 * kMaskSide / std::max(width, height))));
                    const auto low = byte_tensor(morphology(selected, radius, true), kMaskSide, kMaskSide)
                                         .to(Device::GPU, stream)
                                         .to(DataType::Float32)
                                         .reshape({1, 1, kMaskSide, kMaskSide});
                    const auto circle_tensor = byte_tensor(circle_keep, height, width)
                                                   .to(Device::GPU, stream)
                                                   .gt(0)
                                                   .reshape({1, 1, height, width});
                    auto seed = core::nn::resize2d(low, height, width, core::nn::ResizeMode::Bilinear,
                                                   core::nn::CoordTransform::HalfPixel)
                                    .gt(0.5f)
                                    .logical_and(circle_tensor);
                    seed_mask = seed.to(DataType::Float32).to(Device::CPU, stream);
                }
                LFS_CUDA_CHECK(cudaStreamSynchronize(stream));

                if (!seed_mask.is_valid()) {
                    detection.status = "unconfirmed_operator";
                    warn();
                    if (options.require_operator)
                        return std::unexpected(failure(ErrorCode::FailedPrecondition, result.warning));
                    for (const int image : sequence)
                        write_mask(frames[size_t(image)].mask_path, circle_keep, input.width(image), input.height(image));
                }
            }
            std::vector<int> decode_order;
            if (seed_mask.is_valid()) {
                for (int index = seed_index; index < int(sequence.size()); ++index)
                    decode_order.push_back(sequence[size_t(index)]);
                for (int index = seed_index; index >= 0; --index)
                    decode_order.push_back(sequence[size_t(index)]);
            }
            std::optional<MaskFrameLoader> frame_loader;
            if (!decode_order.empty())
                frame_loader.emplace(frames, std::move(decode_order), decode);
            const auto read_frame = [&](int image) -> io::ReadyImage {
                auto ready = frame_loader->get(image);
                if (!ready.error.empty())
                    throw std::runtime_error(ready.error);
                if (ready.tensor.ndim() != 3 || ready.tensor.size(0) != 3 ||
                    ready.tensor.size(1) != size_t(input.height(image)) ||
                    ready.tensor.size(2) != size_t(input.width(image)))
                    throw std::invalid_argument("Decoded video frame dimensions or channels changed");
                if (ready.image_ready)
                    ready.image_ready->wait_on(static_cast<void*>(stream));
                ready.tensor.set_stream(stream);
                return ready;
            };
            std::deque<std::future<void>> pending_masks;
            const auto finish_mask = [&] {
                pending_masks.front().get();
                pending_masks.pop_front();
            };
            int cuda_device = 0;
            LFS_CUDA_CHECK(cudaGetDevice(&cuda_device));
            if (seed_mask.is_valid()) {
                const int seed_image = sequence[size_t(seed_index)];
                const int width = input.width(seed_image), height = input.height(seed_image);
                const auto circle_tensor = byte_tensor(circle_keep, height, width).to(Device::GPU, stream).gt(0);
                const auto seed = seed_mask.to(Device::GPU, stream);
                std::optional<OperatorView> view;
                Tensor tracking_seed = seed;
                if (options.recenter_fisheye && circle.radius_x > 0 && circle.radius_y > 0) {
                    const auto values = seed_mask.contiguous().to_vector();
                    double sum_x = 0, sum_y = 0, count = 0;
                    for (size_t pixel = 0; pixel < values.size(); ++pixel) {
                        if (values[pixel] <= 0)
                            continue;
                        sum_x += pixel % width;
                        sum_y += pixel / width;
                        ++count;
                    }
                    if (count > 0) {
                        view = make_operator_view(width, height, float(sum_x / count), float(sum_y / count),
                                                  {float(circle.center_x), float(circle.center_y),
                                                   float(circle.radius_x), float(circle.radius_y)},
                                                  options.fisheye_fov_degrees, options.view_fov_degrees, 1024);
                        tracking_seed = render_operator_view(seed, *view).gt(0.5f).to(DataType::Float32);
                    }
                }
                for (const bool reverse : {false, true}) {
                    model->reset_video();
                    detail::MaskRecovery recovery;
                    Tensor last_good_mask;
                    const int step = reverse ? -1 : 1;
                    for (int index = seed_index; index >= 0 && index < int(sequence.size()); index += step) {
                        const int image = sequence[size_t(index)];
                        input.notify(reverse ? VideoMaskStage::Reverse : VideoMaskStage::Forward,
                                     sam_frames, frames.size());
                        auto source = std::make_shared<MaskSource>(stream);
                        source->image = read_frame(image);
                        auto rgb = source->image.tensor.unsqueeze(0);
                        if (view) {
                            rgb = render_operator_view(rgb, *view);
                        }
                        if (view && index == seed_index && !reverse) {
                            // Host decisions require a completed download; explicit-stream transfers are asynchronous.
                            const auto support = tracking_seed.to(Device::CPU).contiguous().to_vector();
                            size_t closest = 0, area = 0;
                            double distance = std::numeric_limits<double>::infinity();
                            for (size_t pixel = 0; pixel < support.size(); ++pixel) {
                                if (support[pixel] <= 0)
                                    continue;
                                ++area;
                                const double dx = double(pixel % view->size) - view->size * 0.5;
                                const double dy = double(pixel / view->size) - view->size * 0.5;
                                if (dx * dx + dy * dy < distance) {
                                    distance = dx * dx + dy * dy;
                                    closest = pixel;
                                }
                            }
                            if (area) {
                                if (const auto encoded = model->set_image(rgb); !encoded)
                                    return std::unexpected(encoded.error());
                                const auto refined = model->predict(
                                    {{float(closest % view->size), float(closest / view->size), 1}}, std::nullopt, false);
                                if (!refined)
                                    return std::unexpected(refined.error());
                                const auto candidate = refined->masks.gt(0).to(DataType::Float32).unsqueeze(0);
                                const auto pixels = candidate.to(Device::CPU).contiguous().to_vector();
                                size_t covered = 0, proposed_area = 0;
                                for (size_t pixel = 0; pixel < pixels.size(); ++pixel) {
                                    proposed_area += pixels[pixel] > 0;
                                    covered += pixels[pixel] > 0 && support[pixel] > 0;
                                }
                                if (covered >= 0.8 * area && proposed_area <= 3 * area)
                                    tracking_seed = candidate;
                            }
                        }
                        {
                            if (const auto encoded = model->set_video_image(rgb); !encoded)
                                return std::unexpected(encoded.error());
                        }
                        auto prediction = [&] {
                            return index == seed_index ? model->seed_video(tracking_seed, index, int(sequence.size()), reverse)
                                                       : model->track_video(index);
                        }();
                        if (!prediction)
                            return std::unexpected(prediction.error());
                        const auto project_mask = [&](const Tensor& masks) {
                            const auto logits = view ? restore_operator_mask(masks.unsqueeze(0), *view).squeeze(0) : masks;
                            return logits.gt(0).logical_and(circle_tensor).to(DataType::Float32).unsqueeze(0);
                        };
                        auto raw_mask = project_mask(prediction->masks);
                        {
                            const auto quality_of = [&](const core::nn::models::Sam2Prediction& value, const Tensor& raw) {
                                if (!value.object_score_logits.is_valid())
                                    throw std::runtime_error("SAM2 image predictions must expose object scores for automatic re-seeding");
                                const auto host_pixels = raw.sum().to(Device::CPU, stream);
                                const auto host_area = value.low_res_logits.gt(0).to(DataType::Float32).sum().div(256.0f * 256.0f).to(Device::CPU, stream);
                                const auto host_iou = value.iou.to(Device::CPU, stream).contiguous();
                                const auto host_score = value.object_score_logits.to(Device::CPU, stream).contiguous();
                                // Only scalar diagnostics cross to the host. The previous mask stays on CUDA
                                // and is downloaded at 256 square only when a recovery box is needed.
                                LFS_CUDA_CHECK(cudaStreamSynchronize(stream));
                                return detail::RecoveryQuality{host_iou.ptr<float>()[0], host_score.ptr<float>()[0],
                                                               host_area.ptr<float>()[0], host_pixels.ptr<float>()[0] == 0};
                            };
                            const auto quality = quality_of(*prediction, raw_mask);
                            const auto reason = recovery.observe(quality, kSam2MinIou, kSam2MaxArea, 3);
                            if (quality.rejection(kSam2MinIou, kSam2MaxArea).empty())
                                last_good_mask = prediction->low_res_logits.gt(0).to(DataType::Float32).reshape({1, 1, 256, 256});
                            if (!reason.empty()) {
                                detail::RecoveryQuality candidate_quality;
                                std::string_view recovery_result = "no_reliable_mask";
                                if (last_good_mask.is_valid()) {
                                    const auto support = last_good_mask.to(Device::CPU).contiguous().to_vector();
                                    const auto box = detail::recovery_box(support, 256, int(rgb.size(3)), int(rgb.size(2)));
                                    if (box) {
                                        // Image prediction must see the current image, without temporal memory.
                                        if (const auto encoded = model->set_image(rgb); !encoded)
                                            return std::unexpected(encoded.error());
                                        const auto prompt = last_good_mask.mul(20).sub(10);
                                        auto candidate = model->predict({}, box, false, &prompt);
                                        if (!candidate)
                                            return std::unexpected(candidate.error());
                                        const auto candidate_raw = project_mask(candidate->masks);
                                        candidate_quality = quality_of(*candidate, candidate_raw);
                                        recovery_result = candidate_quality.rejection(kSam2MinIou, kSam2MaxArea);
                                        if (recovery_result.empty()) {
                                            const auto new_seed = candidate->masks.gt(0).to(DataType::Float32).unsqueeze(0);
                                            if (const auto encoded = model->set_video_image(rgb); !encoded)
                                                return std::unexpected(encoded.error());
                                            prediction = model->seed_video(new_seed, index, int(sequence.size()), reverse);
                                            if (!prediction)
                                                return std::unexpected(prediction.error());
                                            raw_mask = project_mask(prediction->masks);
                                            const auto restarted = quality_of(*prediction, raw_mask);
                                            const bool accepted = restarted.rejection(kSam2MinIou, kSam2MaxArea).empty();
                                            recovery_result = accepted ? "accepted" : "empty_after_seed_filter";
                                            if (accepted) {
                                                recovery.accept();
                                                last_good_mask = prediction->low_res_logits.gt(0).to(DataType::Float32).reshape({1, 1, 256, 256});
                                            }
                                        }
                                    }
                                }
                                LOG_INFO("SAM2 re-seed {} reverse {}: {} -> {} (IoU {}, object {}, area {})",
                                         core::path_to_utf8(frames[size_t(image)].image_path.filename()), reverse,
                                         reason, recovery_result, candidate_quality.iou, candidate_quality.object_score, candidate_quality.area);
                            }
                        }
                        source->completion.emplace(stream);
                        if (reverse && index == seed_index) {
                            auto completion = std::make_shared<MaskCompletion>(stream);
                            pending_masks.push_back(std::async(std::launch::async,
                                                               [source, completion, cuda_device]() mutable {
                                                                   LFS_CUDA_CHECK(cudaSetDevice(cuda_device));
                                                                   LFS_CUDA_CHECK(cudaEventSynchronize(completion->ready));
                                                                   source.reset();
                                                               }));
                            if (pending_masks.size() >= kPendingMaskWrites)
                                finish_mask();
                            continue;
                        }
                        const auto object = core::nn::dilate_binary_disk(raw_mask, options.dilation_radius).squeeze(0).squeeze(0).gt(0).logical_and(circle_tensor);
                        const auto keep = (!object).logical_and(circle_tensor).to(DataType::Float32).mul(255).to(DataType::UInt8).to(Device::CPU, stream).contiguous();
                        auto completion = std::make_shared<MaskCompletion>(stream);
                        ++sam_frames;
                        pending_masks.push_back(std::async(std::launch::async,
                                                           [keep, completion, source, cuda_device, path = frames[size_t(image)].mask_path,
                                                            width = input.width(image), height = input.height(image)]() mutable {
                                                               LFS_CUDA_CHECK(cudaSetDevice(cuda_device));
                                                               LFS_CUDA_CHECK(cudaEventSynchronize(completion->ready));
                                                               // Decoder leases are no longer needed by CUDA. Release
                                                               // them before PNG work, even if the future is collected later.
                                                               source.reset();
                                                               write_mask(path, {keep.ptr<uint8_t>(), keep.numel()}, width, height);
                                                           }));
                        if (pending_masks.size() >= kPendingMaskWrites)
                            finish_mask();
                    }
                }
            }

            while (!pending_masks.empty())
                finish_mask();
            input.notify(VideoMaskStage::Writing, frames.size(), frames.size());
            result.mask_wall_seconds = seconds(wall_start);
            return result;
        } catch (const MaskingCancelled&) {
            return std::unexpected(failure(ErrorCode::Cancelled, "Video masking cancelled"));
        } catch (const std::exception& error) {
            return std::unexpected(failure(ErrorCode::Internal, "SAM2 video masking failed", error.what()));
        }
    }

} // namespace lfs::preprocessing
