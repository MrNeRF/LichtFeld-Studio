/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/image_io.hpp"
#include "core/tensor_backend.hpp"
#include "masked_image_fill.hpp"

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <algorithm>
#include <bit>
#include <climits>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

// The rule of fill_masked_image in cuda/masked_image_fill.cu: an ignored pixel takes the RGB of the
// kept pixel nearest by squared distance, the lower linear index on a tie, within the square window of
// radius 16; without one, the first kept pixel of the image; without any kept pixel, zero. The CUDA
// kernel searches rings around each pixel. Here a column pass finds the best kept pixel of every
// column of the window, so each ignored pixel compares at most 33 candidates. Both select the
// minimum of (distance, index) over the same window.
namespace lfs::io {
    namespace {
        using lfs::core::DataType;
        using lfs::core::Device;
        using lfs::core::Tensor;

        constexpr int radius = 16;
        constexpr std::int8_t no_row = INT8_MAX;
        constexpr float high_band_cutoff = 250.5f / 255.0f;

        // IEEE comparisons on the bit patterns: subnormal values and cutoffs keep their order even
        // where a thread flushes them to zero, as the CUDA classifier's non-FTZ comparisons do.
        bool positive(const float value) {
            return std::bit_cast<std::uint32_t>(value) - 1u < 0x7f800000u;
        }
        bool at_least(const float value, const float positive_cutoff) {
            const auto bits = std::bit_cast<std::uint32_t>(value);
            return bits <= 0x7f800000u && bits >= std::bit_cast<std::uint32_t>(positive_cutoff);
        }

        struct KeepRule {
            bool invert = false;
            bool high_band = false;
            bool thresholded = false;
            float threshold = 0.f;

            explicit KeepRule(const MaskedImageFill& policy)
                : invert(policy.invert),
                  high_band(policy.keep_high_band),
                  thresholded(positive(policy.threshold)),
                  threshold(policy.threshold) {}

            bool operator()(float value) const {
                if (invert)
                    value = 1.f - value;
                if (high_band)
                    return value > high_band_cutoff;
                return thresholded ? at_least(value, threshold) : positive(value);
            }
        };

        // Correctly rounded k / 255 and k / 65535, as normalize_mask_byte and normalize_mask_u16_value.
        template <typename T>
        std::vector<std::uint8_t> keep_table(const KeepRule& rule) {
            constexpr std::uint32_t top = std::numeric_limits<T>::max();
            std::vector<std::uint8_t> table(top + 1);
            for (std::uint32_t k = 0; k <= top; ++k)
                table[k] = rule(static_cast<float>(k) / static_cast<float>(top));
            return table;
        }

        template <typename T>
        void classify(const T* values, const size_t stride, const size_t count, const KeepRule& rule,
                      std::uint8_t* keep) {
            if constexpr (std::is_same_v<T, float>) {
                tbb::parallel_for(tbb::blocked_range<size_t>(0, count, 1 << 14), [&](const auto& range) {
                    for (size_t i = range.begin(); i < range.end(); ++i)
                        keep[i] = rule(values[i * stride]);
                });
            } else {
                const auto table = keep_table<T>(rule);
                tbb::parallel_for(tbb::blocked_range<size_t>(0, count, 1 << 14), [&](const auto& range) {
                    for (size_t i = range.begin(); i < range.end(); ++i)
                        keep[i] = table[values[i * stride]];
                });
            }
        }

        // [rows, columns] of the mask after fill_masked_image's squeeze, if it is 2-D.
        std::optional<std::pair<size_t, size_t>> mask_grid(const Tensor& mask) {
            const auto& shape = mask.shape();
            if (mask.ndim() == 2)
                return std::pair{shape[0], shape[1]};
            if (mask.ndim() == 3 && shape[0] == 1)
                return std::pair{shape[1], shape[2]};
            if (mask.ndim() == 3 && shape[2] == 1)
                return std::pair{shape[0], shape[1]};
            return std::nullopt;
        }

        // A mask on the source grid, or on the output grid of resize_factor and max_width. Others
        // leave the image as decoded.
        bool on_fill_grid(const MaskedImageFill& policy, const int width, const int height) {
            if (!policy.mask.is_valid())
                return false;
            const auto grid = mask_grid(policy.mask);
            if (!grid)
                return false;
            if (grid->first == static_cast<size_t>(height) && grid->second == static_cast<size_t>(width))
                return true;
            const auto [target_width, target_height] =
                lfs::core::resized_image_dimensions(width, height, policy.resize_factor, policy.max_width);
            return grid->first == static_cast<size_t>(target_height) && grid->second == static_cast<size_t>(target_width);
        }

        // Source of each destination index in a nearest, asymmetric resize, as nn::resize2d.
        std::vector<int> nearest_sources(const int size, const int target) {
            std::vector<int> sources(static_cast<size_t>(target), 0);
            for (int i = 0; target > 1 && i < target; ++i)
                sources[i] = static_cast<int>(std::min<std::int64_t>(std::int64_t{i} * size / target, size - 1));
            return sources;
        }

        std::vector<std::uint8_t> keep_from_mask(const MaskedImageFill& policy, const KeepRule& rule,
                                                 const int width, const int height) {
            Tensor mask = policy.mask.device() == Device::CPU ? policy.mask : policy.mask.cpu();
            mask = mask.contiguous();
            const auto [rows, columns] = *mask_grid(mask);
            if (mask.dtype() != DataType::Float32 && mask.dtype() != DataType::UInt8)
                mask = mask.to(DataType::Float32);
            std::vector<std::uint8_t> classified(rows * columns);
            if (mask.dtype() == DataType::UInt8)
                classify(mask.ptr<std::uint8_t>(), 1, classified.size(), rule, classified.data());
            else
                classify(mask.ptr<float>(), 1, classified.size(), rule, classified.data());
            if (rows == static_cast<size_t>(height) && columns == static_cast<size_t>(width))
                return classified;
            // An output-grid mask is expanded onto the source grid. The nearest resize only copies
            // values, so classifying first is equivalent.
            const auto source_rows = nearest_sources(static_cast<int>(rows), height);
            const auto source_columns = nearest_sources(static_cast<int>(columns), width);
            std::vector<std::uint8_t> keep(static_cast<size_t>(width) * height);
            tbb::parallel_for(tbb::blocked_range<int>(0, height, 16), [&](const auto& range) {
                for (int y = range.begin(); y < range.end(); ++y) {
                    const auto* source = classified.data() + static_cast<size_t>(source_rows[y]) * columns;
                    auto* destination = keep.data() + static_cast<size_t>(y) * width;
                    for (int x = 0; x < width; ++x)
                        destination[x] = source[source_columns[x]];
                }
            });
            return keep;
        }

        // Row offset of the best kept pixel in each pixel's column within the radius: the nearest,
        // the upper one on a tie, which also has the lower index.
        std::vector<std::int8_t> column_candidates(const std::vector<std::uint8_t>& keep, const int width,
                                                   const int height) {
            std::vector<std::int8_t> offsets(keep.size());
            tbb::parallel_for(tbb::blocked_range<int>(0, width, 256), [&](const auto& range) {
                const int first = range.begin(), count = range.end() - range.begin();
                std::vector<int> kept_row(static_cast<size_t>(count), -1);
                for (int y = 0; y < height; ++y) {
                    const size_t base = static_cast<size_t>(y) * width + first;
                    for (int i = 0; i < count; ++i) {
                        if (keep[base + i])
                            kept_row[i] = y;
                        offsets[base + i] = kept_row[i] >= 0 && y - kept_row[i] <= radius
                                                ? static_cast<std::int8_t>(kept_row[i] - y)
                                                : no_row;
                    }
                }
                std::fill(kept_row.begin(), kept_row.end(), -1);
                for (int y = height - 1; y >= 0; --y) {
                    const size_t base = static_cast<size_t>(y) * width + first;
                    for (int i = 0; i < count; ++i) {
                        if (keep[base + i])
                            kept_row[i] = y;
                        const int above = offsets[base + i] == no_row ? radius + 1 : -offsets[base + i];
                        if (kept_row[i] >= 0 && kept_row[i] - y < above)
                            offsets[base + i] = static_cast<std::int8_t>(kept_row[i] - y);
                    }
                }
            });
            return offsets;
        }

        template <typename T>
        void fill(T* pixels, const int width, const int height, const int channels, const bool planar,
                  const MaskedImageFill& policy) {
            if ((channels != 3 && channels != 4) || (policy.alpha && channels != 4))
                throw std::invalid_argument(masked_image_fill_error());
            if (width <= 0 || height <= 0)
                return;
            const size_t count = static_cast<size_t>(width) * static_cast<size_t>(height);
            // As on CUDA, a source beyond this index domain keeps its decode.
            if (count > static_cast<size_t>(INT_MAX))
                return;
            const size_t pixel_stride = planar ? 1 : static_cast<size_t>(channels);
            const size_t channel_stride = planar ? count : 1;
            const KeepRule rule(policy);
            std::vector<std::uint8_t> keep;
            if (policy.alpha) {
                keep.resize(count);
                classify(pixels + 3 * channel_stride, pixel_stride, count, rule, keep.data());
            } else {
                if (!on_fill_grid(policy, width, height))
                    return;
                keep = keep_from_mask(policy, rule, width, height);
            }

            const auto first = std::find(keep.begin(), keep.end(), std::uint8_t{1});
            if (first == keep.end()) {
                for (size_t i = 0; i < count; ++i)
                    for (int c = 0; c < 3; ++c)
                        pixels[i * pixel_stride + c * channel_stride] = T{};
                return;
            }
            const auto first_kept = static_cast<size_t>(first - keep.begin());
            const auto offsets = column_candidates(keep, width, height);
            tbb::parallel_for(tbb::blocked_range<int>(0, height, 8), [&](const auto& range) {
                for (int y = range.begin(); y < range.end(); ++y) {
                    const size_t base = static_cast<size_t>(y) * width;
                    const auto* row = offsets.data() + base;
                    // Columns with a candidate within [x - radius, x + radius].
                    int window = 0;
                    for (int column = 0; column <= std::min(radius, width - 1); ++column)
                        window += row[column] != no_row;
                    for (int x = 0; x < width; ++x) {
                        if (x > 0) {
                            if (x + radius < width)
                                window += row[x + radius] != no_row;
                            if (x - radius - 1 >= 0)
                                window -= row[x - radius - 1] != no_row;
                        }
                        if (keep[base + x])
                            continue;
                        size_t nearest = first_kept;
                        int best = INT_MAX;
                        // Columns dx away hold no candidate nearer than dx * dx.
                        for (int dx = 0; window && dx <= radius && dx * dx <= best; ++dx) {
                            for (int side = dx ? -1 : 1; side <= 1; side += 2) {
                                const int column = x + side * dx;
                                if (column < 0 || column >= width || row[column] == no_row)
                                    continue;
                                const int dy = row[column];
                                const int distance = dx * dx + dy * dy;
                                const size_t index = static_cast<size_t>(y + dy) * width + column;
                                if (distance < best || (distance == best && index < nearest)) {
                                    best = distance;
                                    nearest = index;
                                }
                            }
                        }
                        // Kept pixels are never written, so rows may fill concurrently.
                        for (int c = 0; c < 3; ++c)
                            std::memcpy(pixels + (base + x) * pixel_stride + c * channel_stride,
                                        pixels + nearest * pixel_stride + c * channel_stride, sizeof(T));
                    }
                }
            });
        }
    } // namespace

    void fill_masked_image_host(std::uint8_t* pixels, const int width, const int height, const int channels,
                                const bool planar, const MaskedImageFill& policy) {
        fill(pixels, width, height, channels, planar, policy);
    }

    void fill_masked_image_host(std::uint16_t* pixels, const int width, const int height, const int channels,
                                const bool planar, const MaskedImageFill& policy) {
        fill(pixels, width, height, channels, planar, policy);
    }

    void fill_masked_image_host(float* pixels, const int width, const int height, const int channels,
                                const bool planar, const MaskedImageFill& policy) {
        fill(pixels, width, height, channels, planar, policy);
    }

    void fill_masked_image_host(Tensor& image, const bool planar, const MaskedImageFill& policy) {
        if (!image.is_valid() || image.device() != Device::CPU || !image.is_contiguous() || image.ndim() != 3)
            throw std::invalid_argument(masked_image_fill_error());
        const auto& shape = image.shape();
        const int channels = static_cast<int>(planar ? shape[0] : shape[2]);
        const int height = static_cast<int>(planar ? shape[1] : shape[0]);
        const int width = static_cast<int>(planar ? shape[2] : shape[1]);
        switch (image.dtype()) {
        case DataType::UInt8:
            return fill(image.ptr<std::uint8_t>(), width, height, channels, planar, policy);
        case DataType::Float16:
            return fill(static_cast<std::uint16_t*>(image.data_ptr()), width, height, channels, planar, policy);
        case DataType::Float32:
            return fill(image.ptr<float>(), width, height, channels, planar, policy);
        default:
            throw std::invalid_argument(masked_image_fill_error());
        }
    }

    bool masked_image_fill_applies(const MaskedImageFill& policy, const int width, const int height) {
        if (width <= 0 || height <= 0 || static_cast<size_t>(width) * static_cast<size_t>(height) > INT_MAX)
            return false;
        return policy.alpha || on_fill_grid(policy, width, height);
    }

    void fill_masked_planar_image(Tensor& image, const MaskedImageFill& policy) {
        if (!image.is_valid() || image.ndim() != 3)
            throw std::invalid_argument(masked_image_fill_error());
        if (!masked_image_fill_applies(policy, static_cast<int>(image.shape()[2]), static_cast<int>(image.shape()[1])))
            return;
        if (image.device() == Device::CPU) {
            image = image.contiguous();
            return fill_masked_image_host(image, true, policy);
        }
        const auto backend = lfs::core::gpu_backend_of(image);
        auto host = image.cpu().contiguous();
        fill_masked_image_host(host, true, policy);
        const lfs::core::GpuBackendScope scope(backend.value_or(lfs::core::default_gpu_backend()));
        image = host.to(Device::GPU);
    }
} // namespace lfs::io
