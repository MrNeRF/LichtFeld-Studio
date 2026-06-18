/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Compiled with --fmad=false: the CPU decoders round every multiply and add
// separately, and contraction here would break bit-parity for the pure
// arithmetic encodings (r8, s8, meta dequant frames).

#include "lod_page_dequant_cuda.hpp"

#include "core/cuda/sh_layout.cuh"
#include "io/formats/rad_dequant_math.hpp"

namespace lfs::vis {
    namespace {

        using lfs::io::RadPackedEncoding;
        using lfs::io::RadPackedKind;
        using lfs::io::RadPackedProperty;
        using lfs::io::RadPagePackedDesc;
        namespace radmath = lfs::io::radmath;

        constexpr std::uint32_t kReorder = lfs::core::kShReorderSize;

        __device__ inline std::uint32_t packHalf2(const float a, const float b) {
            return static_cast<std::uint32_t>(radmath::floatToHalf(a)) |
                   (static_cast<std::uint32_t>(radmath::floatToHalf(b)) << 16);
        }

        __device__ inline std::int8_t quantizeS8(const float v, const float max_abs) {
            const float scaled = v / max_abs * 127.0f;
            const float clamped = radmath::clampf(rintf(scaled), -127.0f, 127.0f);
            return static_cast<std::int8_t>(clamped);
        }

        // Dimension-major plane read for splat i, dimension d, with the same
        // per-element math as rad.cpp's PropertyDecoder dispatch.
        __device__ float readPlane(const std::uint8_t* const plane,
                                   const RadPackedProperty& prop,
                                   const std::uint32_t dims,
                                   const std::uint32_t count,
                                   const std::uint32_t d,
                                   const std::uint32_t i) {
            const std::uint32_t e = d * count + i;
            switch (static_cast<RadPackedEncoding>(prop.encoding)) {
            case RadPackedEncoding::F32: {
                std::uint32_t bits;
                memcpy(&bits, plane + static_cast<std::size_t>(e) * 4u, 4u);
                return radmath::bitsToFloat(bits);
            }
            case RadPackedEncoding::F32LeBytes: {
                const std::uint32_t stride = count * dims;
                const std::uint32_t bits =
                    static_cast<std::uint32_t>(plane[e]) |
                    (static_cast<std::uint32_t>(plane[stride + e]) << 8) |
                    (static_cast<std::uint32_t>(plane[2u * stride + e]) << 16) |
                    (static_cast<std::uint32_t>(plane[3u * stride + e]) << 24);
                return radmath::bitsToFloat(bits);
            }
            case RadPackedEncoding::F16: {
                std::uint16_t h;
                memcpy(&h, plane + static_cast<std::size_t>(e) * 2u, 2u);
                return radmath::halfToFloat(h);
            }
            case RadPackedEncoding::F16LeBytes: {
                const std::uint32_t stride = count * dims;
                const std::uint16_t h = static_cast<std::uint16_t>(
                    plane[e] | (static_cast<std::uint16_t>(plane[stride + e]) << 8));
                return radmath::halfToFloat(h);
            }
            case RadPackedEncoding::R8:
                return radmath::dequantR8(plane[e], prop.min_val, prop.max_val - prop.min_val);
            case RadPackedEncoding::S8:
                return radmath::dequantS8(static_cast<std::int8_t>(plane[e]),
                                          radmath::shMaxAbs(prop.min_val, prop.max_val, prop.scale));
            case RadPackedEncoding::Ln0R8:
                return radmath::dequantLn0R8(plane[e], prop.min_val, prop.max_val);
            case RadPackedEncoding::LnF16: {
                std::uint16_t h;
                memcpy(&h, plane + static_cast<std::size_t>(e) * 2u, 2u);
                return std::exp(radmath::halfToFloat(h));
            }
            default:
                return 0.0f;
            }
        }

        struct PropSlots {
            const RadPackedProperty* by_kind[8];
        };

        struct ShBand {
            const RadPackedProperty* prop;
            std::uint32_t local;
            std::uint32_t dims;
        };

        __device__ inline ShBand shBandOf(const PropSlots& props, const std::uint32_t coeff) {
            if (coeff < 3u) {
                return {props.by_kind[static_cast<std::uint32_t>(RadPackedKind::Sh1)], coeff, 9u};
            }
            if (coeff < 8u) {
                return {props.by_kind[static_cast<std::uint32_t>(RadPackedKind::Sh2)], coeff - 3u, 15u};
            }
            return {props.by_kind[static_cast<std::uint32_t>(RadPackedKind::Sh3)], coeff - 8u, 21u};
        }

        // Raw s8 byte for canonical SH float index c — the passthrough path
        // (file s8 -> pool s8, bit-exact). Returns 0 for absent bands.
        __device__ inline std::uint8_t readShS8Raw(const std::uint8_t* const slot,
                                                   const PropSlots& props,
                                                   const std::uint32_t count,
                                                   const std::uint32_t i,
                                                   const std::uint32_t c) {
            const ShBand band = shBandOf(props, c / 3u);
            if (band.prop == nullptr) {
                return 0u;
            }
            const std::uint32_t d = band.local * 3u + (c % 3u);
            if (static_cast<RadPackedEncoding>(band.prop->encoding) == RadPackedEncoding::S8) {
                return slot[band.prop->plane_offset + static_cast<std::size_t>(d) * count + i];
            }
            // Non-s8 band (outside the converter profile but representable):
            // dequantize and requantize against the band's own scale.
            const float v = readPlane(slot + band.prop->plane_offset, *band.prop, band.dims, count, d, i);
            const float max_abs =
                radmath::shMaxAbs(band.prop->min_val, band.prop->max_val, band.prop->scale);
            return static_cast<std::uint8_t>(quantizeS8(v, max_abs));
        }

        __device__ inline std::size_t shNSlotIndex(const std::uint32_t page,
                                                   const std::uint32_t page_splats,
                                                   const std::uint32_t i,
                                                   const std::uint32_t s,
                                                   const std::uint32_t dst_slots) {
            const std::uint32_t block = i / kReorder;
            const std::uint32_t lane = i % kReorder;
            return (static_cast<std::size_t>(page) * page_splats / kReorder + block) *
                       dst_slots * kReorder +
                   static_cast<std::size_t>(s) * kReorder + lane;
        }

        __global__ void lodPageDequantKernel(const std::uint8_t* const __restrict__ slot,
                                             const RadPagePackedDesc desc,
                                             const LodPoolDeviceView pool,
                                             const std::uint32_t page,
                                             const std::uint32_t page_splats) {
            const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= page_splats) {
                return;
            }
            const std::size_t dst = static_cast<std::size_t>(page) * page_splats + i;
            const bool live = i < desc.count;

            PropSlots props{};
            for (std::uint32_t p = 0; p < desc.property_count; ++p) {
                props.by_kind[desc.props[p].kind] = &desc.props[p];
            }
            const auto plane_of = [&](const RadPackedKind kind) {
                return props.by_kind[static_cast<std::uint32_t>(kind)];
            };

            if (i == 0u && pool.page_frames != nullptr) {
                lodq::PageFrame frame{};
                for (std::uint32_t band = 0; band < 3u; ++band) {
                    const auto* const prop = props.by_kind[static_cast<std::uint32_t>(RadPackedKind::Sh1) + band];
                    frame.sh_max[band] =
                        prop != nullptr
                            ? radmath::shMaxAbs(prop->min_val, prop->max_val, prop->scale)
                            : 0.0f;
                }
                for (std::uint32_t d = 0; d < 3u; ++d) {
                    frame.bbox_min[d] = desc.frame.bbox_min[d];
                    frame.bbox_extent[d] = desc.frame.bbox_extent[d];
                }
                frame.log_size_min = desc.frame.log_size_min;
                frame.log_size_range = desc.frame.log_size_range;
                const auto* const src = reinterpret_cast<const float4*>(&frame);
                float4* const out = pool.page_frames +
                                    static_cast<std::size_t>(page) * lodq::kPageFrameFloat4s;
                for (std::uint32_t f = 0; f < lodq::kPageFrameFloat4s; ++f) {
                    out[f] = src[f];
                }
            }

            if (const auto* const prop = plane_of(RadPackedKind::Means); pool.means != nullptr) {
                for (std::uint32_t d = 0; d < 3u; ++d) {
                    pool.means[dst * 3u + d] =
                        live && prop != nullptr
                            ? readPlane(slot + prop->plane_offset, *prop, 3u, desc.count, d, i)
                            : 0.0f;
                }
            }
            if (const auto* const prop = plane_of(RadPackedKind::Sh0); pool.sh0 != nullptr) {
                float v[3] = {0.0f, 0.0f, 0.0f};
                if (live && prop != nullptr) {
                    for (std::uint32_t d = 0; d < 3u; ++d) {
                        v[d] = radmath::sh0Transform(
                            readPlane(slot + prop->plane_offset, *prop, 3u, desc.count, d, i));
                    }
                }
                pool.sh0[dst] = uint2{packHalf2(v[0], v[1]), packHalf2(v[2], 0.0f)};
            }
            if (const auto* const prop = plane_of(RadPackedKind::Scales); pool.scaling != nullptr) {
                std::uint16_t h[3] = {0u, 0u, 0u};
                if (live && prop != nullptr) {
                    if (static_cast<RadPackedEncoding>(prop->encoding) == RadPackedEncoding::LnF16) {
                        // The file already stores log-domain f16 — bit-exact
                        // passthrough instead of the old f16->exp->log round
                        // trip.
                        for (std::uint32_t d = 0; d < 3u; ++d) {
                            memcpy(&h[d],
                                   slot + prop->plane_offset +
                                       (static_cast<std::size_t>(d) * desc.count + i) * 2u,
                                   2u);
                        }
                    } else {
                        for (std::uint32_t d = 0; d < 3u; ++d) {
                            const float linear =
                                readPlane(slot + prop->plane_offset, *prop, 3u, desc.count, d, i);
                            h[d] = radmath::floatToHalf(radmath::scaleLog(linear));
                        }
                    }
                }
                pool.scaling[dst] = uint2{
                    static_cast<std::uint32_t>(h[0]) | (static_cast<std::uint32_t>(h[1]) << 16),
                    static_cast<std::uint32_t>(h[2])};
            }
            if (const auto* const prop = plane_of(RadPackedKind::Alpha); pool.opacity != nullptr) {
                float v = 0.0f;
                if (live && prop != nullptr) {
                    const float raw =
                        readPlane(slot + prop->plane_offset, *prop, 1u, desc.count, 0u, i);
                    v = desc.lod_opacity != 0u ? radmath::opacityLodEncoded(raw)
                                               : radmath::opacityLogit(raw);
                }
                pool.opacity[dst] = radmath::floatToHalf(v);
            }
            if (const auto* const prop = plane_of(RadPackedKind::Rotation); pool.rotation != nullptr) {
                float xyzw[4] = {0.0f, 0.0f, 0.0f, 1.0f};
                if (live && prop != nullptr) {
                    if (static_cast<RadPackedEncoding>(prop->encoding) == RadPackedEncoding::Oct88R8) {
                        const std::uint8_t* const q = slot + prop->plane_offset +
                                                      static_cast<std::size_t>(i) * 3u;
                        radmath::dequantQuatOct88R8(q[0], q[1], q[2], xyzw);
                    } else {
                        for (std::uint32_t d = 0; d < 3u; ++d) {
                            xyzw[d] = readPlane(slot + prop->plane_offset, *prop, 3u, desc.count, d, i);
                        }
                        xyzw[3] = radmath::quatWFromXyz(xyzw[0], xyzw[1], xyzw[2]);
                    }
                }
                // Pool order is (w, x, y, z), as in decode_chunk_properties.
                pool.rotation[dst] = uint2{packHalf2(xyzw[3], xyzw[0]), packHalf2(xyzw[1], xyzw[2])};
            }
            if (pool.shN != nullptr && pool.dst_slots > 0u) {
                // writeSwizzledShPage semantics: destination component k holds
                // canonical source float k while k < src_rest*3, else zero.
                const std::uint32_t active_floats = desc.sh_coeffs_rest * 3u;
                for (std::uint32_t s = 0; s < pool.dst_slots; ++s) {
                    std::uint32_t packed = 0u;
                    if (live) {
                        const std::uint32_t c0 = s * 4u;
                        for (std::uint32_t comp = 0; comp < 4u; ++comp) {
                            const std::uint8_t b =
                                c0 + comp < active_floats
                                    ? readShS8Raw(slot, props, desc.count, i, c0 + comp)
                                    : 0u;
                            packed |= static_cast<std::uint32_t>(b) << (comp * 8u);
                        }
                    }
                    pool.shN[shNSlotIndex(page, page_splats, i, s, pool.dst_slots)] = packed;
                }
            }

            // Sidecar records pass through untouched; the selector
            // dequantizes against the page frame and derives logical from
            // page_to_chunk (slack nodes decode to logical >= node_count).
            if (pool.meta_bounds != nullptr && pool.meta_links != nullptr) {
                if (i < desc.meta_node_count) {
                    uint2 bq;
                    memcpy(&bq, slot + desc.meta_bounds_offset + static_cast<std::size_t>(i) * 8u, 8u);
                    pool.meta_bounds[dst] = bq;
                    std::uint32_t lq[3];
                    memcpy(lq, slot + desc.meta_links_offset + static_cast<std::size_t>(i) * 12u, 12u);
                    for (std::uint32_t w = 0; w < 3u; ++w) {
                        pool.meta_links[dst * 3u + w] = lq[w];
                    }
                } else {
                    pool.meta_bounds[dst] = uint2{0u, 0u};
                    for (std::uint32_t w = 0; w < 3u; ++w) {
                        pool.meta_links[dst * 3u + w] = 0xFFFFFFFFu;
                    }
                }
            }
        }

        // Pass 1 of the from-tensors fill: per-band |max| over the page's
        // canonical shN, accumulated into the page frame via atomicMax on the
        // non-negative float bit patterns.
        __global__ void lodPageShBandMaxKernel(const LodPageTensorSources src,
                                               float4* const frame) {
            const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= src.count || src.shN == nullptr) {
                return;
            }
            const std::uint32_t floats = src.src_rest * 3u;
            float band_max[3] = {0.0f, 0.0f, 0.0f};
            for (std::uint32_t c = 0; c < floats; ++c) {
                const float v = fabsf(src.shN[static_cast<std::size_t>(i) * floats + c]);
                const std::uint32_t coeff = c / 3u;
                const std::uint32_t band = coeff < 3u ? 0u : (coeff < 8u ? 1u : 2u);
                if (v > band_max[band]) {
                    band_max[band] = v;
                }
            }
            auto* const words = reinterpret_cast<std::uint32_t*>(frame);
            for (std::uint32_t band = 0; band < 3u; ++band) {
                if (band_max[band] > 0.0f) {
                    atomicMax(&words[band], radmath::floatToBits(band_max[band]));
                }
            }
        }

        __global__ void lodPageQuantizeKernel(const LodPageTensorSources src,
                                              const LodPoolDeviceView pool,
                                              const std::uint32_t page,
                                              const std::uint32_t page_splats) {
            const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= page_splats) {
                return;
            }
            const std::size_t dst = static_cast<std::size_t>(page) * page_splats + i;
            const bool live = i < src.count;
            const float4* const frame =
                pool.page_frames + static_cast<std::size_t>(page) * lodq::kPageFrameFloat4s;

            if (pool.means != nullptr) {
                for (std::uint32_t d = 0; d < 3u; ++d) {
                    pool.means[dst * 3u + d] =
                        live ? src.means[static_cast<std::size_t>(i) * 3u + d] : 0.0f;
                }
            }
            if (pool.sh0 != nullptr) {
                const float x = live ? src.sh0[static_cast<std::size_t>(i) * 3u + 0u] : 0.0f;
                const float y = live ? src.sh0[static_cast<std::size_t>(i) * 3u + 1u] : 0.0f;
                const float z = live ? src.sh0[static_cast<std::size_t>(i) * 3u + 2u] : 0.0f;
                pool.sh0[dst] = uint2{packHalf2(x, y), packHalf2(z, 0.0f)};
            }
            if (pool.scaling != nullptr) {
                std::uint16_t h[3] = {0u, 0u, 0u};
                if (live) {
                    for (std::uint32_t d = 0; d < 3u; ++d) {
                        h[d] = radmath::floatToHalf(src.scaling[static_cast<std::size_t>(i) * 3u + d]);
                    }
                }
                pool.scaling[dst] = uint2{
                    static_cast<std::uint32_t>(h[0]) | (static_cast<std::uint32_t>(h[1]) << 16),
                    static_cast<std::uint32_t>(h[2])};
            }
            if (pool.opacity != nullptr) {
                pool.opacity[dst] = radmath::floatToHalf(live ? src.opacity[i] : 0.0f);
            }
            if (pool.rotation != nullptr) {
                float wxyz[4] = {1.0f, 0.0f, 0.0f, 0.0f};
                if (live) {
                    for (std::uint32_t d = 0; d < 4u; ++d) {
                        wxyz[d] = src.rotation[static_cast<std::size_t>(i) * 4u + d];
                    }
                }
                pool.rotation[dst] = uint2{packHalf2(wxyz[0], wxyz[1]), packHalf2(wxyz[2], wxyz[3])};
            }
            if (pool.shN != nullptr && pool.dst_slots > 0u) {
                const std::uint32_t active_floats =
                    src.shN != nullptr ? src.src_rest * 3u : 0u;
                for (std::uint32_t s = 0; s < pool.dst_slots; ++s) {
                    std::uint32_t packed = 0u;
                    if (live) {
                        const std::uint32_t c0 = s * 4u;
                        for (std::uint32_t comp = 0; comp < 4u; ++comp) {
                            const std::uint32_t c = c0 + comp;
                            std::uint8_t b = 0u;
                            if (c < active_floats) {
                                const std::uint32_t coeff = c / 3u;
                                const std::uint32_t band = coeff < 3u ? 0u : (coeff < 8u ? 1u : 2u);
                                const float max_abs =
                                    fmaxf(reinterpret_cast<const float*>(frame)[band], 1e-6f);
                                b = static_cast<std::uint8_t>(quantizeS8(
                                    src.shN[static_cast<std::size_t>(i) * active_floats + c],
                                    max_abs));
                            }
                            packed |= static_cast<std::uint32_t>(b) << (comp * 8u);
                        }
                    }
                    pool.shN[shNSlotIndex(page, page_splats, i, s, pool.dst_slots)] = packed;
                }
            }
        }

    } // namespace

    cudaError_t launchLodPageDequant(const std::uint8_t* const device_slot,
                                     const lfs::io::RadPagePackedDesc& desc,
                                     const LodPoolDeviceView& pool,
                                     const std::uint32_t page,
                                     const std::uint32_t page_splats,
                                     const cudaStream_t stream) {
        constexpr std::uint32_t kBlock = 256;
        const std::uint32_t grid = (page_splats + kBlock - 1) / kBlock;
        lodPageDequantKernel<<<grid, kBlock, 0, stream>>>(device_slot, desc, pool, page, page_splats);
        return cudaGetLastError();
    }

    cudaError_t launchLodPageQuantizeFromTensors(const LodPageTensorSources& src,
                                                 const LodPoolDeviceView& pool,
                                                 const std::uint32_t page,
                                                 const std::uint32_t page_splats,
                                                 const cudaStream_t stream) {
        if (pool.page_frames == nullptr) {
            return cudaErrorInvalidValue;
        }
        constexpr std::uint32_t kBlock = 256;
        float4* const frame =
            pool.page_frames + static_cast<std::size_t>(page) * lodq::kPageFrameFloat4s;
        // Only float4 [0] (sh band maxima) belongs to this path; the render
        // thread owns the bounds frame slots for sync-published pages.
        if (const cudaError_t status = cudaMemsetAsync(frame, 0, 16u, stream);
            status != cudaSuccess) {
            return status;
        }
        if (src.shN != nullptr && src.src_rest > 0u && src.count > 0u) {
            const std::uint32_t grid = (src.count + kBlock - 1) / kBlock;
            lodPageShBandMaxKernel<<<grid, kBlock, 0, stream>>>(src, frame);
            if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) {
                return status;
            }
        }
        const std::uint32_t grid = (page_splats + kBlock - 1) / kBlock;
        lodPageQuantizeKernel<<<grid, kBlock, 0, stream>>>(src, pool, page, page_splats);
        return cudaGetLastError();
    }

} // namespace lfs::vis
