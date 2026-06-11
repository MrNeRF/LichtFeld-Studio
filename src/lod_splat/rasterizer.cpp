/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rasterizer.hpp"
#include "util.hpp"

#include <algorithm>
#include <atomic>
#include <cstdio>

namespace lfs::lod {

    namespace {

        constexpr int kTile = 16;

        struct Projected {
            float px, py;     // screen position
            float ca, cb, cc; // inverse 2D covariance (conic): a, b, c
            float radius;     // 3-sigma screen radius
            float alpha;
            Vec3 color;
            float depth;
        };

    } // namespace

    void rasterize(const Camera& cam, const std::vector<RasterInput>& inputs, Image& out,
                   RasterStats& stats, unsigned threads) {
        const double t_start = nowMs();
        if (threads == 0)
            threads = std::thread::hardware_concurrency();
        stats = RasterStats{};
        out.init(cam.width, cam.height);

        uint64_t total = 0;
        for (const RasterInput& in : inputs)
            total += in.count;
        stats.splats_in = total;

        // ---- project all splats ----
        std::vector<Projected> proj(total);
        std::vector<uint8_t> valid(total, 0);
        const float f = cam.focalPx();
        const float cx = cam.width * 0.5f, cy = cam.height * 0.5f;

        // flatten input ranges for indexed access
        std::vector<uint64_t> range_start(inputs.size());
        {
            uint64_t acc = 0;
            for (size_t r = 0; r < inputs.size(); ++r) {
                range_start[r] = acc;
                acc += inputs[r].count;
            }
        }

        for (size_t r = 0; r < inputs.size(); ++r) {
            const RasterInput& in = inputs[r];
            const SplatCloud& cl = *in.cloud;
            parallelFor(0, in.count, threads, [&](size_t lo, size_t hi) {
                for (size_t k = lo; k < hi; ++k) {
                    const size_t s = in.offset + k;
                    const size_t o = range_start[r] + k;
                    const Vec3 t = cam.worldToView(cl.means[s]);
                    const float depth = -t.z;
                    if (depth < cam.znear || depth > cam.zfar)
                        continue;
                    const float vx = t.x, vy = t.y, vz = depth;
                    const float ppx = f * vx / vz + cx;
                    const float ppy = f * vy / vz + cy;
                    if (ppx < -200 || ppx > cam.width + 200 || ppy < -200 || ppy > cam.height + 200)
                        continue;

                    // world covariance -> camera (x right, y up, z = depth forward)
                    const Sym3 cov3 = Sym3::fromCovariance(cl.rotations[s].toMat3(), cl.linearScale(s));
                    const Mat3 C = cov3.toMat3();
                    Mat3 A = cam.rotation.transposed(); // world-to-view
                    for (int col = 0; col < 3; ++col)
                        A.at(2, col) = -A.at(2, col); // z forward
                    const Mat3 Ccam = A * C * A.transposed();

                    // perspective Jacobian
                    const float j00 = f / vz, j02 = -f * vx / (vz * vz);
                    const float j11 = f / vz, j12 = -f * vy / (vz * vz);
                    const float c00 = Ccam.at(0, 0), c01 = Ccam.at(0, 1), c02 = Ccam.at(0, 2);
                    const float c11 = Ccam.at(1, 1), c12 = Ccam.at(1, 2), c22 = Ccam.at(2, 2);
                    float s00 = j00 * (j00 * c00 + j02 * c02) + j02 * (j00 * c02 + j02 * c22);
                    float s01 = j00 * (j11 * c01 + j12 * c02) + j02 * (j11 * c12 + j12 * c22);
                    float s11 = j11 * (j11 * c11 + j12 * c12) + j12 * (j11 * c12 + j12 * c22);
                    s00 += 0.3f; // low-pass dilation as in 3DGS
                    s11 += 0.3f;
                    const float det = s00 * s11 - s01 * s01;
                    if (det <= 1e-12f)
                        continue;
                    const float inv_det = 1.f / det;

                    const float mid = 0.5f * (s00 + s11);
                    const float lmax = mid + std::sqrt(std::max(0.01f, mid * mid - det));
                    const float radius = std::ceil(3.f * std::sqrt(lmax));

                    Projected& pr = proj[o];
                    pr.px = ppx;
                    pr.py = ppy;
                    pr.ca = s11 * inv_det;
                    pr.cb = -s01 * inv_det;
                    pr.cc = s00 * inv_det;
                    pr.radius = radius;
                    // lodOpacity merged splats can carry alpha > 1; clamp only
                    // per-pixel after the Gaussian falloff
                    pr.alpha = std::min(8.f, cl.opacity(s) * in.weight);
                    pr.color = cl.color(s);
                    pr.depth = depth;
                    valid[o] = pr.alpha > 1.f / 255.f;
                }
            });
        }

        // ---- global depth sort (front to back) ----
        std::vector<uint32_t> order;
        order.reserve(total);
        {
            ScopedTimer t(stats.sort_ms);
            std::vector<uint64_t> keys;
            keys.reserve(total);
            std::vector<uint32_t> idx;
            idx.reserve(total);
            for (uint64_t i = 0; i < total; ++i)
                if (valid[i]) {
                    uint32_t dk;
                    float d = proj[i].depth;
                    std::memcpy(&dk, &d, 4);
                    keys.push_back(dk); // positive floats sort correctly as uints
                    idx.push_back((uint32_t)i);
                }
            radixSortPairs(keys, idx);
            order = std::move(idx);
        }
        stats.splats_projected = order.size();

        // ---- bin into tiles (preserving depth order) ----
        const int tiles_x = (cam.width + kTile - 1) / kTile;
        const int tiles_y = (cam.height + kTile - 1) / kTile;
        std::vector<std::vector<uint32_t>> bins(tiles_x * tiles_y);
        {
            ScopedTimer t(stats.bin_ms);
            for (uint32_t oi : order) {
                const Projected& pr = proj[oi];
                const int x0 = std::max(0, (int)((pr.px - pr.radius) / kTile));
                const int x1 = std::min(tiles_x - 1, (int)((pr.px + pr.radius) / kTile));
                const int y0 = std::max(0, (int)((pr.py - pr.radius) / kTile));
                const int y1 = std::min(tiles_y - 1, (int)((pr.py + pr.radius) / kTile));
                for (int ty = y0; ty <= y1; ++ty)
                    for (int tx = x0; tx <= x1; ++tx)
                        bins[ty * tiles_x + tx].push_back(oi);
            }
        }

        // ---- blend per tile ----
        std::atomic<uint64_t> contrib_total{0};
        {
            ScopedTimer t(stats.blend_ms);
            parallelFor(0, bins.size(), threads, [&](size_t lo, size_t hi) {
                float T[kTile * kTile];
                float acc[kTile * kTile * 3];
                uint64_t contribs = 0;
                for (size_t bi = lo; bi < hi; ++bi) {
                    const auto& bin = bins[bi];
                    if (bin.empty())
                        continue;
                    const int tx = (int)(bi % tiles_x), ty = (int)(bi / tiles_x);
                    const int px0 = tx * kTile, py0 = ty * kTile;
                    const int tw = std::min(kTile, cam.width - px0);
                    const int th = std::min(kTile, cam.height - py0);
                    std::fill(T, T + kTile * kTile, 1.f);
                    std::fill(acc, acc + kTile * kTile * 3, 0.f);
                    int live = tw * th;
                    for (uint32_t oi : bin) {
                        if (live <= 0)
                            break;
                        const Projected& pr = proj[oi];
                        const int lx0 = std::max(0, (int)(pr.px - pr.radius) - px0);
                        const int lx1 = std::min(tw - 1, (int)(pr.px + pr.radius) - px0);
                        const int ly0 = std::max(0, (int)(pr.py - pr.radius) - py0);
                        const int ly1 = std::min(th - 1, (int)(pr.py + pr.radius) - py0);
                        for (int y = ly0; y <= ly1; ++y) {
                            for (int x = lx0; x <= lx1; ++x) {
                                const int li = y * kTile + x;
                                float& trans = T[li];
                                if (trans < 1e-3f)
                                    continue;
                                const float dx = (float)(px0 + x) + 0.5f - pr.px;
                                const float dy = (float)(py0 + y) + 0.5f - pr.py;
                                const float power =
                                    0.5f * (pr.ca * dx * dx + pr.cc * dy * dy) + pr.cb * dx * dy;
                                if (power > 4.6f)
                                    continue; // exp(-4.6) ~ 0.01
                                if (power < 0.f)
                                    continue;
                                const float a = std::min(0.99f, pr.alpha * std::exp(-power));
                                if (a < 1.f / 255.f)
                                    continue;
                                const float w = trans * a;
                                acc[li * 3 + 0] += w * pr.color.x;
                                acc[li * 3 + 1] += w * pr.color.y;
                                acc[li * 3 + 2] += w * pr.color.z;
                                trans *= 1.f - a;
                                ++contribs;
                                if (trans < 1e-3f)
                                    --live;
                            }
                        }
                    }
                    for (int y = 0; y < th; ++y)
                        for (int x = 0; x < tw; ++x) {
                            const size_t gi = ((size_t)(py0 + y) * cam.width + (px0 + x)) * 3;
                            const int li = y * kTile + x;
                            out.rgb[gi + 0] = acc[li * 3 + 0];
                            out.rgb[gi + 1] = acc[li * 3 + 1];
                            out.rgb[gi + 2] = acc[li * 3 + 2];
                        }
                }
                contrib_total += contribs;
            });
        }

        stats.mean_contrib_per_px = (double)contrib_total / ((double)cam.width * cam.height);
        stats.total_ms = nowMs() - t_start;
    }

    double psnr(const Image& a, const Image& b) {
        double mse = 0;
        for (size_t i = 0; i < a.rgb.size(); ++i) {
            const double d = (double)a.rgb[i] - b.rgb[i];
            mse += d * d;
        }
        mse /= (double)a.rgb.size();
        if (mse < 1e-12)
            return 99.0;
        return 10.0 * std::log10(1.0 / mse);
    }

    double meanAbsDiff(const Image& a, const Image& b) {
        double s = 0;
        for (size_t i = 0; i < a.rgb.size(); ++i)
            s += std::fabs((double)a.rgb[i] - b.rgb[i]);
        return s / (double)a.rgb.size();
    }

    bool writePpm(const std::string& path, const Image& img) {
        FILE* f = std::fopen(path.c_str(), "wb");
        if (!f)
            return false;
        std::fprintf(f, "P6\n%d %d\n255\n", img.w, img.h);
        std::vector<uint8_t> row(img.w * 3);
        for (int y = 0; y < img.h; ++y) {
            for (int x = 0; x < img.w * 3; ++x) {
                const float v = img.rgb[(size_t)y * img.w * 3 + x];
                row[x] = (uint8_t)std::clamp(v * 255.f, 0.f, 255.f);
            }
            std::fwrite(row.data(), 1, row.size(), f);
        }
        std::fclose(f);
        return true;
    }

} // namespace lfs::lod
