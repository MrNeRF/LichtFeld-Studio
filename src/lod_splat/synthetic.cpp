/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "synthetic.hpp"

#include <cmath>

namespace lfs::lod {

    namespace {

        struct Rng {
            uint64_t s;
            explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ULL + 1) {}
            uint64_t next() {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                return s;
            }
            float uniform() { return (float)(next() >> 40) / (float)(1 << 24); }
            float uniform(float a, float b) { return a + (b - a) * uniform(); }
        };

        struct Rect {
            Vec3 origin, t1, t2; // corner + two edge vectors (full lengths)
            Vec3 normal;
            Vec3 base_color;
        };

        Quat frameToQuat(const Vec3& a1, const Vec3& a2, const Vec3& n) {
            Mat3 m;
            const Vec3 c0 = a1.normalized(), c1 = a2.normalized(), c2 = n.normalized();
            m.at(0, 0) = c0.x;
            m.at(1, 0) = c0.y;
            m.at(2, 0) = c0.z;
            m.at(0, 1) = c1.x;
            m.at(1, 1) = c1.y;
            m.at(2, 1) = c1.z;
            m.at(0, 2) = c2.x;
            m.at(1, 2) = c2.y;
            m.at(2, 2) = c2.z;
            if (m.det() < 0.f) {
                m.at(0, 2) = -m.at(0, 2);
                m.at(1, 2) = -m.at(1, 2);
                m.at(2, 2) = -m.at(2, 2);
            }
            return Quat::fromMat3(m);
        }

        // cheap value noise for surface color variation
        float hashNoise(float x, float y, uint32_t salt) {
            uint32_t h = (uint32_t)(x * 73856093.f) ^ (uint32_t)(y * 19349663.f) ^ salt;
            h ^= h >> 13;
            h *= 0x85EBCA6Bu;
            h ^= h >> 16;
            return (float)(h & 0xFFFF) / 65535.f;
        }

    } // namespace

    void generateCityScene(const SyntheticSceneParams& p, SplatCloud& out) {
        Rng rng(p.seed);
        std::vector<Rect> rects;

        const float world = p.grid * p.block_size;
        const Vec3 ground_color{0.35f, 0.34f, 0.32f};
        rects.push_back({{0, 0, 0}, {world, 0, 0}, {0, 0, world}, {0, 1, 0}, ground_color});

        for (int gx = 0; gx < p.grid; ++gx) {
            for (int gz = 0; gz < p.grid; ++gz) {
                const float side = p.block_size * p.building_fill * rng.uniform(0.8f, 1.1f);
                const float h = rng.uniform(p.min_height, p.max_height);
                const float cx = (gx + 0.5f) * p.block_size, cz = (gz + 0.5f) * p.block_size;
                const float x0 = cx - side * 0.5f, x1 = cx + side * 0.5f;
                const float z0 = cz - side * 0.5f, z1 = cz + side * 0.5f;
                const Vec3 col{rng.uniform(0.25f, 0.85f), rng.uniform(0.25f, 0.85f), rng.uniform(0.25f, 0.85f)};
                // four walls + roof
                rects.push_back({{x0, 0, z0}, {x1 - x0, 0, 0}, {0, h, 0}, {0, 0, -1}, col});
                rects.push_back({{x0, 0, z1}, {x1 - x0, 0, 0}, {0, h, 0}, {0, 0, 1}, col});
                rects.push_back({{x0, 0, z0}, {0, 0, z1 - z0}, {0, h, 0}, {-1, 0, 0}, col});
                rects.push_back({{x1, 0, z0}, {0, 0, z1 - z0}, {0, h, 0}, {1, 0, 0}, col});
                rects.push_back({{x0, h, z0}, {x1 - x0, 0, 0}, {0, 0, z1 - z0}, {0, 1, 0}, col * 0.7f});
            }
        }

        double total_area = 0;
        for (const Rect& r : rects)
            total_area += (double)r.t1.norm() * r.t2.norm();
        const float spacing = (float)std::sqrt(total_area / (double)p.target_splats);

        out.reserve(p.target_splats + p.target_splats / 8);
        const float tangent_sigma = spacing * 0.6f; // neighbors overlap for watertight blending
        const float normal_sigma = spacing * 0.08f; // thin surfel
        const Vec3 log_scale{std::log(tangent_sigma), std::log(tangent_sigma), std::log(normal_sigma)};

        for (uint32_t ri = 0; ri < rects.size(); ++ri) {
            const Rect& r = rects[ri];
            const float l1 = r.t1.norm(), l2 = r.t2.norm();
            const int n1 = std::max(1, (int)(l1 / spacing));
            const int n2 = std::max(1, (int)(l2 / spacing));
            const Vec3 u1 = r.t1 * (1.f / n1), u2 = r.t2 * (1.f / n2);
            const Quat rot = frameToQuat(r.t1, r.t2, r.normal);
            for (int i = 0; i < n1; ++i) {
                for (int j = 0; j < n2; ++j) {
                    const float jx = rng.uniform(0.25f, 0.75f), jy = rng.uniform(0.25f, 0.75f);
                    const Vec3 pos = r.origin + u1 * (i + jx) + u2 * (j + jy);
                    // window/stripe pattern on walls, mild noise everywhere
                    const float noise = hashNoise((float)i, (float)j, ri * 7919u);
                    float shade = 0.85f + 0.3f * (noise - 0.5f);
                    if (std::fabs(r.normal.y) < 0.5f && ((i / 2 + j / 3) % 4 == 0))
                        shade *= 0.45f; // darker "window" bands on facades
                    const Vec3 c = r.base_color * shade;
                    const Vec3 dc{(c.x - 0.5f) / SH_C0, (c.y - 0.5f) / SH_C0, (c.z - 0.5f) / SH_C0};
                    out.push(pos, log_scale, rot, logit(0.9f), dc);
                }
            }
        }
    }

} // namespace lfs::lod
