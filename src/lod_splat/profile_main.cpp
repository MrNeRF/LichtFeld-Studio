/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// LOD splat profiling harness.
//
// Pipeline: synthetic city scene (or a real 3DGS PLY via --ply) -> offline
// octree hierarchy with moment-matched representatives -> per-frame budgeted
// cut selection with hysteresis/cross-fade -> simulated streaming -> per-frame
// depth sort of the working set. Reports timings, budget adherence, streaming
// behavior; optional rasterizer-based quality validation (--validate).
#include "bhatt_ref.hpp"
#include "cut.hpp"
#include "hierarchy.hpp"
#include "ply_io.hpp"
#include "rasterizer.hpp"
#include "streaming.hpp"
#include "synthetic.hpp"

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <string>

using namespace lfs::lod;

namespace {

    struct Args {
        size_t splats = 3'000'000;
        float quality = 3.f;
        float tau_px = 1.f;
        float hysteresis = 1.5f;
        uint64_t budget = 1'500'000;
        float dissolve_band = 0.18f;
        int frames = 600;
        int width = 1280, height = 720;
        std::string ply; // optional input scene
        std::string out_dir = "lod_profile_out";
        bool validate = false;
        bool compare = false;
        bool compare_lite = false; // hybrid-vs-truth only (skip original + bhatt builds)
        uint32_t leaf = 64;
        bool no_streaming = false;
        size_t resident_mb = 256;
        double bandwidth_mb_s = 800.0;
        double latency_ms = 2.0;
    };

    Args parseArgs(int argc, char** argv) {
        Args a;
        for (int i = 1; i < argc; ++i) {
            auto next = [&]() -> const char* { return (i + 1 < argc) ? argv[++i] : ""; };
            if (!strcmp(argv[i], "--splats"))
                a.splats = strtoull(next(), nullptr, 10);
            else if (!strcmp(argv[i], "--quality"))
                a.quality = strtof(next(), nullptr);
            else if (!strcmp(argv[i], "--tau"))
                a.tau_px = strtof(next(), nullptr);
            else if (!strcmp(argv[i], "--hysteresis"))
                a.hysteresis = strtof(next(), nullptr);
            else if (!strcmp(argv[i], "--budget"))
                a.budget = strtoull(next(), nullptr, 10);
            else if (!strcmp(argv[i], "--band"))
                a.dissolve_band = strtof(next(), nullptr);
            else if (!strcmp(argv[i], "--compare"))
                a.compare = true;
            else if (!strcmp(argv[i], "--compare-lite")) {
                a.compare = true;
                a.compare_lite = true;
            } else if (!strcmp(argv[i], "--leaf"))
                a.leaf = (uint32_t)strtoul(next(), nullptr, 10);
            else if (!strcmp(argv[i], "--frames"))
                a.frames = atoi(next());
            else if (!strcmp(argv[i], "--width"))
                a.width = atoi(next());
            else if (!strcmp(argv[i], "--height"))
                a.height = atoi(next());
            else if (!strcmp(argv[i], "--ply"))
                a.ply = next();
            else if (!strcmp(argv[i], "--out"))
                a.out_dir = next();
            else if (!strcmp(argv[i], "--validate"))
                a.validate = true;
            else if (!strcmp(argv[i], "--no-streaming"))
                a.no_streaming = true;
            else if (!strcmp(argv[i], "--resident-mb"))
                a.resident_mb = strtoull(next(), nullptr, 10);
            else if (!strcmp(argv[i], "--bandwidth"))
                a.bandwidth_mb_s = strtod(next(), nullptr);
            else if (!strcmp(argv[i], "--latency"))
                a.latency_ms = strtod(next(), nullptr);
            else {
                std::fprintf(stderr, "unknown arg %s\n", argv[i]);
                exit(2);
            }
        }
        return a;
    }

    // Flythrough: street-level run down a canyon, then climb to an overview orbit.
    // Scene spans [0, world]^2 in xz, ground at y=0. street_x must lie on a block
    // boundary so the low pass flies down a canyon instead of through facades.
    Camera cameraAt(double u, float world, float street_x, int w, int h) {
        Vec3 pos, look;
        if (u < 0.55) {
            const float z = 6.f + (world - 24.f) * (float)(u / 0.55);
            pos = {street_x, 9.f, z};
            look = {street_x, 8.f, z + 40.f};
        } else if (u < 0.8) {
            const float t = (float)((u - 0.55) / 0.25);
            const float s = t * t * (3.f - 2.f * t); // smoothstep
            const Vec3 p0{street_x, 9.f, world - 18.f};
            const Vec3 p1{world * 0.5f, world * 0.7f, world * 1.35f};
            pos = p0 + (p1 - p0) * s;
            const Vec3 l0{street_x, 8.f, world + 22.f};
            const Vec3 l1{world * 0.5f, 0.f, world * 0.5f};
            look = l0 + (l1 - l0) * s;
        } else {
            const float t = (float)((u - 0.8) / 0.2);
            const float ang = t * 1.7f;
            const float r = world * 0.85f;
            const Vec3 c{world * 0.5f, 0.f, world * 0.5f};
            pos = {c.x + r * std::sin(ang), world * 0.7f, c.z + r * std::cos(ang)};
            look = c;
        }
        return Camera::lookAt(pos, look, {0, 1, 0}, 0.9f, w, h);
    }

    struct Percentiles {
        double p50 = 0, p95 = 0, max = 0, mean = 0;
    };
    Percentiles percentiles(std::vector<double> v) {
        Percentiles p;
        if (v.empty())
            return p;
        std::sort(v.begin(), v.end());
        p.p50 = v[v.size() / 2];
        p.p95 = v[(size_t)(v.size() * 0.95)];
        p.max = v.back();
        for (double x : v)
            p.mean += x;
        p.mean /= (double)v.size();
        return p;
    }

    // Depth keys + radix sort over the frame's working set — the per-frame cost
    // that the splat budget is supposed to bound.
    double depthSortMs(const Hierarchy& h, const std::vector<DrawEntry>& draw, const Camera& cam,
                       double* std_sort_ms = nullptr) {
        static std::vector<uint32_t> keys32; // reused across frames
        static std::vector<uint32_t> idx;
        keys32.clear();
        idx.clear();
        const Vec3 fwd = cam.forward();
        const double t0 = nowMs();
        for (const DrawEntry& e : draw) {
            const SplatCloud& cl = e.from_leaves ? h.leaves : h.merged;
            for (uint32_t i = e.offset; i < e.offset + e.count; ++i) {
                const float d = fwd.dot(cl.means[i] - cam.position);
                uint32_t k;
                std::memcpy(&k, &d, 4);
                // map float to monotonic uint (handles negative depths behind camera)
                k = (k & 0x80000000u) ? ~k : (k | 0x80000000u);
                keys32.push_back(k);
                idx.push_back((uint32_t)idx.size());
            }
        }
        const double key_ms = nowMs() - t0;
        if (std_sort_ms) {
            std::vector<std::pair<uint32_t, uint32_t>> pairs(keys32.size());
            for (size_t i = 0; i < keys32.size(); ++i)
                pairs[i] = {keys32[i], idx[i]};
            const double s0 = nowMs();
            std::sort(pairs.begin(), pairs.end());
            *std_sort_ms = nowMs() - s0;
        }
        const double r0 = nowMs();
        radixSortPairs(keys32, idx);
        return key_ms + (nowMs() - r0);
    }

    std::vector<RasterInput> drawToRaster(const Hierarchy& h, const std::vector<DrawEntry>& draw,
                                          const StreamingManager* stream) {
        std::vector<RasterInput> in;
        in.reserve(draw.size());
        for (const DrawEntry& e : draw) {
            if (stream) {
                if (const SplatCloud* c = stream->chunk(e.node)) {
                    in.push_back({c, 0, (uint32_t)c->size(), e.weight});
                    continue;
                }
            }
            in.push_back({e.from_leaves ? &h.leaves : &h.merged, e.offset, e.count, e.weight});
        }
        return in;
    }

    void selfTest(const Hierarchy& h) {
        // Every leaf splat must be covered by exactly one node of the initial cut
        // after a converged selection. Run a selection and check coverage.
        CutParams cp;
        cp.quality = 2.f;
        cp.splat_budget = 1u << 30;
        CutSelector sel(h, cp);
        Camera cam = Camera::lookAt({10, 10, 10}, {50, 0, 50}, {0, 1, 0}, 0.9f, 640, 360);
        std::vector<DrawEntry> draw;
        CutStats st;
        for (int i = 0; i < 30; ++i)
            sel.update(cam, nullptr, draw, st);

        std::vector<uint8_t> covered(h.leaves.size(), 0);
        std::vector<uint32_t> stack;
        for (uint32_t u : sel.cutNodes())
            stack.push_back(u);
        while (!stack.empty()) {
            const uint32_t u = stack.back();
            stack.pop_back();
            const Node& n = h.nodes[u];
            if (n.is_leaf) {
                for (uint32_t i = n.rep_offset; i < n.rep_offset + n.rep_count; ++i) {
                    if (covered[i]) {
                        std::fprintf(stderr, "SELF-TEST FAIL: double cover\n");
                        exit(1);
                    }
                    covered[i] = 1;
                }
            } else {
                for (int c = 0; c < n.child_count; ++c)
                    stack.push_back(n.children[c]);
            }
        }
        for (size_t i = 0; i < covered.size(); ++i)
            if (!covered[i]) {
                std::fprintf(stderr, "SELF-TEST FAIL: uncovered leaf splat\n");
                exit(1);
            }

        // Integrated alpha conservation at the root (moment matching target):
        double leaf_alpha_area = 0, root_alpha_area = 0;
        for (size_t i = 0; i < h.leaves.size(); ++i)
            leaf_alpha_area += h.leaves.opacity(i) * h.leaves.meanArea(i);
        const Node& root = h.nodes[0];
        const SplatCloud& rc = h.cloudFor(root);
        for (uint32_t i = root.rep_offset; i < root.rep_offset + root.rep_count; ++i)
            root_alpha_area += rc.opacity(i) * rc.meanArea(i);
        const double ratio = root_alpha_area / leaf_alpha_area;
        std::printf("[self-test] cut covers all %zu leaf splats exactly once\n", h.leaves.size());
        std::printf("[self-test] integrated alpha root/leaves = %.3f (1.0 = exact conservation)\n", ratio);
    }

} // namespace

int main(int argc, char** argv) {
    const Args args = parseArgs(argc, argv);

    std::string mk = "mkdir -p '" + args.out_dir + "'";
    if (system(mk.c_str()) != 0)
        return 1;

    // ---------- scene ----------
    SplatCloud cloud;
    double scene_ms = 0;
    float world = 0.f;
    {
        ScopedTimer t(scene_ms);
        if (!args.ply.empty()) {
            std::string err;
            if (!loadPly(args.ply, cloud, &err)) {
                std::fprintf(stderr, "PLY load failed: %s\n", err.c_str());
                return 1;
            }
        } else {
            SyntheticSceneParams sp;
            sp.target_splats = args.splats;
            generateCityScene(sp, cloud);
            world = sp.grid * sp.block_size;
        }
    }
    std::printf("scene: %zu splats (%.0f ms)%s\n", cloud.size(), scene_ms,
                args.ply.empty() ? " [synthetic city]" : args.ply.c_str());

    // ---------- hierarchy ----------
    Hierarchy h;
    HierarchyParams hp;
    hp.max_leaf_splats = args.leaf;
    buildHierarchy(std::move(cloud), hp, h);
    const BuildStats& bs = h.stats;
    std::printf("\n=== hierarchy build ===\n");
    std::printf("total %.0f ms  (morton %.0f, sort %.0f, reorder %.0f, topology %.0f, merge %.0f)\n",
                bs.total_ms, bs.morton_ms, bs.sort_ms, bs.reorder_ms, bs.topology_ms, bs.merge_ms);
    std::printf("nodes %zu, leaf splats %zu, merged reps %zu (+%.1f%% memory)\n", h.nodes.size(),
                h.leaves.size(), h.merged.size(), 100.0 * h.merged.size() / h.leaves.size());
    std::printf("%-6s %-8s %-12s %s\n", "depth", "nodes", "reps", "density vs next finer");
    for (size_t d = 0; d < bs.nodes_per_depth.size(); ++d) {
        double ratio = 0;
        if (d + 1 < bs.reps_per_depth.size() && bs.reps_per_depth[d + 1])
            ratio = (double)bs.reps_per_depth[d] / (double)bs.reps_per_depth[d + 1];
        std::printf("%-6zu %-8u %-12" PRIu64 " %.3f\n", d, bs.nodes_per_depth[d],
                    bs.reps_per_depth[d], ratio);
    }

    if (world == 0.f) {
        const Vec3 e = h.scene_bounds.extent();
        world = std::max(e.x, e.z);
    }
    // block boundary nearest to 1/4 of the way across (synthetic block = 18)
    const float street_x = args.ply.empty() ? 18.f * std::max(1.f, std::floor(world / 4.f / 18.f))
                                            : world / 4.f;

    selfTest(h);

    // ---------- A/B comparison mode: original scheme vs hybrid (ported) ----------
    if (args.compare) {
        std::printf("\n=== A/B comparison: original vs hybrid scheme, budget %llu ===\n",
                    (unsigned long long)args.budget);
        // `h` was built with the ported defaults (similarity pairing +
        // lodOpacity). Build the original scheme from a fresh scene copy.
        Hierarchy h1;
        if (!args.compare_lite) {
            SplatCloud cloud1;
            if (!args.ply.empty()) {
                std::string err;
                loadPly(args.ply, cloud1, &err);
            } else {
                SyntheticSceneParams sp2;
                sp2.target_splats = args.splats;
                generateCityScene(sp2, cloud1);
            }
            HierarchyParams hp1;
            hp1.similarity_pairing = false;
            hp1.lod_opacity = false;
            buildHierarchy(std::move(cloud1), hp1, h1);
        }
        // LichtFeld's shipped method (CPU port of bhatt_lod + selection shader)
        BhattRefTree bhatt;
        if (!args.compare_lite) {
            SplatCloud cloud2;
            if (!args.ply.empty()) {
                std::string err;
                loadPly(args.ply, cloud2, &err);
            } else {
                SyntheticSceneParams sp3;
                sp3.target_splats = args.splats;
                generateCityScene(sp3, cloud2);
            }
            buildBhattRef(cloud2, 1.25f, bhatt);
            std::printf("bhatt_lod (LichtFeld) build: %.0f ms, %zu merges, %zu output nodes\n",
                        bhatt.build_ms, bhatt.total_merges, bhatt.output_count);
        }
        const int vw = 1280, vh = 720;
        const struct {
            const char* name;
            double u;
        } views[2] = {{"street", 0.25},
                      {"overview", 0.92}};
        for (const auto& v : views) {
            Camera cam = cameraAt(v.u, world, street_x, vw, vh);
            Image img_full;
            RasterStats rs_full;
            std::vector<RasterInput> full{{&h.leaves, 0, (uint32_t)h.leaves.size(), 1.f}};
            rasterize(cam, full, img_full, rs_full);
            writePpm(args.out_dir + "/cmp_" + v.name + "_full.ppm", img_full);
            std::printf("[%s] ground truth: %" PRIu64 " splats, %.1f contrib/px\n", v.name,
                        rs_full.splats_in, rs_full.mean_contrib_per_px);

            const struct {
                const char* tag;
                const Hierarchy* hh;
            } schemes[2] = {
                {"original", &h1},
                {"hybrid", &h}};
            for (const auto& sc : schemes) {
                if (args.compare_lite && sc.hh == &h1)
                    continue;
                CutParams vp;
                vp.quality = args.quality;
                vp.tau_px = args.tau_px;
                vp.hysteresis = args.hysteresis;
                vp.splat_budget = args.budget;
                vp.dissolve_band = args.dissolve_band;
                CutSelector vsel(*sc.hh, vp);
                std::vector<DrawEntry> draw;
                CutStats st;
                for (int i = 0; i < 40; ++i)
                    vsel.update(cam, nullptr, draw, st);
                Image img;
                RasterStats rs;
                rasterize(cam, drawToRaster(*sc.hh, draw, nullptr), img, rs);
                std::printf("[%s] %-8s: %8" PRIu64 " splats, %5.1f contrib/px, PSNR %6.2f dB"
                            "  (select %.2f ms, cut %u nodes)\n",
                            v.name, sc.tag, st.drawn_splats, rs.mean_contrib_per_px,
                            psnr(img_full, img), st.select_ms, st.cut_nodes);
                writePpm(args.out_dir + "/cmp_" + std::string(v.name) + "_" + sc.tag + ".ppm", img);
            }
            if (!args.compare_lite) {
                const double s0 = nowMs();
                const float limit = bhattLimitForBudget(bhatt, cam, args.budget);
                BhattSelection bsel;
                bhattSelect(bhatt, cam, limit, bsel);
                const double sel_ms = nowMs() - s0;
                SplatCloud gathered;
                bhattGather(bhatt, bsel, gathered);
                std::vector<RasterInput> in{{&gathered, 0, (uint32_t)gathered.size(), 1.f}};
                Image img;
                RasterStats rs;
                rasterize(cam, in, img, rs);
                std::printf("[%s] %-8s: %8zu splats, %5.1f contrib/px, PSNR %6.2f dB"
                            "  (limit %.2e, select+search %.0f ms)\n",
                            v.name, "lichtfeld", bsel.nodes.size(), rs.mean_contrib_per_px,
                            psnr(img_full, img), limit, sel_ms);
                writePpm(args.out_dir + "/cmp_" + std::string(v.name) + "_lichtfeld.ppm", img);
            }
        }
        return 0;
    }

    // ---------- streaming ----------
    StreamParams sp;
    sp.bandwidth_mb_s = args.bandwidth_mb_s;
    sp.base_latency_ms = args.latency_ms;
    sp.resident_budget_bytes = args.resident_mb << 20;
    double stream_build_ms = 0;
    StreamingManager* stream = nullptr;
    {
        ScopedTimer t(stream_build_ms);
        if (!args.no_streaming)
            stream = new StreamingManager(h, sp);
    }
    if (stream) {
        const size_t raw = h.leaves.size() * 56 + h.merged.size() * 56;
        std::printf("\n=== streaming store ===\n");
        std::printf("encode %.0f ms, store %.1f MB (raw f32 SoA %.1f MB, %.2fx)\n", stream_build_ms,
                    stream->encodedStoreBytes() / 1048576.0, raw / 1048576.0,
                    (double)raw / stream->encodedStoreBytes());
    }

    // ---------- frame loop ----------
    CutParams cp;
    cp.quality = args.quality;
    cp.tau_px = args.tau_px;
    cp.hysteresis = args.hysteresis;
    cp.splat_budget = args.budget;
    cp.dissolve_band = args.dissolve_band;
    CutSelector sel(h, cp);

    FILE* csv = fopen((args.out_dir + "/frames.csv").c_str(), "w");
    fprintf(csv, "frame,select_ms,sort_ms,cut_nodes,visible_nodes,drawn_splats,refines,coarsens,"
                 "forced,blocked,band_parents,saturated,max_err,resident_chunks,resident_mb,"
                 "bytes_fetched_mb,evictions\n");

    std::vector<double> v_select, v_sort, v_drawn, v_total;
    std::vector<double> v_std_sort;
    const double dt_ms = 1000.0 / 60.0;
    double sim_time = 0;
    uint64_t prev_bytes = 0;

    std::printf("\n=== frame loop: %d frames @ %dx%d, quality=%.1f, budget=%" PRIu64 " ===\n",
                args.frames, args.width, args.height, args.quality, args.budget);
    for (int f = 0; f < args.frames; ++f) {
        const double u = (double)f / args.frames;
        Camera cam = cameraAt(u, world, street_x, args.width, args.height);
        const Camera cam_next = cameraAt((double)(f + 1) / args.frames, world, street_x, args.width, args.height);
        const Vec3 vel = (cam_next.position - cam.position) * (1000.f / (float)dt_ms);

        const double f0 = nowMs();
        std::vector<DrawEntry> draw;
        CutStats st;
        sel.update(cam, stream, draw, st);
        if (stream) {
            stream->prefetch(cam, vel, cp, sel.cutNodes());
            stream->tick(sim_time, (uint32_t)f);
        }
        double std_ms = 0;
        const bool sample_std = (f % 60 == 30);
        const double sort_ms = depthSortMs(h, draw, cam, sample_std ? &std_ms : nullptr);
        const double total_ms = nowMs() - f0 - std_ms;

        v_select.push_back(st.select_ms);
        v_sort.push_back(sort_ms);
        v_drawn.push_back((double)st.drawn_splats);
        v_total.push_back(total_ms);
        if (sample_std)
            v_std_sort.push_back(std_ms);

        const StreamStats* ss = stream ? &stream->stats() : nullptr;
        fprintf(csv, "%d,%.3f,%.3f,%u,%u,%" PRIu64 ",%u,%u,%u,%u,%u,%d,%.2f,%zu,%.1f,%.2f,%" PRIu64 "\n",
                f, st.select_ms, sort_ms, st.cut_nodes, st.visible_nodes, st.drawn_splats,
                st.refines, st.coarsens, st.forced_coarsens, st.blocked_by_residency,
                st.band_parents, st.budget_saturated ? 1 : 0, st.max_unresolved_error,
                ss ? ss->resident_chunks : 0, ss ? ss->resident_bytes / 1048576.0 : 0.0,
                ss ? (ss->bytes_fetched - prev_bytes) / 1048576.0 : 0.0, ss ? ss->evictions : 0);
        if (ss)
            prev_bytes = ss->bytes_fetched;
        sim_time += dt_ms;
    }
    fclose(csv);

    auto pr = [](const char* name, const Percentiles& p, const char* unit) {
        std::printf("%-28s mean %8.2f  p50 %8.2f  p95 %8.2f  max %8.2f %s\n", name, p.mean, p.p50,
                    p.p95, p.max, unit);
    };
    std::printf("\n=== per-frame profile ===\n");
    pr("cut selection", percentiles(v_select), "ms");
    pr("depth sort (radix)", percentiles(v_sort), "ms");
    pr("depth sort (std::sort)", percentiles(v_std_sort), "ms  [sampled]");
    pr("selection+sort total", percentiles(v_total), "ms");
    pr("drawn splats", percentiles(v_drawn), "");
    if (stream) {
        const StreamStats& ss = stream->stats();
        std::printf("\n=== streaming ===\n");
        std::printf("requests: %" PRIu64 " demand + %" PRIu64 " prefetch, completed %" PRIu64 "\n",
                    ss.requests_demand, ss.requests_prefetch, ss.completed);
        std::printf("fetched %.1f MB, evicted %" PRIu64 " chunks (%.1f MB), resident end %.1f MB / %zu chunks\n",
                    ss.bytes_fetched / 1048576.0, ss.evictions, ss.evicted_bytes / 1048576.0,
                    ss.resident_bytes / 1048576.0, ss.resident_chunks);
        std::printf("prefetched chunks later used by cut: %" PRIu64 " (%.0f%% of prefetches)\n",
                    ss.prefetch_useful,
                    ss.requests_prefetch ? 100.0 * ss.prefetch_useful / ss.requests_prefetch : 0.0);
        std::printf("avg request wait %.2f ms, decode total %.0f ms\n", ss.avg_wait_ms, ss.decode_ms);
    }

    // ---------- validation ----------
    if (args.validate) {
        std::printf("\n=== quality validation (CPU reference rasterizer) ===\n");
        const int vw = 960, vh = 540;
        const double frames_u[3] = {0.25, 0.62, 0.92}; // street / climb / overview
        const char* names[3] = {"street", "climb", "overview"};
        for (int k = 0; k < 3; ++k) {
            Camera cam = cameraAt(frames_u[k], world, street_x, vw, vh);
            // ground truth: all leaf splats
            std::vector<RasterInput> full{{&h.leaves, 0, (uint32_t)h.leaves.size(), 1.f}};
            Image img_full;
            RasterStats rs_full;
            rasterize(cam, full, img_full, rs_full);

            // converged LOD cut at this camera (fresh selector, no fades)
            CutParams vp = cp;
            CutSelector vsel(h, vp);
            std::vector<DrawEntry> draw;
            CutStats st;
            for (int i = 0; i < 40; ++i)
                vsel.update(cam, nullptr, draw, st);
            Image img_lod;
            RasterStats rs_lod;
            rasterize(cam, drawToRaster(h, draw, nullptr), img_lod, rs_lod);

            std::printf("[%s] full %" PRIu64 " splats (%.0f ms, %.1f contrib/px) | "
                        "LOD %" PRIu64 " splats (%.0f ms, %.1f contrib/px) | PSNR %.2f dB\n",
                        names[k], rs_full.splats_in, rs_full.total_ms, rs_full.mean_contrib_per_px,
                        st.drawn_splats, rs_lod.total_ms, rs_lod.mean_contrib_per_px,
                        psnr(img_full, img_lod));
            writePpm(args.out_dir + "/" + names[k] + "_full.ppm", img_full);
            writePpm(args.out_dir + "/" + names[k] + "_lod.ppm", img_lod);
        }

        // overdraw vs quality knob
        std::printf("\nquality knob -> measured overdraw (overview, budget unbounded):\n");
        Camera cam = cameraAt(0.92, world, street_x, vw, vh);
        for (float q : {1.f, 2.f, 4.f, 6.f}) {
            CutParams vp = cp;
            vp.quality = q;
            vp.splat_budget = 1u << 30; // unbounded: measure the knob itself
            CutSelector vsel(h, vp);
            std::vector<DrawEntry> draw;
            CutStats st;
            for (int i = 0; i < 40; ++i)
                vsel.update(cam, nullptr, draw, st);
            Image img;
            RasterStats rs;
            rasterize(cam, drawToRaster(h, draw, nullptr), img, rs);
            std::printf("  q=%.0f: %8" PRIu64 " splats, %.2f contributions/px\n", q,
                        st.drawn_splats, rs.mean_contrib_per_px);
        }

        // temporal stability: oscillating dolly crosses refine/coarsen
        // thresholds both ways — exactly where naive switching pops and
        // oscillates. Image metric: frame-to-frame mean |dRGB| in excess of
        // pure camera motion (measured on a fixed full-detail render baseline).
        // Temporal stability, unsaturated (huge budget) so threshold logic acts,
        // not the budget loop. Popping metric: frame-to-frame mean |dRGB| in
        // EXCESS of a full-detail render of the same camera path (pure motion
        // baseline). Two scenarios:
        //   advance — steady forward dolly; refines fire continuously; cross-
        //             fade should soften the level-switch pops.
        //   hover   — fixed pose with ±2m positional jitter; nodes near the
        //             threshold flicker refine/coarsen every frame unless
        //             hysteresis holds them.
        const int tn = 60;
        struct Scenario {
            const char* name;
            Camera (*cam)(int, float, float);
        };
        // Both scenarios sit on the climb view, where rings of several LOD
        // levels are visible at once and motion along the view axis crosses
        // refine/coarsen thresholds.
        const Scenario scenarios[2] = {
            {"recede-approach", [](int f, float world, float street_x) {
                 const double u = 0.62 + 0.06 * std::sin((double)f / 30.0 * 6.28318);
                 return cameraAt(u, world, street_x, 640, 360);
             }},
            {"hover+jitter", [](int f, float world, float street_x) {
                 Camera c = cameraAt(0.62, world, street_x, 640, 360);
                 uint32_t s = (uint32_t)f * 2654435761u + 12345u;
                 auto jit = [&]() {
                     s ^= s << 13;
                     s ^= s >> 17;
                     s ^= s << 5;
                     return ((float)(s & 0xFFFF) / 65535.f - 0.5f) * 4.f; // +/-2m
                 };
                 c.position = c.position + Vec3{jit(), jit() * 0.25f, jit()};
                 return c;
             }},
        };
        struct Mode {
            const char* name;
            float hyst;
            float band;
        };
        const Mode modes[3] = {{"naive (no hysteresis, no dissolve)", 1.f, 0.f},
                               {"hysteresis only", 1.5f, 0.f},
                               {"hysteresis + dissolve", 1.5f, 0.18f}};
        for (const Scenario& sc : scenarios) {
            std::printf("\ntemporal stability [%s] (60 frames, 640x360, unsaturated):\n", sc.name);
            std::vector<double> base_diff(tn, 0.0);
            {
                Image prev, curi;
                std::vector<RasterInput> full{{&h.leaves, 0, (uint32_t)h.leaves.size(), 1.f}};
                for (int f = 0; f < tn; ++f) {
                    RasterStats rs;
                    rasterize(sc.cam(f, world, street_x), full, curi, rs);
                    if (f > 0)
                        base_diff[f] = meanAbsDiff(prev, curi);
                    std::swap(prev, curi);
                }
            }
            std::printf("  %-34s %12s %12s %10s %10s %8s\n", "mode", "excess|d|", "peak excess",
                        "refines/f", "coarsen/f", "thr/f");
            for (const Mode& m : modes) {
                CutParams vp = cp;
                vp.hysteresis = m.hyst;
                vp.dissolve_band = m.band;
                vp.splat_budget = 1u << 30;
                CutSelector vsel(h, vp);
                Image prev, cur;
                double sum = 0, peak = 0;
                uint64_t n_ref = 0, n_coar = 0, n_thr = 0;
                int count = 0;
                for (int f = 0; f < tn; ++f) {
                    Camera c = sc.cam(f, world, street_x);
                    std::vector<DrawEntry> draw;
                    CutStats st;
                    vsel.update(c, nullptr, draw, st);
                    RasterStats rs;
                    rasterize(c, drawToRaster(h, draw, nullptr), cur, rs);
                    if (f > 10) { // skip warmup
                        const double excess = meanAbsDiff(prev, cur) - base_diff[f];
                        sum += excess;
                        peak = std::max(peak, excess);
                        n_ref += st.refines;
                        n_coar += st.coarsens + st.forced_coarsens;
                        n_thr += st.coarsens_threshold;
                        ++count;
                    }
                    std::swap(prev, cur);
                }
                std::printf("  %-34s %12.5f %12.5f %10.1f %10.1f %8.1f\n", m.name, sum / count,
                            peak, (double)n_ref / count, (double)n_coar / count,
                            (double)n_thr / count);
            }
        }
    }

    delete stream;
    std::printf("\nCSV written to %s/frames.csv\n", args.out_dir.c_str());
    return 0;
}
