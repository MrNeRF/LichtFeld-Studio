/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "streaming.hpp"

#include <algorithm>
#include <cstring>

namespace lfs::lod {

    namespace {

        // IEEE 754 binary16 conversion (round to nearest even not needed here;
        // truncation error is far below quantization error elsewhere).
        uint16_t f32_to_f16(float f) {
            uint32_t x;
            std::memcpy(&x, &f, 4);
            const uint32_t sign = (x >> 16) & 0x8000;
            int32_t exp = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
            uint32_t mant = (x >> 13) & 0x3FF;
            if (exp <= 0)
                return (uint16_t)sign; // flush denormals/underflow
            if (exp >= 31)
                return (uint16_t)(sign | 0x7BFF); // clamp to max finite
            return (uint16_t)(sign | (exp << 10) | mant);
        }
        float f16_to_f32(uint16_t hv) {
            const uint32_t sign = (uint32_t)(hv & 0x8000) << 16;
            const uint32_t exp = (hv >> 10) & 0x1F;
            const uint32_t mant = hv & 0x3FF;
            uint32_t x;
            if (exp == 0) {
                x = sign; // zero (denormals flushed)
            } else {
                x = sign | ((exp - 15 + 127) << 23) | (mant << 13);
            }
            float f;
            std::memcpy(&f, &x, 4);
            return f;
        }

        constexpr size_t kCoarseBytesPerSplat = 20; // u16 pos, f16 logscale, i8 quat, u8 alpha, u8 rgb
        constexpr size_t kFineBytesPerSplat = 29;   // f32 pos, f16 logscale, i8 quat, u8 alpha, f16 sh0
        constexpr size_t kDecodedBytesPerSplat = sizeof(Vec3) * 3 + sizeof(Quat) + sizeof(float);

        struct ByteWriter {
            std::vector<uint8_t>& b;
            void u8(uint8_t v) { b.push_back(v); }
            void i8(int8_t v) { b.push_back((uint8_t)v); }
            void u16(uint16_t v) {
                b.push_back(v & 0xFF);
                b.push_back(v >> 8);
            }
            void f32(float v) {
                uint32_t x;
                std::memcpy(&x, &v, 4);
                for (int i = 0; i < 4; ++i)
                    b.push_back((x >> (8 * i)) & 0xFF);
            }
        };
        struct ByteReader {
            const uint8_t* p;
            uint8_t u8() { return *p++; }
            int8_t i8() { return (int8_t)*p++; }
            uint16_t u16() {
                uint16_t v = p[0] | (p[1] << 8);
                p += 2;
                return v;
            }
            float f32() {
                uint32_t x = p[0] | (p[1] << 8) | (p[2] << 16) | ((uint32_t)p[3] << 24);
                p += 4;
                float f;
                std::memcpy(&f, &x, 4);
                return f;
            }
        };

    } // namespace

    StreamingManager::StreamingManager(const Hierarchy& h, const StreamParams& p)
        : h_(h), params_(p), lod_opacity_(h.merged.lod_opacity_encoded) {
        const uint32_t n = (uint32_t)h.nodes.size();
        store_.resize(n);
        coarse_format_.resize(n);
        queued_or_flying_.assign(n, 0);

        parallelFor(0, n, std::thread::hardware_concurrency(), [&](size_t lo, size_t hi) {
            for (size_t i = lo; i < hi; ++i) {
                const Node& nd = h.nodes[i];
                const SplatCloud& src = h.cloudFor(nd);
                const bool coarse = !nd.is_leaf;
                coarse_format_[i] = coarse;
                std::vector<uint8_t>& blob = store_[i];
                blob.reserve(nd.rep_count * (coarse ? kCoarseBytesPerSplat : kFineBytesPerSplat));
                ByteWriter w{blob};
                const Vec3 bmin = nd.bounds.mn;
                const Vec3 bext = nd.bounds.extent();
                const Vec3 binv{bext.x > 1e-12f ? 65535.f / bext.x : 0.f,
                                bext.y > 1e-12f ? 65535.f / bext.y : 0.f,
                                bext.z > 1e-12f ? 65535.f / bext.z : 0.f};
                for (uint32_t s = nd.rep_offset; s < nd.rep_offset + nd.rep_count; ++s) {
                    const Vec3& m = src.means[s];
                    if (coarse) {
                        w.u16((uint16_t)std::clamp((m.x - bmin.x) * binv.x, 0.f, 65535.f));
                        w.u16((uint16_t)std::clamp((m.y - bmin.y) * binv.y, 0.f, 65535.f));
                        w.u16((uint16_t)std::clamp((m.z - bmin.z) * binv.z, 0.f, 65535.f));
                    } else {
                        w.f32(m.x);
                        w.f32(m.y);
                        w.f32(m.z);
                    }
                    w.u16(f32_to_f16(src.log_scales[s].x));
                    w.u16(f32_to_f16(src.log_scales[s].y));
                    w.u16(f32_to_f16(src.log_scales[s].z));
                    const Quat q = src.rotations[s].normalized();
                    w.i8((int8_t)std::clamp(q.w * 127.f, -127.f, 127.f));
                    w.i8((int8_t)std::clamp(q.x * 127.f, -127.f, 127.f));
                    w.i8((int8_t)std::clamp(q.y * 127.f, -127.f, 127.f));
                    w.i8((int8_t)std::clamp(q.z * 127.f, -127.f, 127.f));
                    // coarse lodOpacity chunks store alpha in [0,4] (may exceed 1)
                    const float a = src.opacity(s);
                    w.u8((uint8_t)std::clamp((coarse && lod_opacity_ ? a * 0.25f : a) * 255.f, 0.f, 255.f));
                    if (coarse) {
                        const Vec3 c = src.color(s);
                        w.u8((uint8_t)std::clamp(c.x * 255.f, 0.f, 255.f));
                        w.u8((uint8_t)std::clamp(c.y * 255.f, 0.f, 255.f));
                        w.u8((uint8_t)std::clamp(c.z * 255.f, 0.f, 255.f));
                    } else {
                        w.u16(f32_to_f16(src.sh0[s].x));
                        w.u16(f32_to_f16(src.sh0[s].y));
                        w.u16(f32_to_f16(src.sh0[s].z));
                    }
                }
            }
        });
        for (const auto& blob : store_)
            store_total_bytes_ += blob.size();

        // root is pinned resident so frame 0 always has something to draw
        ResidentChunk root;
        decodeChunk(0, root);
        stats_.resident_bytes += root.bytes;
        resident_.emplace(0, std::move(root));
        stats_.resident_chunks = resident_.size();
    }

    void StreamingManager::decodeChunk(uint32_t node, ResidentChunk& out) const {
        const double t0 = nowMs();
        const Node& nd = h_.nodes[node];
        const bool coarse = coarse_format_[node];
        out.data.resize(nd.rep_count);
        out.data.lod_opacity_encoded = coarse && lod_opacity_;
        ByteReader r{store_[node].data()};
        const Vec3 bmin = nd.bounds.mn;
        const Vec3 bext = nd.bounds.extent();
        for (uint32_t i = 0; i < nd.rep_count; ++i) {
            if (coarse) {
                out.data.means[i] = {bmin.x + bext.x * (r.u16() / 65535.f),
                                     bmin.y + bext.y * (r.u16() / 65535.f),
                                     bmin.z + bext.z * (r.u16() / 65535.f)};
            } else {
                out.data.means[i] = {r.f32(), r.f32(), r.f32()};
            }
            out.data.log_scales[i] = {f16_to_f32(r.u16()), f16_to_f32(r.u16()), f16_to_f32(r.u16())};
            out.data.rotations[i] = {r.i8() / 127.f, r.i8() / 127.f, r.i8() / 127.f, r.i8() / 127.f};
            if (coarse && lod_opacity_) {
                out.data.opacity_raw[i] = (r.u8() / 255.f) * 4.f;
            } else {
                out.data.opacity_raw[i] = logit(r.u8() / 255.f);
            }
            if (coarse) {
                const Vec3 c{r.u8() / 255.f, r.u8() / 255.f, r.u8() / 255.f};
                out.data.sh0[i] = {(c.x - 0.5f) / SH_C0, (c.y - 0.5f) / SH_C0, (c.z - 0.5f) / SH_C0};
            } else {
                out.data.sh0[i] = {f16_to_f32(r.u16()), f16_to_f32(r.u16()), f16_to_f32(r.u16())};
            }
        }
        out.bytes = nd.rep_count * kDecodedBytesPerSplat;
        out.last_touch_frame = frame_idx_;
        const_cast<StreamingManager*>(this)->stats_.decode_ms += nowMs() - t0;
    }

    void StreamingManager::touch(uint32_t node) {
        auto it = resident_.find(node);
        if (it == resident_.end())
            return;
        it->second.last_touch_frame = frame_idx_;
        if (it->second.from_prefetch && !it->second.counted_useful) {
            it->second.counted_useful = true;
            stats_.prefetch_useful++;
        }
    }

    void StreamingManager::enqueue(uint32_t node, float priority, bool prefetch) {
        if (node >= queued_or_flying_.size())
            return;
        if (queued_or_flying_[node] || resident_.count(node))
            return;
        queued_or_flying_[node] = 1;
        pending_.push_back({node, priority, prefetch, -1.0});
        (prefetch ? stats_.requests_prefetch : stats_.requests_demand)++;
    }

    void StreamingManager::tick(double sim_now_ms, uint32_t frame_idx) {
        frame_idx_ = frame_idx;
        for (Pending& p : pending_)
            if (p.enqueue_ms < 0)
                p.enqueue_ms = sim_now_ms;

        // complete arrivals
        for (size_t i = 0; i < in_flight_.size();) {
            if (in_flight_[i].done_ms <= sim_now_ms) {
                const InFlight fl = in_flight_[i];
                in_flight_[i] = in_flight_.back();
                in_flight_.pop_back();
                ResidentChunk rc;
                decodeChunk(fl.node, rc);
                rc.from_prefetch = fl.prefetch;
                stats_.resident_bytes += rc.bytes;
                resident_.emplace(fl.node, std::move(rc));
                queued_or_flying_[fl.node] = 0;
                stats_.completed++;
                stats_.bytes_fetched += store_[fl.node].size();
                wait_accum_ms_ += fl.done_ms - fl.issue_ms;
            } else {
                ++i;
            }
        }

        // issue queued fetches, demand before prefetch, high priority first; the
        // shared-bandwidth serialization is the contention model, so everything
        // can be handed to the (simulated) IO queue immediately
        if (!pending_.empty()) {
            std::sort(pending_.begin(), pending_.end(), [](const Pending& a, const Pending& b) {
                if (a.prefetch != b.prefetch)
                    return b.prefetch;
                return a.priority > b.priority;
            });
            const double bytes_per_ms = params_.bandwidth_mb_s * 1048576.0 / 1000.0;
            for (const Pending& p : pending_) {
                const double xfer_ms = store_[p.node].size() / bytes_per_ms;
                const double start = std::max(sim_now_ms, bandwidth_free_at_ms_);
                const double done = start + params_.base_latency_ms + xfer_ms;
                bandwidth_free_at_ms_ = start + xfer_ms;
                in_flight_.push_back({p.node, p.enqueue_ms, done, p.prefetch});
            }
            pending_.clear();
        }

        evictIfNeeded();
        stats_.resident_chunks = resident_.size();
        if (stats_.completed)
            stats_.avg_wait_ms = wait_accum_ms_ / (double)stats_.completed;
    }

    void StreamingManager::prefetch(const Camera& cam, const Vec3& velocity_per_s,
                                    const CutParams& cp, const std::vector<uint32_t>& cut_nodes) {
        Camera pred = cam;
        pred.position = cam.position + velocity_per_s * params_.prefetch_lookahead_s;
        const Frustum fr = Frustum::fromCamera(pred);
        const float qroot = std::sqrt(cp.quality);
        auto err = [&](const Node& n) {
            const float d = std::max(n.bounds.distance(pred.position), pred.znear);
            return n.rep_extent * pred.focalPx() / d * qroot / cp.tau_px;
        };
        // Request the children the predicted camera is about to refine into
        // (error already in the [band, 1] approach window or beyond), bounded per
        // frame so prefetch can't starve demand fetches of bandwidth.
        int requested = 0;
        const int cap = params_.prefetch_max_chunks_per_frame;
        constexpr float kApproachBand = 0.8f;
        for (uint32_t u : cut_nodes) {
            if (requested >= cap)
                break;
            const Node& n = h_.nodes[u];
            if (!n.is_leaf && fr.intersects(n.bounds)) {
                const float e = err(n);
                if (e > kApproachBand) {
                    for (int c = 0; c < n.child_count && requested < cap; ++c)
                        if (!resident_.count(n.children[c]) && !queued_or_flying_[n.children[c]]) {
                            enqueue(n.children[c], e, true);
                            ++requested;
                        }
                }
            }
            // parent needed if the predicted cut would coarsen
            if (n.parent != kInvalidNode && requested < cap) {
                const Node& pn = h_.nodes[n.parent];
                if ((!fr.intersects(pn.bounds) || err(pn) < 1.f / std::max(cp.hysteresis, 1.f)) &&
                    !resident_.count(n.parent) && !queued_or_flying_[n.parent]) {
                    enqueue(n.parent, 1.f, true);
                    ++requested;
                }
            }
        }
    }

    void StreamingManager::evictIfNeeded() {
        if (stats_.resident_bytes <= params_.resident_budget_bytes)
            return;
        // gather eviction candidates: not root, not touched recently
        std::vector<std::pair<uint32_t, uint32_t>> cands; // (last_touch, node)
        for (const auto& [node, rc] : resident_) {
            if (node == 0)
                continue;
            if (rc.last_touch_frame + params_.protect_frames >= frame_idx_)
                continue;
            cands.push_back({rc.last_touch_frame, node});
        }
        std::sort(cands.begin(), cands.end()); // oldest first
        for (const auto& [touch, node] : cands) {
            if (stats_.resident_bytes <= params_.resident_budget_bytes)
                break;
            auto it = resident_.find(node);
            stats_.resident_bytes -= it->second.bytes;
            stats_.evicted_bytes += it->second.bytes;
            stats_.evictions++;
            resident_.erase(it);
        }
    }

} // namespace lfs::lod
