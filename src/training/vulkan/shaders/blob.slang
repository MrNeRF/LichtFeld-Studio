/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
[SpecializationConstant] const uint operation = 0;
struct Params {
    uint64_t a, b, c, d, e, f;
    int width, height, source_width, source_height;
    uint count, capacity, factor, byte_input, radius, horizontal, x_max, y_max;
    int view, neighbors;
    float weights[7];
};
struct SweepView {
    float R[9], t[3];
    float fx, fy, cx, cy;
    int width, height;
    uint64_t bitmap_offset;
    float density[2], z_min, z_max;
};

struct Push {
    uint64_t parameters;
};
[[vk::push_constant]] Push push;
float bs_read(uint64_t a, int i) { return ((Ptr<float, Access.Read>)a)[i]; }
uint bs_byte(uint64_t a, int i) { return uint(((Ptr<uint8_t, Access.Read>)a)[i]); }
uint bs_uint(uint64_t a, uint64_t i) { return ((Ptr<uint, Access.Read>)a)[i]; }
int bs_int(uint64_t a, int i) { return ((Ptr<int, Access.Read>)a)[i]; }
void bs_write(uint64_t a, int i, float v) { ((Ptr<float, Access.ReadWrite>)a)[i] = v; }
void bs_write_byte(uint64_t a, int i, uint v) { ((Ptr<uint8_t, Access.ReadWrite>)a)[i] = uint8_t(v); }
void bs_write_uint(uint64_t a, int i, uint v) { ((Ptr<uint, Access.ReadWrite>)a)[i] = v; }
uint bs_add(uint64_t a, int i, uint v) {
    uint old;
    InterlockedAdd(((Ptr<uint, Access.ReadWrite>)a)[i], v, old);
    return old;
}
SweepView bs_view(uint64_t a, int i) { return ((Ptr<SweepView, Access.Read>)a)[i]; }
groupshared float score[1024], visible[1024], smoothed[1024];
int bs_reflect(int i, int n) {
    i %= 2 * n;
    if (i < 0)
        i += 2 * n;
    return i < n ? i : 2 * n - 1 - i;
}

[numthreads(256, 1, 1)] void main(uint3 gid : SV_GroupID, uint lane : SV_GroupIndex) {
    uint group = gid.x;
    Params p = ((Ptr<Params, Access.Read>)push.parameters)[0];
    uint i = group * 256u + lane;
    int n = p.width * p.height;
    if (operation == 0) {
        if (i >= uint(n))
            return;
        int x = int(i) % p.width, y = int(i) / p.width;
        for (int c = 0; c < 3; ++c) {
            float sum = 0;
            for (int dy = 0; dy < int(p.factor); ++dy)
                for (int dx = 0; dx < int(p.factor); ++dx) {
                    int j = (c * p.source_height + y * int(p.factor) + dy) * p.source_width + x * int(p.factor) + dx;
                    sum += p.byte_input != 0 ? float(bs_byte(p.a, j)) : bs_read(p.a, j);
                }
            bs_write(p.b, c * n + int(i), sum / float(p.factor * p.factor) / (p.byte_input != 0 ? 255.0f : 1.0f));
        }
        return;
    }
    if (operation == 1 || operation == 2) {
        if (i >= uint(n))
            return;
        int x = int(i) % p.width, y = int(i) / p.width;
        float acc = 0;
        for (int k = -3; k <= 3; ++k) {
            int j = operation == 1 ? y * p.width + bs_reflect(x + k, p.width) : bs_reflect(y + k, p.height) * p.width + x;
            float v = operation == 1 ? 255.0f * (0.299f * bs_read(p.a, j) + 0.587f * bs_read(p.a, n + j) + 0.114f * bs_read(p.a, 2 * n + j)) : bs_read(p.a, j);
            acc += p.weights[k + 3] * v;
        }
        if (operation == 1)
            bs_write(p.b, i, acc);
        else {
            bs_write(p.b, 2 * i, acc);
            bs_write(p.b, 2 * i + 1, acc);
        }
        return;
    }
    if (operation == 3) {
        if (i >= uint(n))
            return;
        int x = int(i) % p.width, y = int(i) / p.width;
        float a = bs_read(p.a, 2 * i), b = bs_read(p.a, 2 * i + 1);
        for (int k = -int(p.radius); k <= int(p.radius); ++k) {
            int sx = p.horizontal != 0 ? clamp(x + k, 0, p.width - 1) : x;
            int sy = p.horizontal != 0 ? y : clamp(y + k, 0, p.height - 1);
            int j = 2 * (sy * p.width + sx);
            float ax = bs_read(p.a, j), by = bs_read(p.a, j + 1);
            a = p.x_max != 0 ? max(a, ax) : min(a, ax);
            b = p.y_max != 0 ? max(b, by) : min(b, by);
        }
        bs_write(p.b, 2 * i, a);
        bs_write(p.b, 2 * i + 1, b);
        return;
    }
    if (operation == 4) {
        if (i >= uint(n))
            return;
        float s = bs_read(p.a, 2 * i);
        bs_write(p.c, 2 * i, s - bs_read(p.b, 2 * i));
        bs_write(p.c, 2 * i + 1, bs_read(p.b, 2 * i + 1) - s);
        return;
    }
    if (operation == 5) {
        if (i >= uint(n))
            return;
        float a = bs_read(p.a, 2 * i), b = bs_read(p.a, 2 * i + 1);
        bool bright = a == bs_read(p.b, 2 * i) && a >= 28.0f, dark = b == bs_read(p.b, 2 * i + 1) && b >= 22.0f;
        bs_write_byte(p.d, i, uint(bright) | (uint(dark) << 1));
        for (int polarity = 0; polarity < 2; ++polarity) {
            if (!(polarity == 0 ? bright : dark))
                continue;
            uint slot = bs_add(p.f, 0, 1);
            if (slot >= p.capacity)
                continue;
            int j = int(slot) * 7;
            bs_write(p.e, j, float(i % uint(p.width)));
            bs_write(p.e, j + 1, float(i / uint(p.width)));
            bs_write(p.e, j + 2, float(p.view));
            bs_write(p.e, j + 3, float(polarity));
            bs_write(p.e, j + 4, bs_read(p.c, i));
            bs_write(p.e, j + 5, bs_read(p.c, n + i));
            bs_write(p.e, j + 6, bs_read(p.c, 2 * n + i));
        }
        return;
    }
    if (operation == 6) {
        if (i >= uint((n + 15) / 16))
            return;
        uint bits = 0;
        for (int j = 0; j < 16; ++j) {
            int index = int(i) * 16 + j;
            if (index >= n)
                break;
            int x = index % p.width, y = index / p.width;
            uint dilated = 0;
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    int sx = x + dx, sy = y + dy;
                    if (sx >= 0 && sx < p.width && sy >= 0 && sy < p.height)
                        dilated |= bs_byte(p.a, sy * p.width + sx);
                }
            bits |= dilated << (2 * j);
        }
        bs_write_uint(p.b, i, bits);
        bs_add(p.c, 1, countbits(bits & 0x55555555u));
        bs_add(p.c, 2, countbits(bits & 0xaaaaaaaau));
        return;
    }
    if (operation == 7) {
        int peak = int(group) * 7, view_index = int(bs_read(p.a, peak + 2)), polarity = int(bs_read(p.a, peak + 3));
        SweepView v = bs_view(p.b, view_index);
        bool has_range = v.z_min > 0.0f && v.z_max > v.z_min;
        int samples = has_range ? min(1024, int(ceil(log(v.z_max / v.z_min) / log(1.0085f))) + 1) : 0;
        if (samples < 15) {
            if (lane == 0) {
                bs_write(p.e, group * 2, 0);
                bs_write(p.e, group * 2 + 1, 0);
            }
            return;
        }
        float log_step = log(v.z_max / v.z_min) / float(samples - 1);
        float dx = (bs_read(p.a, peak) + 0.5f - v.cx) / v.fx, dy = (bs_read(p.a, peak + 1) + 0.5f - v.cy) / v.fy;
        float cx = -(v.R[0] * v.t[0] + v.R[3] * v.t[1] + v.R[6] * v.t[2]);
        float cy = -(v.R[1] * v.t[0] + v.R[4] * v.t[1] + v.R[7] * v.t[2]);
        float cz = -(v.R[2] * v.t[0] + v.R[5] * v.t[1] + v.R[8] * v.t[2]);
        float wx = v.R[0] * dx + v.R[3] * dy + v.R[6], wy = v.R[1] * dx + v.R[4] * dy + v.R[7], wz = v.R[2] * dx + v.R[5] * dy + v.R[8];
        for (int j = int(lane); j < samples; j += 256) {
            float z = v.z_min * exp(log_step * float(j)), px = cx + z * wx, py = cy + z * wy, pz = cz + z * wz;
            float s = 0, seen = 0;
            for (int k = 0; k < p.neighbors; ++k) {
                int neighbor = bs_int(p.c, view_index * p.neighbors + k);
                if (neighbor < 0)
                    break;
                SweepView o = bs_view(p.b, neighbor);
                float x = o.R[0] * px + o.R[1] * py + o.R[2] * pz + o.t[0];
                float y = o.R[3] * px + o.R[4] * py + o.R[5] * pz + o.t[1];
                float depth = o.R[6] * px + o.R[7] * py + o.R[8] * pz + o.t[2];
                if (depth <= 0.1f)
                    continue;
                int u = int(round(o.fx * x / depth + o.cx - 0.5f)), w = int(round(o.fy * y / depth + o.cy - 0.5f));
                if (u < 0 || u >= o.width || w < 0 || w >= o.height)
                    continue;
                uint pixel = uint(w * o.width + u);
                uint hit = (bs_uint(p.d, o.bitmap_offset + pixel / 16) >> (2 * (pixel % 16) + uint(polarity))) & 1u;
                seen += 1.0f;
                s += float(hit) - o.density[polarity];
            }
            score[j] = s;
            visible[j] = seen;
        }
        GroupMemoryBarrierWithGroupSync();
        for (int j = int(lane); j < samples; j += 256) {
            float a = score[j > 0 ? j - 1 : 0], b = score[j], c = score[j + 1 < samples ? j + 1 : samples - 1];
            smoothed[j] = visible[j] < 8.0f ? -1e9f : (a + b + c) / 3.0f;
        }
        GroupMemoryBarrierWithGroupSync();
        if (lane != 0)
            return;
        float cand_score[6], cand_rate[6], cand_depth[6];
        int found = 0;
        for (int j = 0; j < samples; ++j) {
            float sj = smoothed[j];
            if (!(sj > 3.0f))
                continue;
            bool local_max = true;
            for (int o = -7; o <= 7 && local_max; ++o)
                local_max = smoothed[bs_reflect(j + o, samples)] <= sj;
            if (!local_max)
                continue;
            int pos = found < 6 ? found++ : 6;
            while (pos > 0 && cand_score[pos - 1] < sj) {
                if (pos < 6) {
                    cand_score[pos] = cand_score[pos - 1];
                    cand_rate[pos] = cand_rate[pos - 1];
                    cand_depth[pos] = cand_depth[pos - 1];
                }
                --pos;
            }
            if (pos < 6) {
                cand_score[pos] = sj;
                cand_rate[pos] = sj / max(visible[j], 1.0f);
                cand_depth[pos] = v.z_min * exp(log_step * float(j));
            }
        }
        int best = -1;
        for (int j = 0; j < found && best < 0; ++j)
            if (cand_rate[j] >= 0.25f && cand_score[j] >= 5.0f)
                best = j;
        bs_write(p.e, group * 2, best >= 0 ? cand_depth[best] : 0.0f);
        bs_write(p.e, group * 2 + 1, best >= 0 ? 1.0f : 0.0f);
    }
}
