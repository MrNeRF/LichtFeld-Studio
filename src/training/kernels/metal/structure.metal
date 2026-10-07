/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
constant uint kStructureOperation [[function_constant(0)]];
struct StructureParams {
    ulong a, b, c, d, e, f, g;
    int width, height, first_row, rows;
    uint stride, blocks, image_byte, mask_byte, radius, first_scale, valid_padding, reserved;
    float gain, sigma_sq;
    float filter_g[21], filter_d1[21], filter_d2[21];
};

inline float st_read(ulong a, int i) { return ((device const float*)a)[i]; }
inline uint st_byte(ulong a, int i) { return uint(((device const uchar*)a)[i]); }
inline void st_write(ulong a, int i, float v) { ((device float*)a)[i] = v; }
inline float st_value(ulong a, int i, uint byte_value) { return byte_value != 0 ? float(st_byte(a, i)) * (1.0f / 255.0f) : st_read(a, i); }
float st_luma(constant StructureParams& p, int i) {
    int n = p.width * p.height;
    float r = p.image_byte != 0 ? float(st_byte(p.a, i)) : st_read(p.a, i);
    float g = p.image_byte != 0 ? float(st_byte(p.a, n + i)) : st_read(p.a, n + i);
    float b = p.image_byte != 0 ? float(st_byte(p.a, 2 * n + i)) : st_read(p.a, 2 * n + i);
    float value = fma(0.0722f, b, fma(0.2126f, r, 0.7152f * g));
    return p.image_byte != 0 ? value * (1.0f / 255.0f) : value;
}
int st_reflect(int i, int n) {
    if (i < 0)
        i = -i - 1;
    i %= 2 * n;
    return i < n ? i : 2 * n - 1 - i;
}
bool st_interior(constant StructureParams& p, int x, int y) { return x > 0 && x < p.width - 1 && y > 0 && y < p.height - 1; }
float st_sx(int x, int y) { return float(x) * (y == 0 ? 2.0f : 1.0f) * 0.125f; }
float st_sy(int x, int y) { return float(y) * (x == 0 ? 2.0f : 1.0f) * 0.125f; }
float st_mask(constant StructureParams& p, int i) {
    if (p.c == 0)
        return 1.0f;
    return p.mask_byte != 0 ? (st_byte(p.c, i) != 0 ? 1.0f : 0.0f) : st_read(p.c, i);
}
float st_residual(constant StructureParams& p, int i) {
    int n = p.width * p.height;
    return 0.2126f * (st_read(p.a, i) - st_value(p.b, i, p.image_byte)) +
           0.7152f * (st_read(p.a, n + i) - st_value(p.b, n + i, p.image_byte)) +
           0.0722f * (st_read(p.a, 2 * n + i) - st_value(p.b, 2 * n + i, p.image_byte));
}

kernel void structure_main(constant StructureParams& p [[buffer(0)]],
                           uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    const uint operation = kStructureOperation;
    threadgroup float3 reduction[256];
    threadgroup float residual[432], confidence[432], coefficient_x[340], coefficient_y[340];
    uint i = group * 256u + lane;
    int n = p.width * p.height;
    if (operation == 0) {
        if (i >= uint((p.rows + 2 * int(p.radius)) * p.width))
            return;
        int x = int(i) % p.width, row = st_reflect(p.first_row + int(i) / p.width - int(p.radius), p.height);
        float g = 0, d1 = 0, d2 = 0;
        for (int k = -int(p.radius); k <= int(p.radius); ++k) {
            float v = st_luma(p, row * p.width + st_reflect(x + k, p.width));
            int j = k + int(p.radius);
            g = fma(v, p.filter_g[j], g);
            d1 = fma(v, p.filter_d1[j], d1);
            d2 = fma(v, p.filter_d2[j], d2);
        }
        st_write(p.b, i, g);
        st_write(p.b, p.stride + i, d1);
        st_write(p.b, 2 * p.stride + i, d2);
        return;
    }
    if (operation == 1) {
        if (i >= uint(p.rows * p.width))
            return;
        int x = int(i) % p.width, by = int(i) / p.width, y = p.first_row + by;
        if (y >= p.height)
            return;
        float hxx = 0, hxy = 0, hyy = 0;
        for (int k = -int(p.radius); k <= int(p.radius); ++k) {
            int j = k + int(p.radius), source = (by + j) * p.width + x;
            hxx = fma(st_read(p.b, 2 * p.stride + source), p.filter_g[j], hxx);
            hxy = fma(st_read(p.b, p.stride + source), p.filter_d1[j], hxy);
            hyy = fma(st_read(p.b, source), p.filter_d2[j], hyy);
        }
        hxy *= p.sigma_sq;
        hyy *= p.sigma_sq;
        float difference = fma(hxx, p.sigma_sq, -hyy);
        float trace = fma(hxx, p.sigma_sq, hyy);
        float disc = sqrt(max(fma(difference, difference, (4.0f * hxy) * hxy), 0.0f));
        float a = abs(0.5f * (trace + disc)), b = abs(0.5f * (trace - disc));
        float hi = max(a, b), lo = min(a, b), response = hi * (hi - lo) / (hi + lo + 1e-6f);
        int index = y * p.width + x;
        st_write(p.c, index, p.first_scale != 0 ? response : max(st_read(p.c, index), response));
        return;
    }
    if (operation == 2 || operation == 3 || operation == 6 || operation == 7) {
        float3 value = float3(0.0f, 0.0f, 0.0f);
        if (operation == 2 || operation == 3)
            value = float3(0.0f, INFINITY, -INFINITY);
        if (operation == 2) {
            for (uint k = i; k < uint(n); k += p.blocks * 256u) {
                float l = st_luma(p, int(k));
                value.x += st_read(p.c, k);
                value.y = min(value.y, l);
                value.z = max(value.z, l);
            }
        } else if (operation == 3) {
            for (uint k = lane; k < p.blocks; k += 256u) {
                value.x += st_read(p.d, k);
                value.y = min(value.y, st_read(p.d, 1024 + k));
                value.z = max(value.z, st_read(p.d, 2048 + k));
            }
        } else if (operation == 6) {
            for (uint k = i; k < uint(n); k += p.blocks * 256u) {
                int x = int(k) % p.width, y = int(k) / p.width;
                bool valid = st_interior(p, x, y);
                float gx = 0, gy = 0;
                if (valid) {
                    for (int dy = -1; dy <= 1; ++dy)
                        for (int dx = -1; dx <= 1; ++dx) {
                            int j = int(k) + dy * p.width + dx;
                            valid = valid && st_mask(p, j) > 0.0f;
                            float r = st_residual(p, j);
                            gx += st_sx(dx, dy) * r;
                            gy += st_sy(dx, dy) * r;
                        }
                }
                float m = valid ? st_mask(p, int(k)) : 0.0f;
                value.x += m * (sqrt(gx * gx + (0.001f * 0.001f)) + sqrt(gy * gy + (0.001f * 0.001f)) - 2.0f * 0.001f);
                value.y += m;
            }
        } else {
            for (uint k = lane; k < p.blocks; k += 256u) {
                value.x += st_read(p.d, k);
                value.y += st_read(p.d, 1024 + k);
            }
        }
        reduction[lane] = value;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Warp XOR followed by the eight warp totals, matching the CUDA reduction order.
        for (uint offset = 16; offset > 0; offset >>= 1) {
            float3 other = reduction[lane ^ offset];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            value.x += other.x;
            if (operation == 2 || operation == 3) {
                value.y = min(value.y, other.y);
                value.z = max(value.z, other.z);
            } else
                value.y += other.y;
            reduction[lane] = value;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        value = lane < 8 ? reduction[lane * 32] : float3(0.0f, 0.0f, 0.0f);
        if ((operation == 2 || operation == 3) && lane >= 8)
            value = float3(0.0f, INFINITY, -INFINITY);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        reduction[lane] = value;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint offset = 4; offset > 0; offset >>= 1) {
            float3 other = reduction[lane ^ offset];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            value.x += other.x;
            if (operation == 2 || operation == 3) {
                value.y = min(value.y, other.y);
                value.z = max(value.z, other.z);
            } else
                value.y += other.y;
            reduction[lane] = value;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        value = reduction[0];
        if (operation == 2) {
            if (lane == 0) {
                st_write(p.d, group, value.x);
                st_write(p.d, 1024 + group, value.y);
                st_write(p.d, 2048 + group, value.z);
            }
        } else if (operation == 3) {
            if (i < uint(n))
                st_write(p.c, i, st_read(p.c, i) * (value.x > 0.0f && value.z > value.y ? float(n) / value.x : 0.0f));
        } else if (operation == 6) {
            if (lane == 0) {
                st_write(p.d, group, value.x);
                st_write(p.d, 1024 + group, value.y);
            }
        } else if (lane == 0) {
            float scale = value.y > 0.0f ? p.gain / (2.0f * value.y) : 0.0f;
            st_write(p.e, 0, scale);
            st_write(p.e, 1, value.y);
            st_write(p.f, 0, value.x * scale);
        }
        return;
    }
    if (operation == 4) {
        if (i >= uint(n))
            return;
        int x = int(i) % p.width, y = int(i) / p.width;
        bool inside = p.valid_padding == 0 || p.width <= 10 || p.height <= 10 || (x >= 5 && x < p.width - 5 && y >= 5 && y < p.height - 5);
        float base = 1.0f;
        if (p.b != 0)
            base = p.mask_byte != 0 ? (st_byte(p.b, i) != 0 ? 1.0f : 0.0f) : st_read(p.b, i);
        st_write(p.c, i, inside ? base * (1.0f + p.gain * min(st_read(p.a, i), 4.0f)) : 0.0f);
        return;
    }
    if (operation == 5) {
        if (i < uint(n))
            st_write(p.a, i, st_read(p.a, i) * (1.0f + p.gain * clamp(st_read(p.b, i), 0.0f, 4.0f)));
        return;
    }
    if (operation == 8) {
        int tiles_x = (p.width + 31) / 32;
        int x0 = int(group) % tiles_x * 32, y0 = int(group) / tiles_x * 8;
        for (uint k = lane; k < 12u * 36u; k += 256u) {
            int y = y0 + int(k) / 36 - 2, x = x0 + int(k) % 36 - 2;
            bool inside = x >= 0 && x < p.width && y >= 0 && y < p.height;
            residual[k] = inside ? st_residual(p, y * p.width + x) : 0.0f;
            confidence[k] = inside ? st_mask(p, y * p.width + x) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = lane; k < 10u * 34u; k += 256u) {
            int ly = int(k) / 34, lx = int(k) % 34;
            bool valid = st_interior(p, x0 + lx - 1, y0 + ly - 1);
            float gx = 0, gy = 0;
            if (valid)
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx) {
                        int j = (ly + 1 + dy) * 36 + lx + 1 + dx;
                        valid = valid && confidence[j] > 0.0f;
                        float r = residual[j];
                        gx += st_sx(dx, dy) * r;
                        gy += st_sy(dx, dy) * r;
                    }
            float m = valid ? confidence[(ly + 1) * 36 + lx + 1] : 0.0f;
            coefficient_x[k] = m * gx / sqrt(gx * gx + (0.001f * 0.001f));
            coefficient_y[k] = m * gy / sqrt(gy * gy + (0.001f * 0.001f));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        int tx = int(lane) % 32, ty = int(lane) / 32, x = x0 + tx, y = y0 + ty;
        if (x >= p.width || y >= p.height)
            return;
        int index = y * p.width + x;
        float value = 0;
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx)
                if (st_interior(p, x - dx, y - dy)) {
                    int j = (ty - dy + 1) * 34 + tx - dx + 1;
                    value += st_sx(dx, dy) * coefficient_x[j] + st_sy(dx, dy) * coefficient_y[j];
                }
        value *= st_read(p.e, 0);
        st_write(p.g, index, st_read(p.g, index) + 0.2126f * value);
        st_write(p.g, n + index, st_read(p.g, n + index) + 0.7152f * value);
        st_write(p.g, 2 * n + index, st_read(p.g, 2 * n + index) + 0.0722f * value);
    }
}
