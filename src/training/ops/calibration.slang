// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
// Calibration correction: a smooth per-photo displacement field, its resampling and its fit.
// Byte images, masks and weights use the *_u8 pointers; an absent input is a null pointer.
struct Tile {
    int width, height, x, y, w, h, side;
    float sx, sy;
    int wrap;
};
struct Parameters {
    Ptr<float2, Access.ReadWrite> field;
    Ptr<float, Access.Read> photo;
    Ptr<uint8_t, Access.Read> photo_u8;
    Ptr<float, Access.Read> render;
    Ptr<float, Access.ReadWrite> output;
    Ptr<float, Access.Read> mask;
    Ptr<uint8_t, Access.Read> mask_u8;
    Ptr<float, Access.ReadWrite> out_mask;
    Ptr<uint8_t, Access.ReadWrite> out_mask_u8;
    Ptr<float, Access.Read> structure;
    Ptr<float, Access.ReadWrite> out_structure;
    Ptr<float, Access.Read> edge;
    Ptr<float, Access.ReadWrite> out_edge;
    Ptr<float, Access.Read> alpha, background, stable_background;
    Ptr<float, Access.ReadWrite> stable, grad, partials;
    int width, height, x, y, w, h, side; // the Tile, flattened for the tensor binding ABI
    float sx, sy;
    int wrap;
    int slot, background_image, chunks, margin, first, channels;
    float values, step;
};
[[vk::push_constant]] ConstantBuffer<Parameters> p;
Tile tile() { return {p.width, p.height, p.x, p.y, p.w, p.h, p.side, p.sx, p.sy, p.wrap}; }
bool present(Ptr<float, Access.Read> x) { return (uint64_t)x != 0; }
bool present(Ptr<float, Access.ReadWrite> x) { return (uint64_t)x != 0; }
bool present(Ptr<uint8_t, Access.Read> x) { return (uint64_t)x != 0; }
static const int threads = 128, chunk = 512, partial_size = 21;
static const float epsilon = .03f;
int fieldBase() { return p.slot * p.side * p.side; }
struct Location {
    int cell;
    float u, v, x, y;
    float weight(int k) {
        return (k & 1 ? u : 1 - u) * (k & 2 ? v : 1 - v);
    }
};
Location locate(const Tile t, int x, int y) {
    const int side = t.side, cells = side - 1;
    // An exact cell boundary must agree with the integer partition used by the reduction.
    const float gx = ((x + .5f) * cells) / float(t.width);
    const float gy = ((y + .5f) * cells) / float(t.height);
    const int i = min(int(gx), cells - 1), j = min(int(gy), cells - 1);
    Location l = {i + j * side, gx - i, gy - j, float(x), float(y)};
    float dx = 0, dy = 0;
    for (int k = 0; k < 4; ++k) {
        float2 d = p.field[fieldBase() + l.cell + (k & 1) + (k >> 1) * side];
        dx = fma(l.weight(k), d.x, dx);
        dy = fma(l.weight(k), d.y, dy);
    }
    l.x = fma(dx, t.sx, l.x);
    l.y = fma(dy, t.sy, l.y);
    return l;
}
// A panorama continues across its left and right edge; rows, and the columns of other images, end there.
int column(Tile t, int x) {
    if (!t.wrap)
        return clamp(x, 0, t.width - 1);
    // The cubic halo can reach two texels past a one-column image.
    if (t.width == 1)
        return 0;
    return x < 0 ? x + t.width : (x >= t.width ? x - t.width : x);
}
int line(Tile t, int y) {
    return min(max(y, 0), t.height - 1);
}
// The pixel left of and above a sample inside the photograph and the fractions towards the next ones.
struct Source {
    int x, y;
    float u, v;
};
bool source(Location l, Tile t, out Source s) {
    s = (Source)0;
    if (!(l.y >= 0 && l.y <= t.height - 1))
        return false;
    float x = l.x;
    if (t.wrap) {
        if (!isfinite(x))
            return false;
        x -= floor(x / t.width) * t.width;
        if (x >= t.width)
            x -= t.width;
    } else if (!(x >= 0 && x <= t.width - 1)) {
        return false;
    }
    s = {int(x), int(l.y), 0, 0};
    s.u = x - s.x;
    s.v = l.y - s.y;
    return true;
}
float cubic(float a, float b, float c, float d, float f) {
    return b + .5f * f * ((c - a) + f * ((2 * a - 5 * b + 4 * c - d) + f * (3 * (b - c) + d - a)));
}
float linear(Ptr<float, Access.Read> src, Tile t, Source s) {
    const int x1 = column(t, s.x + 1), y1 = line(t, s.y + 1);
    const float top = (1 - s.u) * src[s.y * t.width + s.x] + s.u * src[s.y * t.width + x1];
    const float bottom = (1 - s.u) * src[y1 * t.width + s.x] + s.u * src[y1 * t.width + x1];
    return (1 - s.v) * top + s.v * bottom;
}
float tap(int c, int i, float background) {
    const bool bytes = present(p.photo_u8);
    int j = c * p.width * p.height + i;
    float value = bytes ? float(uint(p.photo_u8[j])) : p.photo[j];
    if (!present(p.alpha))
        return value;
    float alpha = saturate(p.alpha[i]);
    return value * (bytes ? 1.f / 255.f : 1.f) * alpha + background * (1 - alpha);
}
float sample_cubic(Source s, Tile t, int channel, float background) {
    if (s.u == 0 && s.v == 0)
        return tap(channel, s.y * t.width + s.x, background);
    float taps[4][4], lines[4];
    for (int j = 0; j < 4; ++j) {
        const int row = line(t, s.y + j - 1) * t.width;
        for (int k = 0; k < 4; ++k)
            taps[j][k] = tap(channel, row + column(t, s.x + k - 1), background);
        lines[j] = cubic(taps[j][0], taps[j][1], taps[j][2], taps[j][3], s.u);
    }
    const float value = cubic(lines[0], lines[1], lines[2], lines[3], s.v);
    const float low = min(min(taps[1][1], taps[1][2]), min(taps[2][1], taps[2][2]));
    const float high = max(max(taps[1][1], taps[1][2]), max(taps[2][1], taps[2][2]));
    return min(max(value, low), high);
}

[shader("compute")]
[numthreads(128, 1, 1)]
void resample(uint3 id : SV_DispatchThreadID) {
    const Tile t = tile();
    const int a = int(id.x), n = t.w * t.h;
    if (a >= n)
        return;
    Location loc = locate(t, a % t.w + t.x, a / t.w + t.y);
    Source s;
    bool valid = source(loc, t, s);
    int pixels = t.width * t.height, dest = (a / t.w + t.y) * t.width + a % t.w + t.x;
    const bool alpha = present(p.alpha);
    for (int c = 0; c < 3; c++) {
        float v = p.render[c * n + a];
        if (valid) {
            float bg = alpha ? p.background[p.background_image ? c * pixels + dest : c] : 0;
            v = sample_cubic(s, t, c, bg);
            if (!alpha && present(p.photo_u8))
                v *= 1.f / 255.f;
        }
        p.output[c * n + a] = v;
        if (present(p.stable))
            p.stable[c * pixels + dest] = valid ? sample_cubic(s, t, c, p.stable_background[c]) : v;
    }
    const bool mask_bytes = present(p.mask_u8);
    if (mask_bytes || present(p.mask)) {
        int q = valid ? int(loc.y + .5f) * t.width + column(t, int(s.x + s.u + .5f)) : 0;
        if (mask_bytes)
            p.out_mask_u8[a] = uint8_t(valid ? uint(p.mask_u8[q]) : 0);
        else
            p.out_mask[a] = valid ? p.mask[q] : 0;
    }
    if (present(p.structure))
        p.out_structure[a] = valid ? linear(p.structure, t, s) : 0;
    if (present(p.edge))
        p.out_edge[a] = valid ? linear(p.edge, t, s) : 0;
}

[shader("compute")]
[numthreads(128, 1, 1)]
void follow(uint3 id : SV_DispatchThreadID) {
    const Tile t = tile();
    const int a = int(id.x), n = t.w * t.h, channels = p.channels;
    let src = p.photo;
    if (a >= n)
        return;
    Source s;
    const bool inside = source(locate(t, a % t.w, a / t.w), t, s);
    if (inside && s.u == 0 && s.v == 0) {
        for (int c = 0; c < channels; ++c)
            p.output[c * n + a] = src[c * n + s.y * t.w + s.x];
        return;
    }
    float sum[3] = {0, 0, 0}, total = 0;
    for (int k = 0; inside && k < 4; ++k) {
        const float w = (k & 1 ? s.u : 1 - s.u) * (k & 2 ? s.v : 1 - s.v);
        const int i = line(t, s.y + (k >> 1)) * t.w + column(t, s.x + (k & 1));
        float tap[3] = {0, 0, 0}, length = 0;
        bool finite = true;
        for (int c = 0; c < channels; ++c) {
            tap[c] = src[c * n + i];
            finite = finite && isfinite(tap[c]);
            length += tap[c] * tap[c];
        }
        const bool valid = finite && (channels == 1 ? tap[0] > 0 : sqrt(length) >= 0.5f);
        if (!(w > 0 && valid))
            continue;
        for (int c = 0; c < channels; ++c)
            sum[c] += w * tap[c];
        total += w;
    }
    float length = 0;
    for (int c = 0; c < channels; ++c) {
        sum[c] = total > 0 ? sum[c] / total : 0;
        length += sum[c] * sum[c];
    }
    const float norm = channels == 3 && length > 0 ? rsqrt(length) : 1.f;
    for (int c = 0; c < channels; ++c)
        p.output[c * n + a] = sum[c] * norm;
}

groupshared float laneSums[128][20];
groupshared uint nonzero[128];
[shader("compute")]
[numthreads(128, 1, 1)]
void accumulate(uint3 groupId : SV_GroupID, uint laneId : SV_GroupIndex) {
    const Tile t = tile();
    const int group = int(groupId.x), lane = int(laneId);
    const bool byte_weight = present(p.mask_u8), float_weight = present(p.mask);
    const bool has_grad = present(p.grad);
    int chunks = p.chunks, margin = p.margin;
    float values = p.values;
    let image = p.photo;
    let rendered = p.render;
    let grad = p.grad;
    const int cells = t.side - 1;
    const int cell = group / chunks, col = cell % cells, r = cell / cells;
    const int x0 = (2 * col * t.width + cells - 1) / (2 * cells), x1 = (2 * (col + 1) * t.width + cells - 1) / (2 * cells);
    const int y0 = (2 * r * t.height + cells - 1) / (2 * cells), y1 = (2 * (r + 1) * t.height + cells - 1) / (2 * cells);
    // The loss's valid padding leaves the pixels this close to the tile's edge without a term of their own.
    const int mx = t.w > 2 * margin ? margin : 0, my = t.h > 2 * margin ? margin : 0;
    // A tile spanning a panorama's full width continues across the seam for the central differences.
    const bool seam = t.wrap && t.w == t.width;
    float sums[4][5];
    for (int k = 0; k < 4; k++)
        for (int f = 0; f < 5; f++)
            sums[k][f] = 0;
    bool differs = false;
    for (int j = lane; j < chunk; j += threads) {
        const int idx = chunk * (group % chunks) + j;
        if (idx >= (x1 - x0) * (y1 - y0))
            break;
        const int x = x0 + idx % (x1 - x0), y = y0 + idx / (x1 - x0);
        const int xx = x - t.x, yy = y - t.y;
        if (xx < 0 || yy < 0 || xx >= t.w || yy >= t.h)
            continue;
        const int a = yy * t.w + xx;
        Location l = locate(t, x, y);
        Source s;
        if (!source(l, t, s))
            continue;
        if ((!seam && (xx == 0 || xx == t.w - 1)) || yy == 0 || yy == t.h - 1)
            continue;
        float w = 1;
        if (byte_weight || float_weight) {
            w = float_weight ? p.mask[a] : float(uint(p.mask_u8[a]) > 0);
            if (!(w > 0))
                continue;
        }
        const int left = xx == 0 ? t.w - 1 : -1, right = xx == t.w - 1 ? 1 - t.w : 1;
        float q[5] = {0, 0, 0, 0, 0}, ss = 0;
        for (int c = 0; c < 3; ++c) {
            const int i = c * t.w * t.h + a;
            const float dx = .5f * t.sx * (image[i + right] - image[i + left]);
            const float dy = .5f * t.sy * (image[i + t.w] - image[i - t.w]);
            const float residual = image[i] - rendered[i];
            ss += residual * residual;
            const float g = has_grad ? -epsilon * values * grad[i] : residual;
            q[0] += dx * g;
            q[1] += dy * g;
            q[2] += dx * dx;
            q[3] += dx * dy;
            q[4] += dy * dy;
        }
        if (!isfinite(ss))
            continue;
        const float rho = epsilon * rsqrt(ss / 3 + epsilon * epsilon);
        // Weighted least squares: a loss gradient already carries the pixel's weight, the curvature needs it too.
        const bool counted = xx >= mx && yy >= my && xx < t.w - mx && yy < t.h - my;
        const float curvature = counted ? rho * w : 0.f;
        for (int f = 2; f < 5; ++f)
            q[f] *= curvature;
        if (!has_grad)
            for (int f = 0; f < 2; ++f)
                q[f] *= rho * w;
        if (!(isfinite(q[0]) && isfinite(q[1]) && isfinite(q[2]) && isfinite(q[3]) && isfinite(q[4])))
            continue;
        differs = differs || ss > 0;
        // The group owns this cell's four partials. Form the distance from
        // that cell in integers so a reciprocal rounded at its left/top edge
        // cannot put a boundary pixel's weight on the opposite control point.
        l.u = float((2 * x + 1) * cells - 2 * t.width * col) / float(2 * t.width);
        l.v = float((2 * y + 1) * cells - 2 * t.height * r) / float(2 * t.height);
        for (int k = 0; k < 4; ++k)
            for (int f = 0; f < 5; ++f)
                sums[k][f] += l.weight(k) * q[f];
    }

    for (int k = 0; k < 4; k++)
        for (int f = 0; f < 5; f++)
            laneSums[lane][k * 5 + f] = sums[k][f];
    nonzero[lane] = uint(differs);
    GroupMemoryBarrierWithGroupSync();
    // Reproduce the four logical 32-lane reductions independently of subgroup width.
    for (int stride = 16; stride > 0; stride /= 2) {
        if ((lane % 32) < stride) {
            for (int f = 0; f < 20; f++)
                laneSums[lane][f] += laneSums[lane + stride][f];
            nonzero[lane] |= nonzero[lane + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    let partials = p.partials;
    int base = group * partial_size;
    if (lane < 20) {
        float sum = 0;
        for (int w = 0; w < 4; w++)
            sum += laneSums[w * 32][lane];
        partials[base + lane] = p.first ? sum : partials[base + lane] + sum;
    }
    if (lane == 20) {
        bool anyDiff = (nonzero[0] | nonzero[32] | nonzero[64] | nonzero[96]) != 0;
        partials[base + 20] = float(anyDiff || (!p.first && partials[base + 20] != 0));
    }
}
groupshared float2 old[81];
groupshared float trace[81];
[shader("compute")]
[numthreads(128, 1, 1)]
void update(uint laneId : SV_GroupIndex) {
    const Tile t = tile();
    const int i = int(laneId);
    const int side = t.side, cells = side - 1, points = side * side, x = i % side, y = i / side, chunks = p.chunks;
    const int row = fieldBase();
    let partials = p.partials;
    float q[5] = {0, 0, 0, 0, 0};
    bool differs = false;
    for (int k = 0; i < points && k < 4; ++k) {
        int cx = x - (k & 1);
        const int cy = y - (k >> 1);
        // A panorama's first and last control column are one longitude: both gather the cells on either side.
        if (t.wrap && cx < 0)
            cx = cells - 1;
        if (t.wrap && cx == cells)
            cx = 0;
        if (cx < 0 || cx >= cells || cy < 0 || cy >= cells)
            continue;
        for (int j = 0; j < chunks; ++j) {
            int base = ((cy * cells + cx) * chunks + j) * partial_size;
            differs = differs || partials[base + 20] != 0;
            for (int f = 0; f < 5; ++f)
                q[f] += partials[base + k * 5 + f];
        }
    }
    if (i < points)
        old[i] = p.field[row + i];
    if (i < points)
        trace[i] = q[2] + q[4];
    nonzero[i] = uint(differs);
    GroupMemoryBarrierWithGroupSync();
    if (i >= points)
        return;
    uint anyDiff = 0;
    for (int j = 0; j < points; j++)
        anyDiff |= nonzero[j];
    if (!anyDiff) {
        // Equal images can leave roundoff in a windowed loss derivative.
        q[0] = q[1] = 0;
    }
    float mean = 0;
    for (int j = 0; j < points; ++j)
        mean += trace[j];
    mean /= 2 * points;
    if (!(mean > 0 && isfinite(mean)))
        return;
    float2 lap = float2(0, 0);
    int degree = 0;
    // A panorama's seam column has the column across the seam as its outer neighbour.
    const int seam = t.wrap ? cells - 1 : 0;
    const int offsets[4] = {x > 0 ? -1 : seam, x < cells ? 1 : -seam, y > 0 ? -side : 0, y < cells ? side : 0};
    for (int j = 0; j < 4; ++j)
        if (offsets[j]) {
            int offset = offsets[j];
            lap.x += old[i].x - old[i + offset].x;
            lap.y += old[i].y - old[i + offset].y;
            ++degree;
        }
    const float diagonal = mean * (2 * degree * .05f + .01f + .1f);
    const float a = q[2] + diagonal, b = q[3], c = q[4] + diagonal, det = a * c - b * b;
    const float gx = q[0] + mean * (.05f * lap.x + .01f * old[i].x);
    const float gy = q[1] + mean * (.05f * lap.y + .01f * old[i].y);
    if (!(det > 0))
        return;
    const float dx = (c * gx - b * gy) / det, dy = (a * gy - b * gx) / det;
    const float moveLength = length(float2(dx * t.sx, dy * t.sy));
    if (isfinite(moveLength)) {
        const float scale = p.step * (moveLength > 1.5f ? 1.5f / moveLength : 1.f);
        float2 moved = float2(old[i].x - scale * dx, old[i].y - scale * dy);
        const float reach = length(moved * float2(t.sx, t.sy)), limit = .05f * min(t.width, t.height);
        if (limit > 0 && reach > limit)
            moved = float2(moved.x * (limit / reach), moved.y * (limit / reach));
        p.field[row + i] = moved;
    }
}

[shader("compute")]
[numthreads(128, 1, 1)]
void clear_outside(uint3 id : SV_DispatchThreadID) {
    const Tile t = tile();
    const int a = int(id.x);
    if (a >= t.w * t.h)
        return;
    Source s;
    if (!source(locate(t, a % t.w + t.x, a / t.w + t.y), t, s))
        for (int c = 0; c < 3; c++)
            p.grad[c * t.w * t.h + a] = 0;
}
