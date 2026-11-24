// SPDX-License-Identifier: GPL-3.0-or-later
[SpecializationConstant] const uint operation = 0;
struct Tile {
    int width, height, x, y, w, h, side;
    float sx, sy;
    int wrap;
};
struct Params {
    uint64_t field, photo, render, output, mask, out_mask, structure, out_structure, edge, out_edge;
    uint64_t alpha, background, stable_background, stable, grad, partials;
    Tile tile;
    int photo_byte, mask_byte, background_image, chunks, margin, first, channels;
    float values, step;
};
struct Push {
    uint64_t parameters;
};
[[vk::push_constant]] Push push;
float readAt(uint64_t address, int index) { return ((Ptr<float, Access.Read>)address)[index]; }
void writeAt(uint64_t address, int index, float value) { ((Ptr<float, Access.ReadWrite>)address)[index] = value; }
uint byteAt(uint64_t address, int index) { return uint(((Ptr<uint8_t, Access.Read>)address)[index]); }
static const int threads = 128, chunk = 512, partial_size = 21;
static const float epsilon = .03f;
struct Location {
    int cell;
    float u, v, x, y;
    float weight(int k) {
        return (k & 1 ? u : 1 - u) * (k & 2 ? v : 1 - v);
    }
};
Location locate(const Tile t, int x, int y, uint64_t row) {
    const int side = t.side, cells = side - 1;
    // An exact cell boundary must agree with the integer partition used by the reduction.
    const float gx = ((x + .5f) * cells) / float(t.width);
    const float gy = ((y + .5f) * cells) / float(t.height);
    const int i = min(int(gx), cells - 1), j = min(int(gy), cells - 1);
    Location p = {i + j * side, gx - i, gy - j, float(x), float(y)};
    float dx = 0, dy = 0;
    for (int k = 0; k < 4; ++k) {
        float2 d = ((Ptr<float2, Access.Read>)row)[p.cell + (k & 1) + (k >> 1) * side];
        dx = fma(p.weight(k), d.x, dx);
        dy = fma(p.weight(k), d.y, dy);
    }
    p.x = fma(dx, t.sx, p.x);
    p.y = fma(dy, t.sy, p.y);
    return p;
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
bool source(Location p, Tile t, out Source s) {
    s = (Source)0;
    if (!(p.y >= 0 && p.y <= t.height - 1))
        return false;
    float x = p.x;
    if (t.wrap) {
        if (!isfinite(x))
            return false;
        x -= floor(x / t.width) * t.width;
        if (x >= t.width)
            x -= t.width;
    } else if (!(x >= 0 && x <= t.width - 1)) {
        return false;
    }
    s = {int(x), int(p.y), 0, 0};
    s.u = x - s.x;
    s.v = p.y - s.y;
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
float tap(Params p, int c, int i, float background) {
    int j = c * p.tile.width * p.tile.height + i;
    float value = p.photo_byte ? float(byteAt(p.photo, j)) : readAt(p.photo, j);
    if (!p.alpha)
        return value;
    float alpha = saturate(readAt(p.alpha, i));
    return value * (p.photo_byte ? 1.f / 255.f : 1.f) * alpha + background * (1 - alpha);
}
float sample_cubic(Params p, Source s, Tile t, int channel, float background) {
    if (s.u == 0 && s.v == 0)
        return tap(p, channel, s.y * t.width + s.x, background);
    float taps[4][4], lines[4];
    for (int j = 0; j < 4; ++j) {
        const int row = line(t, s.y + j - 1) * t.width;
        for (int k = 0; k < 4; ++k)
            taps[j][k] = tap(p, channel, row + column(t, s.x + k - 1), background);
        lines[j] = cubic(taps[j][0], taps[j][1], taps[j][2], taps[j][3], s.u);
    }
    const float value = cubic(lines[0], lines[1], lines[2], lines[3], s.v);
    const float low = min(min(taps[1][1], taps[1][2]), min(taps[2][1], taps[2][2]));
    const float high = max(max(taps[1][1], taps[1][2]), max(taps[2][1], taps[2][2]));
    return min(max(value, low), high);
}
void samplePhoto(Params p, Tile t, int a) {
    int n = t.w * t.h;
    if (a >= n)
        return;
    Location loc = locate(t, a % t.w + t.x, a / t.w + t.y, p.field);
    Source s;
    bool valid = source(loc, t, s);
    int pixels = t.width * t.height, dest = (a / t.w + t.y) * t.width + a % t.w + t.x;
    for (int c = 0; c < 3; c++) {
        float v = readAt(p.render, c * n + a);
        if (valid) {
            float bg = p.alpha ? readAt(p.background, p.background_image ? c * pixels + dest : c) : 0;
            v = sample_cubic(p, s, t, c, bg);
            if (!p.alpha && p.photo_byte)
                v *= 1.f / 255.f;
        }
        writeAt(p.output, c * n + a, v);
        if (p.stable)
            writeAt(p.stable, c * pixels + dest, valid ? sample_cubic(p, s, t, c, readAt(p.stable_background, c)) : v);
    }
    if (p.mask) {
        int q = valid ? int(loc.y + .5f) * t.width + column(t, int(s.x + s.u + .5f)) : 0;
        if (p.mask_byte)
            ((Ptr<uint8_t, Access.ReadWrite>)p.out_mask)[a] = uint8_t(valid ? byteAt(p.mask, q) : 0);
        else
            writeAt(p.out_mask, a, valid ? readAt(p.mask, q) : 0);
    }
    if (p.structure)
        writeAt(p.out_structure, a, valid ? linear((Ptr<float, Access.Read>)p.structure, t, s) : 0);
    if (p.edge)
        writeAt(p.out_edge, a, valid ? linear((Ptr<float, Access.Read>)p.edge, t, s) : 0);
}
void followPrior(Params params, Tile t, int a) {
    int n = t.w * t.h, channels = params.channels;
    let src = (Ptr<float, Access.Read>)params.photo;
    let output = (Ptr<float, Access.ReadWrite>)params.output;
    uint64_t row = params.field;
    if (a >= n)
        return;
    Source s;
    const bool inside = source(locate(t, a % t.w, a / t.w, row), t, s);
    if (inside && s.u == 0 && s.v == 0) {
        for (int c = 0; c < channels; ++c)
            output[c * n + a] = src[c * n + s.y * t.w + s.x];
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
        output[c * n + a] = sum[c] * norm;
}

groupshared float laneSums[128][20];
groupshared uint nonzero[128];
void accumulate(Params params, Tile t, int group, int lane) {
    uint64_t row = params.field, weight = params.mask;
    bool float_weight = !params.mask_byte;
    int chunks = params.chunks, margin = params.margin;
    float values = params.values;
    let image = (Ptr<float, Access.Read>)params.photo;
    let rendered = (Ptr<float, Access.Read>)params.render;
    let grad = (Ptr<float, Access.Read>)params.grad;
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
        Location p = locate(t, x, y, row);
        Source s;
        if (!source(p, t, s))
            continue;
        if ((!seam && (xx == 0 || xx == t.w - 1)) || yy == 0 || yy == t.h - 1)
            continue;
        float w = 1;
        if (weight) {
            w = float_weight ? readAt(weight, a) : float(byteAt(weight, a) > 0);
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
            const float g = params.grad != 0 ? -epsilon * values * grad[i] : residual;
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
        if (!grad)
            for (int f = 0; f < 2; ++f)
                q[f] *= rho * w;
        if (!(isfinite(q[0]) && isfinite(q[1]) && isfinite(q[2]) && isfinite(q[3]) && isfinite(q[4])))
            continue;
        differs = differs || ss > 0;
        // The group owns this cell's four partials. Form the distance from
        // that cell in integers so a reciprocal rounded at its left/top edge
        // cannot put a boundary pixel's weight on the opposite control point.
        p.u = float((2 * x + 1) * cells - 2 * t.width * col) / float(2 * t.width);
        p.v = float((2 * y + 1) * cells - 2 * t.height * r) / float(2 * t.height);
        for (int k = 0; k < 4; ++k)
            for (int f = 0; f < 5; ++f)
                sums[k][f] += p.weight(k) * q[f];
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
    let partials = (Ptr<float, Access.ReadWrite>)params.partials;
    int base = group * partial_size;
    if (lane < 20) {
        float sum = 0;
        for (int w = 0; w < 4; w++)
            sum += laneSums[w * 32][lane];
        partials[base + lane] = params.first ? sum : partials[base + lane] + sum;
    }
    if (lane == 20) {
        bool anyDiff = (nonzero[0] | nonzero[32] | nonzero[64] | nonzero[96]) != 0;
        partials[base + 20] = float(anyDiff || (!params.first && partials[base + 20] != 0));
    }
}
groupshared float2 old[81];
groupshared float trace[81];
void updateRow(Params params, Tile t, int i) {
    const int side = t.side, cells = side - 1, points = side * side, x = i % side, y = i / side, chunks = params.chunks;
    let row = (Ptr<float2, Access.ReadWrite>)params.field;
    let partials = (Ptr<float, Access.Read>)params.partials;
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
        old[i] = row[i];
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
        const float scale = params.step * (moveLength > 1.5f ? 1.5f / moveLength : 1.f);
        float2 moved = float2(old[i].x - scale * dx, old[i].y - scale * dy);
        const float reach = length(moved * float2(t.sx, t.sy)), limit = .05f * min(t.width, t.height);
        if (limit > 0 && reach > limit)
            moved = float2(moved.x * (limit / reach), moved.y * (limit / reach));
        row[i] = moved;
    }
}

[numthreads(128, 1, 1)]
void main(uint3 group : SV_GroupID, uint lane : SV_GroupIndex) {
    Params p = ((Ptr<Params, Access.Read>)push.parameters)[0];
    Tile t = p.tile;
    int a = int(group.x * 128 + lane);
    if (operation == 0)
        samplePhoto(p, t, a);
    if (operation == 1)
        accumulate(p, t, int(group.x), int(lane));
    if (operation == 2)
        updateRow(p, t, int(lane));
    if (operation == 3 && a < t.w * t.h) {
        Source s;
        if (!source(locate(t, a % t.w + t.x, a / t.w + t.y, p.field), t, s))
            for (int c = 0; c < 3; c++)
                writeAt(p.grad, c * t.w * t.h + a, 0);
    }
    if (operation == 4)
        followPrior(p, t, a);
}
