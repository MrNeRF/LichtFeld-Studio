/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

static const int ROW_THREADS = 256;
static const int MAX_RADIUS = 64;
static const int HORIZONTAL_PLANES = 7;
static const int FEATURE_PLANES = 5;

// linear sRGB to XYZ and back, and the XYZ of linear white used as the reference illuminant.
static const float RGB_TO_XYZ[9] = {
    10135552.0f / 24577794.0f, 8788810.0f / 24577794.0f, 4435075.0f / 24577794.0f,
    2613072.0f / 12288897.0f, 8788810.0f / 12288897.0f, 887015.0f / 12288897.0f,
    1425312.0f / 73733382.0f, 8788810.0f / 73733382.0f, 70074185.0f / 73733382.0f};
static const float XYZ_TO_RGB[9] = {
    3.241003275f, -1.537398934f, -0.498615861f,
    -0.969224334f, 1.875930071f, 0.041554224f,
    0.055639423f, -0.204011202f, 1.057148933f};
static const float WHITE_X = RGB_TO_XYZ[0] + RGB_TO_XYZ[1] + RGB_TO_XYZ[2];
static const float WHITE_Z = RGB_TO_XYZ[6] + RGB_TO_XYZ[7] + RGB_TO_XYZ[8];

static const float COLOR_EXPONENT = 0.7f;
static const float FEATURE_EXPONENT = 0.5f;
static const float COLOR_CUTOFF = 0.4f;
static const float COLOR_CUTOFF_ERROR = 0.95f;
static const float FEATURE_WIDTH_DEGREES = 0.082f;

struct Lab {
    float l;
    float a;
    float b;
};

float srgb_to_linear(const float c) {
    return c <= 0.04045f ? c / 12.92f : pow((c + 0.055f) / 1.055f, 2.4f);
}

float lab_f(const float t) {
    constexpr float delta = 6.0f / 29.0f;
    return t > delta * delta * delta ? pow(t, 1.0f / 3.0f) : t / (3.0f * delta * delta) + 4.0f / 29.0f;
}

Lab hunt_adjusted_lab(const float r, const float g, const float b) {
    const float x = (RGB_TO_XYZ[0] * r + RGB_TO_XYZ[1] * g + RGB_TO_XYZ[2] * b) / WHITE_X;
    const float y = RGB_TO_XYZ[3] * r + RGB_TO_XYZ[4] * g + RGB_TO_XYZ[5] * b;
    const float z = (RGB_TO_XYZ[6] * r + RGB_TO_XYZ[7] * g + RGB_TO_XYZ[8] * b) / WHITE_Z;
    const float fx = lab_f(x);
    const float fy = lab_f(y);
    const float fz = lab_f(z);
    const float l = 116.0f * fy - 16.0f;
    Lab result;
    result.l = l;
    result.a = 0.01f * l * 500.0f * (fx - fy);
    result.b = 0.01f * l * 200.0f * (fy - fz);
    return result;
}

float hyab(Lab p, Lab q) {
    const float da = p.a - q.a;
    const float db = p.b - q.b;
    return abs(p.l - q.l) + sqrt(da * da + db * db);
}
struct Params {
    uint64_t image, taps, planes, reference, features, error, output;
    int width, height, csf_radius, feature_radius;
    float max_color_error;
    uint operation, group_offset;
};
struct Push {
    uint64_t parameters;
};
[[vk::push_constant]] Push push;
groupshared float row[4 * (256 + 128)];
int clamp_to_edge(int index, int size) { return min(max(index, 0), size - 1); }
[numthreads(256, 1, 1)] void main(uint3 gid : SV_GroupID, uint lane : SV_GroupIndex) {
    Params p = ((Ptr<Params, Access.Read>)push.parameters)[0];
    uint group = gid.x + p.group_offset;
    uint nx = (uint(p.width) + 255) / 256;
    int width = p.width, height = p.height, y = int(group / nx);
    if (p.operation == 2) {
        uint i = group * 256 + lane, pixels = uint(width) * uint(height);
        if (i >= pixels)
            return;
        float error = ((Ptr<float, Access.Read>)p.image)[i];
        int index = int(clamp(error, 0.0f, 1.0f) * 255.0f + 0.5f);
        for (int c = 0; c < 3; ++c)
            ((Ptr<uint8_t, Access.ReadWrite>)p.output)[uint(c) * pixels + i] = ((Ptr<uint8_t, Access.Read>)p.taps)[3 * index + c];
        return;
    }
    Ptr<float, Access.Read> image = (Ptr<float, Access.Read>)p.image;
    Ptr<float, Access.Read> tap_h = (Ptr<float, Access.Read>)p.taps;
    Ptr<float, Access.Read> tap_v = (Ptr<float, Access.Read>)(p.taps + uint64_t(4 * (2 * p.csf_radius + 1)) * 4);
    Ptr<float, Access.Read> tap_f = (Ptr<float, Access.Read>)(p.taps + uint64_t(8 * (2 * p.csf_radius + 1)) * 4);
    Ptr<float, Access.ReadWrite> planes = (Ptr<float, Access.ReadWrite>)p.planes;
    Ptr<float, Access.Read> reference = (Ptr<float, Access.Read>)p.reference;
    Ptr<float, Access.ReadWrite> features = (Ptr<float, Access.ReadWrite>)p.features;
    Ptr<float, Access.ReadWrite> error = (Ptr<float, Access.ReadWrite>)p.error;
    if (p.operation == 0) {

        const int radius = max(p.csf_radius, p.feature_radius);
        const int span = ROW_THREADS + 2 * radius;

        const int x0 = int(group % nx) * ROW_THREADS;
        const uint pixels = uint(width) * height;
        for (int i = int(lane); i < span; i += ROW_THREADS) {
            const uint source = uint(y) * width + clamp_to_edge(x0 + i - radius, width);
            const float r = srgb_to_linear(image[source]);
            const float g = srgb_to_linear(image[pixels + source]);
            const float b = srgb_to_linear(image[2 * pixels + source]);
            const float x = (RGB_TO_XYZ[0] * r + RGB_TO_XYZ[1] * g + RGB_TO_XYZ[2] * b) / WHITE_X;
            const float lum = RGB_TO_XYZ[3] * r + RGB_TO_XYZ[4] * g + RGB_TO_XYZ[5] * b;
            const float z = (RGB_TO_XYZ[6] * r + RGB_TO_XYZ[7] * g + RGB_TO_XYZ[8] * b) / WHITE_Z;
            row[i] = 116.0f * lum - 16.0f;
            row[span + i] = 500.0f * (x - lum);
            row[2 * span + i] = 200.0f * (lum - z);
            row[3 * span + i] = lum;
        }
        GroupMemoryBarrierWithGroupSync();
        const int x = x0 + int(lane);
        if (x >= width)
            return;
        const int csf_taps = 2 * p.csf_radius + 1;
        float sums[HORIZONTAL_PLANES];
        for (int j = 0; j < HORIZONTAL_PLANES; ++j)
            sums[j] = 0;
        for (int k = 0; k < csf_taps; ++k) {
            const int i = int(lane) + radius - p.csf_radius + k;
            sums[0] = fma(tap_h[k], row[i], sums[0]);
            sums[1] = fma(tap_h[csf_taps + k], row[span + i], sums[1]);
            sums[2] = fma(tap_h[2 * csf_taps + k], row[2 * span + i], sums[2]);
            sums[3] = fma(tap_h[3 * csf_taps + k], row[2 * span + i], sums[3]);
        }
        const int feature_taps = 2 * p.feature_radius + 1;
        for (int k = 0; k < feature_taps; ++k) {
            const float lum = row[3 * span + int(lane) + radius - p.feature_radius + k];
            sums[4] = fma(tap_f[k], lum, sums[4]);
            sums[5] = fma(tap_f[feature_taps + k], lum, sums[5]);
            sums[6] = fma(tap_f[2 * feature_taps + k], lum, sums[6]);
        }
        const uint out = uint(y) * width + x;
        for (int p = 0; p < HORIZONTAL_PLANES; ++p)
            planes[p * pixels + out] = sums[p];

    } else {
        const int x = int(group % nx) * ROW_THREADS + int(lane);

        if (x >= width)
            return;
        const uint pixels = uint(width) * height;
        const int csf_taps = 2 * p.csf_radius + 1;
        float opponent[3] = {0, 0, 0};
        for (int k = 0; k < csf_taps; ++k) {
            const uint source = uint(clamp_to_edge(y + k - p.csf_radius, height)) * width + x;
            opponent[0] = fma(tap_v[k], planes[source], opponent[0]);
            opponent[1] = fma(tap_v[csf_taps + k], planes[pixels + source], opponent[1]);
            opponent[2] += tap_v[2 * csf_taps + k] * planes[2 * pixels + source] +
                           tap_v[3 * csf_taps + k] * planes[3 * pixels + source];
        }
        const int feature_taps = 2 * p.feature_radius + 1;
        float edge_x = 0.0f, edge_y = 0.0f, point_x = 0.0f, point_y = 0.0f;
        for (int k = 0; k < feature_taps; ++k) {
            const uint source = uint(clamp_to_edge(y + k - p.feature_radius, height)) * width + x;
            const float gaussian = tap_f[2 * feature_taps + k];
            edge_x = fma(gaussian, planes[4 * pixels + source], edge_x);
            point_x = fma(gaussian, planes[5 * pixels + source], point_x);
            edge_y = fma(tap_f[k], planes[6 * pixels + source], edge_y);
            point_y = fma(tap_f[feature_taps + k], planes[6 * pixels + source], point_y);
        }

        const float lum = (opponent[0] + 16.0f) / 116.0f;
        const float cx = (opponent[1] / 500.0f + lum) * WHITE_X;
        const float cz = (lum - opponent[2] / 200.0f) * WHITE_Z;
        const float r = min(max(XYZ_TO_RGB[0] * cx + XYZ_TO_RGB[1] * lum + XYZ_TO_RGB[2] * cz, 0.0f), 1.0f);
        const float g = min(max(XYZ_TO_RGB[3] * cx + XYZ_TO_RGB[4] * lum + XYZ_TO_RGB[5] * cz, 0.0f), 1.0f);
        const float b = min(max(XYZ_TO_RGB[6] * cx + XYZ_TO_RGB[7] * lum + XYZ_TO_RGB[8] * cz, 0.0f), 1.0f);
        const Lab lab = hunt_adjusted_lab(r, g, b);
        const float edge = sqrt(edge_x * edge_x + edge_y * edge_y);
        const float point = sqrt(point_x * point_x + point_y * point_y);
        const uint out = uint(y) * width + x;
        if (p.reference == 0) {
            features[out] = lab.l;
            features[pixels + out] = lab.a;
            features[2 * pixels + out] = lab.b;
            features[3 * pixels + out] = edge;
            features[4 * pixels + out] = point;
            return;
        }

        Lab reference_lab;
        reference_lab.l = reference[out];
        reference_lab.a = reference[pixels + out];
        reference_lab.b = reference[2 * pixels + out];
        const float color = pow(hyab(reference_lab, lab), COLOR_EXPONENT);
        const float cutoff = COLOR_CUTOFF * p.max_color_error;
        const float color_error =
            color < cutoff ? color * (COLOR_CUTOFF_ERROR / cutoff)
                           : COLOR_CUTOFF_ERROR + (color - cutoff) / (p.max_color_error - cutoff) * (1.0f - COLOR_CUTOFF_ERROR);
        const float feature_difference = max(abs(reference[3 * pixels + out] - edge),
                                             abs(reference[4 * pixels + out] - point));
        const float feature_error = pow(feature_difference * 0.70710678f, FEATURE_EXPONENT);
        error[out] = pow(color_error, 1.0f - feature_error);
    }
}
