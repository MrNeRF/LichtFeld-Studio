// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later

// SharedImageOps kernels, ported from io/cuda/image_format_kernels.cu and
// core/cuda/lanczos_resize/lanczos_resize.cu. Undistortion and prior resizes
// reuse image_resample.

constant constexpr float kImageScaleU8 = 1.0f / 255.0f;
constant constexpr float kImageScaleU16 = 1.0f / 65535.f;

static uchar shared_image_sentinel(const uint index, const uint seed) {
    uint value = index ^ seed;
    value ^= value >> 16u;
    value *= 0x7feb352du;
    value ^= value >> 15u;
    value *= 0x846ca68bu;
    value ^= value >> 16u;
    return uchar(value);
}

struct SharedImageSentinelParams {
    device uchar* bytes;
    device uint* unchanged;
    uint count, seed;
};

kernel void shared_image_sentinel_fill(constant SharedImageSentinelParams& p [[buffer(0)]],
                                       uint index [[thread_position_in_grid]]) {
    if (index < p.count)
        p.bytes[index] = shared_image_sentinel(index, p.seed);
}

kernel void shared_image_sentinel_check(constant SharedImageSentinelParams& p [[buffer(0)]],
                                        uint index [[thread_position_in_grid]]) {
    if (index < p.count && p.bytes[index] != shared_image_sentinel(index, p.seed))
        p.unchanged[0] = 0u;
}

// ImageConversion order of core/shared_image_ops.hpp.
enum SharedImageConversion : uint {
    kU8HWCToF32CHW,
    kU16HWCToF32CHW,
    kF32HWCToU16HWC,
    kU16HWCToF32HWC,
    kNormalCHWToJ2KHWC,
    kJ2KHWCToNormalCHW,
    kNormalPriorU8,
    kNormalPriorU16,
    kU8HWCToU8CHW,
    kU16HWCToU8CHW,
    kF32CHWToU8CHW,
    kU8HWToF32HW,
};

struct SharedImageConvertParams {
    device const uchar* source;
    device uchar* destination;
    uint kind, height, width, channels;
    uint srgb, flip_yz, world_to_camera, padding;
    float w2c[9];
};

static float shared_image_srgb_to_linear(const float v) {
    return v <= 0.04045f ? v / 12.92f : pow((v + 0.055f) / 1.055f, 2.4f);
}

kernel void shared_image_convert(constant SharedImageConvertParams& p [[buffer(0)]],
                                 uint index [[thread_position_in_grid]]) {
    device const uchar* u8 = p.source;
    device const ushort* u16 = (device const ushort*)p.source;
    device const float* f32 = (device const float*)p.source;
    device float* out_f32 = (device float*)p.destination;
    const uint pixels = p.height * p.width;
    const uint total = pixels * p.channels;
    // HWC element index -> CHW element index.
    const uint c = index % max(p.channels, 1u), pixel = index / max(p.channels, 1u);
    const uint chw = c * pixels + pixel;
    switch (p.kind) {
    case kU8HWCToF32CHW:
        if (index < total)
            out_f32[chw] = float(u8[index]) * kImageScaleU8;
        return;
    case kU16HWCToF32CHW:
        if (index < total)
            out_f32[chw] = float(u16[index]) * kImageScaleU16;
        return;
    case kF32HWCToU16HWC:
        if (index < total)
            ((device ushort*)p.destination)[index] = ushort(fmin(fmax(f32[index] * 65535.0f, 0.0f), 65535.0f) + 0.5f);
        return;
    case kU16HWCToF32HWC:
        if (index < total)
            out_f32[index] = float(u16[index]) * kImageScaleU16;
        return;
    case kNormalCHWToJ2KHWC:
        if (index < pixels) {
            for (uint k = 0; k < 3; ++k)
                out_f32[index * 3 + k] = fmin(fmax(f32[k * pixels + index] * 0.5f + 0.5f, 0.0f), 1.0f);
        }
        return;
    case kJ2KHWCToNormalCHW:
        if (index < pixels) {
            for (uint k = 0; k < 3; ++k)
                out_f32[k * pixels + index] = f32[index * 3 + k] * 2.0f - 1.0f;
        }
        return;
    case kNormalPriorU8:
    case kNormalPriorU16: {
        if (index >= pixels)
            return;
        float n[3];
        for (uint k = 0; k < 3; ++k) {
            float encoded = p.kind == kNormalPriorU16 ? float(u16[index * 3 + k]) * kImageScaleU16
                                                      : float(u8[index * 3 + k]) * kImageScaleU8;
            if (p.srgb != 0)
                encoded = shared_image_srgb_to_linear(encoded);
            n[k] = encoded * 2.0f - 1.0f;
        }
        if (p.flip_yz != 0) {
            n[1] = -n[1];
            n[2] = -n[2];
        }
        if (p.world_to_camera != 0) {
            const float x = n[0], y = n[1], z = n[2];
            n[0] = p.w2c[0] * x + p.w2c[1] * y + p.w2c[2] * z;
            n[1] = p.w2c[3] * x + p.w2c[4] * y + p.w2c[5] * z;
            n[2] = p.w2c[6] * x + p.w2c[7] * y + p.w2c[8] * z;
        }
        for (uint k = 0; k < 3; ++k)
            out_f32[k * pixels + index] = n[k];
        return;
    }
    case kU8HWCToU8CHW:
        if (index < total)
            p.destination[chw] = u8[index];
        return;
    case kU16HWCToU8CHW:
        if (index < total)
            p.destination[chw] = uchar((uint(u16[index]) * 255u + 32767u) / 65535u);
        return;
    case kF32CHWToU8CHW:
        if (index < total)
            p.destination[index] = uchar(fmin(fmax(f32[index], 0.0f), 1.0f) * 255.0f + 0.5f);
        return;
    case kU8HWToF32HW:
        if (index < pixels)
            out_f32[index] = float(u8[index]) * kImageScaleU8;
        return;
    }
}

struct SharedImageRgbaParams {
    device const uchar* rgba;
    device uchar* rgb;
    device float* alpha;
    uint pixels, rgb_bytes;
};

kernel void shared_image_rgba_split(constant SharedImageRgbaParams& p [[buffer(0)]],
                                    uint index [[thread_position_in_grid]]) {
    if (index >= p.pixels)
        return;
    device const uchar* src = p.rgba + index * 4;
    for (uint k = 0; k < 3; ++k) {
        if (p.rgb_bytes != 0)
            p.rgb[k * p.pixels + index] = src[k];
        else
            ((device float*)p.rgb)[k * p.pixels + index] = float(src[k]) * kImageScaleU8;
    }
    p.alpha[index] = float(src[3]) * kImageScaleU8;
}

struct SharedImageMaskParams {
    device float* mask;
    uint count, threshold_mode;
    float threshold;
};

kernel void shared_image_mask(constant SharedImageMaskParams& p [[buffer(0)]], uint index [[thread_position_in_grid]]) {
    if (index >= p.count)
        return;
    const float v = p.mask[index];
    p.mask[index] = p.threshold_mode != 0 ? (v >= p.threshold ? 1.0f : 0.0f) : 1.0f - v;
}

// Lanczos resampling: normalized per-output tap weights along each axis, then
// the separable product over the input box of every output pixel.

static float shared_image_sinc(const float x) {
    if (fabs(x) < 1e-12f)
        return 1.0f;
    return sin(M_PI_F * x) / (M_PI_F * x);
}

static float shared_image_lanczos(const float x, const float a) {
    if (x <= -a || x >= a)
        return 0.0f;
    return shared_image_sinc(x) * shared_image_sinc(x / a);
}

struct SharedImageLanczosCoefParams {
    device float* coefficients;
    int input_size, output_size, kernel_size;
    uint stride;
};

kernel void shared_image_lanczos_coefficients(constant SharedImageLanczosCoefParams& p [[buffer(0)]],
                                              uint index [[thread_position_in_grid]]) {
    if (int(index) >= p.output_size)
        return;
    const float scale = 1.0f * float(p.input_size) / float(p.output_size);
    const float center = (float(index) + 0.5f) * scale;
    const int lo = max(int(center - float(p.kernel_size) * scale + 0.5f), 0);
    const int hi = min(int(center + float(p.kernel_size) * scale + 0.5f), p.input_size);
    device float* values = p.coefficients + index * p.stride;
    float norm = 0.0f;
    for (int i = lo; i < hi; ++i) {
        const float value = shared_image_lanczos((float(i) + 0.5f - center) / scale, float(p.kernel_size));
        values[i - lo] = value;
        norm += value;
    }
    for (int i = lo; i < hi; ++i)
        values[i - lo] /= norm;
}

// Layouts: 0 HWC RGB (uint8 divided by 255, or float), 1 HW gray (uint8 times
// 1/255, or float), 2 CHW RGB float. Output is CHW or HW float.
struct SharedImageLanczosParams {
    device const uchar* input;
    device float* output;
    device const float* coef_x;
    device const float* coef_y;
    int input_h, input_w, output_h, output_w, kernel_size;
    uint stride_x, stride_y, layout, bytes;
};

kernel void shared_image_lanczos_resample(constant SharedImageLanczosParams& p [[buffer(0)]],
                                          uint index [[thread_position_in_grid]]) {
    if (int(index) >= p.output_h * p.output_w)
        return;
    const int px = int(index) % p.output_w, py = int(index) / p.output_w;
    const float scale_h = 1.0f * float(p.input_h) / float(p.output_h);
    const float scale_w = 1.0f * float(p.input_w) / float(p.output_w);
    const float cx = (float(px) + 0.5f) * scale_w, cy = (float(py) + 0.5f) * scale_h;
    const int lx = max(int(cx - float(p.kernel_size) * scale_w + 0.5f), 0);
    const int ly = max(int(cy - float(p.kernel_size) * scale_h + 0.5f), 0);
    const int rx = min(int(cx + float(p.kernel_size) * scale_w + 0.5f), p.input_w);
    const int ry = min(int(cy + float(p.kernel_size) * scale_h + 0.5f), p.input_h);
    const int channels = p.layout == 1 ? 1 : 3;
    const uint input_plane = uint(p.input_h * p.input_w);
    device const float* f32 = (device const float*)p.input;
    float acc[3] = {0.0f, 0.0f, 0.0f};
    for (int y = ly; y < ry; ++y) {
        const float ky = p.coef_y[uint(py) * p.stride_y + uint(y - ly)];
        for (int x = lx; x < rx; ++x) {
            const uint pixel = uint(p.input_w * y + x);
            const float k = ky * p.coef_x[uint(px) * p.stride_x + uint(x - lx)];
            for (int ch = 0; ch < channels; ++ch) {
                float v;
                if (p.layout == 2)
                    v = f32[uint(ch) * input_plane + pixel];
                else if (p.layout == 1)
                    v = p.bytes != 0 ? float(p.input[pixel]) * kImageScaleU8 : f32[pixel];
                else
                    v = p.bytes != 0 ? float(p.input[pixel * 3 + uint(ch)]) / 255.0f : f32[pixel * 3 + uint(ch)];
                acc[ch] += v * k;
            }
        }
    }
    const uint output_plane = uint(p.output_h * p.output_w);
    for (int ch = 0; ch < channels; ++ch)
        p.output[uint(ch) * output_plane + index] = acc[ch];
}
