#include "half_bits.slangh"
#include "byte_read.slangh"
#include "byte_word.slangh"

[SpecializationConstant]
const uint kKind = 0;

struct Parameters {
    uint64_t source;
    uint64_t destination;
    uint64_t unchanged;
    uint64_t coefficientX;
    uint64_t coefficientY;
    uint count;
    uint height;
    uint width;
    uint channels;
    uint kind;
    uint layout;
    uint bytes;
    uint srgb;
    uint flipYZ;
    uint worldToCamera;
    uint inputHeight;
    uint inputWidth;
    uint outputHeight;
    uint outputWidth;
    uint kernelSize;
    uint strideX;
    uint strideY;
    float threshold;
    float w2c[9];
};
struct Push { uint64_t parameters; };

void storeImageByte(uint64_t address, uint64_t index, uint value)
{
    const uint64_t byteAddress = address + index;
    const uint shift = uint(byteAddress & uint64_t(3)) * 8u;
    const uint mask = 255u << shift;
    Ptr<uint, Access.ReadWrite> word =
        (Ptr<uint, Access.ReadWrite>)(byteAddress & ~uint64_t(3));
    uint assumed = word[0];
    uint observed;
    do
    {
        InterlockedCompareExchange(word[0], assumed,
                                   (assumed & ~mask) | ((value & 255u) << shift), observed);
        if (observed == assumed) return;
        assumed = observed;
    } while (true);
}

uint sentinelByte(uint index, uint seed) {
    uint value = index ^ seed;
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    value ^= value >> 16;
    return value & 255u;
}

float sinc(float x) {
    return abs(x) < 1e-12 ? 1.0 : sin(3.14159265358979323846 * x) / (3.14159265358979323846 * x);
}
float lanczos(float x, float a) {
    return x <= -a || x >= a ? 0.0 : sinc(x) * sinc(x / a);
}
float srgbToLinear(float v) {
    return v <= 0.04045 ? v / 12.92 : pow((v + 0.055) / 1.055, 2.4);
}

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID, uniform Push push) {
    Parameters p = *((Ptr<Parameters, Access.Read>)push.parameters);
    const uint i = dispatchThreadId.x;
    Ptr<uint16_t, Access.Read> u16 = (Ptr<uint16_t, Access.Read>)p.source;
    Ptr<float, Access.Read> f32 = (Ptr<float, Access.Read>)p.source;
    Ptr<uint16_t, Access.ReadWrite> out16 = (Ptr<uint16_t, Access.ReadWrite>)p.destination;
    Ptr<float, Access.ReadWrite> out32 = (Ptr<float, Access.ReadWrite>)p.destination;

    if (kKind == 0) {
        const uint wordCount = (p.count + 3u) / 4u;
        if (i < wordCount) {
            uint packed = 0u;
            uint mask = 0u;
            for (uint lane = 0u; lane < 4u; ++lane) {
                const uint byteIndex = i * 4u + lane;
                if (byteIndex < p.count) {
                    packed |= sentinelByte(byteIndex, p.kind) << (lane * 8u);
                    mask |= 255u << (lane * 8u);
                }
            }
            Ptr<uint, Access.ReadWrite> output =
                (Ptr<uint, Access.ReadWrite>)(p.destination + uint64_t(i) * 4u);
            if (mask != 0xffffffffu)
                packed = (output[0] & ~mask) | packed;
            output[0] = packed;
        }
        return;
    }
    if (kKind == 1) {
        if (i < p.count && loadByte(p.source, i) != sentinelByte(i, p.kind))
            ((Ptr<uint32_t, Access.ReadWrite>)p.unchanged)[0] = 0;
        return;
    }
    if (kKind == 2) {
        const uint pixels = p.height * p.width;
        const uint total = pixels * p.channels;
        if (i >= total) return;
        const uint channel = i % p.channels;
        const uint pixel = i / p.channels;
        const uint chw = channel * pixels + pixel;
        switch (p.kind) {
        case 0: out32[chw] = float(loadByte(p.source, i)) * (1.0 / 255.0); return;
        case 1: out32[chw] = float(u16[i]) * (1.0 / 65535.0); return;
        case 2: out16[i] = uint16_t(clamp(f32[i] * 65535.0, 0.0, 65535.0) + 0.5); return;
        case 3: out32[i] = float(u16[i]) * (1.0 / 65535.0); return;
        case 8: storeImageByte(p.destination, chw, loadByte(p.source, i)); return;
        case 9: storeImageByte(p.destination, chw, (uint(u16[i]) * 255u + 32767u) / 65535u); return;
        case 10: storeImageByte(p.destination, i, uint(clamp(f32[i], 0.0, 1.0) * 255.0 + 0.5)); return;
        case 11: out32[i] = float(loadByte(p.source, i)) * (1.0 / 255.0); return;
        }
        if (p.kind == 4 || p.kind == 5) {
            if (i >= pixels) return;
            for (uint c = 0; c < 3; ++c) {
                const float value = f32[p.kind == 4 ? c * pixels + i : i * 3 + c];
                out32[p.kind == 4 ? i * 3 + c : c * pixels + i] =
                    p.kind == 4 ? clamp(value * 0.5 + 0.5, 0.0, 1.0) : value * 2.0 - 1.0;
            }
            return;
        }
        if (p.kind == 6 || p.kind == 7) {
            if (i >= pixels) return;
            float3 normal;
            for (uint c = 0; c < 3; ++c) {
                float encoded = p.kind == 6 ? float(loadByte(p.source, i * 3 + c)) * (1.0 / 255.0)
                                            : float(u16[i * 3 + c]) * (1.0 / 65535.0);
                if (p.srgb != 0) encoded = srgbToLinear(encoded);
                normal[c] = encoded * 2.0 - 1.0;
            }
            if (p.flipYZ != 0) { normal.y = -normal.y; normal.z = -normal.z; }
            if (p.worldToCamera != 0) {
                const float3 n = normal;
                normal = float3(p.w2c[0] * n.x + p.w2c[1] * n.y + p.w2c[2] * n.z,
                                p.w2c[3] * n.x + p.w2c[4] * n.y + p.w2c[5] * n.z,
                                p.w2c[6] * n.x + p.w2c[7] * n.y + p.w2c[8] * n.z);
            }
            for (uint c = 0; c < 3; ++c) out32[c * pixels + i] = normal[c];
            return;
        }
        return;
    }
    if (kKind == 3) {
        if (i >= p.count) return;
        const uint pixels = p.height * p.width;
        for (uint c = 0; c < 3; ++c) {
            const uint value = loadByte(p.source, i * 4 + c);
            if (p.bytes != 0) storeImageByte(p.destination, c * pixels + i, value);
            else out32[c * pixels + i] = float(value) * (1.0 / 255.0);
        }
        ((Ptr<float, Access.ReadWrite>)p.unchanged)[i] =
            float(loadByte(p.source, i * 4 + 3)) * (1.0 / 255.0);
        return;
    }
    if (kKind == 4) {
        if (i >= p.count) return;
        const float value = out32[i];
        out32[i] = p.kind == 0 ? 1.0 - value : (value >= p.threshold ? 1.0 : 0.0);
        return;
    }
    if (kKind == 5) {
        const uint pixels = p.outputHeight * p.outputWidth;
        if (i >= pixels) return;
        const uint px = i % p.outputWidth, py = i / p.outputWidth;
        const float scaleX = float(p.inputWidth) / float(p.outputWidth);
        const float scaleY = float(p.inputHeight) / float(p.outputHeight);
        const float centerX = (float(px) + 0.5) * scaleX;
        const float centerY = (float(py) + 0.5) * scaleY;
        const int left = max(int(centerX - float(p.kernelSize) * scaleX + 0.5), 0);
        const int top = max(int(centerY - float(p.kernelSize) * scaleY + 0.5), 0);
        const int right = min(int(centerX + float(p.kernelSize) * scaleX + 0.5), int(p.inputWidth));
        const int bottom = min(int(centerY + float(p.kernelSize) * scaleY + 0.5), int(p.inputHeight));
        const uint channels = p.layout == 1 ? 1u : 3u;
        const uint inputPlane = p.inputHeight * p.inputWidth;
        Ptr<float, Access.Read> cx = (Ptr<float, Access.Read>)p.coefficientX;
        Ptr<float, Access.Read> cy = (Ptr<float, Access.Read>)p.coefficientY;
        float3 accum = 0.0;
        for (int y = top; y < bottom; ++y) {
            const float wy = cy[py * p.strideY + uint(y - top)];
            for (int x = left; x < right; ++x) {
                const uint pixel = uint(y) * p.inputWidth + uint(x);
                const float weight = wy * cx[px * p.strideX + uint(x - left)];
                for (uint c = 0; c < channels; ++c) {
                    float value;
                    if (p.layout == 2) value = f32[c * inputPlane + pixel];
                    else if (p.layout == 1) value = p.bytes != 0 ? float(loadByte(p.source, pixel)) / 255.0 : f32[pixel];
                    else value = p.bytes != 0 ? float(loadByte(p.source, pixel * 3 + c)) / 255.0 : f32[pixel * 3 + c];
                    accum[c] += value * weight;
                }
            }
        }
        const uint outputPlane = pixels;
        for (uint c = 0; c < channels; ++c) out32[c * outputPlane + i] = accum[c];
        return;
    }
}
