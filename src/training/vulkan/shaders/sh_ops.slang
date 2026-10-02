#include "../../../core/include/core/rounded_math.slangh"

#include "sh_codec.slangh"

[SpecializationConstant] const uint operation = 0;

groupshared float lows[256];
groupshared float highs[256];

float minNumber(float a, float b) { return isnan(a) ? b : isnan(b) ? a : min(a, b); }
float maxNumber(float a, float b) { return isnan(a) ? b : isnan(b) ? a : max(a, b); }
uint q16Code(float value, float lo, float hi) {
    float scaled = roundedMul(roundedMul(65535.0f, roundedSub(value,lo)), 1.0f / maxNumber(roundedSub(hi,lo), 1e-20f));
    float base = floor(scaled);
    return uint(minNumber(maxNumber(base + (scaled - base >= 0.5f ? 1.0f : 0.0f), 0.0f), 65535.0f));
}

struct ReencodeParams {
    uint64_t codes, bounds, canonical, destinations;
    uint64_t uniqueBlocks, offsets, runCount, canonicalOrder;
    uint sortedCount, primitives, decodeRows, rest;
    uint reserved0, reserved1, reserved2, reserved3;
};
uint q16Offset(uint row, uint cell, uint rest) {
    return ((row / 32 * (rest * 3) + cell) * 32 + row % 32);
}

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint lane : SV_GroupIndex, uniform Parameters p) {
    uint3 groups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
    if (operation == 12) {
        ReencodeParams rp = *((Ptr<ReencodeParams, Access.Read>)p.source);
        uint runCount = uint(((Ptr<int, Access.Read>)rp.runCount)[0]);
        for (uint run = gid.x; run < runCount; run += groups.x) {
            uint block = uint(((Ptr<int, Access.Read>)rp.uniqueBlocks)[run]);
            uint start = uint(((Ptr<int, Access.Read>)rp.offsets)[run]);
            uint end = run + 1 < runCount
                           ? uint(((Ptr<int, Access.Read>)rp.offsets)[run + 1])
                           : rp.sortedCount;
            float2 oldBounds = ((Ptr<float2, Access.Read>)rp.bounds)[block];
            uint row = block * 256 + lane;
            uint canonicalRow = 0xffffffffu;
            // The last matching destination wins, including duplicate rows.
            for (uint sorted = start; sorted < end; ++sorted) {
                if (((Ptr<int64_t, Access.Read>)rp.destinations)[sorted] == int64_t(row))
                    canonicalRow = rp.canonicalOrder != 0
                                       ? uint(((Ptr<int64_t, Access.Read>)rp.canonicalOrder)[sorted])
                                       : sorted;
            }
            float values[45];
            float lo = 1e30f, hi = -1e30f;
            if (row < rp.primitives) {
                for (uint cell = 0; cell < rp.rest * 3; ++cell) {
                    float value = 0.0f;
                    if (canonicalRow != 0xffffffffu)
                        value = ((Ptr<float, Access.Read>)rp.canonical)[uint64_t(canonicalRow) * rp.rest * 3 + cell];
                    else if (row < rp.decodeRows) {
                        uint code = ((Ptr<uint16_t, Access.Read>)rp.codes)[q16Offset(row, cell, rp.rest)];
                        value = fma(oldBounds.y - oldBounds.x, float(code) * (1.0f / 65535.0f), oldBounds.x);
                    }
                    values[cell] = value;
                    lo = minNumber(lo, value);
                    hi = maxNumber(hi, value);
                }
            }
            lows[lane] = lo;
            highs[lane] = hi;
            GroupMemoryBarrierWithGroupSync();
            for (uint stride = 128; stride > 0; stride >>= 1) {
                if (lane < stride) {
                    lows[lane] = minNumber(lows[lane], lows[lane + stride]);
                    highs[lane] = maxNumber(highs[lane], highs[lane + stride]);
                }
                GroupMemoryBarrierWithGroupSync();
            }
            lo = lows[0]; hi = highs[0];
            if (lo > hi) { lo = 0.0f; hi = 0.0f; }
            if (lane == 0)
                ((Ptr<float2, Access.ReadWrite>)rp.bounds)[block] = float2(lo, hi);
            if (row < rp.primitives)
                for (uint cell = 0; cell < rp.rest * 3; ++cell)
                    ((Ptr<uint16_t, Access.ReadWrite>)rp.codes)[q16Offset(row, cell, rp.rest)] =
                        uint16_t(q16Code(values[cell], lo, hi));
            GroupMemoryBarrierWithGroupSync();
        }
        return;
    }
    if (operation == 0) { // fp32 float4 swizzle -> Q16 values
        uint row = gid.x * 256 + lane;
        float values[45];
        float lo = 1e30f, hi = -1e30f;
        if (row < p.count) {
            [unroll] for (uint c = 0; c < destinationRest * 3; ++c) {
                float v = readCell(p, row, c);
                values[c] = v;
                lo = minNumber(lo, v);
                hi = maxNumber(hi, v);
            }
        }
        lows[lane] = lo; highs[lane] = hi;
        GroupMemoryBarrierWithGroupSync();
        for (uint stride = 128; stride > 0; stride >>= 1) {
            if (lane < stride) { lows[lane] = minNumber(lows[lane], lows[lane + stride]); highs[lane] = maxNumber(highs[lane], highs[lane + stride]); }
            GroupMemoryBarrierWithGroupSync();
        }
        lo = lows[0]; hi = highs[0];
        if (lo > hi) { lo = 0; hi = 0; }
        if (lane == 0) { ((Ptr<float, Access.ReadWrite>)p.destinationBounds)[gid.x * 2] = lo; ((Ptr<float, Access.ReadWrite>)p.destinationBounds)[gid.x * 2 + 1] = hi; }
        if (row < ((p.count + 31) / 32) * 32) {
            [unroll] for (uint c = 0; c < destinationRest * 3; ++c) {
                uint code = row < p.count ? q16Code(values[c], lo, hi) : 0;
                ((Ptr<uint16_t, Access.ReadWrite>)p.destination)[shOffset(row, c, destinationRest, true)] = uint16_t(code);
            }
        }
        return;
    }
    if (operation == 1) { // Q16 -> fp32 float4 swizzle
        uint width = ((sourceRest * 3 + 3) / 4) * 4;
        uint total = ((p.count + 31) / 32) * 32 * width;
        for (uint i = gid.x * 256 + lane; i < total; i += groups.x * 256) {
            uint row = (i / (width * 32)) * 32 + (i / 4) % 32;
            uint c = ((i / 128) % (width / 4)) * 4 + i % 4;
            float v = c < sourceRest * 3 ? readCell(p, row, c) : 0.0f;
            ((Ptr<float, Access.ReadWrite>)p.destination)[i] = v;
        }
        return;
    }
    if (operation == 2) { // block id = float32 floor(row / 256)
        for (uint i = gid.x * 256 + lane; i < p.count; i += groups.x * 256)
            ((Ptr<float, Access.ReadWrite>)p.destination)[i] = floor(float(((Ptr<int64_t, Access.Read>)p.source)[i]) * (1.0f / 256.0f));
        return;
    }
    if (operation == 3) { // sorted block runs
        if (gid.x != 0 || lane != 0) return;
        Ptr<float, Access.Read> input = (Ptr<float, Access.Read>)p.source;
        Ptr<int, Access.ReadWrite> ids = (Ptr<int, Access.ReadWrite>)p.destination;
        Ptr<int, Access.ReadWrite> offsets = (Ptr<int, Access.ReadWrite>)p.destinationBounds;
        Ptr<int, Access.ReadWrite> runCount = (Ptr<int, Access.ReadWrite>)p.sourceBounds;
        uint runs = 0;
        for (uint i = 0; i < p.count; ++i) {
            if (i == 0 || input[i] != input[i - 1]) { ids[runs] = int(input[i]); offsets[runs] = int(i); ++runs; }
        }
        runCount[0] = int(runs);
        return;
    }
    if (operation == 4) { // decode canonical scalar range
        uint64_t first = p.sourceOffset;
        uint64_t count = p.destinationOffset;
        uint destinationRestCount = p.destinationRows;
        uint layoutRestCount = p.sourceRows;
        for (uint i = gid.x * 256 + lane; i < count; i += groups.x * 256) {
            uint64_t canonical = first + i;
            uint row = uint(canonical / (uint64_t(destinationRestCount) * 3));
            uint cell = uint(canonical % (uint64_t(destinationRestCount) * 3));
            float value = 0.0f;
            if (row < p.count && cell < layoutRestCount * 3) {
                if (sourceFormat == 3) {
                    Parameters source = p;
                    source.sourceRows = p.count;
                    value = readCell(source, row, cell);
                }
                else if (sourceFormat == 0) value = ((Ptr<float, Access.Read>)p.source)[row * layoutRestCount * 3 + cell];
                else if (sourceFormat == 2) value = f16tof32(((Ptr<uint16_t, Access.Read>)p.source)[shOffset(row, cell, layoutRestCount, false)]);
                else value = ((Ptr<float, Access.Read>)p.source)[shOffset(row, cell, layoutRestCount, false)];
            }
            ((Ptr<float, Access.ReadWrite>)p.destination)[i] = value;
        }
        return;
    }
    if (operation == 5) { // zero selected fp32 swizzled rows
        uint rest = sourceRest;
        for (uint i = gid.x * 256 + lane; i < p.count; i += groups.x * 256) {
            uint row = uint(((Ptr<int, Access.Read>)p.indices)[i]);
            [unroll] for (uint c = 0; c < ((rest * 3 + 3) / 4) * 4; ++c)
                ((Ptr<float, Access.ReadWrite>)p.destination)[shOffset(row, c, rest, false)] = 0.0f;
        }
        return;
    }
    if (operation == 6) { // gather swizzled float or byte rows
        uint srcRest = sourceRest, dstRest = destinationRest;
        uint srcWidth = sourceFormat == 0 ? srcRest * 3 : ((srcRest * 3 + 3) / 4) * 4;
        uint dstWidth = destinationFormat == 0 ? dstRest * 3 : ((dstRest * 3 + 3) / 4) * 4;
        uint logical = p.count * dstWidth;
        for (uint i = gid.x * 256 + lane; i < logical; i += groups.x * 256) {
            uint outRow = i / dstWidth, c = i % dstWidth;
            uint srcRow = uint(indexAt(p, outRow));
            uint dstRow = p.destinationOffset + outRow;
            uint srcOff = sourceFormat == 0 ? srcRow * srcWidth + c : shOffset(srcRow, c, srcRest, false);
            uint dstOff = destinationFormat == 0 ? dstRow * dstWidth + c : shOffset(dstRow, c, dstRest, false);
            if (sourceFormat == 4) {
                ((Ptr<uint8_t, Access.ReadWrite>)p.destination)[dstOff] = c < srcWidth
                    ? ((Ptr<uint8_t, Access.Read>)p.source)[srcOff] : uint8_t(0);
            } else {
                ((Ptr<float, Access.ReadWrite>)p.destination)[dstOff] = c < srcWidth
                    ? ((Ptr<float, Access.Read>)p.source)[srcOff] : 0.0f;
            }
        }
        return;
    }
    if (operation == 7 || operation == 8 || operation == 9) { // canonical gather / append / scatter
        uint srcRest = sourceRest, dstRest = destinationRest;
        uint width = operation == 7 ? dstRest*3 : ((dstRest*3+3)/4)*4;
        uint total = p.count * width;
        for (uint i = gid.x * 256 + lane; i < total; i += groups.x * 256) {
            uint row = i / width;
            uint cell = i % width;
            uint srcRow = operation == 7 ? uint(((Ptr<int64_t, Access.Read>)p.indices)[row]) : row;
            uint dstRow = operation == 7 ? row : operation == 8 ? p.destinationOffset + row : uint(((Ptr<int, Access.Read>)p.indices)[row]);
            if (operation == 7) {
                ((Ptr<float, Access.ReadWrite>)p.destination)[i] = cell < srcRest * 3
                    ? ((Ptr<float, Access.Read>)p.source)[shOffset(srcRow, cell, srcRest, false)] : 0.0f;
            } else {
                ((Ptr<float, Access.ReadWrite>)p.destination)[shOffset(dstRow, cell, dstRest, false)] = cell < srcRest*3 ? ((Ptr<float, Access.Read>)p.source)[row*srcRest*3+cell] : 0.0f;
            }
        }
        return;
    }
    if (operation == 10) { // exact byte fill
        Ptr<uint8_t, Access.ReadWrite> dst = (Ptr<uint8_t, Access.ReadWrite>)p.destination;
        uint value = p.sourceOffset & 255u;
        for (uint i = gid.x * 256 + lane; i < p.count; i += groups.x * 256) dst[i] = uint8_t(value);
        return;
    }
    if (operation == 11) { // copy a logical part into the arena view
        Ptr<uint8_t, Access.Read> src = (Ptr<uint8_t, Access.Read>)p.source;
        Ptr<uint8_t, Access.ReadWrite> dst = (Ptr<uint8_t, Access.ReadWrite>)p.destination;
        for (uint i = gid.x * 256 + lane; i < p.count; i += groups.x * 256)
            dst[p.destinationOffset + i] = src[i];
        return;
    }
}
