/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "lpips_common.slangh"
groupshared half matrixWeights[4096];
groupshared half matrixInputs[4096];
groupshared float matrixOutputs[4096];
[numthreads(512,1,1)]
void main(uint3 group : SV_GroupID, uint lane : SV_GroupIndex, uniform Parameters p) {
    uint tile = uint(p.g) / 512u + group.x;
    uint plane = p.out_height * p.out_width;
    uint spatialTiles = (plane + 63u) / 64u;
    uint channelTiles = (p.out_channels + 63u) / 64u;
    uint spatialBase = (tile % spatialTiles) * 64u;
    uint channelBase = ((tile / spatialTiles) % channelTiles) * 64u;
    uint batch = tile / (spatialTiles * channelTiles);
    uint warp = lane / 32u;
    uint sh = pair_lo(p.stride), sw = pair_hi(p.stride);
    uint ph = pair_lo(p.padding), pw = pair_hi(p.padding);
    uint dh = pair_lo(p.dilation), dw = pair_hi(p.dilation);
    bool replicate = ((p.flags >> 8) & 1u) != 0u;
    uint reduction = p.channels * 9u;
    bool rgb = (p.flags & (1u << 9)) == 0u;
    var acc = linalg.CoopMat<float, MemoryScope.Subgroup, 16, 16, linalg.CoopMatMatrixUse.MatrixAccumulator>(0.0);
    for (uint base = 0; base < reduction; base += 64u) {
        for (uint t = lane; t < 4096u; t += 512u) {
            uint oc = channelBase + t / 64u, k = base + t % 64u;
            matrixWeights[t] = oc < p.out_channels && k < reduction ? ((Ptr<half,Access.Read>)p.b)[oc * reduction + k] : half(0.0f);
        }
        for (uint t = lane; t < 4096u; t += 512u) {
            uint k = base + t / 64u, pixel = spatialBase + t % 64u;
            float value = 0.0f;
            if (k < reduction && pixel < plane) {
                int y = int((pixel / p.out_width) * sh + ((k / 3u) % 3u) * dh) - int(ph);
                int x = int((pixel % p.out_width) * sw + (k % 3u) * dw) - int(pw);
                if (replicate) { y = clamp(y, 0, int(p.height)-1); x = clamp(x, 0, int(p.width)-1); }
                if (y >= 0 && x >= 0 && y < int(p.height) && x < int(p.width)) {
                    uint ic = k / 9u;
                    uint source = ((batch * p.channels + ic) * p.height + uint(y)) * p.width + uint(x);
                    if (rgb) {
                        value = ((Ptr<float,Access.Read>)p.a)[source];
                        if (p.flags >> 16 != 0u) value = 2.0f * value - 1.0f;
                        float shift = ic == 0u ? p.shift0 : ic == 1u ? p.shift1 : p.shift2;
                        float scale = ic == 0u ? p.scale0 : ic == 1u ? p.scale1 : p.scale2;
                        value = (value - shift) / scale;
                    } else {
                        matrixInputs[t] = ((Ptr<half,Access.Read>)p.a)[source];
                        continue;
                    }
                }
            }
            matrixInputs[t] = half(value);
        }
        GroupMemoryBarrierWithGroupSync();
        for (uint k = 0u; k < 64u; k += 16u) {
            var a = linalg.CoopMat<half, MemoryScope.Subgroup, 16, 16, linalg.CoopMatMatrixUse.MatrixA>.Load<linalg.CoopMatMatrixLayout.RowMajor,4096>(matrixWeights, (warp / 4u) * 1024u + k, 64u);
            var b = linalg.CoopMat<half, MemoryScope.Subgroup, 16, 16, linalg.CoopMatMatrixUse.MatrixB>.Load<linalg.CoopMatMatrixLayout.RowMajor,4096>(matrixInputs, k * 64u + (warp % 4u) * 16u, 64u);
            acc = linalg.coopMatMulAdd<float,false,half,half,float,MemoryScope.Subgroup,16,16,16>(a,b,acc);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    acc.Store<linalg.CoopMatMatrixLayout.RowMajor,4096>(matrixOutputs,(warp / 4u) * 1024u + (warp % 4u) * 16u,64u);
    GroupMemoryBarrierWithGroupSync();
    for (uint t = lane; t < 4096u; t += 512u) {
        uint oc = channelBase+t/64u, pixel = spatialBase+t%64u;
        if (oc < p.out_channels && pixel < plane) {
            float value = matrixOutputs[t];
            if (p.c != 0) value += load_half(p.c,oc);
            store_half(p.d,(batch*p.out_channels+oc)*plane+pixel,activation(value,p.flags&255u));
        }
    }
}
