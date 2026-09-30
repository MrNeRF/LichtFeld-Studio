/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "lpips_common.slangh"
// A 4x16 output tile reads each 16-pixel matrix column directly from the
// transposed halo, avoiding a shared-memory repack for every kernel tap.
// Padded channel rows avoid bank conflicts during cooperative matrix loads.
groupshared half matrixWeights[2560];
groupshared half halo[4320];
groupshared float matrixOutputs[4096];
[numthreads(512,1,1)]
void main(uint3 group : SV_GroupID, uint lane : SV_GroupIndex, uniform Parameters p) {
    uint tile = uint(p.g / 512u) + group.x;
    uint tilesX = (p.out_width + 15u) / 16u;
    uint tilesY = (p.out_height + 3u) / 4u;
    uint spatialTiles = tilesX * tilesY;
    uint channelTiles = (p.out_channels + 63u) / 64u;
    uint spatial = tile % spatialTiles;
    uint xBase = (spatial % tilesX) * 16u;
    uint yBase = (spatial / tilesX) * 4u;
    uint channelBase = ((tile / spatialTiles) % channelTiles) * 64u;
    uint batch = tile / (spatialTiles * channelTiles);
    uint warp = lane / 32u;
    uint ph = pair_lo(p.padding), pw = pair_hi(p.padding);
    bool replicate = ((p.flags >> 8) & 1u) != 0u;
    var acc = linalg.CoopMat<float, MemoryScope.Subgroup, 16, 16, linalg.CoopMatMatrixUse.MatrixAccumulator>(0.0);
    for (uint base = 0; base < p.channels; base += 32u) {
        for (uint t = lane; t < 3456u; t += 512u) {
            uint ic = base + t / 108u, pixel = t % 108u;
            int y = int(yBase + pixel / 18u) - int(ph);
            int x = int(xBase + pixel % 18u) - int(pw);
            if (replicate) { y = clamp(y, 0, int(p.height)-1); x = clamp(x, 0, int(p.width)-1); }
            half value = half(0.0f);
            if (ic < p.channels && y >= 0 && x >= 0 && y < int(p.height) && x < int(p.width))
                value = ((Ptr<half,Access.Read>)p.a)[((batch*p.channels+ic)*p.height+uint(y))*p.width+uint(x)];
            halo[pixel*40u+t/108u] = value;
        }
        GroupMemoryBarrierWithGroupSync();
        for (uint tap = 0; tap < 9u; ++tap) {
            for (uint t = lane; t < 2048u; t += 512u) {
                uint oc = channelBase + t / 32u, ic = base + t % 32u;
                half value = half(0.0f);
                if (oc < p.out_channels && ic < p.channels)
                    value = p.e != 0
                        ? ((Ptr<half,Access.Read>)p.e)[(tap*p.out_channels+oc)*p.channels+ic]
                        : ((Ptr<half,Access.Read>)p.b)[(oc*p.channels+ic)*9u+tap];
                matrixWeights[(t/32u)*40u+t%32u] = value;
            }
            GroupMemoryBarrierWithGroupSync();
            for (uint k = 0u; k < 32u; k += 16u) {
                var a = linalg.CoopMat<half, MemoryScope.Subgroup, 16, 16, linalg.CoopMatMatrixUse.MatrixA>.Load<linalg.CoopMatMatrixLayout.RowMajor,2560>(matrixWeights, (warp/4u)*640u+k, 40u);
                var b = linalg.CoopMat<half, MemoryScope.Subgroup, 16, 16, linalg.CoopMatMatrixUse.MatrixB>.Load<linalg.CoopMatMatrixLayout.ColumnMajor,4320>(halo, ((warp%4u+tap/3u)*18u+tap%3u)*40u+k, 40u);
                acc = linalg.coopMatMulAdd<float,false,half,half,float,MemoryScope.Subgroup,16,16,16>(a,b,acc);
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }
    acc.Store<linalg.CoopMatMatrixLayout.RowMajor,4096>(matrixOutputs,(warp/4u)*1024u+(warp%4u)*16u,64u);
    GroupMemoryBarrierWithGroupSync();
    for (uint t = lane; t < 4096u; t += 512u) {
        uint oc = channelBase+t/64u, pixel = t%64u;
        uint y = yBase+pixel/16u, x = xBase+pixel%16u;
        if (oc < p.out_channels && y < p.out_height && x < p.out_width) {
            float value = matrixOutputs[t];
            if (p.c != 0) value += load_half(p.c,oc);
            store_half(p.d,((batch*p.out_channels+oc)*p.out_height+y)*p.out_width+x,activation(value,p.flags&255u));
        }
    }
}
