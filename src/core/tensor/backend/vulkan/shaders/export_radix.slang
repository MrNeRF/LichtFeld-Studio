// Stable 4-bit LSD radix over uint64 keys with int32 payloads.
// Phase 0 writes per-block digit counts. Phase 1 scatters using an exclusive
// prefix of those counts in digit-major order (digit * blocks + block).
static const uint LFS_LOCAL_SIZE_X = 256;
static const uint LFS_PUSH_CONSTANT_SIZE = 64;
static const uint kElems = 8;

[SpecializationConstant]
const uint kPhase = 0;

struct Parameters {
    uint64_t keysIn;
    uint64_t keysOut;
    uint64_t indexIn;
    uint64_t indexOut;
    uint64_t counts;
    uint64_t scratch;
    uint n;
    uint blocks;
    uint shift;
    uint padding;
};

groupshared uint scanIn[256];
groupshared uint scanOut[256];

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform Parameters p) {
    const uint3 groups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
    // Exclusive prefix of the digit-major block counts. Chunk totals sit past ncounts.
    if (kPhase == 2) {
        const uint ncounts = p.blocks * 16u;
        const uint nchunks = (ncounts + 255u) / 256u;
        Ptr<uint, Access.Read> counts = (Ptr<uint, Access.Read>)p.counts;
        Ptr<uint, Access.ReadWrite> scratch = (Ptr<uint, Access.ReadWrite>)p.scratch;
        if (p.padding == 1) {
            if (gid.x == 0 && tid.x == 0) {
                uint sum = 0;
                for (uint c = 0; c < nchunks; ++c) {
                    const uint value = scratch[ncounts + c];
                    scratch[ncounts + c] = sum;
                    sum += value;
                }
            }
            return;
        }
        if (p.padding == 2) {
            for (uint idx = gid.x * 256u + tid.x; idx < ncounts; idx += groups.x * 256u)
                scratch[idx] += scratch[ncounts + idx / 256u];
            return;
        }
        for (uint chunk = gid.x; chunk < nchunks; chunk += groups.x) {
            const uint idx = chunk * 256u + tid.x;
            const uint value = idx < ncounts ? counts[idx] : 0u;
            scanIn[tid.x] = value;
            GroupMemoryBarrierWithGroupSync();
            [ForceUnroll] for (uint offset = 1; offset < 256u; offset <<= 1) {
                const uint add = tid.x >= offset ? scanIn[tid.x - offset] : 0u;
                GroupMemoryBarrierWithGroupSync();
                scanOut[tid.x] = scanIn[tid.x] + add;
                GroupMemoryBarrierWithGroupSync();
                scanIn[tid.x] = scanOut[tid.x];
                GroupMemoryBarrierWithGroupSync();
            }
            if (idx < ncounts)
                scratch[idx] = scanIn[tid.x] - value;
            if (tid.x == 255)
                scratch[ncounts + chunk] = scanIn[255];
            GroupMemoryBarrierWithGroupSync();
        }
        return;
    }
    Ptr<uint64_t, Access.Read> kin = (Ptr<uint64_t, Access.Read>)p.keysIn;
    Ptr<int, Access.Read> iin = (Ptr<int, Access.Read>)p.indexIn;
    uint localIndex[8];
    uint localDigit[8];
    for (uint block = gid.x; block < p.blocks; block += groups.x) {
        uint mine[16];
        [ForceUnroll] for (uint d = 0; d < 16; ++d) mine[d] = 0;
        [ForceUnroll] for (uint e = 0; e < kElems; ++e) {
            const uint local = tid.x * kElems + e;
            const uint index = block * (256u * kElems) + local;
            uint digit = 0;
            if (index < p.n) {
                const uint64_t key = kin[index];
                digit = uint((key >> p.shift) & 15u);
                mine[digit] += 1;
            }
            localIndex[e] = index;
            localDigit[e] = digit;
        }
        if (kPhase == 0) {
            Ptr<uint, Access.ReadWrite> cout = (Ptr<uint, Access.ReadWrite>)p.counts;
            [ForceUnroll] for (uint d = 0; d < 16; ++d) {
                scanIn[tid.x] = mine[d];
                GroupMemoryBarrierWithGroupSync();
                [ForceUnroll] for (uint offset = 1; offset < 256u; offset <<= 1) {
                    const uint add = tid.x >= offset ? scanIn[tid.x - offset] : 0;
                    GroupMemoryBarrierWithGroupSync();
                    scanOut[tid.x] = scanIn[tid.x] + add;
                    GroupMemoryBarrierWithGroupSync();
                    scanIn[tid.x] = scanOut[tid.x];
                    GroupMemoryBarrierWithGroupSync();
                }
                if (tid.x == 255)
                    cout[d * p.blocks + block] = scanIn[255];
                GroupMemoryBarrierWithGroupSync();
            }
        } else {
            Ptr<uint, Access.Read> prefix = (Ptr<uint, Access.Read>)p.counts;
            Ptr<uint64_t, Access.ReadWrite> kout = (Ptr<uint64_t, Access.ReadWrite>)p.keysOut;
            Ptr<int, Access.ReadWrite> iout = (Ptr<int, Access.ReadWrite>)p.indexOut;
            uint base[16];
            [ForceUnroll] for (uint d = 0; d < 16; ++d) {
                scanIn[tid.x] = mine[d];
                GroupMemoryBarrierWithGroupSync();
                [ForceUnroll] for (uint offset = 1; offset < 256u; offset <<= 1) {
                    const uint add = tid.x >= offset ? scanIn[tid.x - offset] : 0;
                    GroupMemoryBarrierWithGroupSync();
                    scanOut[tid.x] = scanIn[tid.x] + add;
                    GroupMemoryBarrierWithGroupSync();
                    scanIn[tid.x] = scanOut[tid.x];
                    GroupMemoryBarrierWithGroupSync();
                }
                base[d] = prefix[d * p.blocks + block] + scanIn[tid.x] - mine[d];
                GroupMemoryBarrierWithGroupSync();
            }
            uint seen[16];
            [ForceUnroll] for (uint d = 0; d < 16; ++d) seen[d] = 0;
            [ForceUnroll] for (uint e = 0; e < kElems; ++e) {
                const uint index = localIndex[e];
                if (index < p.n) {
                    const uint digit = localDigit[e];
                    const uint at = base[digit] + seen[digit];
                    seen[digit] += 1;
                    kout[at] = kin[index];
                    iout[at] = iin[index];
                }
            }
        }
    }
}
