// One task is 256 swizzled SH3 points that share a super-cluster. Each point
// is scored only against that super's four nearest member lists.
static const uint LFS_LOCAL_SIZE_X = 256;
static const uint LFS_PUSH_CONSTANT_SIZE = 88;

struct Parameters {
    uint64_t sh;
    uint64_t centroids;
    uint64_t norms;
    uint64_t labels;
    uint64_t sorted;
    uint64_t groupOffsets;
    uint64_t taskOffsets;
    uint64_t members;
    uint64_t memberOffsets;
    uint64_t nearest;
    uint n;
    uint slots;
};

uint sh_at(uint p, uint slot, uint slots) {
    return (p / 32u) * (slots * 32u) + slot * 32u + (p % 32u);
}
float sh_dim(Ptr<float, Access.Read> data, uint p, uint dim, uint slots) {
    return data[sh_at(p, dim / 4u, slots) * 4u + (dim % 4u)];
}

groupshared float tile[32 * 45];
groupshared float tileNorm[32];
groupshared int tileId[32];

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform Parameters p) {
    const uint3 groups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
    Ptr<float, Access.Read> sh = (Ptr<float, Access.Read>)p.sh;
    Ptr<float, Access.Read> centroids = (Ptr<float, Access.Read>)p.centroids;
    Ptr<float, Access.Read> norms = (Ptr<float, Access.Read>)p.norms;
    Ptr<int, Access.ReadWrite> labels = (Ptr<int, Access.ReadWrite>)p.labels;
    Ptr<int, Access.Read> sorted = (Ptr<int, Access.Read>)p.sorted;
    Ptr<int, Access.Read> groupOffsets = (Ptr<int, Access.Read>)p.groupOffsets;
    Ptr<int, Access.Read> taskOffsets = (Ptr<int, Access.Read>)p.taskOffsets;
    Ptr<int, Access.Read> members = (Ptr<int, Access.Read>)p.members;
    Ptr<int, Access.Read> memberOffsets = (Ptr<int, Access.Read>)p.memberOffsets;
    Ptr<int, Access.Read> nearest = (Ptr<int, Access.Read>)p.nearest;
    const uint numTasks = uint(taskOffsets[256]);
    for (uint task = gid.x; task < numTasks; task += groups.x) {
        uint lo = 0;
        uint hi = 256;
        while (lo + 1 < hi) {
            const uint mid = (lo + hi) >> 1;
            if (uint(taskOffsets[mid]) <= task) lo = mid;
            else hi = mid;
        }
        const uint group = lo;
        const uint begin = uint(groupOffsets[group]);
        const uint end = uint(groupOffsets[group + 1]);
        const uint local = (task - uint(taskOffsets[group])) * 256u + tid.x;
        const uint sortedPos = begin + local;
        const bool valid = sortedPos < end && sortedPos < p.n;
        const uint point = valid ? uint(sorted[sortedPos]) : 0;
        float vals[45];
        [ForceUnroll] for (uint d = 0; d < 45; ++d)
            vals[d] = valid ? sh_dim(sh, point, d, p.slots) : 0;
        float best = 1e30;
        int bestId = 0x7fffffff;
        uint starts[4], lengths[4];
        uint count = 0;
        [ForceUnroll] for (uint a = 0; a < 4u; ++a) {
            const uint super = uint(nearest[group * 4u + a]);
            starts[a] = uint(memberOffsets[super]);
            lengths[a] = uint(memberOffsets[super + 1u]) - starts[a];
            count += lengths[a];
        }
        const uint tiles = (count + 31u) / 32u;
        for (uint t = 0; t < tiles; ++t) {
            if (tid.x < 32) {
                uint cursor = t * 32u + tid.x;
                int id = -1;
                [ForceUnroll] for (uint a = 0; a < 4u; ++a) {
                    if (cursor < lengths[a]) { id = members[starts[a] + cursor]; break; }
                    cursor -= lengths[a];
                }
                tileId[tid.x] = id;
                tileNorm[tid.x] = id >= 0 ? norms[uint(id)] : 0;
            }
            GroupMemoryBarrierWithGroupSync();
            for (uint i = tid.x; i < 32u * 45u; i += 256u) {
                const uint c = i / 45u;
                const uint d = i - c * 45u;
                const int id = tileId[c];
                tile[c * 45u + d] = id >= 0 ? centroids[uint(id) * 45u + d] : 0;
            }
            GroupMemoryBarrierWithGroupSync();
            if (valid) {
                [ForceUnroll] for (uint c = 0; c < 32; ++c) {
                    const int id = tileId[c];
                    if (id < 0) continue;
                    float dot = 0;
                    [ForceUnroll] for (uint d = 0; d < 45; ++d)
                        dot = fma(vals[d], tile[c * 45u + d], dot);
                    const float dist = fma(-2.0, dot, tileNorm[c]);
                    if (dist < best || (dist == best && id < bestId)) {
                        best = dist;
                        bestId = id;
                    }
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
        if (valid) labels[point] = bestId;
    }
}
