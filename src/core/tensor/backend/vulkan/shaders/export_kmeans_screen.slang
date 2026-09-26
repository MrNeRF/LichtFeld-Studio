// Screened SH3 assignment. Half cooperative-matrix products only reject
// distant centroids; every survivor is scored with ordered FP32 FMAs.
static const uint LFS_LOCAL_SIZE_X = 32;
static const uint LFS_PUSH_CONSTANT_SIZE = 72;

struct Parameters {
    uint64_t sh;
    uint64_t centroids;
    uint64_t halfCentroids;
    uint64_t norms;
    uint64_t labels;
    uint64_t pointOrder;
    uint64_t centroidOrder;
    uint n;
    uint k;
    uint slots;
    uint haveLabels;
};

uint sh_at(uint p, uint slot, uint slots) {
    return (p / 32u) * (slots * 32u) + slot * 32u + (p % 32u);
}
float sh_dim(Ptr<float, Access.Read> data, uint p, uint dim, uint slots) {
    return data[sh_at(p, dim / 4u, slots) * 4u + (dim % 4u)];
}

uint norm_bound(Ptr<float, Access.Read> norms, Ptr<int, Access.Read> order, uint k, float value, bool upper) {
    uint lo = 0, hi = k;
    while (lo < hi) {
        const uint mid = lo + (hi - lo) / 2u;
        const float norm = norms[uint(order[mid])];
        if (norm < value || (upper && norm == value)) lo = mid + 1u;
        else hi = mid;
    }
    return lo;
}

groupshared half packed[768];
groupshared float approx[256];

[numthreads(32, 1, 1)]
[WaveSize(32)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform Parameters p) {
    const uint3 groups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
    const uint warp = tid.x >> 5;
    const uint lane = tid.x & 31u;
    const uint pointLane = lane >> 1;
    const uint ntiles = (p.n + 15u) / 16u;
    const uint warps = groups.x;
    const uint iters = warps == 0 ? 0u : (ntiles + warps - 1u) / warps;
    const uint tile0 = gid.x + warp;
    Ptr<float, Access.Read> sh = (Ptr<float, Access.Read>)p.sh;
    Ptr<float, Access.ReadWrite> centroids = (Ptr<float, Access.ReadWrite>)p.centroids;
    Ptr<half, Access.ReadWrite> halfCentroids = (Ptr<half, Access.ReadWrite>)p.halfCentroids;
    Ptr<float, Access.Read> norms = (Ptr<float, Access.Read>)p.norms;
    Ptr<int, Access.ReadWrite> labels = (Ptr<int, Access.ReadWrite>)p.labels;
    Ptr<int, Access.Read> pointOrder = (Ptr<int, Access.Read>)p.pointOrder;
    Ptr<int, Access.Read> centroidOrder = (Ptr<int, Access.Read>)p.centroidOrder;
    const float lastNorm = norms[uint(centroidOrder[p.k - 1u])];
    const bool normsSafe = lastNorm >= 0 && lastNorm < 4.0e9;
    for (uint iter = 0; iter < iters; ++iter) {
        const uint tile = tile0 + iter * warps;
        const bool active = tile < ntiles;
        const uint point0 = tile * 16u;
        if (lane < 16) {
            for (uint d = 0; d < 48u; ++d) {
                half value = half(0);
                if (active && point0 + lane < p.n && d < 45u)
                    value = half(sh_dim(sh, uint(pointOrder[point0 + lane]), d, p.slots));
                packed[(warp * 16u + lane) * 48u + d] = value;
            }
        }
        GroupMemoryBarrierWithGroupSync();
        if (active) {
            const uint point = point0 + pointLane < p.n ? uint(pointOrder[point0 + pointLane]) : 0;
            float best = 1e30;
            int bestId = int(p.k);
            float vals[45];
            float pointNorm = 0;
            int safe = 1;
            [ForceUnroll] for (uint d = 0; d < 45; ++d) vals[d] = 0;
            if (point0 + pointLane < p.n) {
                [ForceUnroll] for (uint d = 0; d < 45; ++d) {
                    const float v = sh_dim(sh, point, d, p.slots);
                    vals[d] = v;
                    pointNorm = fma(v, v, pointNorm);
                    if (!isfinite(v) || abs(v) > 65000.0) safe = 0;
                }
                if (p.haveLabels != 0) {
                    const int seed = labels[point];
                    if (seed >= 0 && uint(seed) < p.k) {
                        float dot = 0;
                        [ForceUnroll] for (uint d = 0; d < 45; ++d)
                            dot = fma(vals[d], centroids[uint(seed) * 45u + d], dot);
                        best = fma(-2.0, dot, norms[uint(seed)]);
                        bestId = seed;
                    }
                }
            }
            uint first = p.k, last = 0;
            if (point0 + pointLane < p.n) {
                first = 0;
                last = p.k;
                if (safe != 0 && normsSafe && pointNorm < 4.0e9 && isfinite(best) && best < 1e30) {
                    // A winner satisfies ny - 2*sqrt(nx*ny) - m*(nx+ny) <= best + eta.
                    const float m = 0.0011;
                    const float a = 1.0 - m;
                    const float root = sqrt(pointNorm);
                    const float radius = sqrt(m * (2.0 - m) * pointNorm + a * max(pointNorm + best + 1e-8, 0.0));
                    const float pad = 1e-5 * (root + radius) + 1e-10;
                    const float lower = max(0.0, (root - radius - pad) / a);
                    const float upper = (root + radius + pad) / a;
                    first = norm_bound(norms, centroidOrder, p.k, lower * lower, false);
                    last = norm_bound(norms, centroidOrder, p.k, upper * upper, true);
                }
            }
            const uint firstTile = WaveActiveMin(first) / 16u;
            const uint lastTile = (WaveActiveMax(last) + 15u) / 16u;
            var a0 = linalg.CoopMat<half, MemoryScope.Subgroup, 16, 16, linalg.CoopMatMatrixUse.MatrixA>.Load<linalg.CoopMatMatrixLayout.RowMajor, 768>(packed, warp * 768u, 48u);
            var a1 = linalg.CoopMat<half, MemoryScope.Subgroup, 16, 16, linalg.CoopMatMatrixUse.MatrixA>.Load<linalg.CoopMatMatrixLayout.RowMajor, 768>(packed, warp * 768u + 16u, 48u);
            var a2 = linalg.CoopMat<half, MemoryScope.Subgroup, 16, 16, linalg.CoopMatMatrixUse.MatrixA>.Load<linalg.CoopMatMatrixLayout.RowMajor, 768>(packed, warp * 768u + 32u, 48u);
            for (uint t = firstTile; t < lastTile; ++t) {
                var acc = linalg.CoopMat<float, MemoryScope.Subgroup, 16, 16, linalg.CoopMatMatrixUse.MatrixAccumulator>(0.0);
                var b0 = linalg.CoopMat<half, MemoryScope.Subgroup, 16, 16, linalg.CoopMatMatrixUse.MatrixB>.Load<linalg.CoopMatMatrixLayout.ColumnMajor>(halfCentroids, t * 16u * 48u, 48u);
                acc = linalg.coopMatMulAdd<float, false, half, half, float, MemoryScope.Subgroup, 16, 16, 16>(a0, b0, acc);
                var b1 = linalg.CoopMat<half, MemoryScope.Subgroup, 16, 16, linalg.CoopMatMatrixUse.MatrixB>.Load<linalg.CoopMatMatrixLayout.ColumnMajor>(halfCentroids, t * 16u * 48u + 16u, 48u);
                acc = linalg.coopMatMulAdd<float, false, half, half, float, MemoryScope.Subgroup, 16, 16, 16>(a1, b1, acc);
                var b2 = linalg.CoopMat<half, MemoryScope.Subgroup, 16, 16, linalg.CoopMatMatrixUse.MatrixB>.Load<linalg.CoopMatMatrixLayout.ColumnMajor>(halfCentroids, t * 16u * 48u + 32u, 48u);
                acc = linalg.coopMatMulAdd<float, false, half, half, float, MemoryScope.Subgroup, 16, 16, 16>(a2, b2, acc);
                acc.Store<linalg.CoopMatMatrixLayout.ColumnMajor, 256>(approx, warp * 256u, 16u);
                // Matrix elements may be stored by other lanes than the ones that read them.
                GroupMemoryBarrierWithGroupSync();
                if (point0 + pointLane < p.n) {
                    for (uint c = lane & 1u; c < 16u; c += 2u) {
                        const uint sortedId = t * 16u + c;
                        if (sortedId < first || sortedId >= last) continue;
                        const uint id = uint(centroidOrder[sortedId]);
                        const float estimate = fma(-2.0, approx[warp * 256u + c * 16u + pointLane], norms[id]);
                        const float margin = 0.0011 * (pointNorm + norms[id]) + 1e-8;
                        if (safe == 0 || !(norms[id] >= 0.0 && norms[id] < 4.0e9) || !isfinite(estimate) || estimate <= best + margin) {
                            float dot = 0;
                            [ForceUnroll] for (uint d = 0; d < 45; ++d)
                                dot = fma(vals[d], centroids[id * 45u + d], dot);
                            const float dist = fma(-2.0, dot, norms[id]);
                            if (dist < best || (dist == best && int(id) < bestId)) {
                                best = dist;
                                bestId = int(id);
                            }
                        }
                    }
                }
                // The next tile overwrites approx.
                GroupMemoryBarrierWithGroupSync();
            }
            const float otherBest = WaveReadLaneAt(best, lane ^ 1u);
            const int otherId = WaveReadLaneAt(bestId, lane ^ 1u);
            if (otherBest < best || (otherBest == best && otherId < bestId))
                bestId = otherId;
            if ((lane & 1u) == 0 && point0 + pointLane < p.n)
                labels[point] = bestId;
        }
        GroupMemoryBarrierWithGroupSync();
    }
}
