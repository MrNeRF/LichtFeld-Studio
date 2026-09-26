static const uint LFS_LOCAL_SIZE_X = 256;
static const uint LFS_PUSH_CONSTANT_SIZE = 80;

[SpecializationConstant]
const uint kPhase = 0;

#include "export_kmeans_common.slangh"

groupshared float tile[32 * 48];
groupshared float tileNorm[32];

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform Parameters p) {
    const uint3 groups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
    Ptr<float, Access.Read> sh = (Ptr<float, Access.Read>)p.sh;
    Ptr<float, Access.ReadWrite> centroids = (Ptr<float, Access.ReadWrite>)p.centroids;
    Ptr<int, Access.Read> indices = (Ptr<int, Access.Read>)p.indices;
    Ptr<int, Access.ReadWrite> labels = (Ptr<int, Access.ReadWrite>)p.labels;
    Ptr<float, Access.ReadWrite> sums = (Ptr<float, Access.ReadWrite>)p.sums;
    Ptr<int, Access.ReadWrite> counts = (Ptr<int, Access.ReadWrite>)p.counts;
    Ptr<float, Access.Read> norms = (Ptr<float, Access.Read>)p.norms;
    if (kPhase == 17) {
        Ptr<uint64_t, Access.ReadWrite> keys = (Ptr<uint64_t, Access.ReadWrite>)p.sums;
        for (uint i = gid.x * 256u + tid.x; i < p.n; i += groups.x * 256u) {
            keys[i] = uint(indices[i]);
            counts[i] = int(i);
        }
    } else if (kPhase == 18) {
        Ptr<uint64_t, Access.Read> keys = (Ptr<uint64_t, Access.Read>)p.sh;
        for (uint group = gid.x * 256u + tid.x; group < p.k; group += groups.x * 256u) {
            uint lo = 0, hi = p.n;
            while (lo < hi) {
                const uint mid = lo + (hi - lo) / 2u;
                if (keys[mid] < uint64_t(group)) lo = mid + 1u;
                else hi = mid;
            }
            labels[group] = int(lo);
        }
    } else if (kPhase == 19) {
        if (gid.x == 0 && tid.x == 0) {
            int total = 0;
            labels[0] = 0;
            for (uint group = 0; group < 256u; ++group) {
                total += (indices[group + 1u] - indices[group] + 255) / 256;
                labels[group + 1u] = total;
            }
        }
    } else if (kPhase == 20) {
        for (uint group = gid.x * 256u + tid.x; group < p.k; group += groups.x * 256u) {
            float values[45];
            [ForceUnroll] for (uint d = 0; d < 45u; ++d) values[d] = centroids[group * 45u + d];
            float best[4];
            int ids[4];
            [ForceUnroll] for (uint a = 0; a < 4u; ++a) { best[a] = 1e30; ids[a] = 0; }
            for (uint other = 0; other < p.k; ++other) {
                float dot = 0;
                [ForceUnroll] for (uint d = 0; d < 45u; ++d)
                    dot = fma(values[d], centroids[other * 45u + d], dot);
                const float dist = fma(-2.0, dot, norms[other]);
                if (dist > best[3] || (dist == best[3] && int(other) >= ids[3])) continue;
                int at = 3;
                while (at > 0 && (dist < best[at - 1] || (dist == best[at - 1] && int(other) < ids[at - 1]))) {
                    best[at] = best[at - 1]; ids[at] = ids[at - 1]; --at;
                }
                best[at] = dist; ids[at] = int(other);
            }
            [ForceUnroll] for (uint a = 0; a < 4u; ++a) labels[group * 4u + a] = ids[a];
        }
    } else if (kPhase == 0) {
        for (uint c = gid.x * 256u + tid.x; c < p.k; c += groups.x * 256u) {
            const uint src = uint(indices[c]);
            for (uint d = 0; d < p.dims; ++d)
                centroids[c * p.dims + d] = sh_dim(sh, src, d, p.slots);
        }
    } else if (kPhase == 1) {
        for (uint c = gid.x * 256u + tid.x; c < p.k; c += groups.x * 256u) {
            float nrm = 0;
            for (uint d = 0; d < p.dims; ++d) {
                const float v = centroids[c * p.dims + d];
                nrm = fma(v, v, nrm);
            }
            ((Ptr<float, Access.ReadWrite>)p.norms)[c] = nrm;
        }
    } else if (kPhase == 2 || kPhase == 11 || kPhase == 16) {
        const uint dims = (kPhase == 11 || kPhase == 16) ? 45u : p.dims;
        for (uint base = 0; base < p.n; base += groups.x * 256u) {
            const uint point = base + gid.x * 256u + tid.x;
            float vals[45];
            if (kPhase == 11 || kPhase == 16) {
                [ForceUnroll] for (uint d = 0; d < 45u; ++d)
                    vals[d] = point < p.n ? (kPhase == 16 ? sh[point * 45u + d] : sh_dim(sh, point, d, p.slots)) : 0;
            }
            float best = 1e30;
            int bestId = int(p.k);
            const uint tiles = (p.k + 31u) / 32u;
            for (uint t = 0; t < tiles; ++t) {
                for (uint i = tid.x; i < 32u * dims; i += 256u) {
                    const uint c = i / dims;
                    const uint d = i - c * dims;
                    const uint global = t * 32u + c;
                    tile[c * 48u + d] = global < p.k ? centroids[global * dims + d] : 0;
                }
                if (tid.x < 32) {
                    const uint global = t * 32u + tid.x;
                    tileNorm[tid.x] = global < p.k ? norms[global] : 0;
                }
                GroupMemoryBarrierWithGroupSync();
                if (point < p.n) {
                    for (uint c = 0; c < 32u && t * 32u + c < p.k; ++c) {
                        float dot = 0;
                        if (kPhase == 11 || kPhase == 16) {
                            [ForceUnroll] for (uint d = 0; d < 45u; ++d)
                                dot = fma(vals[d], tile[c * 48u + d], dot);
                        } else {
                            for (uint d = 0; d < dims; ++d)
                                dot = fma(sh_dim(sh, point, d, p.slots), tile[c * 48u + d], dot);
                        }
                        const float dist = fma(-2.0, dot, tileNorm[c]);
                        const int id = int(t * 32u + c);
                        if (dist < best || (dist == best && id < bestId)) {
                            best = dist;
                            bestId = id;
                        }
                    }
                }
                GroupMemoryBarrierWithGroupSync();
            }
            if (point < p.n) labels[point] = bestId;
        }
    } else if (kPhase == 21) {
        // Sums over label-sorted runs, for devices without float atomics. indices holds
        // the sorted point order, labels the k + 1 run offsets; seed selects dense rows.
        for (uint c = gid.x * 256u + tid.x; c < p.k; c += groups.x * 256u) {
            const int begin = labels[c];
            const int end = labels[c + 1u];
            float acc[48];
            for (uint d = 0; d < p.dims; ++d) acc[d] = 0;
            for (int r = begin; r < end; ++r) {
                const uint i = uint(indices[r]);
                for (uint d = 0; d < p.dims; ++d)
                    acc[d] += p.seed == 0u ? sh_dim(sh, i, d, p.slots) : sh[i * p.dims + d];
            }
            for (uint d = 0; d < p.dims; ++d) sums[c * p.dims + d] = acc[d];
            counts[c] = end - begin;
        }
    } else if (kPhase == 4) {
        for (uint c = gid.x * 256u + tid.x; c < p.k; c += groups.x * 256u) {
            const int count = counts[c];
            if (count > 0) {
                for (uint d = 0; d < p.dims; ++d)
                    centroids[c * p.dims + d] = sums[c * p.dims + d] / float(count);
            } else {
                uint rng = p.seed ^ (c * 1664525u + 1013904223u);
                rng = rng * 1664525u + 1013904223u;
                const uint pick = p.n == 0 ? 0 : rng % p.n;
                for (uint d = 0; d < p.dims; ++d)
                    centroids[c * p.dims + d] = sh_dim(sh, pick, d, p.slots);
            }
        }
    } else if (kPhase == 13 || kPhase == 14) {
        Ptr<uint64_t, Access.ReadWrite> keys = (Ptr<uint64_t, Access.ReadWrite>)p.sums;
        Ptr<int, Access.ReadWrite> order = (Ptr<int, Access.ReadWrite>)p.counts;
        const uint count = kPhase == 13 ? p.n : p.k;
        for (uint i = gid.x * 256u + tid.x; i < count; i += groups.x * 256u) {
            float norm = 0;
            if (kPhase == 13) {
                [ForceUnroll] for (uint d = 0; d < 45u; ++d) {
                    const float v = sh_dim(sh, i, d, p.slots);
                    norm = fma(v, v, norm);
                }
            } else {
                norm = norms[i];
            }
            const uint bits = norm == 0 ? 0u : asuint(norm);
            keys[i] = uint64_t(kPhase == 13 ? bits >> 16u : bits);
            order[i] = int(i);
        }
    } else if (kPhase == 7 || kPhase == 9 || kPhase == 10) {
        // Dense [row, dims] points. Super-clusters are ordinary centroids, not swizzled SH.
        if (kPhase == 9) {
            for (uint c = gid.x * 256u + tid.x; c < p.k; c += groups.x * 256u) {
                const uint src = uint(indices[c]);
                for (uint d = 0; d < p.dims; ++d)
                    centroids[c * p.dims + d] = sh[src * p.dims + d];
            }
        } else if (kPhase == 10) {
            for (uint c = gid.x * 256u + tid.x; c < p.k; c += groups.x * 256u) {
                const int count = counts[c];
                if (count > 0) {
                    for (uint d = 0; d < p.dims; ++d)
                        centroids[c * p.dims + d] = sums[c * p.dims + d] / float(count);
                } else {
                    uint rng = p.seed ^ (c * 1664525u + 1013904223u);
                    rng = rng * 1664525u + 1013904223u;
                    const uint pick = p.n == 0 ? 0 : rng % p.n;
                    for (uint d = 0; d < p.dims; ++d)
                        centroids[c * p.dims + d] = sh[pick * p.dims + d];
                }
            }
        } else {
            for (uint base = 0; base < p.n; base += groups.x * 256u) {
                const uint point = base + gid.x * 256u + tid.x;
                float best = 1e30;
                int bestId = int(p.k);
                const uint tiles = (p.k + 31u) / 32u;
                for (uint t = 0; t < tiles; ++t) {
                    for (uint i = tid.x; i < 32u * p.dims; i += 256u) {
                        const uint c = i / p.dims;
                        const uint d = i - c * p.dims;
                        const uint global = t * 32u + c;
                        tile[c * 48u + d] = global < p.k ? centroids[global * p.dims + d] : 0;
                    }
                    if (tid.x < 32) {
                        const uint global = t * 32u + tid.x;
                        tileNorm[tid.x] = global < p.k ? norms[global] : 0;
                    }
                    GroupMemoryBarrierWithGroupSync();
                    if (point < p.n) {
                        for (uint c = 0; c < 32u && t * 32u + c < p.k; ++c) {
                            float dot = 0;
                            for (uint d = 0; d < p.dims; ++d)
                                dot = fma(sh[point * p.dims + d], tile[c * 48u + d], dot);
                            const float dist = fma(-2.0, dot, tileNorm[c]);
                            const int id = int(t * 32u + c);
                            if (dist < best || (dist == best && id < bestId)) {
                                best = dist;
                                bestId = id;
                            }
                        }
                    }
                    GroupMemoryBarrierWithGroupSync();
                }
                if (point < p.n) labels[point] = bestId;
            }
        }
    }
}
