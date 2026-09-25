// Float-atomic accumulation for the export k-means; loaded only when the device adds floats atomically.
static const uint LFS_LOCAL_SIZE_X = 256;
static const uint LFS_PUSH_CONSTANT_SIZE = 80;

[SpecializationConstant]
const uint kPhase = 3;

#include "export_kmeans_common.slangh"

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform Parameters p) {
    const uint3 groups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
    Ptr<float, Access.Read> sh = (Ptr<float, Access.Read>)p.sh;
    Ptr<int, Access.ReadWrite> labels = (Ptr<int, Access.ReadWrite>)p.labels;
    Ptr<float, Access.ReadWrite> sums = (Ptr<float, Access.ReadWrite>)p.sums;
    Ptr<int, Access.ReadWrite> counts = (Ptr<int, Access.ReadWrite>)p.counts;
    // Phase 3 reads swizzled SH points, phase 8 dense [row, dims] points.
    for (uint i = gid.x * 256u + tid.x; i < p.n; i += groups.x * 256u) {
        const int label = labels[i];
        if (label < 0 || uint(label) >= p.k) continue;
        for (uint d = 0; d < p.dims; ++d)
            InterlockedAdd(sums[uint(label) * p.dims + d], kPhase == 3 ? sh_dim(sh, i, d, p.slots) : sh[i * p.dims + d]);
        InterlockedAdd(counts[uint(label)], 1);
    }
}
