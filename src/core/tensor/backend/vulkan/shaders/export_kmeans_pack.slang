// Half-precision centroid tiles for the screened SH3 assignment, which already requires FP16.
static const uint LFS_LOCAL_SIZE_X = 256;
static const uint LFS_PUSH_CONSTANT_SIZE = 80;

#include "export_kmeans_common.slangh"

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID, uniform Parameters p) {
    const uint3 groups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
    Ptr<float, Access.Read> centroids = (Ptr<float, Access.Read>)p.centroids;
    Ptr<int, Access.Read> indices = (Ptr<int, Access.Read>)p.indices;
    Ptr<half, Access.ReadWrite> packed = (Ptr<half, Access.ReadWrite>)p.sums;
    for (uint c = gid.x * 256u + tid.x; c < p.k; c += groups.x * 256u) {
        for (uint d = 0; d < 48u; ++d)
            packed[c * 48u + d] = d < p.dims ? half(centroids[uint(indices[c]) * p.dims + d]) : half(0);
    }
}
