static const uint LFS_LOCAL_SIZE_X = 256;
static const uint LFS_PUSH_CONSTANT_SIZE = 64;

// C[m][n] = A[m][k] * B, with B stored as [k][n] (kTransposeB 0) or as
// [n][k] (kTransposeB 1). kBiasRelu applies max(value + bias[row], 0).
[SpecializationConstant]
const uint kTransposeB = 0;
[SpecializationConstant]
const uint kBiasRelu = 0;

static const uint kTileM = 64;
static const uint kTileN = 64;
static const uint kTileK = 16;
static const uint kThreadTile = 4;
static const uint kThreadsPerRow = kTileN / kThreadTile;

struct Parameters
{
    uint64_t aAddress;
    uint64_t bAddress;
    uint64_t cAddress;
    uint64_t biasAddress;
    uint m;
    uint n;
    uint k;
    uint batch;
    uint strideA;
    uint strideB;
    uint strideC;
    uint pad0;
};

groupshared float tileA[kTileM][kTileK];
groupshared float tileB[kTileK][kTileN];

[numthreads(LFS_LOCAL_SIZE_X, 1, 1)]
void main(uint3 groupId : SV_GroupID, uint3 groupThreadId : SV_GroupThreadID,
          uniform Parameters parameters)
{
    const uint thread = groupThreadId.x;
    const uint threadCol = thread % kThreadsPerRow;
    const uint threadRow = thread / kThreadsPerRow;
    const uint rowBase = groupId.y * kTileM;
    const uint colBase = groupId.x * kTileN;
    const uint m = parameters.m;
    const uint n = parameters.n;
    const uint k = parameters.k;
    const uint64_t batchA = uint64_t(groupId.z) * parameters.strideA;
    const uint64_t batchB = uint64_t(groupId.z) * parameters.strideB;
    const uint64_t batchC = uint64_t(groupId.z) * parameters.strideC;
    Ptr<float, Access.Read> a = (Ptr<float, Access.Read>)parameters.aAddress;
    Ptr<float, Access.Read> b = (Ptr<float, Access.Read>)parameters.bAddress;
    Ptr<float, Access.ReadWrite> c = (Ptr<float, Access.ReadWrite>)parameters.cAddress;

    float accumulator[kThreadTile][kThreadTile];
    for (uint i = 0; i < kThreadTile; ++i)
        for (uint j = 0; j < kThreadTile; ++j)
            accumulator[i][j] = 0.0f;

    for (uint k0 = 0; k0 < k; k0 += kTileK)
    {
        for (uint i = thread; i < kTileM * kTileK; i += LFS_LOCAL_SIZE_X)
        {
            const uint r = i / kTileK;
            const uint col = i % kTileK;
            const uint gr = rowBase + r;
            const uint gc = k0 + col;
            tileA[r][col] = (gr < m && gc < k) ? a[batchA + uint64_t(gr) * k + gc] : 0.0f;
        }
        for (uint i = thread; i < kTileK * kTileN; i += LFS_LOCAL_SIZE_X)
        {
            const uint r = i / kTileN;
            const uint col = i % kTileN;
            const uint gk = k0 + r;
            const uint gn = colBase + col;
            float value = 0.0f;
            if (gk < k && gn < n)
            {
                if (kTransposeB != 0)
                    value = b[batchB + uint64_t(gn) * k + gk];
                else
                    value = b[batchB + uint64_t(gk) * n + gn];
            }
            tileB[r][col] = value;
        }
        GroupMemoryBarrierWithGroupSync();
        for (uint kk = 0; kk < kTileK; ++kk)
        {
            float aFragment[kThreadTile];
            float bFragment[kThreadTile];
            for (uint i = 0; i < kThreadTile; ++i)
                aFragment[i] = tileA[threadRow * kThreadTile + i][kk];
            for (uint j = 0; j < kThreadTile; ++j)
                bFragment[j] = tileB[kk][threadCol * kThreadTile + j];
            for (uint i = 0; i < kThreadTile; ++i)
                for (uint j = 0; j < kThreadTile; ++j)
                    accumulator[i][j] += aFragment[i] * bFragment[j];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    for (uint i = 0; i < kThreadTile; ++i)
    {
        const uint gr = rowBase + threadRow * kThreadTile + i;
        if (gr >= m)
            continue;
        for (uint j = 0; j < kThreadTile; ++j)
        {
            const uint gc = colBase + threadCol * kThreadTile + j;
            if (gc >= n)
                continue;
            float value = accumulator[i][j];
            if (kBiasRelu != 0)
            {
                value += ((Ptr<float, Access.Read>)parameters.biasAddress)[gr];
                value = value > 0.0f ? value : 0.0f;
            }
            c[batchC + uint64_t(gr) * n + gc] = value;
        }
    }
}
