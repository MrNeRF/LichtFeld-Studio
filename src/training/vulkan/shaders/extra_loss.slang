[SpecializationConstant] const uint operation = 0;

struct Parameters {
    uint64_t source;
    uint64_t gradient;
    uint64_t loss;
    uint64_t scratch;
    uint count;
    uint partialCount;
    float weight;
    uint kind;
    uint accumulate;
};

groupshared float reduction[256];

float activate(float x, uint kind) {
    return kind == 0 ? exp(x) : 1.0f / (1.0f + exp(-x));
}

float sigmoidDerivativeFromLogit(float x) {
    const float e = exp(-abs(x));
    const float d = 1.0f + e;
    return e / (d * d);
}

[numthreads(256, 1, 1)]
void main(uint3 groupId : SV_GroupID, uint lane : SV_GroupIndex, uniform Parameters p) {
    Ptr<float, Access.Read> source = (Ptr<float, Access.Read>)p.source;
    Ptr<float, Access.ReadWrite> gradient = (Ptr<float, Access.ReadWrite>)p.gradient;
    if (operation == 0) {
        const uint blocks = p.partialCount;
        const uint stride = 256u * blocks;
        float sum = 0.0f;
        for (uint i = groupId.x * 256u + lane; i < p.count; i += stride) {
            const float activated = activate(source[i], p.kind);
            sum += activated;
            if (p.accumulate != 0) {
                const float gradScale = p.weight / float(p.count);
                const float derivative = p.kind == 0
                                             ? gradScale * activated
                                             : gradScale * sigmoidDerivativeFromLogit(source[i]);
                gradient[i] += derivative;
            }
        }
        reduction[lane] = sum;
        GroupMemoryBarrierWithGroupSync();
        for (uint offset = 128; offset > 0; offset >>= 1) {
            if (lane < offset) reduction[lane] += reduction[lane + offset];
            GroupMemoryBarrierWithGroupSync();
        }
        if (lane == 0) ((Ptr<float, Access.ReadWrite>)p.scratch)[groupId.x] = reduction[0];
        return;
    }
    if (operation == 1) {
        float sum = 0.0f;
        Ptr<float, Access.Read> partials = (Ptr<float, Access.Read>)p.scratch;
        for (uint i = lane; i < p.partialCount; i += 256u) sum += partials[i];
        reduction[lane] = sum;
        GroupMemoryBarrierWithGroupSync();
        for (uint offset = 128; offset > 0; offset >>= 1) {
            if (lane < offset) reduction[lane] += reduction[lane + offset];
            GroupMemoryBarrierWithGroupSync();
        }
        if (lane == 0) ((Ptr<float, Access.ReadWrite>)p.loss)[0] = p.weight * (reduction[0] / float(p.count));
    }
}
