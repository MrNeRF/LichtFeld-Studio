struct Parameters {
    uint64_t gradOutput, activation, weight, weightGradient, biasGradient, gradInput;
    uint count;
    int m, n;
    float exposurePrior;
};

[SpecializationConstant] const uint operation = 0;

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID, uniform Parameters p) {
    if (operation == 0) {
        Ptr<float, Access.ReadWrite> output = (Ptr<float, Access.ReadWrite>)p.gradInput;
        if (tid.x == 0) output[1600] = p.exposurePrior;
        return;
    }
    if (operation == 1) {
        Ptr<float, Access.Read> grad = (Ptr<float, Access.Read>)p.gradOutput;
        Ptr<float, Access.Read> activation = (Ptr<float, Access.Read>)p.activation;
        Ptr<float, Access.Read> weight = (Ptr<float, Access.Read>)p.weight;
        Ptr<float, Access.ReadWrite> wg = (Ptr<float, Access.ReadWrite>)p.weightGradient;
        Ptr<float, Access.ReadWrite> bg = (Ptr<float, Access.ReadWrite>)p.biasGradient;
        Ptr<float, Access.ReadWrite> gi = (Ptr<float, Access.ReadWrite>)p.gradInput;
        const uint mn = uint(p.m * p.n);
        if (tid.x < mn) {
            const uint row = tid.x / uint(p.n), col = tid.x % uint(p.n);
            wg[tid.x] += grad[row] * activation[col];
        }
        if (tid.x < uint(p.m)) bg[tid.x] += grad[tid.x];
        if (p.gradInput != 0 && tid.x < uint(p.n)) {
            float value = 0.0f;
            for (int row = 0; row < p.m; ++row) value += grad[row] * weight[row * p.n + int(tid.x)];
            gi[tid.x] = activation[tid.x] > 0.0f ? value : 0.0f;
        }
    }
}
