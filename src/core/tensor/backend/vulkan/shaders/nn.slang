static const uint LFS_LOCAL_SIZE_X = 256;
static const uint LFS_PUSH_CONSTANT_SIZE = 72;

// 0 max_pool2d, 1 adaptive_avg_pool2d, 2 bias_add, 3 bias_relu, 4 relu.
[SpecializationConstant]
const uint kKind = 0;

struct Parameters
{
    uint64_t inputAddress;
    uint64_t biasAddress;
    uint64_t outputAddress;
    uint total;
    uint batch;
    uint channels;
    uint inputHeight;
    uint inputWidth;
    uint outputHeight;
    uint outputWidth;
    uint kernel;
    uint stride;
    int padding;
    uint spatial;
    uint pad0;
};

// First NaN wins and equal-magnitude zeros follow the sign rule, matching the
// CUDA max_reduce_op functor used by the pooling kernel.
float ieeeMaximum(float lhs, float rhs)
{
    const uint lhsBits = asuint(lhs);
    const uint rhsBits = asuint(rhs);
    if ((lhsBits & 0x7fffffffu) > 0x7f800000u)
        return lhs;
    if ((rhsBits & 0x7fffffffu) > 0x7f800000u)
        return rhs;
    if ((lhsBits & 0x7fffffffu) == 0u && (rhsBits & 0x7fffffffu) == 0u)
        return asfloat((lhsBits & rhsBits) & 0x80000000u);
    return lhs < rhs ? rhs : lhs;
}

[numthreads(LFS_LOCAL_SIZE_X, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID, uniform Parameters parameters)
{
    const uint3 numWorkgroups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
    const uint stride = numWorkgroups.x * LFS_LOCAL_SIZE_X;
    Ptr<float, Access.Read> input = (Ptr<float, Access.Read>)parameters.inputAddress;
    Ptr<float, Access.Read> bias = (Ptr<float, Access.Read>)parameters.biasAddress;
    Ptr<float, Access.ReadWrite> output = (Ptr<float, Access.ReadWrite>)parameters.outputAddress;
    for (uint index = dispatchThreadId.x; index < parameters.total; index += stride)
    {
        if (kKind == 4)
        {
            const float value = input[index];
            output[index] = value > 0.0f ? value : 0.0f;
        }
        else if (kKind == 2 || kKind == 3)
        {
            const uint channel = (index / parameters.spatial) % parameters.channels;
            const float value = input[index] + bias[channel];
            output[index] = (kKind == 3 && !(value > 0.0f)) ? 0.0f : value;
        }
        else
        {
            const uint wOut = index % parameters.outputWidth;
            const uint hOut = (index / parameters.outputWidth) % parameters.outputHeight;
            const uint channel = (index / (parameters.outputWidth * parameters.outputHeight)) % parameters.channels;
            const uint image = index / (parameters.outputWidth * parameters.outputHeight * parameters.channels);
            const uint64_t plane = (uint64_t(image) * parameters.channels + channel) *
                                   parameters.inputHeight * parameters.inputWidth;
            if (kKind == 0)
            {
                const int hStart = int(hOut * parameters.stride) - parameters.padding;
                const int wStart = int(wOut * parameters.stride) - parameters.padding;
                float best = -1.0f / 0.0f;
                for (uint kh = 0; kh < parameters.kernel; ++kh)
                {
                    const int hIn = hStart + int(kh);
                    if (hIn < 0 || hIn >= int(parameters.inputHeight))
                        continue;
                    for (uint kw = 0; kw < parameters.kernel; ++kw)
                    {
                        const int wIn = wStart + int(kw);
                        if (wIn < 0 || wIn >= int(parameters.inputWidth))
                            continue;
                        best = ieeeMaximum(best, input[plane + uint64_t(hIn) * parameters.inputWidth + uint(wIn)]);
                    }
                }
                output[index] = best;
            }
            else
            {
                const uint hStart = (hOut * parameters.inputHeight) / parameters.outputHeight;
                const uint hEnd = ((hOut + 1) * parameters.inputHeight + parameters.outputHeight - 1) / parameters.outputHeight;
                const uint wStart = (wOut * parameters.inputWidth) / parameters.outputWidth;
                const uint wEnd = ((wOut + 1) * parameters.inputWidth + parameters.outputWidth - 1) / parameters.outputWidth;
                float sum = 0.0f;
                uint count = 0;
                for (uint h = hStart; h < hEnd; ++h)
                {
                    for (uint w = wStart; w < wEnd; ++w)
                    {
                        sum += input[plane + uint64_t(h) * parameters.inputWidth + w];
                        ++count;
                    }
                }
                output[index] = count > 0 ? sum / float(count) : 0.0f;
            }
        }
    }
}
