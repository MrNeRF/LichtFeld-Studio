// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
static const uint LFS_LOCAL_SIZE_X = 256;
static const uint LFS_PUSH_CONSTANT_SIZE = 112;

[SpecializationConstant]
const uint kOperation = 0; // im2col, col2im, resize, pool, activation, grid

struct Parameters
{
    uint64_t inputAddress;
    uint64_t outputAddress;
    uint total;
    uint step;
    int channels;
    int height;
    int width;
    int outHeight;
    int outWidth;
    int kernelH;
    int kernelW;
    int strideH;
    int strideW;
    int padH;
    int padW;
    int dilationH;
    int dilationW;
    int offset;
    int columns;
    int mode;
    int coord;
    int includePad;
    float u0;
    float u1;
    float v0;
    float v1;
};

float sample(float* input, int plane, int y, int x, Parameters p)
{
    return input[(plane * p.height + clamp(y, 0, p.height - 1)) * p.width + clamp(x, 0, p.width - 1)];
}

float coordinate(int i, int insize, int outsize, int mode)
{
    if (outsize == 1) return 0.0;
    if (mode == 2) return outsize == 1 ? 0.0 : float(i) * (insize - 1) / (outsize - 1);
    if (mode == 1) return float(i) * insize / outsize;
    return (float(i) + 0.5) * insize / outsize - 0.5;
}

float cubic(float x)
{
    x = abs(x);
    if (x <= 1.0) return ((1.25 * x - 2.25) * x) * x + 1.0;
    if (x < 2.0) return ((-0.75 * x + 3.75) * x - 6.0) * x + 3.0;
    return 0.0;
}

// Evaluate the erf approximation in fp32 before the output cast.
float erfApprox(float x)
{
    float a = abs(x);
    float t = 1.0 / (1.0 + 0.3275911 * a);
    float r = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                     - 0.284496736) * t + 0.254829592) * t * exp(-a * a);
    return x < 0.0 ? -r : r;
}

[numthreads(LFS_LOCAL_SIZE_X, 1, 1)]
void main(uint3 id : SV_DispatchThreadID, uniform Parameters p)
{
    float* input = (float*)p.inputAddress;
    float* output = (float*)p.outputAddress;
    for (uint index = id.x; index < p.total; index += p.step)
    {
        float value = 0.0;
        if (kOperation == 0)
        {
            int column = int(index) % p.columns + p.offset;
            int tap = int(index) / p.columns;
            int x = (column % p.outWidth) * p.strideW - p.padW + (tap % p.kernelW) * p.dilationW;
            int y = (column / p.outWidth) * p.strideH - p.padH + ((tap / p.kernelW) % p.kernelH) * p.dilationH;
            int c = tap / (p.kernelH * p.kernelW);
            if (p.mode == 1 || (x >= 0 && x < p.width && y >= 0 && y < p.height))
                value = sample(input, c, y, x, p);
        }
        else if (kOperation == 1)
        {
            int x = int(index) % p.outWidth;
            int y = (int(index) / p.outWidth) % p.outHeight;
            int c = int(index) / (p.outWidth * p.outHeight);
            for (int ky = 0; ky < p.kernelH; ++ky)
                for (int kx = 0; kx < p.kernelW; ++kx)
                {
                    int iy = y + p.padH - ky * p.dilationH;
                    int ix = x + p.padW - kx * p.dilationW;
                    if (iy < 0 || ix < 0 || iy % p.strideH != 0 || ix % p.strideW != 0) continue;
                    iy /= p.strideH;
                    ix /= p.strideW;
                    if (iy < p.height && ix < p.width)
                        value += input[((c * p.kernelH + ky) * p.kernelW + kx) * p.height * p.width + iy * p.width + ix];
                }
        }
        else if (kOperation == 2)
        {
            int x = int(index) % p.outWidth;
            int y = (int(index) / p.outWidth) % p.outHeight;
            int plane = int(index) / (p.outWidth * p.outHeight);
            float fx = coordinate(x, p.width, p.outWidth, p.coord);
            float fy = coordinate(y, p.height, p.outHeight, p.coord);
            int ix = int(floor(fx));
            int iy = int(floor(fy));
            if (p.mode == 0)
                value = sample(input, plane, int(floor(fy)), int(floor(fx)), p);
            else if (p.mode == 1)
            {
                float dx = fx - ix;
                float dy = fy - iy;
                float a = sample(input, plane, iy, ix, p) * (1.0 - dx) + sample(input, plane, iy, ix + 1, p) * dx;
                float b = sample(input, plane, iy + 1, ix, p) * (1.0 - dx) + sample(input, plane, iy + 1, ix + 1, p) * dx;
                value = a * (1.0 - dy) + b * dy;
            }
            else
                for (int yy = -1; yy <= 2; ++yy)
                    for (int xx = -1; xx <= 2; ++xx)
                        value += sample(input, plane, iy + yy, ix + xx, p) * cubic(fy - (iy + yy)) * cubic(fx - (ix + xx));
        }
        else if (kOperation == 3)
        {
            int x = (int(index) % p.outWidth) * p.strideW - p.padW;
            int y = ((int(index) / p.outWidth) % p.outHeight) * p.strideH - p.padH;
            int plane = int(index) / (p.outWidth * p.outHeight);
            value = p.mode == 0 ? -1.0 / 0.0 : 0.0;
            int count = 0;
            for (int ky = 0; ky < p.kernelH; ++ky)
                for (int kx = 0; kx < p.kernelW; ++kx)
                    if (y + ky >= 0 && y + ky < p.height && x + kx >= 0 && x + kx < p.width)
                    {
                        float v = sample(input, plane, y + ky, x + kx, p);
                        value = p.mode == 0 ? max(value, v) : value + v;
                        ++count;
                    }
            if (p.mode != 0)
                value /= max(1, p.includePad != 0 ? p.kernelH * p.kernelW : count);
        }
        else if (kOperation == 4)
        {
            float x = input[index];
            if (p.mode == 1) value = max(x, 0.0);
            else if (p.mode == 2) value = 0.5 * x * (1.0 + tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
            else if (p.mode == 3) value = 0.5 * x * (1.0 + erfApprox(x * 0.7071067811865475));
            else if (p.mode == 4) value = x / (1.0 + exp(-x));
            else value = x;
        }
        else if (kOperation == 5)
        {
            int pixel = int(index) % (p.height * p.width);
            if (int(index) < p.height * p.width)
                value = p.width == 1 ? p.u0 : p.u0 + (p.u1 - p.u0) * (pixel % p.width) / (p.width - 1);
            else
                value = p.height == 1 ? p.v0 : p.v0 + (p.v1 - p.v0) * (pixel / p.width) / (p.height - 1);
        }
        output[index] = value;
    }
}
