// Based on https://github.com/MrNeRF/optimized-fused-ssim/blob/main/ssim.cu

#include "config.slang"

#ifndef EXPORT_MODE
#define EXPORT_MODE 0
#endif

#if EXPORT_MODE == EXPORT_MODE_VULKAN_FORWARD
#define ssimForward main
#elif EXPORT_MODE == EXPORT_MODE_VULKAN_BACKWARD
#define ssimBackward main
#endif


static const float GAUSS_COEFFS[6] = {
    0.001028380123898387f,
    0.0075987582094967365f,
    0.036000773310661316f,
    0.10936068743467331f,
    0.21300552785396576f,
    0.26601171493530273f
};

static const uint BLOCK_X = 16;
static const uint BLOCK_Y = 16;
static const uint HALO = 5;
static const uint SHARED_X = BLOCK_X + 2 * HALO;
static const uint SHARED_Y = BLOCK_Y + 2 * HALO;
static const uint CONV_X = BLOCK_X;
static const uint CONV_Y = SHARED_Y;

static const float C1 = 0.01*0.01;
static const float C2 = 0.03*0.03;
static const uint numChannels = 3u;


// Input/Output buffer declarations
layout(binding=0) StructuredBuffer<float4> img1Buffer;
layout(binding=1) StructuredBuffer<uint32_t> img2Buffer;
layout(binding=2) RWStructuredBuffer<float4> ssimMapBuffer;
layout(binding=3) RWStructuredBuffer<float4> dLdimg1Buffer;


// forward
groupshared float sTile[SHARED_Y][SHARED_X][2]; // [img1, img2]
groupshared float xconv[CONV_Y][CONV_X][5]; // 5 values
// backward
groupshared float sData[3][SHARED_Y][SHARED_X]; // [dmu1, dsigma1sq, dsigma12]
groupshared float sScratch[CONV_Y][CONV_X][3]; // 3 values


// Uniform parameters
struct Uniforms {
    uint imageWidth;
    uint imageHeight;
    float gradWeightL1;
    float gradWeightSSIM;
}


// Utility function to safely fetch pixel values with zero padding
[ForceInline]
float3 getPixelValue(StructuredBuffer<float4> buffer, int x, int y, int w, int h) {
    if (x < 0 || x >= w || y < 0 || y >= h)
        return float3(0, 0, 0);
    int idx = y * w + x;
    float4 val = buffer[idx];
    return max(float3(val.rgb), float3(0, 0, 0));
}

[ForceInline]
float3 getPixelValue(StructuredBuffer<half4> buffer, int x, int y, int w, int h) {
    if (x < 0 || x >= w || y < 0 || y >= h)
        return float3(0, 0, 0);
    int idx = y * w + x;
    half4 val = buffer[idx];
    return max(float3(val.rgb), float3(0, 0, 0));
}

[ForceInline]
float3 getPixelValue(StructuredBuffer<uint32_t> buffer, int x, int y, int w, int h, out float alpha) {
    if (x < 0 || x >= w || y < 0 || y >= h) {
        alpha = float(0);
        return float3(0, 0, 0);
    }
    int idx = y * w + x;
    uint32_t val = buffer[idx];
    alpha = float(val >> 24) / float(255);
    return float3(val & 255, (val >> 8) & 255, (val >> 16) & 255) / float(255);
}


// Fully-fused gradient computation (single pass, constant input gradient)
[numthreads(BLOCK_X, BLOCK_Y, 1)]
void ssimForward(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uint3 groupThreadID : SV_GroupThreadID,
    uint3 groupID : SV_GroupID,
    uniform Uniforms uniforms
) {
    const uint imageWidth = uniforms.imageWidth;
    const uint imageHeight = uniforms.imageHeight;

    uint pixX = dispatchThreadID.x;
    uint pixY = dispatchThreadID.y;
    uint pixId = pixY * imageWidth + pixX;
    
    for (uint ci = 0; ci < numChannels; ++ci)
    {
        // Load tile for forward pass
        uint tileSize = SHARED_Y * SHARED_X;
        uint threads = BLOCK_X * BLOCK_Y;
        uint steps = (tileSize + threads - 1) / threads;
        
        uint tileStartY = groupID.y * BLOCK_Y;
        uint tileStartX = groupID.x * BLOCK_X;
        
        for (uint s = 0; s < steps; ++s) {
            uint tid = s * threads + (groupThreadID.y * BLOCK_X + groupThreadID.x);
            if (tid < tileSize) {
                uint localY = tid / SHARED_X;
                uint localX = tid % SHARED_X;
                int gy = int(tileStartY + localY) - int(HALO);
                int gx = int(tileStartX + localX) - int(HALO);
                
                float alpha;
                float val1 = getPixelValue(img1Buffer, gx, gy, imageWidth, imageHeight)[ci];
                float val2 = getPixelValue(img2Buffer, gx, gy, imageWidth, imageHeight, alpha)[ci];
                
                sTile[localY][localX][0] = val1 * alpha;
                sTile[localY][localX][1] = val2 * alpha;
            }
        }
        
        GroupMemoryBarrierWithGroupSync();
        
        /* Forward pass: horizontal convolution */

        uint ly = groupThreadID.y;
        uint lx = groupThreadID.x + HALO;
        
        float sumX = 0.0f, sumX2 = 0.0f, sumY = 0.0f, sumY2 = 0.0f, sumXY = 0.0f;
        
        [ForceUnroll]
        for (int d = 1; d <= int(HALO); ++d) {
            float w = GAUSS_COEFFS[HALO - d];
            float xLeft = sTile[ly][lx - d][0];
            float yLeft = sTile[ly][lx - d][1];
            float xRight = sTile[ly][lx + d][0];
            float yRight = sTile[ly][lx + d][1];
            
            sumX += float((xLeft + xRight) * w);
            sumX2 += float((xLeft * xLeft + xRight * xRight) * w);
            sumY += float((yLeft + yRight) * w);
            sumY2 += float((yLeft * yLeft + yRight * yRight) * w);
            sumXY += float((xLeft * yLeft + xRight * yRight) * w);
        }
        
        float centerX = sTile[ly][lx][0];
        float centerY = sTile[ly][lx][1];
        float wc = GAUSS_COEFFS[HALO];
        sumX += float(centerX * wc);
        sumX2 += float(centerX * centerX * wc);
        sumY += float(centerY * wc);
        sumY2 += float(centerY * centerY * wc);
        sumXY += float(centerX * centerY * wc);
        
        xconv[ly][groupThreadID.x][0] = float(sumX);
        xconv[ly][groupThreadID.x][1] = float(sumX2);
        xconv[ly][groupThreadID.x][2] = float(sumY);
        xconv[ly][groupThreadID.x][3] = float(sumY2);
        xconv[ly][groupThreadID.x][4] = float(sumXY);
        
        GroupMemoryBarrierWithGroupSync();
        
        // Possibly handle second row in same subgroup
        int ly2 = ly + BLOCK_Y;
        if (ly2 < CONV_Y) {
            sumX = 0.f; sumX2 = 0.f; sumY = 0.f; sumY2 = 0.f; sumXY  = 0.f;

            [ForceUnroll]
            for (int d = 1; d <= int(HALO); ++d) {
                float w = GAUSS_COEFFS[HALO-d];
                float xLeft  = sTile[ly2][lx-d][0];
                float yLeft  = sTile[ly2][lx-d][1];
                float xRight = sTile[ly2][lx+d][0];
                float yRight = sTile[ly2][lx+d][1];

                sumX += float((xLeft + xRight) * w);
                sumX2 += float((xLeft * xLeft + xRight * xRight) * w);
                sumY += float((yLeft + yRight) * w);
                sumY2 += float((yLeft * yLeft + yRight * yRight) * w);
                sumXY += float((xLeft * yLeft + xRight * yRight) * w);
            }
        
            float centerX = sTile[ly2][lx][0];
            float centerY = sTile[ly2][lx][1];
            float wc = GAUSS_COEFFS[HALO];
            sumX += float(centerX * wc);
            sumX2 += float(centerX * centerX * wc);
            sumY += float(centerY * wc);
            sumY2 += float(centerY * centerY * wc);
            sumXY += float(centerX * centerY * wc);
            
            xconv[ly2][groupThreadID.x][0] = float(sumX);
            xconv[ly2][groupThreadID.x][1] = float(sumX2);
            xconv[ly2][groupThreadID.x][2] = float(sumY);
            xconv[ly2][groupThreadID.x][3] = float(sumY2);
            xconv[ly2][groupThreadID.x][4] = float(sumXY);
        }
        GroupMemoryBarrierWithGroupSync();

        /* Forward pass: vertical convolution and derivative computation */

        ly = groupThreadID.y + HALO;
        lx = groupThreadID.x;
        
        float out0 = 0.0f, out1 = 0.0f, out2 = 0.0f, out3 = 0.0f, out4 = 0.0f;
        
        [ForceUnroll]
        for (int d = 1; d <= int(HALO); ++d) {
            float w = GAUSS_COEFFS[HALO - d];
            out0 += (xconv[ly-d][lx][0] + xconv[ly+d][lx][0]) * w;
            out1 += (xconv[ly-d][lx][1] + xconv[ly+d][lx][1]) * w;
            out2 += (xconv[ly-d][lx][2] + xconv[ly+d][lx][2]) * w;
            out3 += (xconv[ly-d][lx][3] + xconv[ly+d][lx][3]) * w;
            out4 += (xconv[ly-d][lx][4] + xconv[ly+d][lx][4]) * w;
        }
        
        float wC = GAUSS_COEFFS[HALO];
        out0 += xconv[ly][lx][0] * wC;
        out1 += xconv[ly][lx][1] * wC;
        out2 += xconv[ly][lx][2] * wC;
        out3 += xconv[ly][lx][3] * wC;
        out4 += xconv[ly][lx][4] * wC;
        
        // Compute partial derivatives
        float outSSIM = 0.0;
        float dMdmu1 = 0.0;
        float dMdsigma1Sq = 0.0;
        float dMdsigma12 = 0.0;
        if (pixX < imageWidth && pixY < imageHeight) {
            float mu1 = out0;
            float mu2 = out2;
            float mu1Sq = mu1 * mu1;
            float mu2Sq = mu2 * mu2;
            float sigma1Sq = out1 - mu1Sq;
            float sigma2Sq = out3 - mu2Sq;
            float sigma12 = out4 - mu1 * mu2;
            
            float A = mu1Sq + mu2Sq + C1;
            float B = sigma1Sq + sigma2Sq + C2;
            float C_ = 2.0f * mu1 * mu2 + C1;
            float D_ = 2.0f * sigma12 + C2;

            outSSIM = (C_ * D_) / (A * B);
            
            dMdmu1 = (mu2 * 2.0f * D_) / (A * B) - (mu2 * 2.0f * C_) / (A * B) -
                        (mu1 * 2.0f * C_ * D_) / (A * A * B) + (mu1 * 2.0f * C_ * D_) / (A * B * B);
            dMdsigma1Sq = (-C_ * D_) / (A * B * B);
            dMdsigma12 = (2.0f * C_) / (A * B);

            ssimMapBuffer[pixId*numChannels+ci] = float4(
                float4(outSSIM, dMdmu1, dMdsigma1Sq, dMdsigma12)
            );
        }

    }
}


// Backward gradient computation
[numthreads(BLOCK_X, BLOCK_Y, 1)]
void ssimBackward(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uint3 groupThreadID : SV_GroupThreadID,
    uint3 groupID : SV_GroupID,
    uniform Uniforms uniforms
) {
    const uint imageWidth = uniforms.imageWidth;
    const uint imageHeight = uniforms.imageHeight;
    const float gradWeightL1 = uniforms.gradWeightL1;
    const float gradWeightSSIM = uniforms.gradWeightSSIM;

    uint pixX = dispatchThreadID.x;
    uint pixY = dispatchThreadID.y;
    uint pixId = pixY * imageWidth + pixX;
    
    for (uint ci = 0; ci < numChannels; ++ci) {
        
        // Load and fuse multiplication
        uint startY = groupID.y * BLOCK_Y;
        uint startX = groupID.x * BLOCK_X;
        
        uint tid = groupThreadID.y * BLOCK_X + groupThreadID.x;
        uint totalThreads = BLOCK_X * BLOCK_Y;

        int subgroup_id = tid / SUBGROUP_SIZE;
        int lane_id = tid % SUBGROUP_SIZE;
        int num_subgroups = (totalThreads + SUBGROUP_SIZE - 1) / SUBGROUP_SIZE;

        for (int row = subgroup_id; row < SHARED_Y; row += num_subgroups) {
            for (uint col = lane_id; col < SHARED_X; col += totalThreads) {
                int gy = int(startY + row) - int(HALO);
                int gx = int(startX + col) - int(HALO);
                
                float4 forwardReturns =
                    // (gx >= 0 && gx < int(imageWidth) && gy >= 0 && gy < int(imageHeight)) ?
                    (gx >= int(HALO) && gx < int(imageWidth-HALO) && gy >= int(HALO) && gy < int(imageHeight-HALO)) ?
                    ssimMapBuffer[(gy*imageWidth+gx)*numChannels+ci] : float4(0,0,0,0);
                
                sData[0][row][col] = forwardReturns.y;
                sData[1][row][col] = forwardReturns.z;
                sData[2][row][col] = forwardReturns.w;
            }
        }
        
        GroupMemoryBarrierWithGroupSync();
        
        // Horizontal pass
        uint ly = groupThreadID.y;
        uint lx = groupThreadID.x + HALO;
        
        for (uint pass = 0; pass < 2; ++pass) {
            uint yy = ly + pass * BLOCK_Y;
            if (yy < CONV_Y) {
                float accum0 = 0.0f, accum1 = 0.0f, accum2 = 0.0f;
                
                [ForceUnroll]
                for (int d = 1; d <= int(HALO); ++d) {
                    float w = GAUSS_COEFFS[HALO - d];
                    accum0 += float((sData[0][yy][lx - d] + sData[0][yy][lx + d]) * w);
                    accum1 += float((sData[1][yy][lx - d] + sData[1][yy][lx + d]) * w);
                    accum2 += float((sData[2][yy][lx - d] + sData[2][yy][lx + d]) * w);
                }
                
                float wc = GAUSS_COEFFS[HALO];
                accum0 += float(sData[0][yy][lx] * wc);
                accum1 += float(sData[1][yy][lx] * wc);
                accum2 += float(sData[2][yy][lx] * wc);
                
                sScratch[yy][groupThreadID.x][0] = accum0;
                sScratch[yy][groupThreadID.x][1] = accum1;
                sScratch[yy][groupThreadID.x][2] = accum2;
            }
        }
        
        GroupMemoryBarrierWithGroupSync();
        
        // Vertical pass and finalize
        if (pixX < imageWidth && pixY < imageHeight) {
            ly = groupThreadID.y + HALO;
            lx = groupThreadID.x;
            
            float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f;
            
            [ForceUnroll]
            for (int d = 1; d <= int(HALO); ++d) {
                float w = GAUSS_COEFFS[HALO - d];
                sum0 += (sScratch[ly - d][lx][0] + sScratch[ly + d][lx][0]) * w;
                sum1 += (sScratch[ly - d][lx][1] + sScratch[ly + d][lx][1]) * w;
                sum2 += (sScratch[ly - d][lx][2] + sScratch[ly + d][lx][2]) * w;
            }
            
            float wc = GAUSS_COEFFS[HALO];
            sum0 += sScratch[ly][lx][0] * wc;
            sum1 += sScratch[ly][lx][1] * wc;
            sum2 += sScratch[ly][lx][2] * wc;
            
            float alpha;
            float p1 = float(getPixelValue(img1Buffer, pixX, pixY, imageWidth, imageHeight)[ci]);
            float p2 = float(getPixelValue(img2Buffer, pixX, pixY, imageWidth, imageHeight, alpha)[ci]);
            p1 *= alpha; p2 *= alpha;
            float g_l1 = sign(p1 - p2) * gradWeightL1;
            float g_ssim = (sum0 + (2.0f * p1) * sum1 + p2 * sum2) * gradWeightSSIM;
            float clampDerivative = (p1 > 0.0f) ? 1.0f : 0.0f;
            float dLdpix = (g_l1 + g_ssim) * clampDerivative;
            
            if (ci == 0) dLdimg1Buffer[pixId] = float4(float4(dLdpix, 0, 0, 0));
            else if (ci == 1) dLdimg1Buffer[pixId].y = float(dLdpix);
            else if (ci == 2) dLdimg1Buffer[pixId].z = float(dLdpix);
        }
        
        GroupMemoryBarrierWithGroupSync();
    }
}
