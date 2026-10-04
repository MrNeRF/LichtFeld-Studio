/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "lpips_common.slangh"
[SpecializationConstant] const uint kOperation = 0;
groupshared float convWeights[16][32];
groupshared float convInputs[32][64];

[numthreads(256,1,1)]
void main(uint3 id : SV_DispatchThreadID, uniform Parameters p) {
    uint i=uint(p.g)+id.x;
    if (kOperation == 4u) {
        // Implicit GEMM: reuse each input across 16 output channels and each
        // weight across 64 spatial positions, retaining the scalar FMA order.
        uint lane = id.x % 256u;
        uint tile = i / 256u;
        uint plane = p.out_height * p.out_width;
        uint spatialTiles = (plane + 63u) / 64u;
        uint channelTiles = (p.out_channels + 15u) / 16u;
        uint spatialBase = (tile % spatialTiles) * 64u;
        uint channelBase = ((tile / spatialTiles) % channelTiles) * 16u;
        uint batch = tile / (spatialTiles * channelTiles);
        uint row = lane / 16u, column = lane % 16u;
        uint sh = pair_lo(p.stride), sw = pair_hi(p.stride);
        uint ph = pair_lo(p.padding), pw = pair_hi(p.padding);
        uint dh = pair_lo(p.dilation), dw = pair_hi(p.dilation);
        bool isHalf = ((p.flags >> 9) & 1u) != 0u;
        bool replicate = ((p.flags >> 8) & 1u) != 0u;
        float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        uint reduction = p.channels * 9u;
        for (uint base = 0; base < reduction; base += 32u) {
            for (uint t = lane; t < 512u; t += 256u) {
                uint oc = channelBase + t / 32u, k = base + t % 32u;
                convWeights[t / 32u][t % 32u] = oc < p.out_channels && k < reduction
                    ? load_value(p.b, oc * reduction + k, isHalf) : 0.0f;
            }
            for (uint t = lane; t < 2048u; t += 256u) {
                uint k = base + t / 64u, pixel = spatialBase + t % 64u;
                float value = 0.0f;
                if (k < reduction && pixel < plane) {
                    int y = int((pixel / p.out_width) * sh + ((k / 3u) % 3u) * dh) - int(ph);
                    int x = int((pixel % p.out_width) * sw + (k % 3u) * dw) - int(pw);
                    if (replicate) {
                        y = clamp(y, 0, int(p.height) - 1);
                        x = clamp(x, 0, int(p.width) - 1);
                    }
                    if (y >= 0 && x >= 0 && y < int(p.height) && x < int(p.width))
                        value = load_value(p.a, ((batch * p.channels + k / 9u) * p.height + uint(y)) * p.width + uint(x), isHalf);
                }
                convInputs[t / 64u][t % 64u] = value;
            }
            GroupMemoryBarrierWithGroupSync();
            [unroll] for (uint k = 0; k < 32u; ++k) {
                float weight = convWeights[row][k];
                [unroll] for (uint j = 0; j < 4u; ++j)
                    sums[j] = fma(convInputs[k][column + j * 16u], weight, sums[j]);
            }
            GroupMemoryBarrierWithGroupSync();
        }
        uint oc = channelBase + row;
        [unroll] for (uint j = 0; j < 4u; ++j) {
            uint pixel = spatialBase + column + j * 16u;
            if (oc < p.out_channels && pixel < plane) {
                float value = sums[j];
                if (p.c != 0) value += load_value(p.c, oc, isHalf);
                store_value(p.d, (batch * p.out_channels + oc) * plane + pixel,
                            activation(value, p.flags & 255u), isHalf);
            }
        }
        return;
    }
    if (kOperation==0u) {
        if(i>=p.count)return;
        uint outc=p.out_channels, inc=p.channels;
        uint tap=i/(outc*inc), oc=(i/inc)%outc, ic=i%inc;
        uint kh=tap/3u, kw=tap%3u;
        uint src=((oc*inc+ic)*3u+kh)*3u+kw;
        store_half(p.d,tap*outc*inc+oc*inc+ic,load_half(p.a,src));
        return;
    }
    if(kOperation==1u || kOperation==2u) {
        if(i>=p.count)return;
        uint channels_in=kOperation==1u?3u:p.channels;
        uint channels_out=p.out_channels;
        uint plane=p.out_height*p.out_width;
        uint ox=i%p.out_width, oy=(i/p.out_width)%p.out_height;
        uint oc=(i/plane)%channels_out, batch=i/(plane*channels_out);
        uint sh=pair_lo(p.stride), sw=pair_hi(p.stride);
        uint ph=pair_lo(p.padding), pw=pair_hi(p.padding);
        uint dh=pair_lo(p.dilation), dw=pair_hi(p.dilation);
        uint act=p.flags&255u, replicate=(p.flags>>8)&1u;
        bool is_half=kOperation==1u || ((p.flags>>9)&1u)!=0u;
        float accum=0.0f;
        float partial0=0.0f, partial1=0.0f, partial2=0.0f, partial3=0.0f;
        uint k=0u;
        for(uint ic=0;ic<channels_in;++ic) for(uint ky=0;ky<3u;++ky) for(uint kx=0;kx<3u;++kx,++k) {
            int iy=int(oy*sh+ky*dh)-int(ph), ix=int(ox*sw+kx*dw)-int(pw);
            if(replicate!=0u) { iy=clamp(iy,0,int(p.height)-1); ix=clamp(ix,0,int(p.width)-1); }
            else if(iy<0||ix<0||iy>=int(p.height)||ix>=int(p.width)) continue;
            uint in_index=((batch*channels_in+ic)*p.height+uint(iy))*p.width+uint(ix);
            float value=kOperation==1u?((Ptr<float,Access.Read>)p.a)[in_index]:load_value(p.a,in_index,is_half);
            if(kOperation==1u) {
                if(p.flags>>16!=0u) value=2.0f*value-1.0f;
                float shift=ic==0?p.shift0:ic==1?p.shift1:p.shift2;
                float scale=ic==0?p.scale0:ic==1?p.scale1:p.scale2;
                value=(value-shift)/scale;
                value=halfBitsToFloat(floatToHalfBits(value));
            }
            float weight=load_value(p.b,((oc*channels_in+ic)*3u+ky)*3u+kx,is_half);
            if(kOperation==1u) {
                if((k&3u)==0u) partial0=fma(value,weight,partial0);
                else if((k&3u)==1u) partial1=fma(value,weight,partial1);
                else if((k&3u)==2u) partial2=fma(value,weight,partial2);
                else partial3=fma(value,weight,partial3);
                if((k&15u)==15u) {
                    accum+=((partial0+partial1)+partial2)+partial3;
                    partial0=0.0f; partial1=0.0f; partial2=0.0f; partial3=0.0f;
                }
            } else {
                accum=fma(value,weight,accum);
            }
        }
        if(kOperation==1u) accum+=((partial0+partial1)+partial2)+partial3;
        if(p.c!=0) accum+=load_value(p.c,oc,is_half);
        accum=activation(accum,act);
        store_value(p.d,i,accum,is_half);
        return;
    }
    if(kOperation==3u) {
        uint plane=p.height*p.width;
        if(i<p.count) {
            uint pixel=i%plane, batch=i/plane;
            float sx=0.0f, sy=0.0f;
            for(uint c=0;c<p.channels;++c) {
                float vx=load_half(p.a,batch*p.channels*plane+c*plane+pixel);
                float vy=load_half(p.b,batch*p.channels*plane+c*plane+pixel);
                sx=fma(vx,vx,sx); sy=fma(vy,vy,sy);
            }
            float ix=rsqrt(sx+1e-10f), iy=rsqrt(sy+1e-10f), contribution=0.0f;
            for(uint c=0;c<p.channels;++c) {
                float nx=load_half(p.a,batch*p.channels*plane+c*plane+pixel)*ix;
                float ny=load_half(p.b,batch*p.channels*plane+c*plane+pixel)*iy;
                float diff=nx-ny;
                contribution=fma(load_half(p.c,c),diff*diff,contribution);
            }
            uint y=p.height==0?0:pixel/p.width, x=pixel%p.width;
            uint y0=pair_lo(p.stride), y1=pair_hi(p.stride), x0=pair_lo(p.padding), x1=pair_hi(p.padding);
            if(y>=y0&&y<y1&&x>=x0&&x<x1) {
                InterlockedAdd(((Ptr<float, Access.ReadWrite>)p.d)[0], contribution * p.inverse_count * (p.g != 0 ? ((Ptr<float, Access.Read>)p.g)[(y + p.out_height) * p.out_width + x + p.out_channels] : 1.0f));
            }
        }
        if(p.e!=0 && p.f!=0) {
            uint oh=p.height/2u, ow=p.width/2u, batches=p.count/plane, pooled_count=batches*p.channels*oh*ow;
            uint j=i;
            if(j<pooled_count) {
                uint x=j%ow, y=(j/ow)%oh, ch=(j/(ow*oh))%p.channels, batch=j/(ow*oh*p.channels);
                uint base=batch*p.channels*plane+ch*plane+(2u*y)*p.width+2u*x;
                float mx=max(max(load_half(p.a,base),load_half(p.a,base+1u)),max(load_half(p.a,base+p.width),load_half(p.a,base+p.width+1u)));
                float my=max(max(load_half(p.b,base),load_half(p.b,base+1u)),max(load_half(p.b,base+p.width),load_half(p.b,base+p.width+1u)));
                store_half(p.e,j,mx); store_half(p.f,j,my);
            }
        }
    }
}
