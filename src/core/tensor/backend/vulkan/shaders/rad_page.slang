// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "../../../../include/core/rad_packed_page.hpp"
#include "../../../../include/core/rad_dequant_math.hpp"
using namespace lfs::core;
using namespace lfs::core::radmath;
static const uint LFS_LOCAL_SIZE_X = 256;
static const uint LFS_PUSH_CONSTANT_SIZE = 104;
[SpecializationConstant] const uint kPhase = 0; // packed, clear maxima, reduce maxima, resident
struct Parameters {
    uint64_t means, sh0, shN, rotation, scaling, opacity, frames, bounds, links;
    uint64_t packed, descriptor;
    uint page, pageSplats, slots, padding;
};
struct Sources {
    uint64_t means, sh0, shN, rotation, scaling, opacity, shBounds;
    uint offset, count, rest, halfSH, quantSH, padding;
};
uint word(uint64_t base, uint i) { return ((Ptr<uint, Access.Read>)base)[i]; }
void put(uint64_t base, uint64_t i, uint value) { ((Ptr<uint, Access.ReadWrite>)base)[i] = value; }
uint byteAt(uint64_t base, uint i) { return (word(base,i/4) >> ((i%4)*8)) & 255; }
uint halfAt(uint64_t base, uint i) { return byteAt(base,i) | (byteAt(base,i+1)<<8); }
float scalar(uint64_t base,uint i) { return asfloat(word(base,i)); }
uint pair(float a,float b) { return floatToHalf(a) | (floatToHalf(b)<<16); }
uint propIndex(RadPagePackedDesc d,uint kind) {
    for(uint p=0;p<d.property_count;++p) if(d.props[p].kind==kind) return p;
    return 8;
}
float maximum(RadPackedProperty p) { return shMaxAbs(p.min_val,p.max_val,p.scale); }
float plane(Parameters p, RadPagePackedDesc d,uint prop,uint dims,uint component,uint i) {
    if(prop==8 || i>=d.count) return 0;
    RadPackedProperty a=d.props[prop];
    uint e=component*d.count+i;
    uint64_t base=p.packed+a.plane_offset;
    uint stride=d.count*dims;
    switch(a.encoding) {
    case 0: return scalar(base,e);
    case 1: return asfloat(byteAt(base,e) | (byteAt(base,stride+e)<<8) | (byteAt(base,2*stride+e)<<16) | (byteAt(base,3*stride+e)<<24));
    case 2: return halfToFloat(halfAt(base,e*2));
    case 3: return halfToFloat(byteAt(base,e) | (byteAt(base,stride+e)<<8));
    case 4: return dequantR8(byteAt(base,e),a.min_val,a.max_val-a.min_val);
    case 5: { int v=int(byteAt(base,e)); return dequantS8(v>=128?v-256:v,maximum(a)); }
    case 6: return dequantLn0R8(byteAt(base,e),a.min_val,a.max_val);
    case 7: return exp(halfToFloat(halfAt(base,e*2)));
    default: return 0;
    }
}
uint quantByte(float v,float m) { return uint(int(clamp(round(v/m*127.0),-127.0,127.0))) & 255; }
uint64_t shIndex(Parameters p,uint i,uint s) {
    return ((uint64_t(p.page)*p.pageSplats+i)/32*p.slots+s)*32+i%32;
}
float residentSH(Sources s,uint i,uint c) {
    uint splat=s.offset+i;
    uint index=((splat/32*((s.rest*3+3)/4)+c/4)*32+splat%32)*4+c%4;
    if(s.quantSH!=0) {
        float lo=scalar(s.shBounds,(splat/256)*2), hi=scalar(s.shBounds,(splat/256)*2+1);
        uint cell=(splat/32)*(s.rest*3*32)+c*32+splat%32;
        return lo+(hi-lo)*(float(halfAt(s.shN,cell*2))*(1.0/65535.0));
    }
    return s.halfSH!=0 ? halfToFloat(halfAt(s.shN,index*2)) : scalar(s.shN,index);
}
[numthreads(LFS_LOCAL_SIZE_X,1,1)]
void main(uint3 id:SV_DispatchThreadID, uniform Parameters p) {
    uint i=id.x;
    if(i>=p.pageSplats) return;
    uint64_t dst=uint64_t(p.page)*p.pageSplats+i;
    uint64_t frame=p.frames+uint64_t(p.page)*64;
    if(kPhase!=0) {
        Sources s=((Ptr<Sources,Access.Read>)p.descriptor)[0];
        if(kPhase==1) { if(i<4) put(frame,i,0); return; }
        if(kPhase==2) {
            if(i>=s.count || s.shN==0) return;
            float maxima[3]={0,0,0};
            for(uint c=0;c<s.rest*3;++c) {
                uint band=c<9?0:(c<24?1:2);
                maxima[band]=max(maxima[band],abs(residentSH(s,i,c)));
            }
            for(uint b=0;b<3;++b) InterlockedMax(((Ptr<uint,Access.ReadWrite>)frame)[b],asuint(maxima[b]));
            return;
        }
        bool live=i<s.count;
        for(uint c=0;c<3;++c) put(p.means,dst*3+c,live?word(s.means,(s.offset+i)*3+c):0);
        float rgb[3]={0,0,0}, scale[3]={0,0,0};
        if(live) for(uint c=0;c<3;++c) {
            rgb[c]=scalar(s.sh0,(s.offset+i)*3+c);
            scale[c]=scalar(s.scaling,(s.offset+i)*3+c);
        }
        put(p.sh0,dst*2,pair(rgb[0],rgb[1])); put(p.sh0,dst*2+1,pair(rgb[2],0));
        put(p.scaling,dst*2,pair(scale[0],scale[1])); put(p.scaling,dst*2+1,pair(scale[2],0));
        float q[4]={1,0,0,0};
        if(live) for(uint c=0;c<4;++c) q[c]=scalar(s.rotation,(s.offset+i)*4+c);
        put(p.rotation,dst*2,pair(q[0],q[1])); put(p.rotation,dst*2+1,pair(q[2],q[3]));
        // One invocation owns a complete opacity word, including odd tails.
        if((i&1)==0) {
            float a=live?scalar(s.opacity,s.offset+i):0;
            float b=i+1<s.count?scalar(s.opacity,s.offset+i+1):0;
            put(p.opacity,dst/2,pair(a,b));
        }
        for(uint slot=0;slot<p.slots;++slot) {
            uint packed=0;
            for(uint b=0;b<4;++b) {
                uint c=slot*4+b;
                if(live && s.shN!=0 && c<s.rest*3) {
                    uint band=c<9?0:(c<24?1:2);
                    packed|=quantByte(residentSH(s,i,c),max(scalar(frame,band),1e-6))<<(b*8);
                }
            }
            put(p.shN,shIndex(p,i,slot),packed);
        }
        return;
    }
    RadPagePackedDesc d=((Ptr<RadPagePackedDesc,Access.Read>)p.descriptor)[0];
    bool live=i<d.count;
    uint xyz=propIndex(d,0),alpha=propIndex(d,1),rgb=propIndex(d,2),scales=propIndex(d,3),rot=propIndex(d,4);
    if(i==0) {
        for(uint b=0;b<16;++b) put(frame,b,0);
        for(uint b=0;b<3;++b) {
            uint prop=propIndex(d,5+b);
            put(frame,b,asuint(prop<8?maximum(d.props[prop]):0));
            put(frame,4+b,asuint(d.frame.bbox_min[b]));
            put(frame,8+b,asuint(d.frame.bbox_extent[b]));
        }
        put(frame,7,asuint(d.frame.log_size_min)); put(frame,11,asuint(d.frame.log_size_range));
    }
    for(uint c=0;c<3;++c) put(p.means,dst*3+c,asuint(plane(p,d,xyz,3,c,i)));
    float sh[3]={0,0,0}; uint sc[3]={0,0,0};
    if(live) for(uint c=0;c<3;++c) {
        if(rgb<8) sh[c]=sh0Transform(plane(p,d,rgb,3,c,i));
        if(scales<8) sc[c]=d.props[scales].encoding==7 ? halfAt(p.packed+d.props[scales].plane_offset,(c*d.count+i)*2) : floatToHalf(scaleLog(plane(p,d,scales,3,c,i)));
    }
    put(p.sh0,dst*2,pair(sh[0],sh[1])); put(p.sh0,dst*2+1,pair(sh[2],0));
    put(p.scaling,dst*2,sc[0]|(sc[1]<<16)); put(p.scaling,dst*2+1,sc[2]);
    if((i&1)==0) {
        uint packed=0;
        for(uint b=0;b<2;++b) {
            float v=0;
            if(i+b<d.count && alpha<8) {
                float raw=plane(p,d,alpha,1,0,i+b);
                v=d.lod_opacity!=0?opacityLodEncoded(raw):opacityLogit(raw);
            }
            packed|=floatToHalf(v)<<(b*16);
        }
        put(p.opacity,dst/2,packed);
    }
    float q[4]={0,0,0,1};
    if(live && rot<8) {
        if(d.props[rot].encoding==8) {
            uint64_t base=p.packed+d.props[rot].plane_offset;
            dequantQuatOct88R8(byteAt(base,i*3),byteAt(base,i*3+1),byteAt(base,i*3+2),q);
        } else {
            for(uint c=0;c<3;++c) q[c]=plane(p,d,rot,3,c,i);
            q[3]=quatWFromXyz(q[0],q[1],q[2]);
        }
    }
    put(p.rotation,dst*2,pair(q[3],q[0])); put(p.rotation,dst*2+1,pair(q[1],q[2]));
    for(uint slot=0;slot<p.slots;++slot) {
        uint packed=0;
        for(uint b=0;b<4;++b) {
            uint c=slot*4+b;
            if(live && c<d.sh_coeffs_rest*3) {
                uint band=c<9?0:(c<24?1:2), local=c-(band==0?0:(band==1?9:24));
                uint prop=propIndex(d,5+band);
                if(prop<8) {
                    uint v=d.props[prop].encoding==5 ? byteAt(p.packed+d.props[prop].plane_offset,local*d.count+i) : quantByte(plane(p,d,prop,9+band*6,local,i),maximum(d.props[prop]));
                    packed|=v<<(b*8);
                }
            }
        }
        put(p.shN,shIndex(p,i,slot),packed);
    }
    if(p.bounds!=0 && p.links!=0) {
        for(uint c=0;c<2;++c) put(p.bounds,dst*2+c,i<d.meta_node_count?word(p.packed+d.meta_bounds_offset,i*2+c):0);
        for(uint c=0;c<3;++c) put(p.links,dst*3+c,i<d.meta_node_count?word(p.packed+d.meta_links_offset,i*3+c):0xffffffff);
    }
}
