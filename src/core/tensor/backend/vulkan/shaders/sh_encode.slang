#include "sh_codec.slangh"
groupshared float lows[256];
groupshared float highs[256];
float minNumber(float a,float b) { return isnan(a) ? b : isnan(b) ? a : min(a,b); }
float maxNumber(float a,float b) { return isnan(a) ? b : isnan(b) ? a : max(a,b); }
[numthreads(LFS_LOCAL_SIZE_X,1,1)]
void main(uint3 gid: SV_GroupID, uint lane: SV_GroupIndex, uniform Parameters p) {
    uint row=gid.x*256+lane;
    float cells[45];
    float lo=1e30f,hi=-1e30f;
    if(row<p.count) {
        uint sr=indexType!=0 ? indexAt(p,row) : p.sourceOffset+row;
        [unroll] for(uint c=0;c<destinationRest*3;++c) {
            float v=c<sourceRest*3 ? readCell(p,sr,c) : 0;
            cells[c]=v;
            lo=minNumber(lo,v); hi=maxNumber(hi,v);
        }
    }
    lows[lane]=lo;highs[lane]=hi;
    GroupMemoryBarrierWithGroupSync();
    for(uint stride=128;stride>0;stride>>=1) {
        if(lane<stride) { lows[lane]=minNumber(lows[lane],lows[lane+stride]); highs[lane]=maxNumber(highs[lane],highs[lane+stride]); }
        GroupMemoryBarrierWithGroupSync();
    }
    lo=lows[0];hi=highs[0];
    if(lo>hi) {lo=0;hi=0;}
    if(lane==0) {
        ((Ptr<float, Access.ReadWrite>)p.destinationBounds)[gid.x*2]=lo;
        ((Ptr<float, Access.ReadWrite>)p.destinationBounds)[gid.x*2+1]=hi;
    }
    if(row>=((p.count+31)/32)*32) return;
    [unroll] for(uint c=0;c<destinationRest*3;++c) {
        uint code=0;
        if(row<p.count) {
            float numerator=65535.0f*(cells[c]-lo);
            float scaled=numerator/maxNumber(hi-lo,1e-20f);
            float base=floor(scaled);
            float rounded=base+(scaled-base>=0.5f ? 1.0f : 0.0f);
            code=uint(minNumber(maxNumber(rounded,0.0f),65535.0f));
        }
        ((Ptr<uint16_t, Access.ReadWrite>)p.destination)[shOffset(row,c,destinationRest,true)]=uint16_t(code);
    }
}
