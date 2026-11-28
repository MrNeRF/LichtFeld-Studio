#include "sh_codec.slangh"
[numthreads(LFS_LOCAL_SIZE_X,1,1)]
void main(uint3 tid: SV_DispatchThreadID, uniform Parameters p) {
    uint3 groups=spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
    uint width=destinationFormat==0 ? destinationRest*3 : (destinationRest*3+3)/4*4;
    uint rows=p.count;
    if(scatter==0 && p.destinationOffset+p.count==p.destinationRows && destinationFormat!=0)
        rows+=(32-p.destinationRows%32)%32;
    const uint64_t total=uint64_t((rows+31)/32)*32*width;
    for(uint64_t i=tid.x;i<total;i+=uint64_t(groups.x)*256) {
        // Enumerate swizzled destinations in memory order, including tail lanes.
        uint localRow=uint(destinationFormat==0 ? i/width : (i/(width*32))*32+(i/4)%32);
        uint c=uint(destinationFormat==0 ? i%width : ((i/128)%(width/4))*4+i%4);
        if(localRow>=rows) continue;
        uint sr=indexType!=0 && scatter==0 && localRow<p.count ? indexAt(p,localRow) : p.sourceOffset+localRow;
        uint dr=scatter!=0 ? indexAt(p,localRow) : p.destinationOffset+localRow;
        float value=localRow<p.count && !(c>=destinationRest*3 && sourceFormat!=1) ? readCell(p,sr,c) : 0;
        // The row's 64-bit address and a small index from it (see shRowAddress).
        const uint64_t address=shRowAddress(p.destination,destinationFormat,destinationRest,dr);
        const uint index=shRowIndex(destinationFormat,c);
        if(destinationFormat==2) ((Ptr<uint16_t, Access.ReadWrite>)address)[index]=uint16_t(f32tof16(value));
        else ((Ptr<float, Access.ReadWrite>)address)[index]=value;
    }
}
