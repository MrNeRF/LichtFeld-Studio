/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../../src/training/hip/logical_subgroup32.hpp"
#include <hip/hip_cooperative_groups.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <stdexcept>
using namespace lfs::hip::logical32;
__global__ void probe(unsigned* output) {
    const unsigned linear=threadIdx.x+blockDim.x*(threadIdx.y+blockDim.y*threadIdx.z);
    const unsigned lane=linear&31u,base=linear-lane;
    const unsigned count=blockDim.x*blockDim.y*blockDim.z;
    const unsigned active=count-base>=32?32:count-base;
    const uint32_t mask=active==32?0xffffffffu:((1u<<active)-1u);
    const auto v=linear+1000u;
    auto* result=output+linear*9;
    result[0]=warpSize;
    result[1]=unsigned(ballot(mask,(lane%3)==1));
    result[2]=shuffle(mask,v,0);
    result[3]=shuffle_xor(mask,v,1);
    result[4]=shuffle_up(mask,v,1);
    result[5]=shuffle_down(mask,v,1);
    // Divergent half waves must not observe the other tile's membership.
    if(lane%2==0)result[6]=unsigned(ballot(mask&0x55555555u,lane%4==0));
    else result[6]=unsigned(ballot(mask&0xaaaaaaaau,lane%4==1));
    sync(mask);
    result[7]=base;
    result[8]=0;
    if(count%32==0){
        const auto tile=cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
        unsigned sum=v;
        for(unsigned offset=16;offset;offset/=2)sum+=tile.shfl_xor(sum,offset);
        result[8]=sum;
    }
}
void check(hipError_t code){if(code!=hipSuccess)throw std::runtime_error(hipGetErrorString(code));}
int main(int argc,char** argv)try{
 if(argc!=2)throw std::runtime_error("Expected physical wave size");const unsigned expected=std::atoi(argv[1]);
 for(unsigned lane=0;lane<64;++lane)for(uint32_t bits:{0u,1u,0x80000000u,0x55555555u,0xffffffffu})
  if(local_ballot(physical_mask(bits,lane),lane)!=bits)throw std::runtime_error("CPU mask roundtrip failed");
 for(const auto block:{dim3(128),dim3(16,8),dim3(96),dim3(48),dim3(8,4,3)}){
  const unsigned count=block.x*block.y*block.z;unsigned* device=nullptr;check(hipMalloc(&device,count*9*sizeof(unsigned)));
  probe<<<1,block>>>(device);check(hipGetLastError());std::vector<unsigned> actual(count*9);
  check(hipMemcpy(actual.data(),device,actual.size()*sizeof(unsigned),hipMemcpyDeviceToHost));check(hipFree(device));
  for(unsigned i=0;i<count;++i){const unsigned lane=i&31,base=i-lane,active=std::min(32u,count-base);uint32_t ballot_ref=0,divergent_ref=0;
   for(unsigned j=0;j<active;++j){if(j%3==1)ballot_ref|=1u<<j;if(j%2==lane%2 && j%4==lane%2)divergent_ref|=1u<<j;}
   const auto* v=actual.data()+i*9;
   if(v[0]!=expected||v[1]!=ballot_ref||v[2]!=base+1000||v[3]!=(base+(lane^1u)+1000)||
      (lane>0 && v[4]!=i+999)||(lane+1<active && v[5]!=i+1001)||v[6]!=divergent_ref||v[7]!=base||(count%32==0 && v[8]!=32*(base+1000)+496))
      {std::fprintf(stderr,"thread %u wave %u expected %u ballot %08x/%08x divergent %08x/%08x\n",i,v[0],expected,v[1],ballot_ref,v[6],divergent_ref);throw std::runtime_error("Logical32 crossed subgroup boundaries");}
  }
 }
 std::printf("PASS native physical wave%u logical32: independent masks, ballots, shuffles, divergence, cooperative tiles, partial/2D/3D blocks\n",expected);return 0;
}catch(const std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}
