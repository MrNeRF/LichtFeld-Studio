/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
    {
        LfsHipBuffer *wx=nullptr,*wy=nullptr;
        check(lfs_hip_buffer_allocate(first,65536,&wx,&error));check(lfs_hip_buffer_allocate(first,65536,&wy,&error));
        LfsHipSharedResize resize{};resize.struct_bytes=sizeof(resize);resize.kernel_size=3;
        double max_resize_error=0;
        // Coefficient arithmetic is specified in Float32 by the existing device path.
        // Preserve that rounding in the scalar CPU oracle; evaluate the independent
        // convolution in double. High-magnification support can be ill-conditioned.
        auto sinc_cpu=[](float x){constexpr double pi=3.14159265358979323846;return std::abs(x)<1e-12f?1.f:float(std::sin(float(pi*x))/(pi*x));};
        auto weights_cpu=[&](int input_size,int output_size,int index){std::vector<std::pair<int,double>> result;volatile float scale=float(input_size)/float(output_size);volatile float center=(float(index)+.5f)*scale;volatile float reach=3.f*scale;volatile float left=center-reach,right=center+reach;volatile float low=left+.5f,high=right+.5f;float total=0;int begin=std::max(int(low),0),end=std::min(int(high),input_size);for(int i=begin;i<end;++i){volatile float delta=float(i)+.5f-center;float x=delta/scale;float value=std::abs(x)>=3?0:sinc_cpu(x)*sinc_cpu(x/3);result.emplace_back(i,value);total+=value;}for(auto& item:result)item.second=float(item.second)/total;return result;};
        for(unsigned mode:{0u,1u,2u})for(unsigned dtype:{uint32_t(LFS_HIP_U8),uint32_t(LFS_HIP_F32)})for(unsigned channels:{1u,2u,3u,4u}){
            if((mode==1 && channels!=1) || (mode==2 && (channels!=3 || dtype!=LFS_HIP_F32)))continue;
            for(bool down:{false,true}){
                resize.mode=mode;resize.dtype=dtype;resize.channels=channels;
                resize.source_h=down?37:7;resize.source_w=down?43:9;resize.destination_h=down?17:35;resize.destination_w=down?19:37;
                resize.stride_x=uint32_t((6ull*resize.source_w+resize.destination_w-1)/resize.destination_w+1);
                resize.stride_y=uint32_t((6ull*resize.source_h+resize.destination_h-1)/resize.destination_h+1);
                const size_t sp=size_t(resize.source_h)*resize.source_w,dp=size_t(resize.destination_h)*resize.destination_w;
                std::vector<uint8_t> b(sp*channels);std::vector<float> f(b.size()),got(dp*channels);
                for(size_t i=0;i<b.size();++i){b[i]=uint8_t((i*73)%256);f[i]=float(int((i*79)%313)-20)/271.f;}
                check(lfs_hip_upload(first,input,0,dtype==LFS_HIP_U8?static_cast<void*>(b.data()):static_cast<void*>(f.data()),b.size()*(dtype==LFS_HIP_U8?1:4),&error));
                check(lfs_hip_shared_resize(first,&resize,{input,0},{output,0},{wx,0},{wy,0},&error));
                check(lfs_hip_download(first,output,0,got.data(),got.size()*4,&error));
                for(int y=0;y<resize.destination_h;++y)for(int x=0;x<resize.destination_w;++x){auto ys=weights_cpu(resize.source_h,resize.destination_h,y),xs=weights_cpu(resize.source_w,resize.destination_w,x);for(size_t c=0;c<channels;++c){double expected=0;for(auto [iy,weight_y]:ys)for(auto [ix,weight_x]:xs){size_t pixel=size_t(iy)*resize.source_w+ix,index=mode==2?c*sp+pixel:pixel*channels+c;double value=dtype==LFS_HIP_U8?double(b[index])/255:f[index];expected+=value*weight_x*weight_y;}double delta=std::abs(got[c*dp+size_t(y)*resize.destination_w+x]-expected);max_resize_error=std::max(max_resize_error,delta);if(delta>=2e-5)std::fprintf(stderr,"Lanczos mismatch mode=%u dtype=%u C=%u down=%d xy=%d,%d c=%zu actual=%.9g expected=%.12g error=%.9g\n",mode,dtype,channels,int(down),x,y,c,got[c*dp+size_t(y)*resize.destination_w+x],expected,delta);require(delta<2e-5,"HIP Lanczos differs from independent CPU convolution");}}
            }
        }
        for(unsigned mode:{3u,4u}){
            resize.mode=mode;resize.dtype=LFS_HIP_F32;resize.channels=mode==3?1:3;resize.source_h=7;resize.source_w=9;resize.destination_h=13;resize.destination_w=17;
            const size_t sp=63,dp=221;std::vector<float> f(sp*resize.channels),got(dp*resize.channels);
            for(size_t c=0;c<resize.channels;++c)for(size_t i=0;i<sp;++i)f[c*sp+i]=(i%11==0)?std::numeric_limits<float>::quiet_NaN():(i%7==0)?0.f:mode==3?.1f+float(i%13)*.2f:float(int((i*73+c*37)%157)-78)/70;
            check(lfs_hip_upload(first,input,0,f.data(),f.size()*4,&error));
            check(lfs_hip_shared_resize(first,&resize,{input,0},{output,0},{},{},&error));check(lfs_hip_download(first,output,0,got.data(),got.size()*4,&error));
            auto valid=[&](size_t i){if(mode==3)return std::isfinite(f[i]) && f[i]>0;double squared=0;for(int c=0;c<3;++c){if(!std::isfinite(f[c*sp+i]))return false;squared+=double(f[c*sp+i])*f[c*sp+i];}return squared>=.25;};
            for(int y=0;y<13;++y)for(int x=0;x<17;++x){double sx=std::clamp((x+.5)*9/17-.5,0.,8.),sy=std::clamp((y+.5)*7/13-.5,0.,6.);int x0=int(sx),y0=int(sy);double vx[3]{},weight=0;
                if(valid(size_t(int(sy+.5))*9+int(sx+.5)))for(int j=0;j<2;++j)for(int i=0;i<2;++i){size_t idx=size_t(std::min(y0+j,6))*9+std::min(x0+i,8);if(!valid(idx))continue;double w=(i?sx-x0:1-sx+x0)*(j?sy-y0:1-sy+y0);weight+=w;for(unsigned c=0;c<resize.channels;++c)vx[c]+=w*f[c*sp+idx];}
                if(mode==4)weight=std::sqrt(vx[0]*vx[0]+vx[1]*vx[1]+vx[2]*vx[2]);
                for(unsigned c=0;c<resize.channels;++c){double expected=weight>1e-8?vx[c]/weight:0;require(std::abs(got[c*dp+y*17+x]-expected)<3e-6,"HIP validity-aware prior resize differs from CPU");}
            }
        }
        resize.mode=0;resize.channels=3;resize.dtype=LFS_HIP_F32;resize.source_h=7;resize.source_w=9;resize.destination_h=13;resize.destination_w=17;resize.stride_x=5;resize.stride_y=5;
        rejected(lfs_hip_shared_resize(first,&resize,{input,0},{input,0},{wx,0},{wy,0},&error));
        resize.stride_x=1;rejected(lfs_hip_shared_resize(first,&resize,{input,0},{output,0},{wx,0},{wy,0},&error));
        check(lfs_hip_buffer_release(first,wx,&error));check(lfs_hip_buffer_release(first,wy,&error));
        check(lfs_hip_upload(first,input,0,source.data(),bytes,&error));
        std::printf("PASS HIP shared resize: five modes, up/down, byte/float/channel layouts, invalid priors, workspace guards; Lanczos max error %.9g\n",max_resize_error);
    }
