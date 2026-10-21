/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include <core/tensor.hpp>
#include <core/tensor_backend.hpp>
#include "core/tensor/internal/tensor_impl.hpp"
#include "core/tensor/backend/gpu_backend_ops.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>
using namespace lfs::core;
using namespace lfs::core::internal;
static void require(bool v,const char* message){if(!v)throw std::runtime_error(message);}
static StorageRef ref(const Tensor& t){(void)const_cast<Tensor&>(t).data_ptr();return storage_ref(t);}
static void near(double a,double b,double tolerance,const char* message){if(!std::isfinite(a)||std::abs(a-b)>tolerance*(1+std::abs(b))){std::fprintf(stderr,"%.9g vs %.9g\n",a,b);throw std::runtime_error(message);}}
static std::vector<float> data(size_t count){std::vector<float> v(count);for(size_t i=0;i<count;++i)v[i]=float(int(i*31%137)-68)/79.f;return v;}
int main()try {
    require(bool(set_default_gpu_backend(GpuBackend::HIP)),"Cannot select HIP");
    {
        auto& ops=backend_ops(GpuBackend::HIP);
        {
            PoolProgram p{.batch=2,.channels=3,.input_height=7,.input_width=9,.output_height=4,.output_width=5,.kernel_size=3,.stride=2,.padding=1};
            const auto values=data(2*3*7*9);auto source=Tensor::from_vector(values,{2,3,7,9},Device::GPU),out=Tensor::empty({2,3,4,5},Device::GPU);
            ops.max_pool2d(ref(source),ref(out),p,{});const auto actual=out.cpu().to_vector();
            for(int plane=0;plane<6;++plane)for(int y=0;y<4;++y)for(int x=0;x<5;++x) {
                float expected=-INFINITY;for(int ky=0;ky<3;++ky)for(int kx=0;kx<3;++kx){const int iy=y*2-1+ky,ix=x*2-1+kx;if(iy>=0&&iy<7&&ix>=0&&ix<9)expected=std::fmax(expected,values[(plane*7+iy)*9+ix]);}
                require(actual[(plane*4+y)*5+x]==expected,"HIP legacy max-pooling differs from CPU");
            }
            for(const auto size:{std::array<int,2>{3,4},std::array<int,2>{11,13},std::array<int,2>{1,1}}) {
                p.output_height=size[0];p.output_width=size[1];out=Tensor::empty({2,3,size_t(size[0]),size_t(size[1])},Device::GPU);
                ops.adaptive_avg_pool2d(ref(source),ref(out),p,{});const auto pooled=out.cpu().to_vector();
                for(int plane=0;plane<6;++plane)for(int y=0;y<size[0];++y)for(int x=0;x<size[1];++x) {
                    const int y0=y*7/size[0],y1=((y+1)*7+size[0]-1)/size[0],x0=x*9/size[1],x1=((x+1)*9+size[1]-1)/size[1];double sum=0;
                    for(int iy=y0;iy<y1;++iy)for(int ix=x0;ix<x1;++ix)sum+=values[(plane*7+iy)*9+ix];
                    near(pooled[(plane*size[0]+y)*size[1]+x],sum/((y1-y0)*(x1-x0)),2e-7,"HIP adaptive pooling differs from CPU");
                }
            }
        }
        for(const auto dimensions:{std::array<int,3>{42,3,7},std::array<int,3>{263,5,11},std::array<int,3>{1,1,1}}) {
            const auto [count,channels,spatial]=dimensions;const auto values=data(count),bias_values=data(channels);
            auto source=Tensor::from_vector(values,{size_t(count)},Device::GPU),bias=Tensor::from_vector(bias_values,{size_t(channels)},Device::GPU);
            for(int mode=0;mode<3;++mode)for(bool inplace:{false,true}) {
                auto backing=Tensor::full({size_t(count)+2},99.f,Device::GPU),out=backing.slice(0,1,1+count);
                out.copy_from(source);
                const auto& input=inplace?out:source;
                if(mode==0)ops.bias_add(ref(input),ref(bias),ref(out),count,channels,spatial,{});
                if(mode==1)ops.bias_relu(ref(input),ref(bias),ref(out),count,channels,spatial,{});
                if(mode==2)ops.relu(ref(input),ref(out),count,{});
                const auto actual=out.cpu().to_vector();
                for(int i=0;i<count;++i){const float expected=mode==2?std::fmax(values[i],0.f):values[i]+bias_values[(i/spatial)%channels];near(actual[i],mode==1?std::fmax(expected,0.f):expected,1e-7,"HIP bias/activation differs from CPU");}
                const auto guarded=backing.cpu().to_vector();require(guarded.front()==99&&guarded.back()==99,"HIP bias/activation overwrote output guards");
            }
        }
        for(const auto shape:{std::array<size_t,4>{2,7,11,19},{1,65,67,17},{1,1,259,9},{1,11,9,0}}) {
            const auto [batch,m,n,k]=shape;
            const auto aa=data(batch*m*k),bb=data(batch*k*n),bias=data(n);
            auto a=Tensor::from_vector(aa,{batch,m,k},Device::GPU),b=Tensor::from_vector(bb,{batch,k,n},Device::GPU);
            auto d=Tensor::from_vector(bias,{n},Device::GPU);
            for(unsigned mode=0;mode<4;++mode) {
                auto backing=Tensor::full({batch*m*n+2},99.f,Device::GPU);
                auto c=backing.slice(0,1,1+batch*m*n);
                const GemmProgram program{batch,m,n,k};
                if(mode==0)ops.sgemm(ref(a),ref(b),ref(c),program,{});
                if(mode==1)ops.sgemm_tn(ref(a),ref(b),ref(c),program,{});
                if(mode==2)ops.sgemm_batched(ref(a),ref(b),ref(c),program,{});
                if(mode==3)ops.sgemm_bias_relu(ref(a),ref(b),ref(d),ref(c),program,{});
                const auto actual=c.cpu().to_vector();
                for(size_t z=0;z<batch;++z)for(size_t i=0;i<m;++i)for(size_t j=0;j<n;++j) {
                    double expected=0;
                    for(size_t t=0;t<k;++t)expected+=double(aa[(z*m+i)*k+t])*bb[z*k*n+(mode==1?j*k+t:t*n+j)];
                    if(mode==3)expected=std::max(0.,expected+bias[j]);
                    near(actual[(z*m+i)*n+j],expected,3e-5,"HIP tiled GEMM differs from independent CPU dot products");
                }
                const auto guards=backing.cpu().to_vector();require(guards.front()==99 && guards.back()==99,"HIP GEMM overwrote output guards");
            }
        }
        constexpr int h=7,w=9,c=2,kh=3,kw=2,oh=4,ow=7;
        auto values=data(c*h*w);auto input=Tensor::from_vector(values,{c,h,w},Device::GPU);
        InferenceGeometry g{.channels=c,.height=h,.width=w,.out_height=oh,.out_width=ow,.kernel_h=kh,.kernel_w=kw,
            .stride_h=2,.stride_w=1,.pad_h=1,.pad_w=0,.dilation_h=1,.dilation_w=2,.offset=3,.columns=10};
        for(int padding=0;padding<2;++padding) {
            g.mode=padding;auto result=Tensor::empty({size_t(c*kh*kw*g.columns)},Device::GPU);
            ops.inference(ref(input),ref(result),{InferenceKernel::Im2col,result.numel(),g},{});
            const auto actual=result.cpu().to_vector();
            for(size_t i=0;i<actual.size();++i) {
                int column=int(i)%g.columns+g.offset,tap=int(i)/g.columns;
                int x=column%ow+tap%kw*2,y=column/ow*2-1+(tap/kw)%kh,ch=tap/(kh*kw);
                float expected=0;
                if(padding || (x>=0&&x<w&&y>=0&&y<h))expected=values[(ch*h+std::clamp(y,0,h-1))*w+std::clamp(x,0,w-1)];
                require(actual[i]==expected,"HIP im2col tile/replicate padding differs from CPU");
            }
        }
        auto folded_values=data(c*kh*kw*oh*ow);
        auto columns=Tensor::from_vector(folded_values,{c*kh*kw,oh*ow},Device::GPU),fold=Tensor::empty({c,h,w},Device::GPU);
        g.height=oh;g.width=ow;g.out_height=h;g.out_width=w;
        g.channels=1; // Transposed convolution: original input channels differ from output planes.
        ops.inference(ref(columns),ref(fold),{InferenceKernel::Col2im,fold.numel(),g},{});
        const auto folded=fold.cpu().to_vector();
        for(int ch=0;ch<c;++ch)for(int y=0;y<h;++y)for(int x=0;x<w;++x) {
            double expected=0;
            for(int ky=0;ky<kh;++ky)for(int kx=0;kx<kw;++kx) {
                int iy=y+1-ky,ix=x-kx*2;
                if(iy<0||ix<0||iy%2)continue;iy/=2;
                if(iy<oh && ix<ow)expected+=folded_values[((ch*kh+ky)*kw+kx)*oh*ow+iy*ow+ix];
            }
            near(folded[(ch*h+y)*w+x],expected,2e-7,"HIP col2im differs from CPU fold");
        }
        // Activation and geometry dispatches do not need matrix multiplication.
        for(int activation=0;activation<5;++activation) {
            auto result=Tensor::empty(input.shape(),Device::GPU);
            ops.inference(ref(input),ref(result),{InferenceKernel::Activation,input.numel(),{.mode=activation}},{});
            const auto actual=result.cpu().to_vector();
            for(size_t i=0;i<actual.size();++i) {
                const double x=values[i];double expected=x;
                if(activation==1)expected=std::max(0.,x);
                if(activation==2)expected=.5*x*(1+std::tanh(.7978845608028654*(x+.044715*x*x*x)));
                if(activation==3)expected=.5*x*(1+std::erf(x*.7071067811865475));
                if(activation==4)expected=x/(1+std::exp(-x));
                near(actual[i],expected,3e-7,"HIP activation differs from CPU");
            }
        }
        auto coordinate=[](int i,int in,int out,int mode){return out==1?0.f:mode==2?float(i)*(in-1)/(out-1):mode==1?float(i)*in/out:(float(i)+.5f)*in/out-.5f;};
        auto cubic=[](float v){float x=std::abs(v);return x<=1?((1.25f*x-2.25f)*x)*x+1:x<2?((-0.75f*x+3.75f)*x-6)*x+3:0.f;};
        for(int mode=0;mode<3;++mode)for(int coord=0;coord<3;++coord) {
            constexpr int dh=11,dw=5;
            auto result=Tensor::empty({c,dh,dw},Device::GPU);
            ops.inference(ref(input),ref(result),{InferenceKernel::Resize,result.numel(),{.height=h,.width=w,.out_height=dh,.out_width=dw,.mode=mode,.coord=coord}},{});
            const auto actual=result.cpu().to_vector();
            for(int ch=0;ch<c;++ch)for(int y=0;y<dh;++y)for(int x=0;x<dw;++x) {
                float fx=coordinate(x,w,dw,coord),fy=coordinate(y,h,dh,coord);int ix=int(std::floor(fx)),iy=int(std::floor(fy));
                auto sample=[&](int yy,int xx){return values[(ch*h+std::clamp(yy,0,h-1))*w+std::clamp(xx,0,w-1)];};
                double expected=0;
                if(mode==0)expected=sample(iy,ix);
                else if(mode==1){double dx=fx-ix,dy=fy-iy;expected=(sample(iy,ix)*(1-dx)+sample(iy,ix+1)*dx)*(1-dy)+(sample(iy+1,ix)*(1-dx)+sample(iy+1,ix+1)*dx)*dy;}
                else for(int yy=-1;yy<=2;++yy)for(int xx=-1;xx<=2;++xx)expected+=sample(iy+yy,ix+xx)*cubic(fy-(iy+yy))*cubic(fx-(ix+xx));
                near(actual[(ch*dh+y)*dw+x],expected,1e-6,"HIP resize coordinate/kernel differs from CPU");
            }
        }
        auto grid=Tensor::empty({2,3,7},Device::GPU);
        ops.inference({},ref(grid),{InferenceKernel::Grid,grid.numel(),{.height=3,.width=7,.u0=-1,.u1=2,.v0=-2,.v1=3}},{});
        const auto xy=grid.cpu().to_vector();
        for(int y=0;y<3;++y)for(int x=0;x<7;++x){near(xy[y*7+x],-1.f+3.f*x/6,0,"HIP horizontal grid changed");near(xy[21+y*7+x],-2.f+5.f*y/2,0,"HIP vertical grid changed");}
        auto guard=Tensor::full({c*kh*kw*10},99.f,Device::GPU);
        g={.channels=c,.height=h,.width=w,.out_height=oh,.out_width=ow,.kernel_h=kh,.kernel_w=kw,
           .stride_h=2,.stride_w=1,.dilation_h=1,.dilation_w=1,.offset=oh*ow,.columns=10};
        bool rejected=false;
        try{ops.inference(ref(input),ref(guard),{InferenceKernel::Im2col,guard.numel(),g},{});}catch(const std::exception&){rejected=true;}
        require(rejected,"HIP accepted an im2col tile beyond its image");
        for(float v:guard.cpu().to_vector())require(v==99.f,"Rejected im2col dispatch modified its output");
        rejected=false;
        try{ops.sgemm(ref(input),ref(input),ref(input),{1,3,3,3},{});}catch(const std::exception&){rejected=true;}
        require(rejected && input.cpu().to_vector()==values,"Rejected GEMM alias modified its source");
    }
    require(bool(shutdown_gpu_backend(GpuBackend::HIP)),"HIP NN contract retained storage");
    std::puts("PASS HIP NN: four GEMM paths with batches/odd tile edges/zero K/guards, im2col and col2im CPU references, five activations, nine resize policies and normalized grids");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"HIP NN FAILED: %s\n",e.what());return 1;}
