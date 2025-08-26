#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

#include "Common.h"
#include "QuatScaleToCovar.h"
#include "Utils.cuh"

namespace gsplat {

    namespace cg = cooperative_groups;

    template <typename scalar_t>
    __global__ void quat_to_rotmat_kernel(
        const uint32_t N,
        const scalar_t* __restrict__ quats, // [N, 4]
        // outputs
        scalar_t* __restrict__ rotmats // [N, 3, 3]
    ) {
        // parallelize over N.
        uint32_t idx = cg::this_grid().thread_rank();
        if (idx >= N) {
            return;
        }

        // shift pointers to the current gaussian
        rotmats += idx * 9;

        const vec4 quat = glm::make_vec4(quats + idx * 4);
        const mat3 rotmat = quat_to_rotmat(quat);

#pragma unroll
        for (uint32_t i = 0; i < 3; i++) { // rows
#pragma unroll
            for (uint32_t j = 0; j < 3; j++) { // cols
                rotmats[i * 3 + j] = rotmat[j][i];
            }
        }
    }

    void launch_quats_to_rotmats_kernel(
        // inputs
        const at::Tensor quats, // [N, 4]
        // outputs
        at::Tensor rotmats // [N, 3, 3]
    ) {
        uint32_t N = quats.size(0);

        int64_t n_elements = N;
        dim3 threads(256);
        dim3 grid((n_elements + threads.x - 1) / threads.x);
        int64_t shmem_size = 0; // No shared memory used in this kernel

        if (n_elements == 0) {
            // skip the kernel launch if there are no elements
            return;
        }
        AT_DISPATCH_FLOATING_TYPES(
            quats.scalar_type(),
            "quat_to_rotmat_kernel",
            [&]() {
                quat_to_rotmat_kernel<scalar_t>
                    <<<grid,
                       threads,
                       shmem_size,
                       at::cuda::getCurrentCUDAStream()>>>(
                        N,
                        quats.data_ptr<scalar_t>(),
                        rotmats.data_ptr<scalar_t>());
            });
    }

} // namespace gsplat
