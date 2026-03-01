
#include <cuda_runtime.h>

namespace lfs::training::kernels {

    void launch_grayscale_filter(const float* d_input, float* d_output, const int height, const int width);
    void launch_laplacian_filter(const float* d_input, float* d_output, const int k_size, const int height, const int width);
    void launch_gausssian_blur(const float* d_input, float* d_output, const int k_size, const int height, const int width);
    void launch_sobel_gradient_filter(const float* d_input, float* d_magnitude, float* d_angle, const int height, const int width);
    void launch_nms_kernel(const float* d_magnitude, const float* d_angle, float* d_output, const int height, const int width);

} // namespace lfs::training::kernels