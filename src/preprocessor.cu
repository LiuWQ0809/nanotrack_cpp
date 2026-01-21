#include "nanotrack_cpp/preprocessor.hpp"
#include <device_launch_parameters.h>
#include <cstdint>

namespace nanotrack {

__global__ void crop_resize_kernel(
    const uint8_t* __restrict__ src, int src_w, int src_h, int src_step,
    float* __restrict__ dst, int dst_w, int dst_h,
    float center_x, float center_y, float scale,
    float mean0, float mean1, float mean2
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dst_w || y >= dst_h) return;

    // Calculate source coordinates corresponding to this output pixel
    // Output center (dst_w/2, dst_h/2) maps to (center_x, center_y) in source
    float src_x = center_x + (x - dst_w * 0.5f) * scale;
    float src_y = center_y + (y - dst_h * 0.5f) * scale;

    // Bilinear Interpolation
    int x_low = floorf(src_x);
    int y_low = floorf(src_y);
    int x_high = x_low + 1;
    int y_high = y_low + 1;

    float dx = src_x - x_low;
    float dy = src_y - y_low;

    float w_tl = (1.0f - dx) * (1.0f - dy);
    float w_tr = dx * (1.0f - dy);
    float w_bl = (1.0f - dx) * dy;
    float w_br = dx * dy;

    // Channel offsets for CHW output
    int area = dst_w * dst_h;
    float* dst_c0 = dst; 
    float* dst_c1 = dst + area; 
    float* dst_c2 = dst + area * 2;
    int dst_idx = y * dst_w + x;

    // Padding Logic (Out of bounds)
    // Check if the 2x2 block is strictly within bounds or partially out?
    // OpenCV copyMakeBorder pads the image first. Here we sample.
    // If we sample outside, use mean.
    
    // Simplification: Point sampling logic for padding check
    // If the top-left sampling point is out, we might be on edge.
    // Let's check center of sample.
    bool valid_tl = (x_low >= 0 && x_low < src_w && y_low >= 0 && y_low < src_h);
    bool valid_tr = (x_high >= 0 && x_high < src_w && y_low >= 0 && y_low < src_h);
    bool valid_bl = (x_low >= 0 && x_low < src_w && y_high >= 0 && y_high < src_h);
    bool valid_br = (x_high >= 0 && x_high < src_w && y_high >= 0 && y_high < src_h);

    // If completely out, fill with mean
    if (!valid_tl && !valid_tr && !valid_bl && !valid_br) {
        dst_c0[dst_idx] = mean0;
        dst_c1[dst_idx] = mean1;
        dst_c2[dst_idx] = mean2;
        return;
    }

    // Helper to fetch pixel
    auto get_pixel = [&](int px, int py, int c) -> float {
        if (px < 0 || px >= src_w || py < 0 || py >= src_h) {
            // If strictly using copyMakeBorder logic:
            // The border is constant. So if we sample coordinate -1, it is Mean.
            if (c==0) return mean0;
            if (c==1) return mean1;
            return mean2;
        }
        return (float)src[py * src_step + px * 3 + c];
    };

    float val0 = w_tl * get_pixel(x_low, y_low, 0) + 
                 w_tr * get_pixel(x_high, y_low, 0) +
                 w_bl * get_pixel(x_low, y_high, 0) +
                 w_br * get_pixel(x_high, y_high, 0);

    float val1 = w_tl * get_pixel(x_low, y_low, 1) + 
                 w_tr * get_pixel(x_high, y_low, 1) +
                 w_bl * get_pixel(x_low, y_high, 1) +
                 w_br * get_pixel(x_high, y_high, 1);

    float val2 = w_tl * get_pixel(x_low, y_low, 2) + 
                 w_tr * get_pixel(x_high, y_low, 2) +
                 w_bl * get_pixel(x_low, y_high, 2) +
                 w_br * get_pixel(x_high, y_high, 2);

    dst_c0[dst_idx] = val0;
    dst_c1[dst_idx] = val1;
    dst_c2[dst_idx] = val2;
}

void cuda_crop_resize(
    const uint8_t* src, int src_w, int src_h, int src_step,
    float* dst, int dst_w, int dst_h,
    const PreprocessorParams& params,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);

    crop_resize_kernel<<<grid, block, 0, stream>>>(
        src, src_w, src_h, src_step,
        dst, dst_w, dst_h,
        params.center_x, params.center_y, params.scale,
        params.img_mean[0], params.img_mean[1], params.img_mean[2]
    );
}

} // namespace nanotrack
