#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <cstdint>

namespace nanotrack {

struct PreprocessorParams {
    float center_x;
    float center_y;
    float scale; // crop_size / output_size
    float img_mean[3]; // For padding
};

// C++ API to launch the kernel
// src: Device pointer to uint8_t image (HWC, usually BGR or RGB). 
//      Python code says "BGR" default.
// dst: Device pointer to float/half buffer (CHW, 1x3xHxW).
// src_step: Bytes per row in source image.
void cuda_crop_resize(
    const uint8_t* src, int src_w, int src_h, int src_step,
    float* dst, int dst_w, int dst_h,
    const PreprocessorParams& params,
    cudaStream_t stream
);

} // namespace nanotrack
