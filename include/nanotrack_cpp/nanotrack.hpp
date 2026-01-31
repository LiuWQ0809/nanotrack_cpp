#pragma once
#include <memory>
#include <vector>
#include <cstdint>
#include <cuda_runtime.h>
#include "nanotrack_cpp/tensorrt_engine.hpp"
#include "nanotrack_cpp/preprocessor.hpp"
#include "nanotrack_cpp/kalman_filter.hpp"

namespace nanotrack {

struct BBox {
    float x1, y1, width, height;
};

struct TrackResult {
    BBox bbox;
    float score;
};

class NanoTrack {
public:
    NanoTrack(const std::string& backbone_template_path, 
              const std::string& backbone_search_path, 
              const std::string& head_path);
    ~NanoTrack();

    bool load();

    // Input: Host pointer to image data (BGR/RGB default BGR in python)
    // Runs entirely asynchronously on provided stream if possible, 
    // but post-processing is currently CPU blocking (can be optimized later).
    void init(const uint8_t* h_img, int width, int height, int step, const BBox& roi, cudaStream_t stream = 0);
    
    // Returns tracking result
    TrackResult track(const uint8_t* h_img, int width, int height, int step, cudaStream_t stream = 0);
    void set_track_lost_threshold(float threshold);
    void set_persistence_frames(int frames);

private:
    // Constants
    const int TEMPLATE_SIZE = 127;
    const int SEARCH_SIZE = 255;
    const int STRIDE = 16;
    const float CONTEXT_AMOUNT = 0.5f;
    const float WINDOW_INFLUENCE = 0.455f;
    const float SCALE_LR = 0.37f;
    const float PENALTY_K = 0.15f;
    float track_lost_threshold_ = 0.4f;
    int persistence_frames_ = 10;
    
    int grid_size_ = 16;
    
    // State
    bool is_initialized_ = false;
    float center_x_ = 0, center_y_ = 0;
    float target_w_ = 0, target_h_ = 0;
    float resize_scale_ = 1.0f;
    int origin_w_ = 0, origin_h_ = 0;
    
    // Engines
    std::shared_ptr<TrtEngine> backbone_template_;
    std::shared_ptr<TrtEngine> backbone_search_;
    std::shared_ptr<TrtEngine> head_;
    
    // Kalman
    SimpleKalmanFilter kf_;
    
    // Helpers
    std::vector<float> hanning_window_;
    std::vector<float> grid_x_;
    std::vector<float> grid_y_;

    // Device Buffers
    float *d_template_input_ = nullptr; // 1x3x127x127
    float *d_search_input_ = nullptr;   // 1x3x255x255
    float *d_template_features_ = nullptr; 
    float *d_search_features_ = nullptr;
    float *d_head_cls_ = nullptr;
    float *d_head_reg_ = nullptr;
    
    // Temporary full image buffer on GPU
    uint8_t *d_img_buffer_ = nullptr;
    size_t d_img_buffer_size_ = 0;
    
    // Host buffers for post-processing
    std::vector<float> h_cls_;
    std::vector<float> h_reg_;

    void generate_grids();
    void create_window();
    float subwindow_scale(float w, float h);
    
    // Logic
    void pre_process(const uint8_t* h_img, int w, int h, int step, 
                     float* d_output, int sz, 
                     float& scale_out, 
                     cudaStream_t stream);

    float size_cal(float w, float h);
    
    // State Machine
    enum State {
        TRACKING = 0,
        PERSISTENCE = 1,
        GLOBAL_SEARCH = 2
    } state_ = TRACKING;
    int lost_counter_ = 0;
};

} // namespace nanotrack
