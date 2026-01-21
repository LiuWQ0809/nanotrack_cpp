#pragma once
#include <vector>
#include <array>

namespace nanotrack {

class SimpleKalmanFilter {
public:
    SimpleKalmanFilter();
    
    // Initialize with state and defaults
    void init(float x, float y);
    
    // Predict next state
    // Returns {x, y, dx, dy}
    std::array<float, 4> predict();
    
    // Correct with measurement {x, y}
    // Returns updated {x, y, dx, dy}
    std::array<float, 4> correct(float x, float y);

private:
    // State: [x, y, dx, dy]T
    // Measurement: [x, y]T
    
    // Matrices flattened row-major
    // F: 4x4
    float F[16];
    // H: 2x4
    float H[8];
    // P: 4x4 (Error Covariance)
    float P[16];
    // Q: 4x4 (Process Noise)
    float Q[16];
    // R: 2x2 (Measurement Noise)
    float R[4];
    // State vector x
    float state[4];
    
    // Helpers
    void mat_mul(const float* A, int rA, int cA, const float* B, int rB, int cB, float* C);
    void mat_add(const float* A, const float* B, int size, float* C);
    void mat_sub(const float* A, const float* B, int size, float* C);
    void mat_transpose(const float* A, int r, int c, float* AT);
    bool mat_inv2x2(const float* A, float* A_inv);
};

} // namespace nanotrack
