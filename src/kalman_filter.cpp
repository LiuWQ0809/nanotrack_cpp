#include "nanotrack_cpp/kalman_filter.hpp"
#include <cstring>
#include <cmath>

namespace nanotrack {

SimpleKalmanFilter::SimpleKalmanFilter() {
    // Default Initialization similar to Python code
    // F (Transition) = Identity + dt stuff
    // 1 0 1 0
    // 0 1 0 1
    // 0 0 1 0
    // 0 0 0 1
    std::memset(F, 0, sizeof(F));
    F[0] = 1; F[2] = 1;
    F[5] = 1; F[7] = 1;
    F[10] = 1;
    F[15] = 1;

    // H (Measurement) = Identity 2x4 (Measure x, y)
    // 1 0 0 0
    // 0 1 0 0
    std::memset(H, 0, sizeof(H));
    H[0] = 1;
    H[5] = 1; // H is 2x4. Index 5 is row 1, col 1 (if row-major 2x4: 0,1,2,3 | 4,5,6,7). Yes.

    // Q (Process Noise) = 0.01 * Eye
    std::memset(Q, 0, sizeof(Q));
    Q[0] = 0.01f; Q[5] = 0.01f; Q[10] = 0.01f; Q[15] = 0.01f;

    // R (Measurement Noise) = 0.1 * Eye
    std::memset(R, 0, sizeof(R));
    R[0] = 0.1f; R[3] = 0.1f;

    // P (Covariance) = Identity
    std::memset(P, 0, sizeof(P));
    P[0] = 1; P[5] = 1; P[10] = 1; P[15] = 1;
    
    std::memset(state, 0, sizeof(state));
}

void SimpleKalmanFilter::init(float x, float y) {
    state[0] = x;
    state[1] = y;
    state[2] = 0;
    state[3] = 0;
    
    // Reset P
    std::memset(P, 0, sizeof(P));
    P[0] = 1; P[5] = 1; P[10] = 1; P[15] = 1;
}

std::array<float, 4> SimpleKalmanFilter::predict() {
    float new_state[4];
    // x = F * x
    mat_mul(F, 4, 4, state, 4, 1, new_state);
    std::memcpy(state, new_state, sizeof(state));

    // P = F * P * F^T + Q
    float FT[16];
    mat_transpose(F, 4, 4, FT);
    
    float FP[16];
    mat_mul(F, 4, 4, P, 4, 4, FP);
    
    float FPFt[16];
    mat_mul(FP, 4, 4, FT, 4, 4, FPFt);
    
    mat_add(FPFt, Q, 16, P);
    
    return {state[0], state[1], state[2], state[3]};
}

std::array<float, 4> SimpleKalmanFilter::correct(float zx, float zy) {
    float z[2] = {zx, zy};
    
    // y = z - H * x (Innovation)
    float Hx[2];
    mat_mul(H, 2, 4, state, 4, 1, Hx);
    
    float y[2];
    mat_sub(z, Hx, 2, y);
    
    // S = H * P * H^T + R
    float HT[8];
    mat_transpose(H, 2, 4, HT);
    
    float HP[8]; // 2x4
    mat_mul(H, 2, 4, P, 4, 4, HP);
    
    float HPHt[4]; // 2x2
    mat_mul(HP, 2, 4, HT, 4, 2, HPHt);
    
    float S[4];
    mat_add(HPHt, R, 4, S);
    
    // K = P * H^T * S^-1
    float S_inv[4];
    if (!mat_inv2x2(S, S_inv)) {
        // Fallback if singular (rare)
        return {state[0], state[1], state[2], state[3]};
    }
    
    float PHt[8]; // 4x2
    mat_mul(P, 4, 4, HT, 4, 2, PHt);
    
    float K[8]; // 4x2
    mat_mul(PHt, 4, 2, S_inv, 2, 2, K);
    
    // x = x + K * y
    float Ky[4];
    mat_mul(K, 4, 2, y, 2, 1, Ky);
    mat_add(state, Ky, 4, state);
    
    // P = (I - K * H) * P
    float KH[16]; // 4x4
    mat_mul(K, 4, 2, H, 2, 4, KH);
    
    float I[16] = {0};
    I[0]=1; I[5]=1; I[10]=1; I[15]=1;
    
    float I_minus_KH[16];
    mat_sub(I, KH, 16, I_minus_KH);
    
    float new_P[16];
    mat_mul(I_minus_KH, 4, 4, P, 4, 4, new_P);
    std::memcpy(P, new_P, sizeof(P));
    
    return {state[0], state[1], state[2], state[3]};
}

// Minimal matrix utils
void SimpleKalmanFilter::mat_mul(const float* A, int rA, int cA, const float* B, int rB, int cB, float* C) {
    if (cA != rB) return; 
    for(int i=0; i<rA; ++i) {
        for(int j=0; j<cB; ++j) {
            float sum = 0;
            for(int k=0; k<cA; ++k) {
                sum += A[i*cA + k] * B[k*cB + j];
            }
            C[i*cB + j] = sum;
        }
    }
}

void SimpleKalmanFilter::mat_add(const float* A, const float* B, int size, float* C) {
    for(int i=0; i<size; ++i) C[i] = A[i] + B[i];
}
void SimpleKalmanFilter::mat_sub(const float* A, const float* B, int size, float* C) {
    for(int i=0; i<size; ++i) C[i] = A[i] - B[i];
}
void SimpleKalmanFilter::mat_transpose(const float* A, int r, int c, float* AT) {
    for(int i=0; i<r; ++i) {
        for(int j=0; j<c; ++j) {
            AT[j*r + i] = A[i*c + j];
        }
    }
}
bool SimpleKalmanFilter::mat_inv2x2(const float* A, float* A_inv) {
    float det = A[0]*A[3] - A[1]*A[2];
    if (std::abs(det) < 1e-6) return false;
    float invDet = 1.0f / det;
    A_inv[0] = A[3] * invDet;
    A_inv[1] = -A[1] * invDet;
    A_inv[2] = -A[2] * invDet;
    A_inv[3] = A[0] * invDet;
    return true;
}

} // namespace nanotrack
