#pragma once
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <iostream>

namespace nanotrack {

class TrtEngine {
public:
    TrtEngine(const std::string& engine_path);
    ~TrtEngine();

    bool load();
    
    // Inputs: name -> device pointer
    // Outputs: name -> device pointer (pre-allocated)
    // Returns true on success
    bool infer(const std::map<std::string, void*>& inputs, 
               const std::map<std::string, void*>& outputs, 
               cudaStream_t stream);

    nvinfer1::Dims getBindingDims(const std::string& name);
    nvinfer1::DataType getBindingDataType(const std::string& name);
    int getBindingIndex(const std::string& name);

    float getLastGpuTime();

    // Public names for convenience
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::shared_ptr<nvinfer1::IRuntime> runtime_;
    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    std::shared_ptr<nvinfer1::IExecutionContext> context_;

    cudaEvent_t start_event_, end_event_;
    float last_gpu_time_ms_ = 0.0f;
    std::string engine_path_;
    
    // Logger for TRT
    class Logger : public nvinfer1::ILogger {
        void log(Severity severity, const char* msg) noexcept override {
            if (severity <= Severity::kWARNING)
                std::cout << "[TRT] " << msg << std::endl;
        }
    } logger_;
};

} // namespace nanotrack
