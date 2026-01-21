#include "nanotrack_cpp/tensorrt_engine.hpp"
#include <fstream>
#include <iostream>
#include <cassert>

namespace nanotrack {

TrtEngine::TrtEngine(const std::string& engine_path) 
    : engine_path_(engine_path) {
}

TrtEngine::~TrtEngine() {
    if (context_) context_.reset();
    if (engine_) engine_.reset();
    if (runtime_) runtime_.reset();
    cudaEventDestroy(start_event_);
    cudaEventDestroy(end_event_);
}

bool TrtEngine::load() {
    std::ifstream file(engine_path_, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Error opening engine file: " << engine_path_ << std::endl;
        return false;
    }
    
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);
    
    std::vector<char> engine_data(size);
    file.read(engine_data.data(), size);
    file.close();

    runtime_ = std::shared_ptr<nvinfer1::IRuntime>(
        nvinfer1::createInferRuntime(logger_), 
        [](nvinfer1::IRuntime* ptr){ delete ptr; }
    );
    
    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime_->deserializeCudaEngine(engine_data.data(), size),
        [](nvinfer1::ICudaEngine* ptr){ delete ptr; }
    );

    if (!engine_) return false;

    context_ = std::shared_ptr<nvinfer1::IExecutionContext>(
        engine_->createExecutionContext(),
        [](nvinfer1::IExecutionContext* ptr){ delete ptr; }
    );

    cudaEventCreate(&start_event_);
    cudaEventCreate(&end_event_);

    // Populate names
    input_names.clear();
    output_names.clear();
    int nbTensors = engine_->getNbIOTensors();
    for (int i=0; i<nbTensors; ++i) {
        const char* name = engine_->getIOTensorName(i);
        if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            input_names.push_back(name);
        } else {
            output_names.push_back(name);
        }
    }

    return true;
}

bool TrtEngine::infer(const std::map<std::string, void*>& inputs, 
           const std::map<std::string, void*>& outputs, 
           cudaStream_t stream) 
{
    if (!context_) return false;

    for (const auto& pair : inputs) {
        if (!context_->setTensorAddress(pair.first.c_str(), pair.second)) {
            std::cerr << "Failed to set input tensor address: " << pair.first << std::endl;
            return false;
        }
    }

    for (const auto& pair : outputs) {
        if (!context_->setTensorAddress(pair.first.c_str(), pair.second)) {
            std::cerr << "Failed to set output tensor address: " << pair.first << std::endl;
            return false;
        }
    }

    cudaEventRecord(start_event_, stream);
    bool status = context_->enqueueV3(stream);
    cudaEventRecord(end_event_, stream);
    
    if (status) {
        cudaEventSynchronize(end_event_);
        float ms = 0;
        cudaEventElapsedTime(&ms, start_event_, end_event_);
        last_gpu_time_ms_ = ms;
    }
    
    return status;
}

nvinfer1::Dims TrtEngine::getBindingDims(const std::string& name) {
    return engine_->getTensorShape(name.c_str());
}

nvinfer1::DataType TrtEngine::getBindingDataType(const std::string& name) {
    return engine_->getTensorDataType(name.c_str());
}

int TrtEngine::getBindingIndex(const std::string& name) {
    // Note: getBindingIndex is deprecated/removed in TRT 10. 
    // This is a dummy implementation returning a unique-ish ID if needed, 
    // but names are preferred.
    for (int i=0; i<engine_->getNbIOTensors(); ++i) {
        if (name == engine_->getIOTensorName(i)) return i;
    }
    return -1;
}

float TrtEngine::getLastGpuTime() {
    return last_gpu_time_ms_;
}

} // namespace nanotrack
