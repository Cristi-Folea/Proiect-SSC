//
// Created by crist on 11/11/2026.
//

#ifndef TENSORRT_INFERENCE_TENSORENGINE_H
#define TENSORRT_INFERENCE_TENSORENGINE_H


#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <string>
#include <vector>

class TensorEngine {


public:
    TensorEngine(const std::string& enginePath); // constructor
    TensorEngine(const TensorEngine&) = delete; // copie
    TensorEngine& operator=(const TensorEngine&) = delete; // atribuire
    ~TensorEngine(); // destructor

    bool inference(const std::vector<float> & input, std::vector<float> &output);

private:

    // Logger
    class Logger;
    Logger * logger = nullptr;

    // Engine
    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;

    // Cuda
    cudaStream_t stream = nullptr;
    void* inputBuffer = nullptr;
    void* outputBuffer = nullptr;

    std::string inputName;
    std::string outputName;

    size_t inputSize  = 0;   // numar de elemente float
    size_t outputSize = 0;

};



#endif //TENSORRT_INFERENCE_TENSORENGINE_H
