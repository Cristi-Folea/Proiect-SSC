//
// Created by crist on 11/11/2026.
//

#include "TensorEngine.h"


#include <fstream>
#include <vector>
#include <iostream>

std::streamsize getFileSize(std::ifstream &inputFile) {
    std::streamsize fileSize;
    std::streamsize currentPosition = inputFile.tellg();
    if (currentPosition < 0) {
        return -1;
    }

    inputFile.seekg(0, std::ios::end);
    fileSize = inputFile.tellg();

    inputFile.seekg(currentPosition, std::ios::beg);
    return fileSize;
}

class TensorEngine::Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, nvinfer1::AsciiChar const *msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "TensorEngine: " << msg << '\n';
        }
    }
};

TensorEngine::TensorEngine(const std::string &enginePath) {
    logger = new Logger();
    std::ifstream inputEngineFile;
    inputEngineFile.open(enginePath, std::ios::binary);

    if (!inputEngineFile.is_open()) {
        std::cerr << "TensorEngine: Cannot open engine file \n";
        return;
    }
    std::streamsize fileSize = getFileSize(inputEngineFile);

    char *engineData;
    engineData = new char[fileSize];

    inputEngineFile.read(engineData, fileSize);

    runtime = nvinfer1::createInferRuntime(*logger);

    engine = runtime->deserializeCudaEngine(engineData, fileSize);
    delete[] engineData;

    if (engine == nullptr) {
        std::cerr << "TensorEngine: Cannot deserialize engine file \n";
        return;
    }

    context = engine->createExecutionContext();

    int tensorCount = engine->getNbIOTensors();
    int index = 0;

    while (index < tensorCount) {
        const char *tensorName = engine->getIOTensorName(index);
        nvinfer1::TensorIOMode tensorMode = engine->getTensorIOMode(tensorName);

        if (tensorMode == nvinfer1::TensorIOMode::kINPUT) {
            inputName = tensorName;
        }

        if (tensorMode == nvinfer1::TensorIOMode::kOUTPUT) {
            outputName = tensorName;
        }

        index++;
    }

    if (inputName.empty() || outputName.empty()) {
        std::cerr << "TensorEngine: Input or ouput tensor not found\n";
        return;
    }

    nvinfer1::Dims inputDims = engine->getTensorShape(inputName.c_str());
    nvinfer1::Dims outputDims = engine->getTensorShape(outputName.c_str());


    // Numarul total de elemente
    inputSize = 1;
    for (int i = 0; i < inputDims.nbDims; i++) {
        if (inputDims.d[i] <= 0) {
            std::cerr << "TensorEngine: dynamic input dimension \n";
            return;
        }
        inputSize *= inputDims.d[i];
    }

    outputSize = 1;
    for (int i = 0; i < outputDims.nbDims; i++) {
        if (outputDims.d[i] <= 0) {
            std::cerr << "TensorEngine: dynamic output dimension \n";
            return;
        }
        outputSize *= outputDims.d[i];
    }

    // stream CUDA
    cudaError_t error = cudaStreamCreate(&stream);
    if (error != cudaSuccess) {
        std::cerr << "TensorEngine: Cannot create CUDA stream \n";
        return;
    }

    error = cudaMalloc(&inputBuffer, inputSize * sizeof(float));
    if (error != cudaSuccess) {
        std::cerr << "TensorEngine: Cannot allocate CUDA input buffer \n";
        return;
    }

    error = cudaMalloc(&outputBuffer, outputSize * sizeof(float));
    if (error != cudaSuccess) {
        std::cerr << "TensorEngine: Cannot allocate CUDA output buffer \n";
        return;
    }

    if (!context->setTensorAddress(inputName.c_str(), inputBuffer) ||
    !context->setTensorAddress(outputName.c_str(), outputBuffer)) {
        std::cerr << "TensorEngine: Cannot set tensor address \n";
        return;
    }



    std::cout << "TensorEngine: loaded " << enginePath
            << " | input= " << inputSize
            << " | output= " << outputSize << "\n";
}

TensorEngine::~TensorEngine() {
    if (stream) {
        cudaStreamDestroy(stream);
        stream = nullptr;
    }
    if (inputBuffer) {
        cudaFree(inputBuffer);
        inputBuffer = nullptr;
    }
    if (outputBuffer) {
        cudaFree(outputBuffer);
        outputBuffer = nullptr;
    }

    if (context) {
        delete context;
        context = nullptr;
    }
    if (engine) {
        delete engine;
        engine = nullptr;
    }
    if (runtime) {
        delete runtime;
        runtime = nullptr;
    }

    if (logger) {
        delete logger;
        logger = nullptr;
    }
}

bool TensorEngine::inference(const std::vector<float> &input, std::vector<float> &output) {
    if (!context || !engine || !inputBuffer || !outputBuffer || !stream) {
        std::cerr << "TensorEngine: engine not initialized \n";
        return false;
    }

    if (input.size() != inputSize) {
        std::cerr << "TensorEngine: wrong input size: " << input.size() << ", expected " << inputSize << "\n";
        return false;
    }

    cudaError_t error = cudaMemcpyAsync(inputBuffer, input.data(), inputSize * sizeof(float), cudaMemcpyHostToDevice, stream);
    context->enqueueV3(stream);
    output.resize(outputSize);

    cudaMemcpyAsync(output.data(), outputBuffer, sizeof(float) * outputSize, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    return true;
}

bool TensorEngine::inferenceDevice() {
    return context->enqueueV3(stream);
}

