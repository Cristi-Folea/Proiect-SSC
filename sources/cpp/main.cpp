#include "TensorEngine.h"
#include "Preprocess.h"
#include "Postprocess.h"

#include "Preprocess.cuh"
#include "Postprocess.cuh"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << "<CUDA/CPU> <engine_path> [runs]\n";
        return 1;
    }

    const std::string enginePath = argv[2];
    const std::string engineMode = argv[1];

    std::cout << "=== Loading engine: " << enginePath << " ===\n";
    TensorEngine engine(enginePath);

    //----------------Input-----------
    cv::Mat img = cv::imread("bus.jpg", cv::IMREAD_COLOR);
    LetterBox b;



    if (engineMode == "CPU") {

        auto input = preprocess(img, 640, 640, b);

        std::vector<float> output;
        // Warm-up (primul apel are overhead de lazy init CUDA/TRT)
        if (!engine.inference(input, output)) {
            std::cerr << "Warm-up inference FAILED\n";
            return 1;
        }

        std::cout << "\n=== Warm-up OK ===\n";

        auto detections = postprocess(output,b);

        for ( auto detection : detections) {
            cv::rectangle(img,detection.box, cv::Scalar(0,255,0));
        }

        cv::imwrite("resultCPU.jpg", img);

    } else if (engineMode == "CUDA") {

        Preprocess cudaPreprocess;
        Postprocess cudaPostprocess;

        auto* input = static_cast<float *>(engine.getInputBuffer());
        cudaPreprocess.run(img,640,640,b,input,engine.getStream());

        if (!engine.inferenceDevice()) {
            return 1;
        }

        auto* output = (float*)engine.getOutputBuffer();
        auto detections = cudaPostprocess.run(output,b,0.5f,1024,engine.getStream());
        for ( auto detection : detections) {
            cv::rectangle(img,detection.box, cv::Scalar(255,0,0));
        }

        cv::imwrite("resultCUDA.jpg", img);


    } else {
        std::cerr << "Usage: " << argv[0] << "<CUDA/CPU> <engine_path> [runs]\n";
    }

    return 0;
}