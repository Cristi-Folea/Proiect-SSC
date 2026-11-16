#include "TensorEngine.h"
#include "Preprocess.h"
#include "Postprocess.h"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <engine_path> [runs]\n";
        return 1;
    }

    const std::string enginePath = argv[1];
    const int runs = (argc >= 3) ? std::atoi(argv[2]) : 1;

    std::cout << "=== Loading engine: " << enginePath << " ===\n";
    TensorEngine engine(enginePath);


    //----------------Input-----------
    cv::Mat img = cv::imread("bus.jpg", cv::IMREAD_COLOR);
    LetterBox b;

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

    cv::imwrite("result.jpg", img);

    // Benchmark simplu
    // if (runs > 1) {
    //     std::cout << "\n=== Benchmark: " << runs << " runs ===\n";
    //
    //     auto t0 = std::chrono::steady_clock::now();
    //     for (int i = 0; i < runs; ++i) {
    //         if (!engine.inference(input, output)) {
    //             std::cerr << "Inference failed at run " << i << "\n";
    //             return 1;
    //         }
    //     }
    //     auto t1 = std::chrono::steady_clock::now();
    //
    //     double totalMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    //     double avgMs = totalMs / runs;
    //     double fps = 1000.0 / avgMs;
    //
    //     std::cout << "Total: " << totalMs << " ms\n";
    //     std::cout << "Avg latency: " << avgMs << " ms\n";
    //     std::cout << "FPS: " << fps << "\n";
    // }

    return 0;
}