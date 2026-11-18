#include "TensorEngine.h"
#include "Preprocess.h"
#include "Postprocess.h"

#include "Preprocess.cuh"
#include "Postprocess.cuh"

#include <cuda_runtime.h>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <vector>
#include <numeric>
#include <algorithm>
#include <opencv2/opencv.hpp>

struct BenchStats {
    std::vector<double> preMs;
    std::vector<double> infMs;
    std::vector<double> postMs;
    std::vector<double> totalMs;

    void print(const std::string& mode) const {
        auto calcAvg = [](const std::vector<double>& v) {
            return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
        };

        double avgPre = calcAvg(preMs);
        double avgInf = calcAvg(infMs);
        double avgPost = calcAvg(postMs);
        double avgTotal = calcAvg(totalMs);
        double minTotal = *std::min_element(totalMs.begin(), totalMs.end());
        double maxTotal = *std::max_element(totalMs.begin(), totalMs.end());

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "\n================ BENCHMARK (" << mode << " - " << totalMs.size() << " runs) ================\n";
        std::cout << "Preprocess  (avg): " << avgPre << " ms\n";
        std::cout << "Inference   (avg): " << avgInf << " ms\n";
        std::cout << "Postprocess (avg): " << avgPost << " ms\n";
        std::cout << "------------------------------------------------------------------\n";
        std::cout << "Total  (avg): " << avgTotal << " ms  (" << (1000.0 / avgTotal) << " FPS)\n";
        std::cout << "Total  (min): " << minTotal << " ms  |  (max): " << maxTotal << " ms\n";
        std::cout << "==================================================================\n";
    }
};

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <CUDA/CPU> <engine_path> [runs]\n";
        return 1;
    }

    const std::string engineMode = argv[1];
    const std::string enginePath = argv[2];
    const int runs = (argc >= 4) ? std::atoi(argv[3]) : 100;
    const int warmupRuns = 10;

    if (engineMode != "CPU" && engineMode != "CUDA") {
        std::cerr << "Invalid mode. Use CPU or CUDA.\n";
        return 1;
    }

    std::cout << "=== Loading engine: " << enginePath << " ===\n";
    TensorEngine engine(enginePath);

    cv::Mat img = cv::imread("bus.jpg", cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Failed to load bus.jpg\n";
        return 1;
    }

    LetterBox b;
    BenchStats stats;
    stats.preMs.reserve(runs);
    stats.infMs.reserve(runs);
    stats.postMs.reserve(runs);
    stats.totalMs.reserve(runs);

    using Clock = std::chrono::high_resolution_clock;

    if (engineMode == "CPU") {
        std::vector<float> output;
        std::vector<Detection> detections; // Schimba cu tipul tau real daca difera

        // 1. Warm-up
        std::cout << "Running " << warmupRuns << " warm-up iterations...\n";
        for (int i = 0; i < warmupRuns; ++i) {
            auto input = preprocess(img, 640, 640, b);
            if (!engine.inference(input, output)) {
                std::cerr << "Warm-up inference FAILED\n";
                return 1;
            }
            detections = postprocess(output, b);
        }

        // 2. Benchmark Loop
        std::cout << "Running " << runs << " benchmark iterations...\n";
        for (int i = 0; i < runs; ++i) {
            auto t0 = Clock::now();
            auto input = preprocess(img, 640, 640, b);
            auto t1 = Clock::now();

            if (!engine.inference(input, output)) {
                std::cerr << "Inference FAILED at run " << i << "\n";
                return 1;
            }
            auto t2 = Clock::now();

            detections = postprocess(output, b);
            auto t3 = Clock::now();

            stats.preMs.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            stats.infMs.push_back(std::chrono::duration<double, std::milli>(t2 - t1).count());
            stats.postMs.push_back(std::chrono::duration<double, std::milli>(t3 - t2).count());
            stats.totalMs.push_back(std::chrono::duration<double, std::milli>(t3 - t0).count());
        }

        stats.print("CPU");

        for (const auto& detection : detections) {
            cv::rectangle(img, detection.box, cv::Scalar(0, 255, 0));
        }
        cv::imwrite("resultCPU.jpg", img);

    } else if (engineMode == "CUDA") {
        Preprocess cudaPreprocess;
        Postprocess cudaPostprocess;
        cudaStream_t stream = engine.getStream();

        auto* input = static_cast<float*>(engine.getInputBuffer());
        auto* output = static_cast<float*>(engine.getOutputBuffer());
        decltype(cudaPostprocess.run(output, b, 0.5f, 1024, stream)) detections;

        // 1. Warm-up
        std::cout << "Running " << warmupRuns << " warm-up iterations...\n";
        for (int i = 0; i < warmupRuns; ++i) {
            cudaPreprocess.run(img, 640, 640, b, input, stream);
            if (!engine.inferenceDevice()) {
                std::cerr << "Warm-up inferenceDevice FAILED\n";
                return 1;
            }
            detections = cudaPostprocess.run(output, b, 0.5f, 1024, stream);
            cudaStreamSynchronize(stream);
        }

        // 2. Benchmark Loop
        std::cout << "Running " << runs << " benchmark iterations...\n";
        for (int i = 0; i < runs; ++i) {
            auto t0 = Clock::now();
            cudaPreprocess.run(img, 640, 640, b, input, stream);
            cudaStreamSynchronize(stream); // Necesar pentru a izola timpul de preprocess
            auto t1 = Clock::now();

            if (!engine.inferenceDevice()) {
                std::cerr << "Inference FAILED at run " << i << "\n";
                return 1;
            }
            cudaStreamSynchronize(stream); // Necesar pentru a izola timpul de inference
            auto t2 = Clock::now();

            detections = cudaPostprocess.run(output, b, 0.5f, 1024, stream);
            cudaStreamSynchronize(stream); // Necesar pentru a ne asigura ca GPU a terminat
            auto t3 = Clock::now();

            stats.preMs.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            stats.infMs.push_back(std::chrono::duration<double, std::milli>(t2 - t1).count());
            stats.postMs.push_back(std::chrono::duration<double, std::milli>(t3 - t2).count());
            stats.totalMs.push_back(std::chrono::duration<double, std::milli>(t3 - t0).count());
        }

        stats.print("CUDA");

        for (const auto& detection : detections) {
            cv::rectangle(img, detection.box, cv::Scalar(255, 0, 0));
        }
        cv::imwrite("resultCUDA.jpg", img);
    }

    return 0;
}