//
// Created by crist on 11/17/2026.
//

#ifndef TENSORRT_INFERENCE_PREPROCESS_CUH
#define TENSORRT_INFERENCE_PREPROCESS_CUH

#include <opencv2/core/mat.hpp>

#include "Preprocess.h"

class Preprocess {

public:
    Preprocess() = default;
    ~Preprocess() { release(); }

    void init(int srcPixels);
    void release();

    void run(const cv::Mat &img, int targetW, int targetH, LetterBox &info, float *dst, cudaStream_t stream);

private:

    unsigned char* d_src = nullptr;
    int capacity = 0;
};


#endif //TENSORRT_INFERENCE_PREPROCESS_CUH
