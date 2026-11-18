//
// Created by crist on 11/17/2026.
//

#ifndef TENSORRT_INFERENCE_POSTPROCESS_CUH
#define TENSORRT_INFERENCE_POSTPROCESS_CUH
#include <opencv2/core/mat.hpp>

#include "Preprocess.h"
#include "Postprocess.h"
#include <vector>

struct RawDetection {
    int x1,y1,x2,y2;
    float score;
    int classId;
};

class Postprocess {

public:
    Postprocess() {}
    ~Postprocess() { release();}

    void init(int maxDetections);
    void release();

    std::vector<Detection> run(const float *d_output, const LetterBox &info, float confTreshold, int maxDetections, cudaStream_t stream);


private:

    RawDetection* d_detections = nullptr;
    int* d_count = nullptr;
    int maxDetections = 0;

};

#endif //TENSORRT_INFERENCE_POSTPROCESS_CUH
