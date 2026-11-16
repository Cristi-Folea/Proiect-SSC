//
// Created by crist on 11/16/2026.
//

#ifndef TENSORRT_INFERENCE_POSTPROCESS_H
#define TENSORRT_INFERENCE_POSTPROCESS_H

#include <vector>
#include <opencv2/opencv.hpp>

#include "Preprocess.h"

struct Detection {
    cv::Rect box;
    float score;
    int classId;
};

std::vector<Detection> postprocess(const std::vector<float> &ouput, const LetterBox & info);


#endif //TENSORRT_INFERENCE_POSTPROCESS_H
