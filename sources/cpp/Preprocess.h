//
// Created by crist on 11/15/2026.
//

#ifndef TENSORRT_INFERENCE_PROPROCESS_H
#define TENSORRT_INFERENCE_PROPROCESS_H

#include <vector>
#include <opencv2/opencv.hpp>

struct LetterBox {
    float scale;
    int padX;
    int padY;
    int originW;
    int originH;
};

std::vector<float> preprocess(const cv::Mat &img, int targetW, int targetH, LetterBox& info);


#endif //TENSORRT_INFERENCE_PROPROCESS_H
