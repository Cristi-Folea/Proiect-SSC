//
// Created by crist on 11/15/2026.
//

#ifndef TENSORRT_INFERENCE_PROPROCESS_H
#define TENSORRT_INFERENCE_PROPROCESS_H

#include <vector>
#include <opencv2/opencv.hpp>

struct LetterBox {
    float scale;
    float invScale;
    int padX;
    int padY;
    int originW;
    int originH;
    int newH;
    int newW;

};

inline void initLetterBox(LetterBox &info, const cv::Mat &img, int targetW, int targetH) {

    info.originH = img.rows;
    info.originW = img.cols;

    auto originalH = info.originH;
    auto originalW = info.originW;

    info.scale = std::min(static_cast<float>(targetH)/originalH, static_cast<float>(targetW)/originalW);
    info.invScale = 1.0f / info.scale;

    info.newW = (int)(originalW * info.scale);
    info.newH = (int)(originalH * info.scale);

    info.padX = (targetW - info.newW) / 2;
    info.padY = (targetH - info.newH) / 2;
}
std::vector<float> preprocess(const cv::Mat &img, int targetW, int targetH, LetterBox& info);



#endif //TENSORRT_INFERENCE_PROPROCESS_H
