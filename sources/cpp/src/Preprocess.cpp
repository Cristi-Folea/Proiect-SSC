//
// Created by crist on 11/15/2026.
//

#include "Preprocess.h"


std::vector<float> preprocess(const cv::Mat &img, int targetW,  int targetH, LetterBox &info) {

    // info actualizat
    initLetterBox(info,img,targetW,targetH);

    cv::Mat resizedImage;
    cv::resize(img, resizedImage, cv::Size(info.newW, info.newH), 0, 0 ,cv::INTER_AREA);

    // 3 canale si de culoare gri
    cv::Mat canvas(cv::Size(targetW,targetH), CV_8UC3, cv::Scalar(114, 114,114));

    cv::Rect roi(info.padX, info.padY, info.newW, info.newH); // rect of interest, colt stanga sus
    resizedImage.copyTo(canvas(roi));

    std::vector<float> rez(3 * targetH * targetW);
    const int planeSize = canvas.rows * canvas.cols;

    enum RGB {
        RED,
        GREEN,
        BLUE
    };

    // formatul nou CHW
    for ( int y = 0 ; y < canvas.rows; y++) {
        for (int x = 0 ; x < canvas.cols; x++) {
            const cv::Vec3b & pixel = canvas.at<cv::Vec3b>(y, x);

            const int i = y * canvas.cols + x;

            rez[RED * planeSize + i]   = static_cast<float>(pixel[2]) / 255.0f;
            rez[GREEN * planeSize + i] = static_cast<float>(pixel[1]) / 255.0f;
            rez[BLUE * planeSize + i]  = static_cast<float>(pixel[0]) / 255.0f;
        }
    }

    return rez;
}
