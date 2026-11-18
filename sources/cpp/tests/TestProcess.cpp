//
// Created by crist on 11/16/2026.
//


#include "../include/Preprocess.h"

int main() {

    cv::Mat img = cv::imread("bus.jpg", cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Can't read image" << std::endl;
        return 1;
    }

    LetterBox info{};
    const auto output = preprocess(img,640,640,info);





    return 0;
}