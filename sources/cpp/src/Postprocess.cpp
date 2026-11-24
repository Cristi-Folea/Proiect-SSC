//
// Created by crist on 11/16/2026.
//

#include "Postprocess.h"

// 1 * 84 * 8400
/*
 * YoloV8n - are trei scari diferite:
 *      - 80 x 80 ( 6400 celule )
 *      - 40 x 40 ( 1600 celule )
 *      - 20 x 20 ( 400 celule )
 *
 *      Total : 8400 celule
 *
 *      Pentru fiecare celula:
 *          - 4 numere: cx, cy, w, h
 *          - 80 de numere -> probabilitati
 *
 * Output:
 *      [--- 8400 CX ---][--- 8400 CY ---][--- 8400 W ---][--- 8400 H ---][--- 80 * 8400 prob ---]
 */

enum STRUCTURA_OUTPUT {
    CX,
    CY,
    W,
    H,
    PROBS
};

float IoU(Detection a, Detection b) {
    const cv::Rect intersection = a.box & b.box;
    // int intersectionWidth = std::min(a.box.x, b.box.x) - std::max(a.box.x, b.box.x);
    // int intersectionHeigth = std::min(a.box.y, b.box.y) - std::max(a.box.y, b.box.y);
    float unionArea = a.box.area() + b.box.area() - intersection.area();
    return unionArea > 0.0f ? intersection.area() / unionArea : 0.0f;
}


std::vector<Detection> postprocess(const std::vector<float> &output, const LetterBox &info) {

    const int numClasses = 80;
    const int numCells = 8400;
    const float confTreshold = 0.6f;


    std::vector<Detection> detectionsPerClass[80];

    for ( int cell = 0 ; cell < numCells; cell++) {

        float maxScore = 0.0f;
        int maxC = -1;
        for ( int clasa = 0; clasa < numClasses; clasa++) {

            float currentScore = output[ numCells * (PROBS + clasa) + cell];
            if ( currentScore > confTreshold && currentScore > maxScore) {
                maxScore = currentScore;
                maxC = clasa;
            }
        }
        if (maxC >= 0) {

            const float cx = output[ numCells * CX + cell ] ;
            const float cy = output[ numCells * CY + cell ] ;
            const float w = output[ numCells * W + cell ];
            const float h = output[ numCells * H + cell ];

            float x1_640 = cx - ( w / 2.0f);
            float y1_640 = cy - ( h / 2.0f);
            float x2_640 = cx + ( w / 2.0f);
            float y2_640 = cy + ( h / 2.0f);

            int x1 = static_cast<int>((x1_640 - info.padX) / info.scale);
            int y1 = static_cast<int>((y1_640 - info.padY) / info.scale);
            int x2 = static_cast<int>((x2_640 - info.padX) / info.scale);
            int y2 = static_cast<int>((y2_640 - info.padY) / info.scale);

            x1 = std::max(0,x1);
            y1 = std::max(0, y1);
            x2 = std::min(info.originW, x2);
            y2 = std::min(info.originH, y2);

            Detection detection = {cv::Rect(cv::Point(x1, y1), cv::Point(x2,y2)), maxScore, maxC};
            detectionsPerClass[maxC].push_back(detection);
        }

    }

    std::vector<Detection> rez;
    for (auto& detections : detectionsPerClass) {
        std::sort(detections.begin(),detections.end(),[](Detection &a, Detection &b) {
        return a.score > b.score;
    });
        float nmsTreshold = 0.2f;

        std::vector<bool> surpressed(detections.size(), false);

        for ( int i = 0; i < detections.size(); i++) {
            if ( surpressed[i] == false) {
                rez.push_back(detections[i]);

                for (int j = i + 1; j < detections.size(); j++) {
                    if (IoU(detections[i], detections[j]) >= nmsTreshold) {
                        surpressed[j] = true;
                    }
                }
            }
        }
    }

    return rez;
}


/*

    Ceata, obiecte transparente, coliziune intre obiecte -> la cerinta
    de 3p.

    3 astfel de efecte e nevoie pentru 3p. sau obiecte care sa mearga pe traiectorii
    curbe Bezier.

    P(t, P0, P1) = (1-t)P0 + tP1
    BP0P1P2 = B(P0P1(t)) *(1-t) + B(P1P2(t))
//*/

