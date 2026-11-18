//
// Created by crist on 11/17/2026.
//

#include "Preprocess.cuh"


__global__ void letterBoxCHWKernel (const uchar3* src, LetterBox info, float *dst, int dstW, int dstH) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dstW || y >= dstH) {return;}

    int planeSize = dstW * dstH;
    int dstIdx = y * dstW + x;

    enum RGB {
        RED,
        GREEN,
        BLUE
    };

    float r,g,b;
    r = g = b = 114.0f / 255.0f;

    int x_rel = x - info.padX;
    int y_rel = y - info.padY;

    // daca pixelul e in imagine:
    if (x_rel >= 0 && x_rel < info.newW  && y_rel >= 0 && y_rel < info.newH) {

        float fx = x_rel * info.invScale; // vad care pixel din imaginea originala e aici
        float fy = y_rel * info.invScale;

        int x0 = (int) fx;
        int y0 = (int) fy;

        int x1 = min(x0 + 1, info.originW - 1);
        int y1 = min(y0 + 1, info.originH - 1);

        float tx = fx - x0;
        float ty = fy - y0;

        // Ce i 4 pixeli
        uchar3 p00 = src[y0 * info.originW + x0];
        uchar3 p10 = src[y0 * info.originW + x1];
        uchar3 p01 = src[y1 * info.originW + x0];
        uchar3 p11 = src[y1 * info.originW + x1];

        // Interpolare:

        float w00 = (1 - tx) * (1 - ty);
        float w10 = tx * (1 - ty);
        float w01 = (1 - tx) * ty;
        float w11 = tx * ty;

        b = (p00.x * w00 + p01.x * w01 + p10.x * w10 + p11.x * w11) / 255.0f;
        g = (p00.y * w00 + p01.y * w01 + p10.y * w10 + p11.y * w11) / 255.0f;
        r = (p00.z * w00 + p01.z * w01 + p10.z * w10 + p11.z * w11) / 255.0f;

    }

    dst[RED * planeSize + dstIdx] = r;
    dst[GREEN * planeSize + dstIdx] = g;
    dst[BLUE * planeSize + dstIdx] = b;

}


void Preprocess::init(int srcPixels) {
    if (capacity < srcPixels) {
        if (d_src) cudaFree(d_src);
        cudaMalloc(&d_src, srcPixels * 3 * sizeof(unsigned char));
        capacity = srcPixels;
    }
}

void Preprocess::release() {
    if (d_src) {
        cudaFree(d_src);
        d_src = nullptr;
        capacity = 0;
    }
}

void Preprocess::run(const cv::Mat &img, int targetW, int targetH, LetterBox &info, float *dst, cudaStream_t stream) {

    initLetterBox(info, img,targetW, targetH);

    init(info.originH * info.originW);

    cudaMemcpyAsync(d_src, img.data, info.originH * info.originW * 3 * sizeof(unsigned char),cudaMemcpyHostToDevice, stream);

    //nu inteleg:
    dim3 block(16,16);
    dim3 grid((targetW + 15)/ 16, (targetH + 15)/ 16);

    letterBoxCHWKernel<<<grid, block, 0, stream>>>((const uchar3*)d_src, info, dst, targetW, targetH);

}
