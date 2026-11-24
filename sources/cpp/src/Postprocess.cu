//
// Created by crist on 11/17/2026.
//

#include "Postprocess.cuh"

#include "thrust/sort.h"
#include "thrust/system/cuda/execution_policy.h"

enum STRUCTURA_OUTPUT {
    CX,
    CY,
    W,
    H,
    PROBS
};

struct ScoreDescending {
    __host__ __device__
    bool operator()(const RawDetection &a, const RawDetection &b) const {
        return a.score > b.score;
    }
};

__device__ float IoU(RawDetection a, RawDetection b) {
    const float width = max(0,min(a.x2, b.x2) - max(a.x1, b.x1));
    const float height = max(0,min(a.y2, b.y2) - max(a.y1, b.y1));
    const float intersectionArea = width * height;

    const float areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
    const float areaB = (b.x2 - b.x1) * (b.y2 - b.y1);

    const float unionArea = areaA + areaB - intersectionArea;
    return unionArea > 0.0f ? intersectionArea / unionArea : 0.0f;

}

__global__ void Kernel( const float *d_ouput, RawDetection * d_detections, int *d_count, float confTreshold, LetterBox info, int nrCells, int nrClasses, int maxDetections) {

    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= nrCells) {
        return;
    }

    float maxScore = 0.0f;
    int maxC = -1;

    for (int clasa = 0 ; clasa < nrClasses ; clasa++) {
        float score = d_ouput[ nrCells * (PROBS + clasa) + cell];
        if ( score > confTreshold && score > maxScore ) {
            maxScore = score;
            maxC = clasa;
        }
    }

    if (maxC >= 0) {
        const float cx = d_ouput[ nrCells * CX + cell];
        const float cy = d_ouput[ nrCells * CY + cell];
        const float w = d_ouput[ nrCells * W + cell];
        const float h = d_ouput[ nrCells * H + cell];

        float x1_640 = cx - ( w / 2.0f);
        float y1_640 = cy - ( h / 2.0f);
        float x2_640 = cx + ( w / 2.0f);
        float y2_640 = cy + ( h / 2.0f);

        int x1 = static_cast<int>((x1_640 - info.padX) / info.scale);
        int y1 = static_cast<int>((y1_640 - info.padY) / info.scale);
        int x2 = static_cast<int>((x2_640 - info.padX) / info.scale);
        int y2 = static_cast<int>((y2_640 - info.padY) / info.scale);

        x1 = max(0,x1);
        y1 = max(0, y1);
        x2 = min(info.originW, x2);
        y2 = min(info.originH, y2);

        int index = atomicAdd(d_count, 1);
        if (index >= maxDetections) {
            return;
        }
        d_detections[index] = {x1, y1, x2, y2, maxScore, maxC};


    }
}
__global__ void PairWiseOverlapKernel(const RawDetection *candidates, bool* overlap, int count, int maxDetections, float iouTreshold) {

    const int pair = blockIdx.x * blockDim.x + threadIdx.x;
    const int totalPairs = count * count;
    if (pair >= totalPairs) {
        return;
    }

    const int i = pair / count;
    const int j = pair % count;

    bool suppress = false;
    if (i < j && candidates[i].classId == candidates[j].classId) {
        suppress = IoU(candidates[i], candidates[j]) > iouTreshold;
    }
    overlap[i * maxDetections + j] = suppress;

}
__global__ void GreedyNMSKernel(const RawDetection *candidates, RawDetection * keptDetections, bool* overlap, int* keptCount, int count, int maxDetections) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    int kept = 0;

    for (int i = 0 ; i < count ; i++) {
        if (overlap[i * maxDetections + i]) {
            continue;
        }

        keptDetections[kept++] = candidates[i];
        for (int j = i + 1; j < count ; j++) {
            if (overlap[i * maxDetections + j]) {
                overlap[j * maxDetections + j] = true;
            }
        }
    }

    *keptCount = kept;
}
void Postprocess::init(int requestedMaxDetection) {

    if ( requestedMaxDetection <= this->maxDetections) {
        return;
    }
    cudaFree(d_detections);
    cudaFree(d_count);
    cudaFree(d_overlap);
    cudaFree(d_detectionsNMS);

    cudaMalloc(&d_detections, requestedMaxDetection * sizeof(RawDetection));
    cudaMalloc(&d_detectionsNMS, requestedMaxDetection * sizeof(RawDetection));
    cudaMalloc(&d_count, sizeof(int));
    cudaMalloc(&d_overlap, requestedMaxDetection * requestedMaxDetection * sizeof(bool));

    if (d_detections == nullptr || d_count == nullptr || d_detectionsNMS == nullptr || d_overlap == nullptr) {
        if (d_detectionsNMS) {
            cudaFree(d_detectionsNMS);
            d_detectionsNMS = nullptr;
        }
        if (d_detections) {
            cudaFree(d_detections);
            d_detections = nullptr;
        }
        if (d_count) {
            cudaFree(d_count);
            d_count = nullptr;
        }

        if (d_overlap) {
            cudaFree(d_overlap);
            d_overlap = nullptr;
        }

        return;
    }

    maxDetections = requestedMaxDetection;
}


void Postprocess::release() {

    if (d_detectionsNMS) {
        cudaFree(d_detectionsNMS);
        d_detectionsNMS = nullptr;
    }

    if (d_detections) {
        cudaFree(d_detections);
        d_detections = nullptr;
    }

    if (d_count) {
        cudaFree(d_count);
        d_count = nullptr;
    }

    if (d_overlap) {
        cudaFree(d_overlap);
        d_overlap = nullptr;
    }

    maxDetections = 0;
}

std::vector<Detection> Postprocess::run(const float *d_output, const LetterBox &info, float confTreshold, int maxDetections,
    cudaStream_t stream) {

    init(maxDetections);

    cudaMemsetAsync(d_count,0,sizeof(int), stream);

    dim3 block(256);
    dim3 grid( (8400 + block.x - 1) / block.x);

    // kernel

    Kernel<<<grid, block, 0, stream>>>(d_output,d_detections,d_count, confTreshold, info,8400,80, maxDetections);

    int countHost = 0;
    cudaMemcpyAsync(&countHost, d_count, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    countHost = std::min(countHost, maxDetections);

    thrust::sort(thrust::cuda::par.on(stream),d_detections, d_detections + countHost, ScoreDescending{});
    cudaStreamSynchronize(stream);

    const int totalPairs = countHost * countHost;
    if ( totalPairs > 0) {
        dim3 overLapGrid((totalPairs + block.x - 1) / block.x);
        PairWiseOverlapKernel<<<overLapGrid, block, 0, stream>>>(
            d_detections, d_overlap, countHost, maxDetections, 0.45f);
        cudaStreamSynchronize(stream);
    }

    std::vector<unsigned char> overlapHost(maxDetections * maxDetections * sizeof(bool));
    std::vector<RawDetection> detectionsRaw(countHost);

    if (countHost > 0) {

        cudaMemcpy(overlapHost.data(), d_overlap, maxDetections * maxDetections * sizeof(bool), cudaMemcpyDeviceToHost );
        cudaMemcpy(detectionsRaw.data(), d_detections, countHost * sizeof(RawDetection), cudaMemcpyDeviceToHost);

    }


    std::vector<Detection> detections;
    std::vector suppressed(maxDetections, false);

    for (int i = 0 ; i < countHost; i++) {

        if (suppressed[i]) {
            continue;
        }
        const RawDetection candidate = detectionsRaw[i];
        Detection detection = {cv::Rect(cv::Point(candidate.x1,candidate.y1),cv::Point(candidate.x2,candidate.y2)),candidate.score,candidate.classId};
        detections.push_back(detection);

        for (int j = i + 1; j < countHost; j++) {
            int pairIndex = i * maxDetections + j;

            if (overlapHost[pairIndex] != 0) {
                suppressed[j] = true;
            }
        }
    }

    return detections;
}
