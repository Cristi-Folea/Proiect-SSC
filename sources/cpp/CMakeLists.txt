cmake_minimum_required(VERSION 3.28)

project(tensorrt_inference LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)

message(STATUS "OpenCV include directories: ${OpenCV_INCLUDE_DIRS}")

set(CMAKE_CXX_STANDARD 20)

# find_library(headers/)

include_directories(include)

add_executable(tensorrt_inference main.cpp
        include/TensorEngine.h
        src/TensorEngine.cpp
        src/Preprocess.cpp
        include/Preprocess.h
        include/Preprocess.h
        src/Postprocess.cpp
        include/Postprocess.h
        src/Preprocess.cu
        src/Preprocess.cu
        src/Preprocess.cu
        include/Preprocess.cuh
        src/Postprocess.cu
        include/Postprocess.cuh
        include/Postprocess.cuh)

target_link_libraries(tensorrt_inference PRIVATE nvinfer CUDA::cudart ${OpenCV_LIBS} )

add_executable(benchmark benchmark.cpp include/TensorEngine.h
        src/TensorEngine.cpp
        src/Preprocess.cpp
        include/Preprocess.h
        include/Preprocess.h
        src/Postprocess.cpp
        include/Postprocess.h
        src/Preprocess.cu
        src/Preprocess.cu
        src/Preprocess.cu
        include/Preprocess.cuh
        src/Postprocess.cu
        include/Postprocess.cuh
        include/Postprocess.cuh)
target_link_libraries(benchmark PRIVATE nvinfer CUDA::cudart ${OpenCV_LIBS} )


########## TEST
add_executable(testPreprocess tests/TestProcess.cpp src/Preprocess.cpp include/Preprocess.h)
target_link_libraries(testPreprocess PRIVATE ${OpenCV_LIBS})
