cmake_minimum_required(VERSION 3.28)

project(tensorrt_inference LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)

message(STATUS "OpenCV include directories: ${OpenCV_INCLUDE_DIRS}")

set(CMAKE_CXX_STANDARD 20)

# find_library(headers/)

add_executable(tensorrt_inference main.cpp
        TensorEngine.h
        TensorEngine.cpp
        Preprocess.cpp
        Preprocess.h
        Preprocess.h
        Postprocess.cpp
        Postprocess.h)

target_link_libraries(tensorrt_inference PRIVATE nvinfer CUDA::cudart ${OpenCV_LIBS} )



########## TEST
add_executable(testPreprocess TestProcess.cpp Preprocess.cpp Preprocess.h)
target_link_libraries(testPreprocess PRIVATE ${OpenCV_LIBS})
