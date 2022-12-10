cmake_minimum_required(VERSION 3.10)
project(onnx2trt)
add_executable(onnx2trt onnx2trt.cpp)
target_link_libraries(onnx2trt nvinfer)
target_link_libraries(onnx2trt cudart)
target_link_libraries(onnx2trt nvonnxparser)

