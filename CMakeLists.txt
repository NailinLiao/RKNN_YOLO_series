cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)


project(YOLO_RUN LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_FLAGS "-std=c++17 -O3 -pthread -fopenmp")

add_subdirectory(YOLO_V5/deploy_demo) # 假设src目录下有另一个CMakeLists.txt
add_subdirectory(YOLO_V8/deploy_demo) # 假设src目录下有另一个CMakeLists.txt
