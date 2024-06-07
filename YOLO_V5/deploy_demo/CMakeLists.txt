cmake_minimum_required(VERSION 3.10)

project(rknn_yolov5_deploy)

file(GLOB_RECURSE LIB_SRC src/*.cpp src/*.cc src/*.c)
file(GLOB_RECURSE LIB_HEAD include/*.hpp include/*.h)

file(GLOB_RECURSE utils_file utils/*.cpp utils/*.cc utils/*.c utils/*.hpp utils/*.h)


# OpenCV
include_directories(/home/linaro/inst/opencv-4.5.5/include)
include_directories(/home/linaro/inst/opencv-4.5.5/include/opencv4)
link_directories(/home/linaro/inst/opencv-4.5.5/lib)
set(OpenCV_LIBS libopencv_core.so libopencv_features2d.so libopencv_highgui.so libopencv_imgcodecs.so
        libopencv_imgproc.so libopencv_videoio.so libopencv_video.so libopencv_calib3d.so
        libopencv_dnn.so libopencv_flann.so libopencv_gapi.so libopencv_ml.so libopencv_objdetect.so
        libopencv_photo.so libopencv_stitching.so)

# RGA
include_directories(/usr/include/rga)
set(RGA_LIBRARY rga)

# RKNN
# include_directories(/home/linaro/dev/unitTest/3rdparty)
#include_directories(/home/linaro/inst/rknn)
#set(RKNN_LIBRARY rknnrt)


## RKNN 1.6.0
#include_directories(API/librknn_api_1.6.0/include)
#link_directories(API/librknn_api_1.6.0/aarch64)
#set(RKNN_LIBRARY rknnrt)

# RKNN 1.5.2
include_directories(../API/librknn_api_1.5.2/include)
link_directories(../API/librknn_api_1.5.2/aarch64)
set(RKNN_LIBRARY rknnrt)

#rkaiq
link_directories(/home/linaro/inst/rkaiq/)
include_directories(/home/linaro/inst/rkaiq/)
include_directories(/home/linaro/inst/rkaiq/algos)
include_directories(/home/linaro/inst/rkaiq/common)
include_directories(/home/linaro/inst/rkaiq/iq_parser)
include_directories(/home/linaro/inst/rkaiq/iq_parser_v2)
include_directories(/home/linaro/inst/rkaiq/xcore)
include_directories(/home/linaro/inst/rkaiq/uAPI2)
set(RKAIQ_LIBRARY rkaiq dl)


include_directories(3rdparty/stb_image)
include_directories(3rdparty/jpeg_turbo/include)
include_directories(utils)
include_directories(include)

include_directories(/home/linaro/inst/jpeg_turbo/include)
link_directories(/home/linaro/inst/jpeg_turbo/lib)

add_executable(${PROJECT_NAME}
        main.cc
        ${LIB_SRC}
        ${LIB_HEAD}
        ${utils_file}
        ${OPENCV_LIB_SRC}
        ${OPENCV_LIB_HEAD}
        )

target_link_libraries(${PROJECT_NAME}
        pthread
        ${RGA_LIBRARY}
        ${OpenCV_LIBS}
        ${RKNN_LIBRARY}
        ${RKAIQ_LIBRARY}
#        turbojpeg
        )

