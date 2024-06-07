

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "yolov5.h"


/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char **argv) {
    if (argc != 3) {
        printf("%s <model_path> <image_path>\n", argv[0]);
        return -1;
    }

    const char *model_path = argv[1];
    const char *image_path = argv[2];
    int ret;
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    ret = init_yolov5_model(model_path, &rknn_app_ctx);

    if (ret != 0) {
        printf("init_yolov5_model fail! ret=%d model_path=%s\n", ret, model_path);
    }

    cv::Mat src_Mat = cv::imread(image_path);

    if (src_Mat.empty()) {
        std::cerr << "无法加载图像: " << image_path << std::endl;
        return -1; // 或者其他返回码，表示错误
    }

    object_detect_result_list od_results;

//    cv::Mat dst(640, 640, src_Mat.type(), cv::Scalar(0, 0, 0)); // 初始化目标图像为640x640，黑色背景
    cv::Mat dst(544, 960, src_Mat.type(), cv::Scalar(0, 0, 0)); // 初始化目标图像为640x640，黑色背景

    cv::Scalar paddingColor(0, 0, 0); // 黑色填充

    letterbox_t letter_box;
    calculateAndApplyLetterbox(src_Mat, dst, paddingColor, letter_box);


    // 计算100次循环的总耗时
    // 同样的循环操作

    ret = inference_yolov5_model(&rknn_app_ctx, dst, letter_box, &od_results);

    if (ret != 0) {
        printf("inference_yolov5_model fail! ret=%d\n", ret);
    }
    // 画框和概率
    for (int i = 0; i < od_results.count; i++) {
        object_detect_result *det_result = &(od_results.results[i]);
        printf("pred:%d @ (%d %d %d %d) %.3f\n",
               det_result->cls_id,
               det_result->box.left, det_result->box.top,
               det_result->box.right, det_result->box.bottom,
               det_result->prop);
        int class_id = det_result->cls_id;
        int x1 = det_result->box.left;
        int y1 = det_result->box.top;
        int x2 = det_result->box.right;
        int y2 = det_result->box.bottom;
        std::cout << "class id " << class_id << std::endl;
        // 定义边界框颜色和线宽
        cv::Scalar color(0, 255, 0); // 例如，绿色
        int thickness = 2;          // 线条粗细
        // 使用cv::rectangle()绘制边界框
        cv::rectangle(src_Mat, cv::Point(x1, y1), cv::Point(x2, y2), color, thickness);
    }

    cv::imwrite("out.png", src_Mat);
    ret = release_yolov5_model(&rknn_app_ctx);
    if (ret != 0) {
        printf("release_yolov5_model fail! ret=%d\n", ret);
    }

    return 0;
}
