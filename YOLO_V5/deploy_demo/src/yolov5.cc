// Copyright (c) 2023 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "yolov5.h"
#include "common.h"


static void dump_tensor_attr(rknn_tensor_attr *attr) {
    printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
           "zp=%d, scale=%f\n",
           attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
           attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
           get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

int read_data_from_file(const char *path, char **out_data) {
    FILE *fp = fopen(path, "rb");
    if (fp == NULL) {
        printf("fopen %s fail!\n", path);
        return -1;
    }
    fseek(fp, 0, SEEK_END);
    int file_size = ftell(fp);
    char *data = (char *) malloc(file_size + 1);
    data[file_size] = 0;
    fseek(fp, 0, SEEK_SET);
    if (file_size != fread(data, 1, file_size, fp)) {
        printf("fread %s fail!\n", path);
        free(data);
        fclose(fp);
        return -1;
    }
    if (fp) {
        fclose(fp);
    }
    *out_data = data;
    return file_size;
}

int init_yolov5_model(const char *model_path, rknn_app_context_t *app_ctx) {
    int ret;
    int model_len = 0;
    char *model;
    rknn_context ctx = 0;

    // Load RKNN Model
    model_len = read_data_from_file(model_path, &model);
    if (model == NULL) {
        printf("load_model fail!\n");
        return -1;
    }

    ret = rknn_init(&ctx, model, model_len, 0, NULL);
    free(model);
    if (ret < 0) {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }

    // Get Model Input Output Number
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    // Get Model Input Info
    printf("input tensors:\n");
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    // Get Model Output Info
    printf("output tensors:\n");
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(output_attrs[i]));
    }

    // Set to context
    app_ctx->rknn_ctx = ctx;

    // TODO
    if (output_attrs[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC && output_attrs[0].type != RKNN_TENSOR_FLOAT16) {
        app_ctx->is_quant = true;
    } else {
        app_ctx->is_quant = false;
    }

    app_ctx->io_num = io_num;
    app_ctx->input_attrs = (rknn_tensor_attr *) malloc(io_num.n_input * sizeof(rknn_tensor_attr));
    memcpy(app_ctx->input_attrs, input_attrs, io_num.n_input * sizeof(rknn_tensor_attr));
    app_ctx->output_attrs = (rknn_tensor_attr *) malloc(io_num.n_output * sizeof(rknn_tensor_attr));
    memcpy(app_ctx->output_attrs, output_attrs, io_num.n_output * sizeof(rknn_tensor_attr));

    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
        printf("model is NCHW input fmt\n");
        app_ctx->model_channel = input_attrs[0].dims[1];
        app_ctx->model_height = input_attrs[0].dims[2];
        app_ctx->model_width = input_attrs[0].dims[3];
    } else {
        printf("model is NHWC input fmt\n");
        app_ctx->model_height = input_attrs[0].dims[1];
        app_ctx->model_width = input_attrs[0].dims[2];
        app_ctx->model_channel = input_attrs[0].dims[3];
    }
    printf("model input height=%d, width=%d, channel=%d\n",
           app_ctx->model_height, app_ctx->model_width, app_ctx->model_channel);

    return 0;
}

int release_yolov5_model(rknn_app_context_t *app_ctx) {
    if (app_ctx->input_attrs != NULL) {
        free(app_ctx->input_attrs);
        app_ctx->input_attrs = NULL;
    }
    if (app_ctx->output_attrs != NULL) {
        free(app_ctx->output_attrs);
        app_ctx->output_attrs = NULL;
    }
    if (app_ctx->rknn_ctx != 0) {
        rknn_destroy(app_ctx->rknn_ctx);
        app_ctx->rknn_ctx = 0;
    }
    return 0;
}

// 计算缩放、填充等参数
void calculateLetterboxParams(int src_w, int src_h, int dst_w, int dst_h, letterbox_t &letterbox,
                              bool allow_slight_change) {
    float _scale_w = static_cast<float>(dst_w) / src_w;
    float _scale_h = static_cast<float>(dst_h) / src_h;
    float scale = std::min(_scale_w, _scale_h);

    int resize_w = static_cast<int>(src_w * scale);
    int resize_h = static_cast<int>(src_h * scale);

    // 对齐处理
    if (allow_slight_change) {
        if (resize_w % 4 != 0) resize_w -= resize_w % 4;
        if (resize_h % 2 != 0) resize_h -= resize_h % 2;
    }

    int padding_w = dst_w - resize_w;
    int padding_h = dst_h - resize_h;

    // 计算偏移量以中心对齐
    int _left_offset = padding_w / 2;
    int _top_offset = padding_h / 2;

    // 确保为偶数，若有必要则微调
    if (_left_offset % 2 != 0) _left_offset -= _left_offset % 2;
    if (_top_offset % 2 != 0 && _top_offset > 0) _top_offset -= _top_offset % 2;

    // 设置结果
    letterbox.scale = scale;
    letterbox.x_pad = _left_offset;
    letterbox.y_pad = _top_offset;
}

int inference_yolov5_model(rknn_app_context_t *app_ctx, cv::Mat img, letterbox_t letter_box,
                           object_detect_result_list *od_results) {
    int ret;
    rknn_input inputs[app_ctx->io_num.n_input];
    rknn_output outputs[app_ctx->io_num.n_output];
    const float nms_threshold = NMS_THRESH;      // Default NMS threshold
    const float box_conf_threshold = BOX_THRESH; // Default box threshold

    if ((!app_ctx) || (!od_results)) {
        return -1;
    }
    memset(od_results, 0x00, sizeof(*od_results));
    memset(inputs, 0, sizeof(inputs));
    memset(outputs, 0, sizeof(outputs));

    // Set Input Data
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].size = app_ctx->model_width * app_ctx->model_height * app_ctx->model_channel;
    inputs[0].buf = img.data;


    ret = rknn_inputs_set(app_ctx->rknn_ctx, app_ctx->io_num.n_input, inputs);
    if (ret < 0) {
        printf("rknn_input_set fail! ret=%d\n", ret);
        return -1;
    }

    // Run
    ret = rknn_run(app_ctx->rknn_ctx, nullptr);
    if (ret < 0) {
        printf("rknn_run fail! ret=%d\n", ret);
        return -1;
    }

    // Get Output
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < app_ctx->io_num.n_output; i++) {
        outputs[i].index = i;
        outputs[i].want_float = (!app_ctx->is_quant);
    }

    ret = rknn_outputs_get(app_ctx->rknn_ctx, app_ctx->io_num.n_output, outputs, NULL);
    if (ret < 0) {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
    }

    // Post Process
    post_process(app_ctx, outputs, &letter_box, box_conf_threshold, nms_threshold, od_results);
//    convert_image_with_letterbox( 0.333333 0 140) 0.000
    // Remeber to release rknn output
    rknn_outputs_release(app_ctx->rknn_ctx, app_ctx->io_num.n_output, outputs);


    return ret;
}


void calculateAndApplyLetterbox(const cv::Mat &src, cv::Mat &dst, cv::Scalar paddingColor, letterbox_t &letterbox,
                                bool allowSlightChange) {
    int src_w = src.cols;
    int src_h = src.rows;
    int dst_w = dst.cols;
    int dst_h = dst.rows;

    float _scale_w = static_cast<float>(dst_w) / src_w;
    float _scale_h = static_cast<float>(dst_h) / src_h;
    float scale = std::min(_scale_w, _scale_h);

    int resize_w = static_cast<int>(src_w * scale);
    int resize_h = static_cast<int>(src_h * scale);

    // 对齐处理
    if (allowSlightChange) {
        if (resize_w % 4 != 0) resize_w -= resize_w % 4;
        if (resize_h % 2 != 0) resize_h -= resize_h % 2;
    }

    int padding_w = dst_w - resize_w;
    int padding_h = dst_h - resize_h;

    // 计算偏移量以中心对齐
    int _left_offset = padding_w / 2;
    int _top_offset = padding_h / 2;

    // 确保为偶数，若有必要则微调
    if (_left_offset % 2 != 0) _left_offset -= _left_offset % 2;
    if (_top_offset % 2 != 0 && _top_offset > 0) _top_offset -= _top_offset % 2;

    // 设置结果
    letterbox.scale = scale;
    letterbox.x_pad = _left_offset;
    letterbox.y_pad = _top_offset;

    // 调整图像大小
    cv::resize(src, dst, cv::Size(resize_w, resize_h), 0, 0, cv::INTER_LINEAR);

    // 应用填充
    cv::copyMakeBorder(dst, dst, _top_offset, padding_h - _top_offset, _left_offset, padding_w - _left_offset,
                       cv::BORDER_CONSTANT, paddingColor);
}