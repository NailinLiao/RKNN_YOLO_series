## YOLO V5

## 本工程旨在快速训练模型到部署检测教程用例位于瑞芯微RK3588

### 训练

1. 数据准备
    - XXX_DataSet
        - images
            - 0001.jpg
            - 0002.jpg
            - ...
        - labels
            - 0001.txt
            - 0002.txt
            - ...
        - train.txt
            - 0001.jpg
            - ...
        - val.txt
            - 0002.jpg
            - ...
        - XXX_DataSet.yaml

2. yaml文件解释：
    ```
    path: ../datasets/coco   # 写的数据根路径
    train: train2017.txt     # 写的用于训练的图片是哪些
    val: val2017.txt         # 写的用于验证的图片是哪些
    names:                   # 写的标注内label  对应的类别是什么
          0: person
   ```

3.启动训练

- 将该工程代码上传至GPU服务器
    ``` 
  python train.py --epochs 300 --batch-size 128 --weights yolov5s.pt --data ../xxx/xxx.yaml --noautoanchor
    # --noautoanchor表示不再自动计算 推荐anchor的尺寸，这样yolo_deploy_demo 中的 anchor配置信息可以不用更改直接使用
    #               若进行了 自动计算 anchor 需要修改postprocess.cc中的const int anchor 配置信息；
    ```

### 导出onnx

- 将该工程代码上传至GPU服务器

    ```
    cv yolov5_model_train_and_export
    python export.py --rknpu --weight runs/train/exp13/weights/best.pt --data data/CircleTarget.yaml --imgsz 544 960 --opset 12
    # --rknpu 必须有不可改
    # --opset 12  1.5.2 1.6.0测试通过；
    ```

### 转换RKNN

- 将该工程代码上传至RKNN Tools 2 服务器
    ```
    python conver_ONNX2RKNN.py xxx.onnx rk3588
    # 后缀 RK3588不可忘
    ```

### 部署测试
```
    cd deploy_demo
    mkdir build
    cmake .. && make -j 6
    rknn_yolov5_deploy ../model/xxx.rknn ../xxx.jpg
```

