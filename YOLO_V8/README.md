# 训练模型
- 数据准备
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

- yaml文件解释：
    ```
    path: ../datasets/coco   # 写的数据根路径
    train: train2017.txt     # 写的用于训练的图片是哪些
    val: val2017.txt         # 写的用于验证的图片是哪些
    names:                   # 写的标注内label  对应的类别是什么
          0: person
   ```

- 开始训练
    - 命令式激活训练
      ```
      yolo task=detect mode=train model=yolov8s.pt epochs=300 batch=1 data=pedestrian.yaml
      ```
      - **如果gpu训练报错**：CUDA out memory或者Get was unable xxxx，请使用脚本训练
    - 脚本式激活训练
      ```
      import torch
      from ultralytics import YOLO
      model = YOLO("yolov8s.pt")
    
      results = model.train(data="../../airockchip_yolov5/yolov5/data/Car_person.yaml ",device=(2,3), epochs=300, imgsz=640,
      batch=80)
      metrics = model.val()
      ```

- 验证模型
  运行：
  ```
  yolo val detect data=pedestrian.yaml device=cpu model=runs/detect/train6/weights/best.pt
  ```

# 模型转换
- pt转ONNX
  - 管理导出配置文件./ultralytics/cfg/default.yaml
    - 在Export settings 模块，设置 权重、imagesize 、RKNN等
    ```
      format: rknn  # (str) format to export to, choices at https://docs.ultralytics.com/modes/export/#export-formats
      #model: best.pt
      model: /tmp/tmp.BHXHyPiweL/resuorce/pth/YOLOV8_Car_Person.pt
      keras: False  # (bool) use Kera=s
      optimize: False  # (bool) TorchScript: optimize for mobile
      int8: False  # (bool) CoreML/TF INT8 quantization
      dynamic: False  # (bool) ONNX/TF/TensorRT: dynamic axes
      simplify: False  # (bool) ONNX: simplify model
      opset:  # (int, optional) ONNX: opset version
      workspace: 4  # (int) TensorRT: workspace size (GB)
      nms: False  # (bool) CoreML: add NMS
      imgsz: [ 544，960 ]  # (int | list) input images size as int for train and val modes, or list[h,w] for predict and export modes
  
    ```
  - 执行指令：
    ```
    export PYTHONPATH=./
    python ./ultralytics/engine/exporter.py
    ``` 

- onnx转rknn
  ```
    python conver_ONNX2RKNN.py xxx.onnx rk3588
  ```
### 部署测试

```
    cd deploy_demo
    mkdir build
    cmake .. && make -j 6
    rknn_yolov8_deploy ../model/xxx.rknn ../xxx.jpg
```
