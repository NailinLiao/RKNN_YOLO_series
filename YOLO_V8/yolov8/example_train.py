import torch
from ultralytics import YOLO

#根据当前空闲的gpu进行修改
torch.cuda.set_device(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load a model
model = YOLO("yolov8s.pt").to(device)  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="pedestrian.yaml", epochs=100, imgsz=640)
metrics = model.val()
