from roboflow import Roboflow
rf = Roboflow(api_key="AcbNGyDSd9COUvpFvn95")
project = rf.workspace("arif-5cjus").project("steering-detection-s2wqt")
version = project.version(1)
dataset = version.download("yolov8")
from ultralytics import YOLO
import os

# 🛑 CHỈNH SỬA ĐƯỜNG DẪN NÀY CHO PHÙ HỢP VỚI THƯ MỤC CỦA BẠN 🛑
DATA_YAML_PATH = '/content/steering-detection-1/data.yaml'

# 1. Tải mô hình YOLOv8n (nano)
model = YOLO('yolov8n.pt')
print("Bắt đầu huấn luyện YOLOv8 cho phát hiện Vô lăng...")

results = model.train(
    data=DATA_YAML_PATH,
    epochs=50,             # Số lượng epochs (tùy chọn)
    imgsz=640,             # Kích thước ảnh đầu vào
    name='yolo_wheel_final' # Tên thư mục lưu kết quả
)

print("✅ Huấn luyện hoàn tất.")