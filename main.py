from ultralytics import YOLO

# Load a COCO-pretrained YOLOv8n model
model = YOLO("yolov8n-cls.pt")

# Display model information (optional)
model.info()

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data='C:\Kuliah\Course Yolo\YoloClassification\dataset',
                      epochs=1,
                      imgsz=64)




