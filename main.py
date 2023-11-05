from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch

# Use the model
model.train(data="coco128.yaml", epochs=3)  # train the model

class_mapping = {
    'old_class_name1': 'new_class_name1',
    'old_class_name2': 'new_class_name2',
    # Add more mappings as needed
}
