from ultralytics import YOLO


#train code
#yolo task=detect mode=train model=yolov8m.pt imgsz=640 data=data.yaml epochs=50 batch=8 name=pk



# Create a new YOLO model from scratch
model = YOLO('yolov8m.yaml')

# Load a pretrained YOLO model (recommended for training)
model = YOLO('yolov8m.pt')

# Train the model using the 'coco128.yaml' dataset for 3 epochs
results = model.train(data="data.yaml",batch=10,imgsz=640,epochs=80)

# Evaluate the model's performance on the validation set
results = model.val()

# Perform object detection on an image using the model
results = model('val/images')

#Export the model to ONNX format
success = model.export(format='onnx')

#Model.save('model.h1')

