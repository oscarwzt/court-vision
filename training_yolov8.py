from ultralytics import YOLO

model = YOLO("yolov8s.pt")
# model = YOLO("runs/detect/train7/weights/last.pt")
model.train(data = "basketballDetection-21/data.yaml", 
            epochs = 200, 
            batch = 128, 
            imgsz = 720, 
            save_period=20,
            lr0 = 0.001,
            lrf = 0.01,
            workers = 10,
            device = [0, 1],
            patience = 30,
            # resume = True
            )