from ultralytics import YOLO

def train_yolo(model_name, data_path, epochs=100, batch_size=16, img_size=640,
              optimizer='AdamW', hsv_h=0.015, hsv_s=0.0, hsv_v=0.3, dropout=0.2,
              patience=30, lr0=0.001, lrf=0.1, mosaic=0.0, degrees=10, scale=0.5, cos_lr=True,
              deterministic=False, plots=True, fliplr=0.0, perspective=0.0005,
              translate=0.1, 
            ):

    model = YOLO(f'{model_name}.pt')
    
    model.train(data=data_path, epochs=epochs, batch=batch_size, imgsz=img_size,
                optimizer=optimizer, hsv_h=hsv_h, hsv_s=hsv_s, hsv_v=hsv_v, dropout=dropout,
                patience=patience, lr0=lr0, lrf=lrf, mosaic=mosaic, degrees=degrees, 
                scale=scale, cos_lr=cos_lr, deterministic=deterministic, plots=plots,
                fliplr=fliplr, perspective=perspective, translate=translate,
              )

if __name__ == "__main__":
    dataset_yaml = "/content/yolo/yolo/CelebA_Spoof/images"
    
    # Huấn luyện YOLOv8
    train_yolo(
        model_name="yolov8m-cls",  # Có thể chọn: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
        data_path=dataset_yaml,
        epochs=100,
        batch_size=256,
        img_size=224,
    )