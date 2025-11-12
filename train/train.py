import os
import csv
import pathlib
from ultralytics import YOLO

DATASET_PATH: str = "/workspace/evaluation/MOT/data.yaml"
CSV_FILE_PATH: str = "training_results.csv"


def epoch_callback(trainer):
    with open(CSV_FILE_PATH, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([trainer.metrics['metrics/precision(B)'], trainer.metrics['metrics/mAP50(B)'], trainer.metrics['val/box_loss'], trainer.metrics['val/cls_loss']])

if __name__ == "__main__":
    print(f"Datapath: {DATASET_PATH}")
    model = YOLO("yolo11n.pt")
    model.add_callback("on_train_epoch_end", epoch_callback)
    results = model.train(
        data=DATASET_PATH,
        epochs=200,
        imgsz=640,
        single_cls=True,
        plots=True
    )
