import os
import cv2
import glob
import pathlib

from ultralytics import YOLO

CURRENT_DIR = pathlib.Path(__file__).parent.resolve()

OUTPUT_DIR: str = os.path.join(CURRENT_DIR, "results")
TRACKERS_DIR: str = os.path.abspath(os.path.join(CURRENT_DIR, "../eval/trackers"))
MODEL_PATH: str = os.path.abspath(os.path.join(CURRENT_DIR, "../runs/detect/train4/weights/best.pt"))
IMAGE_DIR: str = os.path.abspath(os.path.join(CURRENT_DIR, "../../evaluation/MOTA/MOT15/MOT15-train/Venice-2/img1"))

if __name__ == "__main__":
  if not os.path.exists(os.path.join(OUTPUT_DIR)):
    os.mkdir(OUTPUT_DIR)

  model = YOLO(MODEL_PATH)

  # imgFiles = [os.path.join(IMAGE_DIR, imageName) for imageName in os.listdir(IMAGE_DIR)]
  imgFiles = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")))

  firstFrame = cv2.imread(imgFiles[0])
  height, width = firstFrame.shape[:2]

  output_path = os.path.join(OUTPUT_DIR, "processed.mp4")
  fourcc = cv2.VideoWriter_fourcc(*"mp4v")
  fps = 30
  out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

  for imgPath in imgFiles:
    frame = cv2.imread(imgPath)

    results = model.track(
      frame, 
      tracker=os.path.join(TRACKERS_DIR, "test_tracker.yaml"),
      classes=[0],
      persist=True
    )

    if results and results[0].boxes != None:
      for bbox in results[0].boxes:
        x1, y1, x2, y2 = map(int, bbox.xyxy[0])
        label = model.names[int(bbox.cls[0])]
        conf = float(bbox.conf[0])

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 2)

    out.write(frame)
  
  out.release()