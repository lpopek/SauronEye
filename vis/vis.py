import os
import cv2
import glob
import pathlib

from ultralytics import YOLO

CURRENT_DIR = pathlib.Path(__file__).parent.resolve()
MODEL_DIR: str = os.path.abspath(os.path.join(CURRENT_DIR, "../models"))
TRACKERS_DIR: str = os.path.abspath(os.path.join(CURRENT_DIR, "../trackers"))

if __name__ == "__main__":

  model = YOLO("yolo11n.pt")

  image_folder = os.path.abspath(os.path.join(CURRENT_DIR, "../../datasets/MOT15/KITTI-13/img1"))
  image_paths = sorted(glob.glob(os.path.join(image_folder, "*.jpg")))

  for img_path in image_paths:
    frame = cv2.imread(img_path)

    results = model.track(
      frame, 
      persist=True, 
      tracker=os.path.join(TRACKERS_DIR, "test_tracker.yaml")
    )

    annotated_frame = results[0].plot()

    cv2.imshow("Tracking result", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
      break

  cv2.destroyAllWindows()