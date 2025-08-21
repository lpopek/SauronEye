import cv2
import os
import sys

def images_to_video(image_folder):
    folder_name = os.path.basename(os.path.normpath(image_folder))
    output_path = os.path.join('/workspace/evaluation/video', f"{folder_name}.mp4")

    images = [img for img in os.listdir(image_folder) if img.lower().endswith(".jpg")]
    images.sort()

    if not images:
        print("No .jpg files in dir!")
        return

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, _ = frame.shape
    size = (width, height)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 1, size)

    for image in images:
        frame = cv2.imread(os.path.join(image_folder, image))
        out.write(frame)

    out.release()
    print(f"✅ Video saved at: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Too many arguments: input catalogue path")
    else:
        images_to_video(sys.argv[1])