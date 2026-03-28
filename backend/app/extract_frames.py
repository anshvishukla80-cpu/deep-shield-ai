import cv2
import os

def extract_frames(video_path, output_folder, label, frame_skip=10):
    cap = cv2.VideoCapture(video_path)
    count = 0
    saved = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_skip == 0:
            filename = f"{label}_{saved}.jpg"
            cv2.imwrite(os.path.join(output_folder, filename), frame)
            saved += 1

        count += 1

    cap.release()
    