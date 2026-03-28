import cv2
import torch
from facenet_pytorch import MTCNN

# Initialize globally to save load time.
# IMPORTANT: MTCNN's adaptive pooling is NOT compatible with MPS (Apple Silicon).
# Always use CPU for face detection to prevent silent failures.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn_model = MTCNN(keep_all=False, select_largest=True, device=device)

def detect_face(frame):
    """
    Detects a face in the given image frame using MTCNN (facenet-pytorch).
    Returns the cropped face if found, otherwise returns the original frame.
    """
    try:
        # OpenCV loads in BGR, MTCNN expects RGB (PIL Image or NumPy array in RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        boxes, probs = mtcnn_model.detect(rgb_frame)

        if boxes is not None and len(boxes) > 0:
            # Take the box of the largest face
            x, y, x2, y2 = [int(b) for b in boxes[0]]

            # Ensure boundaries are within bounds
            h, w = frame.shape[:2]
            x, y = max(0, x), max(0, y)
            x2, y2 = min(w, x2), min(h, y2)

            # Crop face area
            if x2 > x and y2 > y:
                return frame[y:y2, x:x2]
    except Exception as e:
        print(f"Face extraction failed: {e}")

    # Fallback to the original center or uncropped frame if detection fails
    return frame