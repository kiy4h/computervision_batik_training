import cv2
from ultralytics import YOLO
import time
import math

# ================= CONFIGURATION =================
# Path to your model (change to 'best.onnx' if you want to test ONNX)
MODEL_PATH = 'best_float32(3) OBJ DETECT.tflite'

# Confidence Threshold (0.0 - 1.0)
# Raise this to 0.5 or 0.6 if you see too many "ghost" detections on walls
CONF_THRESHOLD = 0.4

# Camera Index (0 is usually the default webcam, 1 is external USB)
CAMERA_INDEX = 0
# =================================================


def main():
    # 1. Load the Model
    print(f"Loading model: {MODEL_PATH}...")
    try:
        model = YOLO(MODEL_PATH, task='detect')
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Open Camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Set camera resolution (optional, helps performance)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Starting Camera... Press 'q' to exit.")

    prev_frame_time = 0
    new_frame_time = 0

    while True:
        # 3. Read Frame
        success, frame = cap.read()
        if not success:
            break

        # 4. Run Inference (The Magic Happens Here)
        # stream=True is efficient for video loops
        results = model(frame, conf=CONF_THRESHOLD, stream=True)

        # 5. Draw Results
        for r in results:
            boxes = r.boxes

            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100

                # Class Name
                cls = int(box.cls[0])
                class_name = model.names[cls]

                # --- CUSTOM DRAWING (To make it look like an App) ---

                # Color Setup: Parang (Red), Megamendung (Blue)
                color = (0, 255, 0)  # Default Green
                if "parang" in class_name.lower():
                    color = (0, 165, 255)  # Orange-ish
                elif "mega" in class_name.lower():
                    color = (255, 0, 0)   # Blue

                # Draw Box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

                # Draw Label Background
                label = f'{class_name} {conf}'
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                cv2.rectangle(frame, (x1, y1), c2, color, -
                              1, cv2.LINE_AA)  # Filled

                # Draw Text
                cv2.putText(frame, label, (x1, y1 - 2), 0, 1,
                            [255, 255, 255], thickness=2, lineType=cv2.LINE_AA)

        # 6. Calculate FPS (Frames Per Second)
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        # Display FPS on screen
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 7. Show the Window
        cv2.imshow('Smart Batik Lens Simulator', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Program stopped.")

if __name__ == "__main__":
    main()
