import cv2
from ultralytics import YOLO

# ================= CONFIGURATION =================
# Path to the model
MODEL_PATH = 'best-segmentation.pt'

# Confidence threshold (Higher = fewer hallucinations)
CONF_THRESHOLD = 0.5

# Camera ID (0 is usually the default webcam)
CAMERA_ID = 0
# =================================================


def main():
    # 1. Load the Model
    try:
        print(f"Loading model from {MODEL_PATH}...")
        model = YOLO(MODEL_PATH, task='segment')
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Open Camera
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Set camera resolution (optional, for speed)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Starting Live Segmentation... Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # 3. Run Inference
        # stream=True is efficient for video loops
        # retina_masks=True makes the edges of the mask much smoother (High Quality)
        results = model.predict(
            source=frame,
            conf=CONF_THRESHOLD,
            save=False,
            verbose=False,
            retina_masks=True
        )

        # 4. Visualize (The Magic Part)
        # plot() automatically handles masks if the model supports them.
        # - boxes=False: Hides the square bounding box (Clean look!)
        # - probs=False: Hides the probability score text
        # - alpha=0.4: Transparency of the mask overlay (0.0 to 1.0)
        annotated_frame = results[0].plot(
            boxes=True,
            probs=True,
        )

        # 5. Display
        cv2.imshow('Smart Batik Lens - Segmentation Mode', annotated_frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
