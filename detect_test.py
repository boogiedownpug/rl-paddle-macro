"""
Screen capture + Rocket League ball/player detection test.

Grabs your screen, sends each frame to a pre-trained Roboflow model that
detects the ball, enemies, teammates, and goals, and draws boxes on a
live preview window so you can see what it's finding.

This is a TEST script to confirm detection works -- it does not control
anything yet. Press 'q' in the preview window to quit.

Setup:
    1. In this same folder, create a file called api_key.txt containing
       ONLY your Roboflow API key (no quotes, nothing else). Never share
       this file or paste its contents anywhere.
    2. pip install opencv-python mss inference-sdk pillow numpy
    3. Have Rocket League running in a window (not fullscreen exclusive --
       "Windowed" or "Borderless Windowed" in RL's display settings) so
       this script can actually see it on screen.
"""

import time

import cv2
import numpy as np
import mss
from inference_sdk import InferenceHTTPClient

with open("api_key.txt") as f:
    API_KEY = f.read().strip()

MODEL_ID = "rocket-league-uidod/12"

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=API_KEY,
)

CLASS_COLORS = {
    "ball": (0, 255, 255),
    "enemy": (0, 0, 255),
    "teammate": (255, 0, 0),
    "enemy_goalpost": (0, 128, 255),
    "my_goalpost": (255, 128, 0),
}


def capture_screen(sct, monitor):
    raw = np.array(sct.grab(monitor))
    frame = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
    return frame


def draw_detections(frame, predictions):
    for pred in predictions:
        cls = pred["class"]
        conf = pred["confidence"]
        x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
        x1, y1 = int(x - w / 2), int(y - h / 2)
        x2, y2 = int(x + w / 2), int(y + h / 2)
        color = CLASS_COLORS.get(cls, (200, 200, 200))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{cls} {conf:.2f}"
        cv2.putText(
            frame, label, (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
        )
    return frame


def main():
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # primary monitor; change index if you have multiple

        print("Starting capture + detection. Press 'q' in the preview window to quit.")
        while True:
            loop_start = time.time()

            frame = capture_screen(sct, monitor)
            small = cv2.resize(frame, (640, 360))

            try:
                result = CLIENT.infer(small, model_id=MODEL_ID)
                predictions = result.get("predictions", [])
            except Exception as e:
                print("Detection request failed:", e)
                predictions = []

            annotated = draw_detections(small, predictions)

            cv2.imshow("RL Vision Test", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            elapsed = time.time() - loop_start
            print(f"Frame time: {elapsed:.2f}s  |  Detections: {len(predictions)}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
