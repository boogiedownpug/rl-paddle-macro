"""Local screen-capture preview for the configured Rocket League detector."""

import time

import cv2
import mss
import numpy as np

from vision_paddle_controller_local import CONFIG, load_model


CLASS_COLORS = {
    "ball": (0, 255, 255),
    "enemy": (0, 0, 255),
    "teammate": (255, 0, 0),
    "enemy_goalpost": (0, 128, 255),
    "my_goalpost": (255, 128, 0),
}


def draw_detections(frame, predictions):
    for prediction in predictions:
        class_name = prediction.class_name
        confidence = prediction.confidence
        x1 = int(prediction.x - prediction.width / 2)
        y1 = int(prediction.y - prediction.height / 2)
        x2 = int(prediction.x + prediction.width / 2)
        y2 = int(prediction.y + prediction.height / 2)
        color = CLASS_COLORS.get(class_name, (200, 200, 200))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{class_name} {confidence:.2f}",
                    (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1)
    return frame


def main():
    model = load_model()
    width, height = CONFIG["detection_frame_size"]
    configured_monitor = int(CONFIG["monitor_index"])
    with mss.mss() as capture:
        if configured_monitor <= 0 or configured_monitor >= len(capture.monitors):
            available = len(capture.monitors) - 1
            raise SystemExit(
                f"Monitor {configured_monitor} is unavailable; found {available} monitor(s). "
                "Update monitor_index in config.json."
            )
        monitor = capture.monitors[configured_monitor]
        print(f"Capturing monitor {configured_monitor}. Press Q in the preview to quit.")
        while True:
            started = time.perf_counter()
            frame = np.array(capture.grab(monitor))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            small = cv2.resize(frame, (width, height))
            try:
                predictions = model.infer(small)[0].predictions
            except Exception as error:
                print(f"Detection failed: {error}")
                predictions = []
            preview = draw_detections(small, predictions)
            cv2.imshow("RL Local Vision Test", preview)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            elapsed = time.perf_counter() - started
            print(f"Frame time: {elapsed:.2f}s | Detections: {len(predictions)}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
