"""Vision-assisted virtual controller. Run elevated on Windows."""

import json
import os
from pathlib import Path
import threading
import time

import cv2
import keyboard
import mss
import numpy as np
import vgamepad as vg
import XInput

from macro_core import MacroEngine, NEUTRAL

ROOT = Path(__file__).resolve().parent
DEFAULTS = {
    "model_id": "rocket-league-uidod/12", "monitor_index": 1,
    "detection_frame_size": [640, 360], "ball_stale_seconds": 1.0,
    "ball_min_confidence": 0.35, "aerial_yaw_gain": 3.0,
    "aerial_steering_smoothing": 0.35, "kickoff_key": "f9",
    "aerial_key": "f10", "cancel_key": "f8", "trigger_mode": "toggle",
    "controller_index": "auto", "poll_hz": 250, "debug": False,
}


def load_config():
    result = dict(DEFAULTS)
    path = ROOT / "config.json"
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            result.update(json.load(handle))
    if os.environ.get("RL_MACRO_DEBUG") == "1":
        result["debug"] = True
    return result


CONFIG = load_config()
ball_lock = threading.Lock()
ball_state = {"x_frac": 0.5, "confidence": 0.0, "timestamp": 0.0}
smoothed_yaw = 0.0


def load_model():
    key_path = ROOT / "api_key.txt"
    if not key_path.exists():
        raise SystemExit("Missing api_key.txt. See README.md for setup.")
    os.environ["ROBOFLOW_API_KEY"] = key_path.read_text(encoding="utf-8").strip()
    from inference import get_model
    print("Loading local vision model (first run may download weights)...")
    return get_model(model_id=CONFIG["model_id"])


def detection_loop(model):
    width, height = CONFIG["detection_frame_size"]
    with mss.mss() as capture:
        index = int(CONFIG["monitor_index"])
        if index <= 0 or index >= len(capture.monitors):
            print(f"Monitor {index} unavailable; using primary monitor 1.")
            index = 1
        while True:
            frame = np.array(capture.grab(capture.monitors[index]))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            frame = cv2.resize(frame, (width, height))
            try:
                predictions = model.infer(frame)[0].predictions
            except Exception as error:
                print(f"Detection failed: {error}")
                time.sleep(0.1)
                continue
            balls = [p for p in predictions if p.class_name == "ball"
                     and p.confidence >= float(CONFIG["ball_min_confidence"])]
            if balls:
                best = max(balls, key=lambda p: p.confidence)
                with ball_lock:
                    ball_state.update(x_frac=best.x / width,
                                      confidence=best.confidence,
                                      timestamp=time.monotonic())


def aerial_controls(elapsed):
    global smoothed_yaw
    with ball_lock:
        state = dict(ball_state)
    if time.monotonic() - state["timestamp"] > float(CONFIG["ball_stale_seconds"]):
        return None
    target = max(-1.0, min(1.0, (state["x_frac"] - 0.5) * CONFIG["aerial_yaw_gain"]))
    smoothing = float(CONFIG["aerial_steering_smoothing"])
    smoothed_yaw += (target - smoothed_yaw) * smoothing
    if elapsed < 0.024:
        return dict(NEUTRAL, jump=True)
    if elapsed < 0.044:
        return dict(NEUTRAL)
    if elapsed < 0.068:
        return dict(NEUTRAL, jump=True)
    if elapsed < 0.520:
        return dict(NEUTRAL, pitch=1, boost=True, yaw=smoothed_yaw)
    return None


def choose_controller_index():
    connected = [i for i, value in enumerate(XInput.get_connected()) if value]
    configured = CONFIG["controller_index"]
    if configured != "auto":
        index = int(configured)
        if index not in connected:
            raise SystemExit(f"Configured slot {index} is not connected; found {connected}.")
        return index
    if not connected:
        raise SystemExit("No XInput controller was found.")
    # ViGEm normally gets the lowest slot because it was created first. The
    # physical controller is therefore the highest connected slot.
    index = connected[-1]
    if len(connected) == 1:
        print("Warning: only one slot is visible; verify it is the physical controller.")
    print(f"Connected XInput slots: {connected}; reading slot {index}.")
    return index


def wait_for_controller(gamepad, previous_index):
    """Fail neutral and wait for a disconnected physical pad to return."""
    apply_controls(gamepad, NEUTRAL)
    print(f"Physical controller slot {previous_index} disconnected; output neutralized.")
    while True:
        connected = [i for i, value in enumerate(XInput.get_connected()) if value]
        configured = CONFIG["controller_index"]
        if configured != "auto" and int(configured) in connected:
            print(f"Physical controller restored on configured slot {configured}.")
            return int(configured)
        # In auto mode, do not select the lone ViGEm device and mirror our own
        # output back into itself. Wait until a second controller appears.
        if configured == "auto" and len(connected) >= 2:
            index = connected[-1]
            print(f"Physical controller restored; now reading slot {index}.")
            return index
        time.sleep(0.25)


def axis(value):
    return int(max(-1.0, min(1.0, value)) * 32767)


def set_button(gamepad, pressed, button):
    (gamepad.press_button if pressed else gamepad.release_button)(button)


def apply_controls(gamepad, output):
    # This player's air pitch/yaw are on left stick; right stick is camera.
    x_value = output["steer"] or output["yaw"]
    gamepad.left_joystick(x_value=axis(x_value), y_value=axis(-output["pitch"]))
    gamepad.right_joystick(x_value=0, y_value=0)
    gamepad.left_trigger(0)
    gamepad.right_trigger(255 if output["throttle"] > 0 else 0)
    set_button(gamepad, output["jump"], vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
    set_button(gamepad, output["boost"], vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
    set_button(gamepad, output["roll"] < 0 or output["handbrake"],
               vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER)
    set_button(gamepad, output["roll"] > 0,
               vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER)
    gamepad.update()


def mirror(gamepad, state):
    gamepad.left_joystick(x_value=state.Gamepad.sThumbLX, y_value=state.Gamepad.sThumbLY)
    gamepad.right_joystick(x_value=state.Gamepad.sThumbRX, y_value=state.Gamepad.sThumbRY)
    gamepad.left_trigger(state.Gamepad.bLeftTrigger)
    gamepad.right_trigger(state.Gamepad.bRightTrigger)
    gamepad.report.wButtons = state.Gamepad.wButtons
    gamepad.update()


def run_application():
    model = load_model()
    gamepad = vg.VX360Gamepad()
    controller_index = choose_controller_index()
    threading.Thread(target=detection_loop, args=(model,), daemon=True).start()
    engine = MacroEngine(aerial_controls, trigger_mode=CONFIG["trigger_mode"])
    interval = 1.0 / float(CONFIG["poll_hz"])
    old_mode, last_debug = engine.mode, 0.0
    print(f"Running: kickoff={CONFIG['kickoff_key']}, aerial={CONFIG['aerial_key']}, "
          f"cancel={CONFIG['cancel_key']}, mode={CONFIG['trigger_mode']}")
    print("Ctrl+C stops. A second trigger tap or the cancel key returns control.")
    try:
        while True:
            started = time.perf_counter()
            if keyboard.is_pressed(CONFIG["cancel_key"]):
                engine.cancel()
            output = engine.update(time.monotonic(),
                                   keyboard.is_pressed(CONFIG["kickoff_key"]),
                                   keyboard.is_pressed(CONFIG["aerial_key"]))
            if output is None:
                try:
                    state = XInput.get_state(controller_index)
                except Exception:
                    controller_index = wait_for_controller(gamepad, controller_index)
                    continue
                mirror(gamepad, state)
                if CONFIG["debug"] and time.monotonic() - last_debug > 0.5:
                    print(f"[slot {controller_index}] LX={state.Gamepad.sThumbLX} "
                          f"LY={state.Gamepad.sThumbLY} RT={state.Gamepad.bRightTrigger}")
                    last_debug = time.monotonic()
            else:
                apply_controls(gamepad, output)
            if engine.mode != old_mode:
                print(f"[mode change] {old_mode} -> {engine.mode}")
                old_mode = engine.mode
            time.sleep(max(0.0, interval - (time.perf_counter() - started)))
    finally:
        apply_controls(gamepad, NEUTRAL)


if __name__ == "__main__":
    run_application()
