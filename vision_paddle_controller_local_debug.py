"""
Vision-assisted paddle "tag-in" controller for Rocket League.

Combines everything so far:
  - Mirrors your REAL physical controller onto a VIRTUAL one (ViGEmBus +
    vgamepad) -- Rocket League only ever sees the virtual controller, and
    this script never touches Rocket League's process or memory.
  - PADDLE_KICKOFF: fixed, open-loop speed-flip kickoff sequence (same as
    before -- doesn't need vision, kickoffs always start from the same spot).
  - PADDLE_AERIAL: a fast-aerial macro that ALSO reads the ball's on-screen
    position (via the Roboflow model, running in a background thread) and
    steers left/right toward it while boosting up. This is the vision part.

Nothing here reads Rocket League's memory or files -- ball position comes
purely from a screenshot of your own screen, the same way you'd look at it
yourself.

Requirements:
    pip install vgamepad XInput-Python keyboard opencv-python mss inference-sdk pillow numpy

Setup:
    1. ViGEmBus driver installed: https://github.com/ViGEm/ViGEmBus/releases
    2. api_key.txt in this same folder with your Roboflow API key.
    3. Run AS ADMINISTRATOR (the keyboard library needs it).
    4. Rocket League set to use the virtual "Xbox 360 Controller".
    5. Rocket League in Windowed or Borderless Windowed mode, NOT Fullscreen.

This is a working first draft, not a finished tuned product -- the yaw
correction gain, trigger keys, and macro timings will likely need a real
tuning pass once you can see it react live.
"""

import os
import threading
import time

import cv2
import keyboard
import mss
import numpy as np
import vgamepad as vg
import XInput

MODEL_ID = "rocket-league-uidod/12"
DETECTION_FRAME_SIZE = (640, 360)
BALL_STALE_SECONDS = 1.0
AERIAL_YAW_GAIN = 3.0
PADDLE_KICKOFF_KEY = "f9"
PADDLE_AERIAL_KEY = "f10"
CONTROLLER_INDEX = 0
POLL_HZ = 250

with open("api_key.txt") as f:
    API_KEY = f.read().strip()

# The local inference package reads this environment variable to download
# model weights the first time -- after that first download it's cached
# locally and runs with no network calls at all.
os.environ["ROBOFLOW_API_KEY"] = API_KEY

from inference import get_model  # noqa: E402  (import after setting env var)

print("Loading model locally (first run downloads weights, may take a minute)...")
MODEL = get_model(model_id=MODEL_ID)
print("Model loaded.")

# ---------------------------------------------------------------------------
# Shared ball state (written by the detection thread, read by the control loop)
# ---------------------------------------------------------------------------

ball_state_lock = threading.Lock()
ball_state = {"x_frac": 0.5, "y_frac": 0.5, "size_frac": 0.0, "timestamp": 0.0}


def detection_loop():
    w, h = DETECTION_FRAME_SIZE
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        while True:
            raw = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
            small = cv2.resize(frame, (w, h))

            try:
                result = MODEL.infer(small)[0]
                predictions = result.predictions
            except Exception as e:
                print("Detection failed:", e)
                predictions = []

            ball_preds = [p for p in predictions if p.class_name == "ball"]
            if ball_preds:
                best = max(ball_preds, key=lambda p: p.confidence)
                with ball_state_lock:
                    ball_state["x_frac"] = best.x / w
                    ball_state["y_frac"] = best.y / h
                    ball_state["size_frac"] = max(best.width, best.height) / h
                    ball_state["timestamp"] = time.time()


def get_ball_state():
    with ball_state_lock:
        return dict(ball_state)


# ---------------------------------------------------------------------------
# Macro definitions
# ---------------------------------------------------------------------------

NEUTRAL = dict(
    throttle=0, steer=0, pitch=0, yaw=0, roll=0,
    jump=False, boost=False, handbrake=False,
)


def hold(ticks, **kwargs):
    controls = dict(NEUTRAL)
    controls.update(kwargs)
    return (ticks, controls)


SPEED_FLIP_KICKOFF = [
    hold(55, throttle=1, boost=True),
    hold(20, throttle=1, boost=True, steer=-1),
    hold(10, throttle=1, jump=True, boost=True),
    hold(5,  throttle=1, boost=True),
    hold(5,  throttle=1, yaw=0.8, pitch=-0.7, jump=True, boost=True),
    hold(65, throttle=1, pitch=1, boost=True),
    hold(50, throttle=1, roll=1, pitch=0.5),
]


def aerial_macro_tick(tick_index):
    """Returns a controls dict for the vision-assisted aerial macro at the
    given tick index (0-based), or None once the macro should stop."""
    state = get_ball_state()
    age = time.time() - state["timestamp"]

    if age > BALL_STALE_SECONDS:
        return None  # lost sight of the ball -- bail out to passthrough

    yaw = max(-1.0, min(1.0, (state["x_frac"] - 0.5) * AERIAL_YAW_GAIN))

    if tick_index == 0:
        return dict(NEUTRAL, jump=True)
    if tick_index <= 5:
        return dict(NEUTRAL)
    if tick_index == 6:
        return dict(NEUTRAL, jump=True)
    if tick_index < 130:
        return dict(NEUTRAL, pitch=1, boost=True, yaw=yaw)
    return None  # macro finished naturally


# ---------------------------------------------------------------------------
# Virtual controller output
# ---------------------------------------------------------------------------

gamepad = vg.VX360Gamepad()


def axis_to_int16(v):
    v = max(-1.0, min(1.0, v))
    return int(v * 32767)


def trigger_to_byte(v):
    v = max(0.0, min(1.0, v))
    return int(v * 255)


def apply_macro_controls(controls):
    # Left stick: x = ground steer / air yaw, y = air pitch
    # (this player's AIR STEER LEFT/RIGHT and AIR PITCH UP/DOWN are both
    # bound to the left stick, confirmed from their actual bindings screen)
    x_axis = controls["steer"] or controls["yaw"]
    y_axis = -controls["pitch"]
    gamepad.left_joystick(x_value=axis_to_int16(x_axis), y_value=axis_to_int16(y_axis))
    gamepad.right_joystick(x_value=0, y_value=0)  # camera control, unused by the macro

    gamepad.left_trigger(0)  # Drive Backwards -- unused by this macro
    gamepad.right_trigger(trigger_to_byte(1 if controls["throttle"] > 0 else 0))  # Drive Forward

    if controls["jump"]:
        gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_A)  # Jump
    else:
        gamepad.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_A)

    if controls["boost"]:
        gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_B)  # Boost
    else:
        gamepad.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_B)

    # This player has discrete roll buttons: Air Roll Left = LB, Air Roll
    # Right = RB. LB doubles as Powerslide, but this macro never needs
    # both roll and powerslide in the same step, so there's no conflict.
    if controls["roll"] > 0:
        gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER)
        gamepad.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER)
    elif controls["roll"] < 0:
        gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER)
        gamepad.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER)
    elif controls["handbrake"]:
        gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER)  # Powerslide
        gamepad.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER)
    else:
        gamepad.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER)
        gamepad.release_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER)

    gamepad.update()


def mirror_physical_state(state):
    gamepad.left_joystick(x_value=state.Gamepad.sThumbLX, y_value=state.Gamepad.sThumbLY)
    gamepad.right_joystick(x_value=state.Gamepad.sThumbRX, y_value=state.Gamepad.sThumbRY)
    gamepad.left_trigger(state.Gamepad.bLeftTrigger)
    gamepad.right_trigger(state.Gamepad.bRightTrigger)
    gamepad.report.wButtons = state.Gamepad.wButtons
    gamepad.update()


# ---------------------------------------------------------------------------
# Main control loop
# ---------------------------------------------------------------------------

def run():
    tick_interval = 1.0 / POLL_HZ

    mode = "idle"  # "idle" | "kickoff" | "aerial"
    kickoff_step_index = 0
    kickoff_step_ticks_left = 0
    aerial_tick_index = 0

    kickoff_was_held = False
    aerial_was_held = False
    last_mode = "idle"
    last_debug_print = time.perf_counter()

    print("Vision paddle controller running. Ctrl+C to stop.")
    print(f"Kickoff key: {PADDLE_KICKOFF_KEY!r}  |  Aerial key: {PADDLE_AERIAL_KEY!r}")

    while True:
        loop_start = time.perf_counter()

        kickoff_held = keyboard.is_pressed(PADDLE_KICKOFF_KEY)
        aerial_held = keyboard.is_pressed(PADDLE_AERIAL_KEY)

        if kickoff_held and not kickoff_was_held:
            mode = "kickoff"
            kickoff_step_index = 0
            kickoff_step_ticks_left = SPEED_FLIP_KICKOFF[0][0]
        elif aerial_held and not aerial_was_held:
            mode = "aerial"
            aerial_tick_index = 0

        if mode == "kickoff" and not kickoff_held:
            mode = "idle"
        if mode == "aerial" and not aerial_held:
            mode = "idle"

        if mode == "kickoff":
            _, controls = SPEED_FLIP_KICKOFF[kickoff_step_index]
            apply_macro_controls(controls)
            kickoff_step_ticks_left -= 1
            if kickoff_step_ticks_left <= 0:
                kickoff_step_index += 1
                if kickoff_step_index >= len(SPEED_FLIP_KICKOFF):
                    mode = "idle"
                else:
                    kickoff_step_ticks_left = SPEED_FLIP_KICKOFF[kickoff_step_index][0]

        elif mode == "aerial":
            controls = aerial_macro_tick(aerial_tick_index)
            if controls is None:
                mode = "idle"
            else:
                apply_macro_controls(controls)
                aerial_tick_index += 1

        else:  # idle -- full passthrough
            state = XInput.get_state(CONTROLLER_INDEX)
            mirror_physical_state(state)

            now = time.perf_counter()
            if now - last_debug_print > 0.5:
                lx = state.Gamepad.sThumbLX
                ly = state.Gamepad.sThumbLY
                rt = state.Gamepad.bRightTrigger
                print(f"[idle passthrough] reading slot {CONTROLLER_INDEX}: "
                      f"stickX={lx} stickY={ly} rightTrigger={rt}")
                last_debug_print = now

        if mode != last_mode:
            print(f"[mode change] {last_mode} -> {mode}")
            last_mode = mode

        kickoff_was_held = kickoff_held
        aerial_was_held = aerial_held

        elapsed = time.perf_counter() - loop_start
        time.sleep(max(0.0, tick_interval - elapsed))


if __name__ == "__main__":
    print("=== Controller diagnostic ===")
    connected = XInput.get_connected()
    for i, is_connected in enumerate(connected):
        print(f"XInput slot {i}: {'CONNECTED' if is_connected else 'empty'}")
    print(f"This script is set to read physical input from slot {CONTROLLER_INDEX}")
    print("==============================")

    detection_thread = threading.Thread(target=detection_loop, daemon=True)
    detection_thread.start()
    run()
