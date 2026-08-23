# Rocket League Vision-Assisted Paddle Controller

## What this is

An EAC-compatible "tag-in" controller system for Rocket League. The idea: hold a
paddle/key, and a scripted macro (like a speed-flip kickoff, or a vision-assisted
fast aerial) takes over your inputs. Release it, and you're instantly back in
control. Built for **online private matches with friends** — this only works
because it never touches Rocket League's process, memory, or files. It's pure
OS-level input emulation (a virtual controller), the same technique legitimate
remapping tools like DS4Windows use. It does not use, bypass, or interact with
Easy Anti-Cheat in any way, and it is *not* a bot that reads live game state
(that's structurally impossible in online play — RLBot/BakkesMod-style game
memory access is disabled by Psyonix the moment a match goes online).

Ball detection for the aerial macro comes from **reading your own screen**
(screen capture + a pretrained computer vision model), not from reading any
game data — conceptually the same as a human watching their monitor.

There's a **separate, unrelated project** in this same environment
(`C:\NextoProject`) — an offline, single-player freestyle bot built on top of
the open-source Nexto bot via RLBot, with a "human-likeness" layer (confidence
drift, hesitation, dynamic beta). That one is fully working end-to-end and is
not affected by anything below. Ask if you want that pushed too.

## Architecture

```
Your real controller
   -> Python script polls it (XInput)
   -> if paddle NOT held: mirror your input onto a virtual controller
   -> if paddle held: play back a scripted macro instead
   -> virtual controller (ViGEmBus + vgamepad) -> Windows -> Rocket League
```

Ball detection runs in a background thread: screen capture (`mss`) -> local
CV model (`inference` / Roboflow's `rocket-league-uidod` model, running
**locally on CPU**, no network calls, no ongoing cost) -> shared ball
position state -> read by the aerial macro for basic left/right steering
correction.

## Current status (as of last session)

**Working:**
- Full local (no internet) ball detection via CPU inference — confirmed
  drawing correct boxes around the ball.
- Virtual controller creation and **passthrough** (driving normally through
  the script) — confirmed working with real controller input.
- The controller-slot bug is understood and fixed: when the script's virtual
  controller and your real controller are both connected, Windows may assign
  the virtual one to XInput slot 0 and bump your real controller to slot 1.
  `CONTROLLER_INDEX` in the script is currently set to `1` to account for
  this. **If passthrough ever stops working after future changes, check this
  first** — run the debug script and look at the `Controller diagnostic`
  printout at startup.
- Control mapping has been corrected to match this player's actual custom
  Rocket League bindings (see below) — Rocket League reads pitch/yaw off the
  **left** stick (not right — right stick is camera), and this player has
  discrete air-roll buttons (LB = roll left, RB = roll right), not the more
  common hold-modifier style.

**Not working / where we left off:**
- Pressing F9 (the kickoff macro trigger) is currently doing nothing. Last
  action taken: added a `[mode change]` debug print to confirm whether the
  keypress is even being detected by the `keyboard` library before assuming
  it's a macro-logic bug. **This is the very next thing to debug.** Terminal
  was running as Administrator (required for the `keyboard` library to work),
  so that's not the cause.
- We explored using **HidHide** (a driver to hide the real controller from
  everything except our whitelisted script) to solve an earlier double-input/
  splitscreen issue. It's installed and correctly configured (verified via
  screenshots) but does not appear to reliably work with **XInput** devices
  specifically (it's well-documented as being more reliable for DirectInput/
  raw HID). We backed out of using it — device hiding is currently **disabled**
  in HidHide, and the project works fine without it as long as
  `CONTROLLER_INDEX` correctly points at the real controller's slot.

## This player's actual Rocket League bindings (relevant subset)

| Action | Gamepad button |
|---|---|
| Drive Forward | RT |
| Drive Backward | LT |
| Steer L/R | Left Stick X |
| Air Pitch Up/Down | Left Stick Y |
| Air Steer (Yaw) L/R | Left Stick X |
| Jump | A |
| Boost | B |
| Powerslide | LB |
| Air Roll Left | LB (shared with Powerslide — mutually exclusive by game state) |
| Air Roll Right | RB |
| Focus on Ball (ball cam) | Y |
| Rear View | Right Stick click |

If you remap your own controls in Rocket League, `apply_macro_controls()` in
the script needs matching updates.

## Setup (condensed — ask for the long version if needed)

1. Python 3.11 specifically (not newer — dependency compatibility issues with
   3.14). Installed via `py install 3.11`, venv created with `py -3.11 -m venv venv`.
2. `pip install opencv-python mss inference-sdk pillow numpy` (vision testing)
3. `pip install inference-cpu` (local inference — **use the CPU package, not
   `inference-gpu`**; the GPU package pulls in a huge unrelated dependency
   tree including `pycuda`, which requires the full NVIDIA CUDA Toolkit to
   compile and isn't worth it for this one small model)
4. `pip install vgamepad XInput-Python keyboard` (controller/macro layer —
   if `pip` itself gets blocked by "Device Guard policy," use
   `python -m pip install ...` instead)
5. Install the [ViGEmBus driver](https://github.com/ViGEm/ViGEmBus/releases)
   (required, separate from pip packages)
6. A free [Roboflow](https://roboflow.com) account for the API key (only
   needed once, to download the model weights on first run — after that it's
   fully local and free forever)
7. Put your Roboflow API key in `api_key.txt` in this folder (**never commit
   this file** — it's in `.gitignore`)
8. Rocket League set to **Windowed** or **Borderless Windowed** mode (not
   Fullscreen), and set to use the virtual "Xbox 360 Controller" once the
   script is running

## Files

- `vision_paddle_controller_local_debug.py` — the main script, current
  working version, with extra diagnostic printouts (controller slot info,
  live stick readings, mode-change logging). Use this one, not the older
  non-debug or network-API versions.
- `detect_test.py` — earlier standalone detection-only test (draws boxes on
  a live preview window), kept for reference/re-testing detection in
  isolation if needed.
- `api_key.txt` — your own Roboflow key, **not committed to git**

## Next steps

1. Debug why F9 isn't triggering the macro (check `[mode change]` output)
2. Once the kickoff macro is confirmed working end-to-end, do a real
   in-game test of the vision-assisted aerial macro (F10) with the ball
   actually in the air and in view
3. Tune the aerial macro's steering gain (`AERIAL_YAW_GAIN`) and the
   feasibility thresholds (`AERIAL_MIN_BALL_HEIGHT`, `AERIAL_MAX_DIST`,
   `AERIAL_MIN_BOOST`) based on how it actually looks in play
4. Longer-term: more macros (wave dash, other kickoff variants), and
   eventually attempting an air-dribble/flip-reset macro (acknowledged
   up front as the hardest tier, may never be fully reliable)
