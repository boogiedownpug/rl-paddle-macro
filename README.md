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

## Current status

**Working:**
- Full local (no internet) ball detection via CPU inference — confirmed
  drawing correct boxes around the ball.
- Virtual controller creation and **passthrough** (driving normally through
  the script) — confirmed working with real controller input.
- Controller slots are discovered at startup. In `auto` mode the program uses
  the highest connected slot (normally the physical controller after ViGEm
  takes slot 0). Set `controller_index` explicitly in `config.json` if that
  heuristic is wrong for your setup.
- Control mapping has been corrected to match this player's actual custom
  Rocket League bindings (see below) — Rocket League reads pitch/yaw off the
  **left** stick (not right — right stick is camera), and this player has
  discrete air-roll buttons (LB = roll left, RB = roll right), not the more
  common hold-modifier style.

**Needs physical/in-game validation:**
- F9 now defaults to toggle mode: press once to run the entire kickoff. Press
  it again, press F8, or take over after completion to return to passthrough.
  Set `trigger_mode` to `hold` if release-to-cancel is preferred.
- The speed-flip and aerial timings still need a Free Play tuning pass.
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
2. `python -m pip install -r requirements.txt`
3. The requirements use `inference-cpu` (local inference — **do not replace it
   `inference-gpu`**; the GPU package pulls in a huge unrelated dependency
   tree including `pycuda`, which requires the full NVIDIA CUDA Toolkit to
   compile and isn't worth it for this one small model)
4. Install the [ViGEmBus driver](https://github.com/ViGEm/ViGEmBus/releases)
   (required, separate from pip packages)
5. A free [Roboflow](https://roboflow.com) account for the API key (only
   needed once, to download the model weights on first run — after that it's
   fully local and free forever)
6. Put your Roboflow API key in `api_key.txt` in this folder (**never commit
   this file** — it's in `.gitignore`)
7. Optionally copy `config.example.json` to `config.json` and customize it.
8. Rocket League set to **Windowed** or **Borderless Windowed** mode (not
   Fullscreen), and set to use the virtual "Xbox 360 Controller" once the
   script is running

Xbox Elite paddles are not exposed as distinct buttons by XInput. Configure a
paddle to emit F9/F10 (or another key chosen in `config.json`) using the
controller's supported remapping software.

## Files

- `vision_paddle_controller_local.py` — main local-inference controller.
- `vision_paddle_controller_local_debug.py` — the same controller with live
  passthrough diagnostics enabled.
- `macro_core.py` — dependency-free, elapsed-time macro state machine.
- `config.example.json` — documented runtime defaults; copy to `config.json`.
- `detect_test.py` — local-inference detection preview. It uses the same model
  and `monitor_index` setting as the controller and draws live boxes without
  sending frames to Roboflow's cloud endpoint.
- `api_key.txt` — your own Roboflow key, **not committed to git**

## Next steps

1. Confirm the F9 kickoff trigger and automatic controller slot in Free Play.
2. Once the kickoff macro is confirmed working end-to-end, do a real
   in-game test of the vision-assisted aerial macro (F10) with the ball
   actually in the air and in view
3. Tune `aerial_yaw_gain`, steering smoothing, and macro timings based on how
   it actually looks in play.
4. Longer-term: more macros (wave dash, other kickoff variants), and
   eventually attempting an air-dribble/flip-reset macro (acknowledged
   up front as the hardest tier, may never be fully reliable)
