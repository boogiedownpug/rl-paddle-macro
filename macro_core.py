"""Pure, dependency-free control logic for the Rocket League macro runtime."""

from dataclasses import dataclass, field
from typing import Callable

NEUTRAL = {"throttle": 0.0, "steer": 0.0, "pitch": 0.0, "yaw": 0.0,
           "roll": 0.0, "jump": False, "boost": False, "handbrake": False}


def controls(**changes):
    result = dict(NEUTRAL)
    result.update(changes)
    return result


# Durations are seconds, not loop ticks, so playback remains stable when a
# frame inference or Windows scheduling makes an individual loop run late.
SPEED_FLIP_KICKOFF = [
    (0.220, controls(throttle=1, boost=True)),
    (0.080, controls(throttle=1, boost=True, steer=-1)),
    (0.040, controls(throttle=1, jump=True, boost=True)),
    (0.020, controls(throttle=1, boost=True)),
    (0.020, controls(throttle=1, yaw=0.8, pitch=-0.7, jump=True, boost=True)),
    (0.260, controls(throttle=1, pitch=1, boost=True)),
    (0.200, controls(throttle=1, roll=1, pitch=0.5)),
]


@dataclass
class MacroEngine:
    """Select macro output from trigger edges and elapsed monotonic time."""

    aerial_controls: Callable[[float], dict | None]
    trigger_mode: str = "toggle"
    mode: str = "idle"
    started_at: float = 0.0
    previous_kickoff: bool = False
    previous_aerial: bool = False
    _kickoff_ends: list[float] = field(init=False)

    def __post_init__(self):
        if self.trigger_mode not in {"toggle", "hold"}:
            raise ValueError("trigger_mode must be 'toggle' or 'hold'")
        total = 0.0
        self._kickoff_ends = []
        for duration, _ in SPEED_FLIP_KICKOFF:
            total += duration
            self._kickoff_ends.append(total)

    def cancel(self):
        self.mode = "idle"

    def update(self, now, kickoff_pressed, aerial_pressed):
        kickoff_edge = kickoff_pressed and not self.previous_kickoff
        aerial_edge = aerial_pressed and not self.previous_aerial

        if kickoff_edge:
            if self.trigger_mode == "toggle" and self.mode == "kickoff":
                self.cancel()
            else:
                self.mode, self.started_at = "kickoff", now
        elif aerial_edge:
            if self.trigger_mode == "toggle" and self.mode == "aerial":
                self.cancel()
            else:
                self.mode, self.started_at = "aerial", now

        if self.trigger_mode == "hold":
            if self.mode == "kickoff" and not kickoff_pressed:
                self.cancel()
            elif self.mode == "aerial" and not aerial_pressed:
                self.cancel()

        self.previous_kickoff = kickoff_pressed
        self.previous_aerial = aerial_pressed
        elapsed = max(0.0, now - self.started_at)

        if self.mode == "kickoff":
            for end, (_, output) in zip(self._kickoff_ends, SPEED_FLIP_KICKOFF):
                if elapsed < end:
                    return output
            self.cancel()
        elif self.mode == "aerial":
            output = self.aerial_controls(elapsed)
            if output is not None:
                return output
            self.cancel()
        return None
