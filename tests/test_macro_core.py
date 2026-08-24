import unittest

from macro_core import MacroEngine


class MacroEngineTests(unittest.TestCase):
    def engine(self, trigger_mode="toggle", aerial=lambda elapsed: None):
        return MacroEngine(aerial, trigger_mode=trigger_mode)

    def test_toggle_continues_after_release(self):
        engine = self.engine()
        self.assertIsNotNone(engine.update(10.0, True, False))
        self.assertIsNotNone(engine.update(10.1, False, False))
        self.assertEqual(engine.mode, "kickoff")

    def test_second_press_cancels_toggle(self):
        engine = self.engine()
        engine.update(10.0, True, False)
        engine.update(10.1, False, False)
        self.assertIsNone(engine.update(10.2, True, False))
        self.assertEqual(engine.mode, "idle")

    def test_hold_cancels_on_release(self):
        engine = self.engine("hold")
        engine.update(10.0, True, False)
        self.assertIsNone(engine.update(10.1, False, False))

    def test_kickoff_finishes_by_elapsed_time(self):
        engine = self.engine()
        engine.update(10.0, True, False)
        self.assertIsNone(engine.update(11.0, False, False))

    def test_aerial_aborts_when_callback_returns_none(self):
        aerial = lambda elapsed: {"jump": True} if elapsed < 0.1 else None
        engine = self.engine(aerial=aerial)
        self.assertEqual(engine.update(5.0, False, True), {"jump": True})
        self.assertIsNone(engine.update(5.2, False, False))

    def test_invalid_trigger_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            self.engine("invalid")

    def test_bot_toggles_on_and_off(self):
        engine = self.engine(aerial=lambda elapsed: {"throttle": 1})
        self.assertEqual(engine.update(1.0, False, False, True), {"throttle": 1})
        self.assertEqual(engine.mode, "bot")
        engine.update(1.1, False, False, False)
        self.assertIsNone(engine.update(1.2, False, False, True))
        self.assertEqual(engine.mode, "idle")

    def test_bot_stale_vision_returns_to_manual(self):
        engine = self.engine(aerial=lambda elapsed: None)
        self.assertIsNone(engine.update(1.0, False, False, True))
        self.assertEqual(engine.mode, "idle")


if __name__ == "__main__":
    unittest.main()
