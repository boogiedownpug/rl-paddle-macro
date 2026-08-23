"""Launch the main controller with passthrough diagnostics enabled."""

import os

os.environ["RL_MACRO_DEBUG"] = "1"

from vision_paddle_controller_local import run_application  # noqa: E402

if __name__ == "__main__":
    run_application()
