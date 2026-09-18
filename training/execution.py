"""Training stays disabled unless explicitly enabled by the operator."""
import os


def require_training_enabled():
    if os.environ.get("MAKE_LLM_ALLOW_TRAINING") != "1":
        raise RuntimeError("Training is disabled. Set MAKE_LLM_ALLOW_TRAINING=1 only after explicit authorization.")


def require_single_process():
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("This trainer supports one process/device; distributed launch is not supported")
