import os
import pytest


def pytest_collection_modifyitems(items):
    # Explicitly opt in to model/tokenizer updates; default pytest is non-training.
    if os.environ.get("MAKE_LLM_ALLOW_TRAINING") != "1":
        for item in items:
            if item.get_closest_marker("training"):
                item.add_marker(pytest.mark.skip(reason="Training disabled by default"))
