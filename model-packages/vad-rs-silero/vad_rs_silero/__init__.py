import importlib.metadata
from pathlib import Path

__version__ = importlib.metadata.version("vad-rs-silero")

_MODEL_FILE = Path(__file__).parent / "silero_vad.onnx"


def model_path() -> str:
    return str(_MODEL_FILE)
