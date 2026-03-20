import importlib.metadata
from pathlib import Path

__version__ = importlib.metadata.version("vad-rs-ten-vad")

_MODEL_FILE = Path(__file__).parent / "ten_vad.onnx"


def model_path() -> str:
    return str(_MODEL_FILE)
