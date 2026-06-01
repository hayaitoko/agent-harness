"""Image generation subsystem.

Single backend: OpenRouter → google-vertex, calling Google's Gemini image
models (default Nano Banana 2 = `google/gemini-3.1-flash-image-preview`).
The dispatcher (generate.py) exposes a stable `generate()` entry point and
returns an ImageResult with binary bytes + content type. Caller decides
where the image goes (Telegram reply, file write, web response).
"""
from . import generate  # noqa: F401

__all__ = ["generate"]
