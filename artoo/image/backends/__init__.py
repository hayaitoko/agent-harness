"""Image-generation backend implementations.

Each module here implements one provider. The dispatcher (image/generate.py)
registers them by name and routes calls based on IMAGE_BACKEND env config.

Required interface:
    generate(prompt: str, *, model: str | None, width: int, height: int,
             **kwargs) -> ImageResult

Cloud backends each need their own API key in .env.
"""
