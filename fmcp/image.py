from dataclasses import dataclass


@dataclass
class Image:
    data: bytes
    format: str = "png"
