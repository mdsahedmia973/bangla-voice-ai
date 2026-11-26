"""Abstract TTS engine interface."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Voice:
    id: str
    name: str
    name_bn: str
    engine: str
    gender: str  # "female" | "male"
    style_tags: list[str] = field(default_factory=list)
    available: bool = True
    reason: str = ""


@dataclass
class EngineInfo:
    id: str
    name: str
    available: bool
    reason: str
    voices: list[Voice]
    capabilities: dict
    license: str
    hardware: str
    repo: str


class Engine(ABC):
    id: str = ""
    name: str = ""
    repo: str = ""
    license: str = ""
    hardware: str = ""

    @property
    @abstractmethod
    def available(self) -> bool: ...

    @property
    @abstractmethod
    def reason(self) -> str: ...

    @property
    @abstractmethod
    def voices(self) -> list[Voice]: ...

    @property
    @abstractmethod
    def capabilities(self) -> dict: ...

    def info(self) -> EngineInfo:
        return EngineInfo(
            id=self.id, name=self.name, available=self.available,
            reason=self.reason, voices=self.voices,
            capabilities=self.capabilities, license=self.license,
            hardware=self.hardware, repo=self.repo,
        )

    @abstractmethod
    def synthesize(self, text: str, voice: str,
                   controls: dict | None = None) -> np.ndarray:
        """Return mono float32 waveform at 24 kHz."""
        ...
