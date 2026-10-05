from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class EvidenceChunk:
    id: str
    text: str
    metadata: dict[str, Any]
    score: float


class VectorStore(ABC):
    @abstractmethod
    async def add(self, texts: list[str], metadata: list[dict[str, Any]]) -> int: ...

    @abstractmethod
    async def search(self, query: str, top_k: int) -> list[EvidenceChunk]: ...

    @property
    @abstractmethod
    def readiness(self) -> str: ...
