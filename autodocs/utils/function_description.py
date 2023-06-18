from dataclasses import dataclass
from typing import Optional


@dataclass
class FunctionDescription:
    name: str
    source: str
    docs: str
    arguments: dict[str, str]
    signature: Optional[str]
