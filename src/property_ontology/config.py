import os
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    root: Path = ROOT
    soffice_path: str = ""

    @classmethod
    def load(cls, root: Path = ROOT):
        return cls(root=root, soffice_path=os.getenv("SOFFICE_PATH", ""))

    def estates(self):
        return yaml.safe_load((self.root / "config/estates.yaml").read_text())["estates"]
