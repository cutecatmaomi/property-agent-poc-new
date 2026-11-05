import json
import os
from dataclasses import dataclass
from pathlib import Path

from .config import ROOT


@dataclass(frozen=True)
class RuntimeSettings:
    root: Path
    uri: str
    username: str
    password: str
    database: str
    auth_path: Path
    profile: str = "main"

    @classmethod
    def load(cls, root=ROOT, profile="main"):
        if profile not in {"main", "test"}:
            raise ValueError("profile必须为main或test")
        root = Path(root)
        path = root / f"data/runtime/{profile}/connection.json"
        stored = json.loads(path.read_text()) if path.exists() else {}
        prefix = "POC_TEST_" if profile == "test" else "POC_"
        default_uri = "bolt://127.0.0.1:17687" if profile == "test" else "bolt://127.0.0.1:7687"
        uri = os.getenv(prefix + "NEO4J_URI", stored.get("uri", default_uri))
        password = os.getenv(prefix + "NEO4J_PASSWORD", stored.get("password", ""))
        if not password:
            raise ValueError("请先运行scripts/neo4j_local.py start，或设置POC_NEO4J_PASSWORD")
        return cls(
            root,
            uri,
            os.getenv(prefix + "NEO4J_USERNAME", stored.get("username", "neo4j")),
            password,
            os.getenv(prefix + "NEO4J_DATABASE", "neo4j"),
            root / f"data/runtime/{profile}/auth.sqlite3",
            profile,
        )
