"""Versioned corpus LSA vectors. This is not a pretrained embedding service or an approval cache."""

import json
import math
import os
import re
import threading
import uuid
from collections import Counter
from pathlib import Path

from ..domain import canonical, digest

INDEX_VERSION = "corpus-lsa-v2"
_lock = threading.RLock()
_cache = {}


def terms(text):
    text = re.sub(r"\s+", "", text.lower())
    return Counter([text[i:i + n] for n in (2, 3) for i in range(len(text) - n + 1)]
                   + re.findall(r"[a-z_]+", text))


def fingerprint(rows):
    return digest({"algorithm": INDEX_VERSION, "records": sorted(rows, key=lambda r: r["id"])})


def build(rows, folder):
    import numpy as np

    rows = sorted(rows, key=lambda r: r["id"])
    signature = fingerprint(rows)
    folder = Path(folder)
    target = folder / (signature + ".npz")
    manifest = folder / (signature + ".json")
    if target.exists() and manifest.exists():
        metadata = json.loads(manifest.read_text())
        if metadata.get("fingerprint") == signature and metadata.get("binary_sha256") == digest(target.read_bytes()):
            return metadata
    counts = [terms(" ".join(str(r.get(k) or "") for k in ("question", "title", "service_code", "concepts"))) for r in rows]
    df = Counter(t for c in counts for t in c)
    vocabulary = sorted(t for t, n in df.items() if n >= 2)
    if not vocabulary:
        vocabulary = sorted(df)
    lookup = {t: i for i, t in enumerate(vocabulary)}
    idf = np.array([math.log((1 + len(rows)) / (1 + df[t])) + 1 for t in vocabulary], dtype=np.float32)
    matrix = np.zeros((len(rows), len(vocabulary)), dtype=np.float32)
    for i, c in enumerate(counts):
        for t, n in c.items():
            if t in lookup:
                matrix[i, lookup[t]] = (1 + math.log(n)) * idf[lookup[t]]
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    rank = min(64, max(1, min(matrix.shape) - 1))
    rng = np.random.default_rng(20261030)
    y = matrix @ rng.normal(size=(matrix.shape[1], min(rank + 8, min(matrix.shape)))).astype(np.float32)
    for _ in range(2):
        y = matrix @ (matrix.T @ y)
    q, _ = np.linalg.qr(y, mode="reduced")
    _, _, vt = np.linalg.svd(q.T @ matrix, full_matrices=False)
    projection = vt[:rank].T.astype(np.float32)
    vectors = matrix @ projection
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder / (signature + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("wb") as f:
        np.savez_compressed(f, projection=projection, vectors=vectors, idf=idf)
    os.replace(temporary, target)
    metadata = {"algorithm": INDEX_VERSION, "fingerprint": signature, "rows": rows,
                "vocabulary": vocabulary, "dimensions": rank, "approval_cache": False,
                "binary_sha256": digest(target.read_bytes()),
                "description": "Corpus-trained latent semantic analysis; not pretrained multilingual embeddings."}
    path = folder / (signature + ".json")
    temporary = folder / (signature + "." + uuid.uuid4().hex + ".json.tmp")
    temporary.write_text(canonical(metadata))
    os.replace(temporary, path)
    return metadata


def recall(root, estate_id, rows, question, *, namespace=None):
    """Live graph metadata controls the cache key. No stale row/version can be returned."""
    if any(r["estate_id"] != estate_id or not r["id"].startswith(estate_id + ":") for r in rows):
        raise ValueError("向量索引不能跨楼盘")
    if not rows:
        return {}, {"status": "EMPTY", "algorithm": INDEX_VERSION}
    try:
        import numpy as np
    except ImportError:
        return {}, {"status": "UNAVAILABLE_NUMPY", "algorithm": INDEX_VERSION}
    signature = fingerprint(rows)
    if namespace is not None and not re.fullmatch(r"[A-Z0-9_]+", namespace):
        raise ValueError("invalid index namespace")
    base = Path(root) / "data/retrieval_index"
    folder = (base / namespace if namespace else base) / estate_id
    key = (str(folder), signature)
    with _lock:
        if key not in _cache:
            path = folder / (signature + ".json")
            status = "REUSED"
            if not path.exists() or not (folder / (signature + ".npz")).exists():
                build(rows, folder)
                status = "BUILT_CURRENT_VERSION"
            meta = json.loads(path.read_text())
            if meta["fingerprint"] != signature or fingerprint(meta["rows"]) != signature:
                raise ValueError("索引版本不一致")
            if meta.get("binary_sha256") != digest((folder / (signature + ".npz")).read_bytes()):
                build(rows, folder)
                meta = json.loads(path.read_text())
            with np.load(folder / (signature + ".npz"), allow_pickle=False) as data:
                _cache.clear()
                _cache[key] = (meta, {n: data[n].copy() for n in ("projection", "vectors", "idf")})
        else:
            status = "MEMORY_CURRENT_VERSION"
        meta, data = _cache[key]
    lookup = {t: i for i, t in enumerate(meta["vocabulary"])}
    vector = np.zeros(len(lookup), dtype=np.float32)
    for t, count in terms(question).items():
        if t in lookup:
            vector[lookup[t]] = (1 + math.log(count)) * data["idf"][lookup[t]]
    latent = vector @ data["projection"]
    latent /= max(float(np.linalg.norm(latent)), 1e-12)
    scores = data["vectors"] @ latent
    return {r["id"]: float(scores[i]) for i, r in enumerate(meta["rows"])}, {
        "status": status, "algorithm": INDEX_VERSION, "fingerprint": signature,
        "dimensions": meta["dimensions"], "records": len(rows), "approval_cache": False}
