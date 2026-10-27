"""Downloads MNIST and Fashion-MNIST, checks them against the published MD5s, caches them in
data/ (gitignored) and loads them as uint8 NumPy arrays.
"""

from __future__ import annotations

import gzip
import hashlib
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

MIRRORS = {
    "mnist": "https://raw.githubusercontent.com/fgnt/mnist/master/{}.gz",
    "fashion": "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/{}.gz",
}
MD5 = {
    "mnist": {
        "train-images-idx3-ubyte": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
        "train-labels-idx1-ubyte": "d53e105ee54ea40749a09fcbcd1e9432",
        "t10k-images-idx3-ubyte": "9fb629c4189551a2d022fa330f9573f3",
        "t10k-labels-idx1-ubyte": "ec29112dd5afa0611ce80d1b7f02629c",
    },
    "fashion": {
        "train-images-idx3-ubyte": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
        "train-labels-idx1-ubyte": "25c81989df183df01b3e8a0aad5dffbe",
        "t10k-images-idx3-ubyte": "bef4ecab320f06d8554ea6380940ec79",
        "t10k-labels-idx1-ubyte": "bb300cfdad3c16e7a12a480ee83cd310",
    },
}


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def fetch(dataset: str) -> Path:
    d = DATA / dataset
    d.mkdir(parents=True, exist_ok=True)
    for name, md5 in MD5[dataset].items():
        f = d / f"{name}.gz"
        if not f.exists() or _md5(f) != md5:
            urllib.request.urlretrieve(MIRRORS[dataset].format(name), f)
        got = _md5(f)
        if got != md5:
            f.unlink()
            raise RuntimeError(f"MD5 mismatch for {dataset}/{name}: {got} != {md5}")
    return d


def _idx(path: Path) -> np.ndarray:
    raw = gzip.decompress(path.read_bytes())
    ndim = raw[3]
    dims = [int.from_bytes(raw[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim)]
    assert raw[2] == 0x08, "expected unsigned byte data"
    return np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)


def load(dataset: str):
    """Returns (x_train, y_train, x_test, y_test), with each image flattened to 784 uint8 values."""
    d = fetch(dataset)
    xtr = _idx(d / "train-images-idx3-ubyte.gz").reshape(-1, 784)
    ytr = _idx(d / "train-labels-idx1-ubyte.gz")
    xte = _idx(d / "t10k-images-idx3-ubyte.gz").reshape(-1, 784)
    yte = _idx(d / "t10k-labels-idx1-ubyte.gz")
    return xtr, ytr, xte, yte


if __name__ == "__main__":
    for ds in ("mnist", "fashion"):
        xtr, ytr, xte, yte = load(ds)
        print(ds, xtr.shape, ytr.shape, xte.shape, yte.shape, "MD5 ok")
