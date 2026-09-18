import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from data_pipeline.common import artifact_path, sha256_file
from data_pipeline.pack import DTYPES


class IndexedRows:
    def __init__(self, manifest_path):
        path = Path(manifest_path)
        self.manifest = json.loads(path.read_text(encoding="utf-8"))
        m = self.manifest
        if m.get("kind") != "packed_tokens" or m.get("format_version") != 1:
            raise ValueError("Unsupported packed dataset")
        self.rows, self.length = m["rows"], m["sequence_length"]
        if self.rows <= 0 or self.length < 2: raise ValueError("Invalid dataset shape")
        self.arrays = {}
        for key, dtype in DTYPES.items():
            info = m["arrays"][key]
            if info["dtype"] != dtype: raise ValueError("Unexpected array dtype")
            file = artifact_path(path.parent, info["path"])
            if file.stat().st_size != self.rows*self.length*np.dtype(dtype).itemsize:
                raise ValueError("Array size mismatch")
            if sha256_file(file) != info["sha256"]: raise ValueError("Array checksum mismatch")
            self.arrays[key] = np.memmap(file, mode="r", dtype=dtype, shape=(self.rows,self.length))
        self.fingerprint = sha256_file(path)

    def __len__(self): return self.rows

    def __getitem__(self, index):
        if isinstance(index, slice): return [self[i] for i in range(*index.indices(self.rows))]
        if not 0 <= index < self.rows: raise IndexError(index)
        return {key:torch.tensor(np.array(values[index]), dtype=torch.bool if key=="attention_mask" else torch.long)
                for key,values in self.arrays.items()}


def shuffled_index(cursor, size, seed):
    """Constant-memory affine permutation per epoch; reproducible from cursor alone."""
    if size <= 0 or cursor < 0: raise ValueError("Invalid cursor/dataset length")
    epoch, offset = divmod(cursor,size)
    value = int(hashlib.sha256(f"{seed}:{epoch}".encode()).hexdigest(),16)
    stride = value % size or 1
    while math.gcd(stride,size) != 1: stride += 1
    return (stride*offset+(value//max(size,1)) % size) % size
