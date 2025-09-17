# utils.py
from typing import List, Tuple
import json

class ROIStore:
    def __init__(self):
        self.polys = {}  # label -> list[(x,y)]

    def set_poly(self, label: str, pts: List[Tuple[int, int]]):
        self.polys[label] = [(int(x), int(y)) for x, y in pts]

    def clear(self):
        self.polys.clear()

    def have_all(self) -> bool:
        return set(self.polys.keys()) == {'N','E','S','W'} and all(len(v) >= 3 for v in self.polys.values())

    def to_json(self) -> str:
        return json.dumps(self.polys)

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.polys, f)

    def load(self, path: str):
        with open(path, 'r') as f:
            self.polys = {k: [tuple(p) for p in v] for k, v in json.load(f).items()}