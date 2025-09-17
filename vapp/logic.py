# logic.py
from dataclasses import dataclass, field
import time
from typing import Dict

@dataclass
class Timing:
    base_green: int = 30   # seconds
    bonus: int = 15        # add if current has highest count
    penalty: int = 15      # subtract if another approach higher
    yellow: int = 3        # fixed yellow time
    min_green: int = 10
    max_green: int = 90

@dataclass
class SignalState:
    current: str = 'N'           # 'N','E','S','W'
    phase: str = 'GREEN'         # 'GREEN'|'YELLOW'
    phase_ends_at: float = 0.0   # epoch seconds when current phase ends

class TrafficController:
    def __init__(self, timing: Timing = None):
        self.timing = timing or Timing()
        self.state = SignalState()
        self.order = ['N','E','S','W']
        self._set_phase('GREEN', duration=self.timing.base_green)

    def _now(self):
        return time.time()

    def _set_phase(self, phase: str, duration: int):
        self.state.phase = phase
        self.state.phase_ends_at = self._now() + duration

    def _decide_green_duration(self, counts: Dict[str, int]) -> int:
        cur = self.state.current
        base = self.timing.base_green
        # who is max now?
        max_approach = max(counts, key=lambda k: counts[k])
        if counts.get(cur, 0) >= counts.get(max_approach, 0):
            dur = base + self.timing.bonus
        else:
            dur = max(self.timing.min_green, base - self.timing.penalty)
        return max(self.timing.min_green, min(self.timing.max_green, int(dur)))

    def _rank_next(self, counts: Dict[str, int]) -> str:
        # Choose the highest; break ties by circular order after current
        max_count = max(counts.values())
        candidates = [k for k,v in counts.items() if v == max_count]
        start = (self.order.index(self.state.current) + 1) % 4
        for i in range(4):
            name = self.order[(start + i) % 4]
            if name in candidates:
                return name
        return self.order[start]

    def tick(self, counts: Dict[str, int]):
        """Call this periodically with latest counts. Updates internal state."""
        now = self._now()
        if now < self.state.phase_ends_at:
            return  # nothing to do
        # phase elapsed
        if self.state.phase == 'GREEN':
            # switch to yellow on current approach
            self._set_phase('YELLOW', self.timing.yellow)
        else:
            # end of yellow => choose next approach and green duration
            self.state.current = self._rank_next(counts)
            dur = self._decide_green_duration(counts)
            self._set_phase('GREEN', dur)

    def export(self) -> Dict:
        return {
            'current': self.state.current,
            'phase': self.state.phase,
            'ends_at': self.state.phase_ends_at,
            'now': self._now(),
        }

