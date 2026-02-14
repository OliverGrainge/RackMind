"""Discrete time-step simulation clock."""

import time
from dataclasses import dataclass, field


@dataclass
class SimulationClock:
    """Maintains simulation time and supports real-time throttling."""

    tick_interval_s: float
    realtime_factor: float = 0.0
    current_time: float = 0.0
    tick_count: int = 0

    def tick(self, n: int = 1) -> None:
        """Advance simulation by n ticks."""
        for _ in range(n):
            self.tick_count += 1
            self.current_time += self.tick_interval_s

            if self.realtime_factor > 0:
                sleep_time = self.tick_interval_s * self.realtime_factor
                time.sleep(sleep_time)

    @property
    def elapsed_human_readable(self) -> str:
        """Format elapsed time as HH:MM:SS."""
        total_seconds = int(self.current_time)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
