import time
import datetime

class Timer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.start_time = time.perf_counter()
        self.last_time = time.perf_counter()

    def split(self):
        elapsed_time = time.perf_counter() - self.last_time
        self.last_time = time.perf_counter()
        return elapsed_time

    def total(self) -> float:
        return time.perf_counter() - self.start_time

    def total_human(self) -> str:
        return str(datetime.timedelta(seconds=self.total()))
