# -*- coding: utf-8 -

import time


class Timer:
    def __init__(self):
        self.stime = 0
        self.etime = 0

    def start(self):
        self.stime = time.perf_counter()
        return self.stime

    def stop(self):
        self.etime = time.perf_counter()
        return self.etime

    def get_time(self, ndigits: int = 2):
        return round(self.etime - self.stime, ndigits)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        self.stop()

    def __str__(self):
        t = self.get_time()
        if t < 1:
            return f"{t * 1000:.{2}f}ms"
        else:
            return f"{t:.{2}f}s"
