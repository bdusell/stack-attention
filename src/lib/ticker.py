import time

class Ticker:

    def __init__(self, total):
        super().__init__()
        self.progress = 0
        self.total = total

    def tick(self):
        raise NotImplementedError

    @property
    def fraction(self):
        return self.progress / self.total

    @property
    def percent(self):
        return 100.0 * self.fraction

    @property
    def int_percent(self):
        return 100 * self.progress // self.total

class OnChangeTicker(Ticker):

    def __init__(self, total):
        super().__init__(total)
        self.prev_tick = 0

    def tick(self):
        new_tick = self.new_tick()
        result = new_tick > self.prev_tick
        self.prev_tick = new_tick
        return result

    def new_tick(self):
        raise NotImplementedError

class DividedTicker(OnChangeTicker):

    def __init__(self, total, ticks):
        super().__init__(total)
        self.ticks = ticks

    def new_tick(self):
        return self.progress * self.ticks // self.total

class TimedTicker(OnChangeTicker):

    def __init__(self, total, seconds):
        super().__init__(total)
        self.seconds = seconds
        self.start_time = time.time()

    def new_tick(self):
        return int((time.time() - self.start_time) / self.seconds)
