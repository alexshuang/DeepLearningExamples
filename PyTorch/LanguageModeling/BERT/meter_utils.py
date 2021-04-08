import time
import numpy as np

class StopwatchMeter():
    """ Computes the sum/avg duration of some event in seconds """
    def __init__(self):
        self.reset()

    def iter_start(self):
        self._iter_start = time.perf_counter()

    def iter_stop(self):
        if self._iter_start is not None:
            self.iter_elapsed.append(time.perf_counter() - self._iter_start)
    
    def forward_start(self):
        self.fwd_start = time.perf_counter()

    def forward_stop(self):
        if self.fwd_start is not None:
            self.fwd_elapsed.append(time.perf_counter() - self.fwd_start)
    
    def backward_start(self):
        self.bwd_start = time.perf_counter()

    def backward_stop(self):
        if self.bwd_start is not None:
            self.bwd_elapsed.append(time.perf_counter() - self.bwd_start)

    def optim_start(self):
        self.opt_start = time.perf_counter()

    def optim_stop(self):
        if self.opt_start is not None:
            self.optim_elapsed.append(time.perf_counter() - self.opt_start)

    def dataloader_start(self):
        self.dl_start = time.perf_counter()

    def dataloader_stop(self):
        if self.dl_start is not None:
            self.dataloader_elapsed.append(time.perf_counter() - self.dl_start)

    @property
    def iter_elapsed_total(self): return np.sum(self.iter_elapsed)
    @property
    def iter_elapsed_avg(self): return np.mean(self.iter_elapsed)
    @property
    def fwd_elapsed_total(self): return np.sum(self.fwd_elapsed)
    @property
    def fwd_elapsed_avg(self): return np.mean(self.fwd_elapsed)
    @property
    def bwd_elapsed_total(self): return np.sum(self.bwd_elapsed)
    @property
    def bwd_elapsed_avg(self): return np.mean(self.bwd_elapsed)
    @property
    def optim_elapsed_total(self): return np.sum(self.optim_elapsed)
    @property
    def optim_elapsed_avg(self): return np.mean(self.optim_elapsed)
    @property
    def dataloader_elapsed_total(self): return np.sum(self.dataloader_elapsed)
    @property
    def dataloader_elapsed_avg(self): return np.mean(self.dataloader_elapsed)

    def reset(self):
        self.iter_elapsed, self.fwd_elapsed, self.bwd_elapsed, self.optim_elapsed, self.dataloader_elapsed = [], [], [], [], [] # cumulative time during which stopwatch was active
        self._iter_start, self.fwd_start, self.bwd_start, self.opt_start, self.dl_start = None, None, None, None, None


class CancelBWDException(Exception): pass
class CancelOptimException(Exception): pass
class CancelTrainException(Exception): pass
