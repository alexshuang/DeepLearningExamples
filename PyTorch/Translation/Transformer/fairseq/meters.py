# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import time
import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class TimeMeter(object):
    """Computes the average occurrence of some event per second"""
    def __init__(self, init=0):
        self.reset(init)

    def reset(self, init=0):
        self.init = init
        self.start = time.time()
        self.n = 0
        self.last_update = time.time()

    def update(self, val=1):
        self.n += val
        self.last_update = time.time()

    @property
    def avg(self):
        return self.n / self.elapsed_time

    @property
    def elapsed_time(self):
        return self.init + (time.time() - self.start)

    @property
    def u_avg(self):
        return self.n / (self.last_update - self.start)


     
class StopwatchMeter(object):
    """Computes the sum/avg duration of some event in seconds"""
    def __init__(self):
        self.reset()
        self.intervals = []

    def start(self):
        self.start_time = time.time()

    def stop(self, n=1):
        if self.start_time is not None:
            delta = time.time() - self.start_time
            self.intervals.append(delta)
            self.sum += delta
            self.n += n
            self.start_time = None

    def reset(self):
        self.sum = 0
        self.n = 0
        self.start_time = None
        self.intervals = []

    @property
    def avg(self):
        return self.sum / self.n

    def p(self, i):
        assert i <= 100
        idx = int(len(self.intervals) * i / 100)
        return sorted(self.intervals)[idx]


class IterMeter():
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

    def reset(self):
        #global self.iter_elapsed
        self.iter_elapsed, self.fwd_elapsed, self.bwd_elapsed, self.optim_elapsed = [], [], [], [] # cumulative time during which stopwatch was active
        self._iter_start, self.fwd_start, self.bwd_start, self.opt_start = None, None, None, None
