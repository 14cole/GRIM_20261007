"""Experimental run-local inverse reuse; accurate operators are never cached."""
from collections import OrderedDict
import math
import sys
import threading
from functools import wraps
import numpy as np

# Still at most five percent of the live solve reservation. The previous
# 128-MiB ceiling could not retain even one qualified 10-GHz cubic inverse.
# This capacity is opt-in and is included by solve/admission memory models.
MAX_BYTES=512*1024**2
FREQUENCY_RATIO=1.10
_CACHE_LOCK=threading.RLock()


def _synchronized(function):
    @wraps(function)
    def call(*args,**kwargs):
        # A tree has exactly one owner: a cache entry or one live solve.
        # BoR modes may transfer different entries concurrently.
        with _CACHE_LOCK:return function(*args,**kwargs)
    return call


@_synchronized
def capacity_bytes():
    from ghost_backend.execution.options import option
    from ghost_backend.twod.solver import _solve_memory_limit_gb
    capacity=(max(0,min(MAX_BYTES,int(.05*_solve_memory_limit_gb()*1024**3)))
              if option('frequency_preconditioner','off')=='reuse' else 0)
    from ghost_backend.twod.preparation import run_resources
    resources=run_resources()
    owner=resources.get('frequency_inverse_cache') if resources is not None else None
    if owner is not None:
        owner.capacity=capacity
        while owner.entries and owner.bytes>capacity:
            owner.bytes-=owner.entries.popitem(last=False)[1][0]
    return capacity


def _checkpoint(factor,callback):
    factor.checkpoint=callback
    pending=[factor.root]
    while pending:
        node=pending.pop();node.checkpoint=callback
        if not node.leaf:pending.extend((node.left,node.right))


def _idle():pass


def retained_size(value,seen=None):
    """Include numeric buffers, base allocations and Python tree bookkeeping."""
    seen=set() if seen is None else seen
    if id(value) in seen:return 0
    seen.add(id(value));size=sys.getsizeof(value)
    if isinstance(value,np.ndarray):
        return size+(retained_size(value.base,seen) if value.base is not None else 0)
    if isinstance(value,dict):return size+sum(retained_size(k,seen)+retained_size(v,seen) for k,v in value.items())
    if isinstance(value,(list,tuple)):return size+sum(retained_size(v,seen) for v in value)
    if not callable(value) and hasattr(value,'__dict__'):size+=retained_size(vars(value),seen)
    return size


class InverseCache:
    def __init__(self,capacity):
        self.capacity=capacity;self.bytes=0;self.entries=OrderedDict()

    @_synchronized
    def close(self):
        self.entries.clear();self.bytes=0

    @_synchronized
    def take(self,identity,frequency,budget):
        record=self.entries.pop(identity,None)
        if record is None:return None
        size,original,factor=record;self.bytes-=size
        if (not math.isfinite(frequency) or min(frequency,original)<=0 or
            max(frequency,original)/min(frequency,original)>FREQUENCY_RATIO or factor.bytes>budget):return None
        return factor,original

    @_synchronized
    def put(self,identity,frequency,factor):
        if (identity is None or not factor.inverse_only or not math.isfinite(frequency) or frequency<=0):return False
        size=retained_size(factor)+256
        if size>self.capacity:return False
        previous=self.entries.pop(identity,None)
        if previous:self.bytes-=previous[0]
        for key,record in list(self.entries.items()):
            original=record[1]
            if max(frequency,original)/min(frequency,original)>FREQUENCY_RATIO:
                self.bytes-=self.entries.pop(key)[0]
        # Alternating TE/TM inverses can each fit individually but not together.
        # Preserve the first useful entry instead of evicting it every step.
        if self.bytes+size>self.capacity:return False
        # A cached callback must not retain the completed frequency's CPU state.
        _checkpoint(factor,_idle)
        self.entries[identity]=(size,frequency,factor);self.bytes+=size
        return True


@_synchronized
def cache(create=False):
    from ghost_backend.twod.preparation import run_resources
    resources=run_resources()
    if resources is None:return None
    capacity=capacity_bytes()
    owner=resources.get('frequency_inverse_cache')
    if owner is not None:
        owner.capacity=capacity
        while owner.entries and owner.bytes>capacity:
            owner.bytes-=owner.entries.popitem(last=False)[1][0]
        return owner
    if not capacity:return None
    if create and 'frequency_inverse_cache' not in resources:
        resources['frequency_inverse_cache']=InverseCache(capacity)
    return resources.get('frequency_inverse_cache')


@_synchronized
def live_bytes():
    owner=cache()
    return owner.bytes if owner is not None else 0


@_synchronized
def reserve_current_inverse(needed,available):
    """Discard optional entries before a tight storage cap would prevent work."""
    owner=cache()
    if owner is None:return
    limit=max(0,available-needed)
    while owner.entries and owner.bytes>limit:
        owner.bytes-=owner.entries.popitem(last=False)[1][0]


@_synchronized
def take(identity,frequency,budget,checkpoint):
    owner=cache()
    if owner is None or identity is None:return None
    result=owner.take(identity,frequency,budget)
    if result is not None:_checkpoint(result[0],checkpoint)
    return result


@_synchronized
def save(identity,frequency,factor):
    owner=cache(create=True)
    return owner.put(identity,frequency,factor) if owner is not None else False
