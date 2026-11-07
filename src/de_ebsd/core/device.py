"""Device selection and device locking shared by every entry point that takes ``device=``.

The lock is the caller's: an application whose threads share one accelerator owns
one lock and hands it to every de-ebsd object and call that runs torch work
(``lock=``). de-ebsd only holds it, so it serializes against the application's
other torch users as well as against itself. Apple's MPS is why: two threads
submitting work to it at once crash the process with a native segfault that no
``try`` sees.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
import logging
import os

import torch

__all__ = ["default_device", "holds_lock", "resolve_device", "resolve_dtype", "resolve_lock"]

log = logging.getLogger(__name__)


def resolve_device(device, default="cpu") -> torch.device:
    """*device* as a :class:`torch.device`, or *default* when it is ``None``."""
    return torch.device(default if device is None else device)


def default_device() -> torch.device:
    """The fastest device here: CUDA, then MPS, then the CPU.

    The environment variable ``DE_EBSD_DEVICE`` (``"cpu"``, ``"cuda:1"``, ...)
    overrides it. Indexing is bound by one large matrix product, the shape of work
    an accelerator gains on; at a small problem size the CPU can still be faster
    (measured on an M1's 8 GPU cores: MPS at half the CPU's speed for 256 patterns
    against 1000 entries, 1.16 times faster for 4096 against 20000), so measure
    with the override rather than adding a size threshold here.
    """
    forced = os.environ.get("DE_EBSD_DEVICE")
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(device, dtype):
    """*dtype* (a :class:`torch.dtype` or its name) as *device* can run it.

    Metal has no float64, so on MPS ``float64`` becomes ``float32``, in the form it
    was given; coercing here is better than failing deep inside a loop of tiles.
    """
    if resolve_device(device).type != "mps":
        return dtype
    if dtype is torch.float64:
        log.debug("MPS does not support float64; running in float32")
        return torch.float32
    if dtype == "float64":
        log.debug("MPS does not support float64; running in float32")
        return "float32"
    return dtype


def resolve_lock(lock) -> contextlib.AbstractContextManager:
    """*lock* to hold around device work, or a context that does nothing when it is
    ``None``.

    *lock* is any context manager that can be entered again and again and from
    inside itself, such as a :class:`threading.RLock`: de-ebsd's public calls
    nest (an index refines), and each holds it.
    """
    return contextlib.nullcontext() if lock is None else lock


def holds_lock(function):
    """Run *function* holding a lock (see :func:`resolve_lock`): its own ``lock``
    argument when it takes one, such as a function or a constructor, and otherwise,
    for a method, its instance's ``lock`` attribute."""
    signature = inspect.signature(function)
    takes_lock = "lock" in signature.parameters

    @functools.wraps(function)
    def locked(*args, **kwargs):
        if takes_lock:
            lock = resolve_lock(signature.bind(*args, **kwargs).arguments.get("lock"))
        else:
            lock = args[0].lock
        with lock:
            return function(*args, **kwargs)

    return locked
