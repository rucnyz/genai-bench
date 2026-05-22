# Gevent monkey patching must be done before any other imports
# This is required for proper cooperative multitasking with Locust
# Without this, blocking I/O operations (like HTTP requests) will block
# the entire worker process, causing heartbeat timeouts
from gevent import monkey

monkey.patch_all()


# ── Workaround: multiprocess.resource_tracker + gevent shutdown race ──
#
# huggingface `datasets` transitively pulls in the third-party
# `multiprocess` package (a dill-aware fork of stdlib multiprocessing).
# Its `resource_tracker.ResourceTracker.__del__` runs during interpreter
# shutdown and calls `threading.RLock.acquire()` → `_thread.get_ident()`.
# By that point gevent's monkey patches have begun finalizing greenlets,
# so the patched `_thread.get_ident()` raises
# `RuntimeError: greenlet is being finalized`. The exception in
# `__del__` is normally just warned ("Exception ignored in ..."), but
# in practice the bench process gets stuck in an unkillable R-state
# user-mode loop right after — likely because the tracker's
# subprocess never receives its EOF signal and the parent never
# completes shutdown cleanly. A `SIGKILL` from outside doesn't help
# (kernel can only deliver fatal signals on the syscall return path,
# and the process never makes one).
#
# Catch the RuntimeError so `_stop` returns cleanly. The tracker's
# resources (semaphores, shared-memory segments) are also cleaned
# automatically on process exit by the kernel, so swallowing this
# error doesn't actually leak anything in practice.
def _install_resource_tracker_gevent_safety() -> None:
    try:
        import multiprocess.resource_tracker as _mp_rt
    except ImportError:
        return

    _original = _mp_rt.ResourceTracker._stop

    # The wrapper captures `_original` via closure, not by referencing
    # a module-level global. Module globals get cleared to `None`
    # during interpreter shutdown in an unpredictable order, so a
    # global-referencing wrapper can race the finalizer and explode
    # with `TypeError: 'NoneType' object is not callable`.
    def _stop_safe(self, *args, **kwargs):  # noqa: ANN001
        try:
            return _original(self, *args, **kwargs)
        except RuntimeError:
            # gevent has already finalized greenlets — best-effort
            # cleanup is done; let the interpreter continue exiting.
            return None

    _mp_rt.ResourceTracker._stop = _stop_safe


_install_resource_tracker_gevent_safety()
del _install_resource_tracker_gevent_safety
