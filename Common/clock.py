"""Monotonic seconds across CPython and MicroPython runtimes."""

import time


if hasattr(time, "monotonic"):
    monotonic = time.monotonic
else:
    _last_tick = None
    _elapsed_ms = 0

    def monotonic():
        """Return elapsed seconds using MicroPython's wrap-safe tick counter."""
        global _last_tick, _elapsed_ms
        tick = time.ticks_ms()
        if _last_tick is not None:
            _elapsed_ms += time.ticks_diff(tick, _last_tick)
        _last_tick = tick
        return _elapsed_ms / 1000.0
