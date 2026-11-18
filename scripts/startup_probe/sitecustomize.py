"""Profile each app startup in its real Streamlit script thread when CI enables it."""

import json
import os
from pathlib import Path
import threading
import time

_destination = os.environ.get("KEYCLF_STARTUP_PROFILE_DIR")
_target = os.environ.get("KEYCLF_STARTUP_PROFILE_SCRIPT")
_local = threading.local()


def _trace(frame, event, arg):
    active = getattr(_local, "stack", None)
    if active is None:
        if event != "call" or frame.f_code.co_filename != _target or frame.f_code.co_name != "<module>":
            return
        _local.stack = active = []
        _local.operations = {}
    now = time.perf_counter_ns()
    if event in ("call", "c_call"):
        if event == "call":
            name = f"{frame.f_code.co_filename}:{frame.f_code.co_firstlineno}({frame.f_code.co_name})"
        else:
            name = f"{getattr(arg, '__module__', '')}.{getattr(arg, '__qualname__', repr(arg))}"
        active.append([name, now, 0])
    elif event in ("return", "c_return", "c_exception") and active:
        name, start, child = active.pop()
        elapsed = now - start
        record = _local.operations.setdefault(name, {"calls": 0, "max_ms": 0., "max_self_ms": 0.})
        record["calls"] += 1
        record["max_ms"] = max(record["max_ms"], elapsed / 1e6)
        record["max_self_ms"] = max(record["max_self_ms"], max(0, elapsed - child) / 1e6)
        if active:
            active[-1][2] += elapsed
        else:
            output = Path(_destination)
            output.mkdir(parents=True, exist_ok=True)
            target = output / f"{time.time_ns()}-{threading.get_ident()}.json"
            target.write_text(json.dumps({"operations": _local.operations}))
            _local.stack = None


if _destination and _target:
    threading.setprofile(_trace)
