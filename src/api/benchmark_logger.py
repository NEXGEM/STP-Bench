"""Small structured logger for STPBench benchmark orchestration."""

from __future__ import annotations

import json
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Iterator, Optional

# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------

_R   = "\033[0m"   # reset
_B   = "\033[1m"   # bold
_D   = "\033[2m"   # dim
_RED = "\033[31m"
_GRN = "\033[32m"
_YLW = "\033[33m"
_BLU = "\033[34m"
_CYN = "\033[36m"
_GRY = "\033[90m"

_STATUS = {
    "start":  ("▶", _CYN),
    "done":   ("✓", _GRN),
    "failed": ("✗", _RED),
}
_LEVEL = {
    "INFO":  ("·", _BLU,  _CYN),
    "WARN":  ("⚠", _YLW,  _YLW),
    "ERROR": ("✗", _RED,  _RED),
}


def _color_ok() -> bool:
    out = sys.__stdout__ if sys.__stdout__ is not None else sys.stdout
    return hasattr(out, "isatty") and out.isatty()


def _fmt(record: Dict[str, Any]) -> str:
    color = _color_ok()

    display   = dict(record)
    level     = display.pop("level")
    event     = display.pop("event")
    raw_time  = display.pop("time")
    status    = display.pop("status", None)
    elapsed   = display.pop("elapsed_sec", None)
    error_msg = display.pop("error", None)
    hint_msg  = display.pop("hint", None)

    time_str = raw_time.split("T")[-1] if "T" in raw_time else raw_time

    # Choose icon + color
    if status in _STATUS:
        icon, icon_c = _STATUS[status]
    else:
        icon, icon_c, _ = _LEVEL.get(level, ("·", _BLU, _CYN))
    _, _, prefix_c = _LEVEL.get(level, ("·", _BLU, _CYN))

    # Build field string  (elapsed first, then the rest)
    parts: list[str] = []
    if elapsed is not None:
        parts.append(f"({elapsed}s)")
    for k, v in display.items():
        if v is not None:
            parts.append(f"{k}={v}")
    fields_str = "  ".join(parts)

    if color:
        tag    = f"{_B}{prefix_c}[STPBench]{_R}"
        ts     = f"{_D}{_GRY}{time_str}{_R}"
        ic     = f"{icon_c}{icon}{_R}"
        ev     = f"{_B}{event}{_R}"
        fd     = f"  {_D}{_GRY}{fields_str}{_R}" if fields_str else ""
        line   = f"{tag} {ts}  {ic}  {ev}{fd}"
        indent = "             "  # visual alignment under event text
        extras = []
        if error_msg:
            extras.append(f"{indent}{_RED}error : {error_msg}{_R}")
        if hint_msg:
            extras.append(f"{indent}{_YLW}hint  : {hint_msg}{_R}")
    else:
        line   = f"[STPBench] {time_str}  {icon}  {event}"
        if fields_str:
            line += f"  {fields_str}"
        indent = "             "
        extras = []
        if error_msg:
            extras.append(f"{indent}error : {error_msg}")
        if hint_msg:
            extras.append(f"{indent}hint  : {hint_msg}")

    if extras:
        return line + "\n" + "\n".join(extras)
    return line


def format_stpbench_line(msg: str, level: str = "INFO") -> str:
    """Format a one-off [STPBench] info line (for use outside BenchmarkLogger)."""
    return _fmt({
        "time": datetime.now().isoformat(timespec="seconds"),
        "level": level,
        "event": msg,
    })


# ---------------------------------------------------------------------------
# Logger class
# ---------------------------------------------------------------------------

class BenchmarkLogger:
    """Emit consistent benchmark events to stdout and optional JSONL."""

    def __init__(self, enabled: bool = True, log_file: Optional[str] = None):
        self.enabled = enabled
        self.log_file = log_file
        if self.log_file:
            os.makedirs(os.path.dirname(os.path.abspath(self.log_file)), exist_ok=True)

    def info(self, event: str, **fields: Any) -> None:
        self._emit("INFO", event, fields)

    def warning(self, event: str, **fields: Any) -> None:
        self._emit("WARN", event, fields)

    def error(self, event: str, **fields: Any) -> None:
        self._emit("ERROR", event, fields)

    @contextmanager
    def section(self, event: str, **fields: Any) -> Iterator[None]:
        from api.output_control import parse_error_hint

        start = time.perf_counter()
        self.info(event, status="start", **fields)
        try:
            yield
        except Exception as exc:
            hint = parse_error_hint(exc)
            error_fields: Dict[str, Any] = {
                "status": "failed",
                "elapsed_sec": round(time.perf_counter() - start, 3),
                "error": f"{type(exc).__name__}: {exc}",
                **fields,
            }
            if hint:
                error_fields["hint"] = hint
            self.error(event, **error_fields)
            raise
        self.info(event, status="done", elapsed_sec=round(time.perf_counter() - start, 3), **fields)

    def _emit(self, level: str, event: str, fields: Dict[str, Any]) -> None:
        if not self.enabled and not self.log_file:
            return
        record = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "level": level,
            "event": event,
            **fields,
        }
        if self.enabled:
            # Write to the real terminal even when sys.stdout is redirected
            # to /dev/null by suppress_library_output().
            out = sys.__stdout__ if sys.__stdout__ is not None else sys.stdout
            print(_fmt(record), flush=True, file=out)
        if self.log_file:
            with open(self.log_file, "a") as f:
                # Strip ANSI from JSON log
                clean = dict(record)
                f.write(json.dumps(clean, default=str, ensure_ascii=False) + "\n")
