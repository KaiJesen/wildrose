from __future__ import annotations

from enum import Enum


class Side(str, Enum):
    FLAT = "FLAT"
    LONG = "LONG"
    SHORT = "SHORT"


class ActionType(str, Enum):
    OPEN_LONG = "OPEN_LONG"
    OPEN_SHORT = "OPEN_SHORT"
    ADD = "ADD"
    REDUCE = "REDUCE"
    CLOSE = "CLOSE"
    REVERSE = "REVERSE"
    HOLD = "HOLD"
    BLOCK = "BLOCK"
    FORCE_CLOSE = "FORCE_CLOSE"


class SignalStatus(str, Enum):
    VALID = "VALID"
    INVALID = "INVALID"

