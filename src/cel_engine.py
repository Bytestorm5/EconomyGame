from __future__ import annotations
from decimal import Decimal
from typing import Any, Dict
import celpy
import celpy.celtypes as ct


def to_cel(value: Any) -> Any:
    """Convert Python native types to CEL types."""
    if isinstance(value, bool):
        return ct.BoolType(value)
    if isinstance(value, int):
        return ct.IntType(value)
    if isinstance(value, float):
        return ct.DoubleType(value)
    if isinstance(value, Decimal):
        return ct.DoubleType(float(value))
    if isinstance(value, str):
        return ct.StringType(value)
    return value  # Views and other objects pass through


class CelEngine:
    def __init__(self):
        self._env = celpy.Environment()
        self._cache: Dict[str, Any] = {}

    def compile(self, expr: str) -> Any:
        if expr in self._cache:
            return self._cache[expr]
        ast = self._env.compile(expr)
        prog = self._env.program(ast)
        self._cache[expr] = prog
        return prog

    def evaluate(self, program: Any, activation: Dict[str, Any]) -> Any:
        cel_act = {k: to_cel(v) for k, v in activation.items()}
        return program.evaluate(cel_act)


ENGINE = CelEngine()
