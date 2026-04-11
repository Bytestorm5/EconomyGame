from __future__ import annotations
from typing import Any, Callable, Dict, Type, get_type_hints
from enum import Enum
from pydantic import BaseModel


class Scope(str, Enum):
    WORLD = "world"
    PERSON = "person"
    COMPANY = "company"
    BUILDING = "building"
    MACHINE = "machine"


class GetterSpec(BaseModel):
    name: str
    scope: Scope
    fn: Callable[..., Any]
    args_schema: Dict[str, type]
    return_type: type
    as_property: bool = True


class SelectorSpec(BaseModel):
    name: str
    from_scope: Scope
    to_scope: Scope
    args_schema: Dict[str, type]
    fn: Callable[..., Any]


class EffectSpec(BaseModel):
    name: str
    scope: Scope
    params_model: Type[BaseModel]
    fn: Callable[..., None]


CONDITION_GETTERS: Dict[tuple[str, Scope], GetterSpec] = {}
SELECTORS: Dict[str, SelectorSpec] = {}
EFFECT_ACTIONS: Dict[tuple[str, Scope], EffectSpec] = {}


def condition_getter(name: str, scope: Scope, *, as_property: bool = True):
    def deco(fn: Callable[..., Any]):
        hints = get_type_hints(fn)
        ret = hints.get("return", Any)
        args = {k: v for k, v in hints.items() if k != "return"}
        CONDITION_GETTERS[(name, scope)] = GetterSpec(
            name=name,
            scope=scope,
            fn=fn,
            args_schema=args,
            return_type=ret,
            as_property=as_property,
        )
        return fn
    return deco


def scope_selector(name: str, from_scope: Scope, to_scope: Scope):
    def deco(fn: Callable[..., Any]):
        hints = get_type_hints(fn)
        args = {k: v for k, v in hints.items() if k != "return"}
        SELECTORS[name] = SelectorSpec(
            name=name,
            from_scope=from_scope,
            to_scope=to_scope,
            args_schema=args,
            fn=fn,
        )
        return fn
    return deco


def effect_action(name: str, scope: Scope, params_model: Type[BaseModel]):
    def deco(fn: Callable[..., None]):
        EFFECT_ACTIONS[(name, scope)] = EffectSpec(
            name=name,
            scope=scope,
            params_model=params_model,
            fn=fn,
        )
        return fn
    return deco
