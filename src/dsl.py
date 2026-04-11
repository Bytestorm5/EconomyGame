from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel
import objects as G
from views import WorldView, PersonView, CompanyView, BuildingView, MachineView
from cel_engine import ENGINE
from scripting import Scope, SELECTORS, EFFECT_ACTIONS


class EvalContext(BaseModel):
    world: G._WorldState
    tick: int = 0
    rng: Optional[Any] = None


def _wrap_view(obj: Any) -> Any:
    if isinstance(obj, G._WorldState):
        return WorldView(obj)
    if isinstance(obj, G._PersonInstance):
        return PersonView(obj)
    if isinstance(obj, G._CompanyInstance):
        return CompanyView(obj)
    if isinstance(obj, G._BuildingInstance):
        return BuildingView(obj)
    if isinstance(obj, G._MachineInstance):
        return MachineView(obj)
    return obj


def resolve_scopes(base: Dict[str, Any], scopes_cfg: Optional[Dict[str, Dict[str, Any]]], ctx: EvalContext) -> Dict[str, Any]:
    resolved: Dict[str, Any] = {k: _wrap_view(v) for k, v in base.items()}
    if not scopes_cfg:
        return resolved
    for name, node in scopes_cfg.items():
        sel_name = node.get("selector")
        spec = SELECTORS.get(sel_name)
        if not spec:
            raise ValueError(f"Unknown selector '{sel_name}'")
        args: Dict[str, Any] = {}
        for k, v in (node.get("args") or {}).items():
            if isinstance(v, str) and v in resolved:
                args[k] = resolved[v]
            else:
                args[k] = v
        if "world" in spec.args_schema and "world" not in args:
            args["world"] = WorldView(ctx.world)
        target = spec.fn(**args)
        resolved[name] = _wrap_view(target)
    return resolved


def build_activation(scopes: Dict[str, Any], ctx: EvalContext) -> Dict[str, Any]:
    act = dict(scopes)
    act["world"] = WorldView(ctx.world)
    act["tick"] = ctx.tick
    return act


def _eval_expr(expr: str, activation: Dict[str, Any]) -> bool:
    prog = ENGINE.compile(expr)
    result = ENGINE.evaluate(prog, activation)
    return bool(result)


def evaluate_condition_block(
    cond: Optional[G.ConditionBlock],
    ctx: EvalContext,
    base_scopes: Dict[str, Any],
    scopes_cfg: Optional[Dict[str, Dict[str, Any]]] = None,
) -> bool:
    if cond is None:
        return True
    scopes = resolve_scopes(base_scopes, scopes_cfg, ctx)
    activation = build_activation(scopes, ctx)

    # If a string expression is provided, treat it as CEL
    if isinstance(cond.condition, str):
        return _eval_expr(cond.condition, activation)

    # Logical combinators
    if cond.AND:
        if not all(evaluate_condition_block(c, ctx, base_scopes, scopes_cfg) for c in cond.AND):
            return False
    if cond.OR:
        if not any(evaluate_condition_block(c, ctx, base_scopes, scopes_cfg) for c in cond.OR):
            return False
    if cond.NOT:
        if evaluate_condition_block(cond.NOT, ctx, base_scopes, scopes_cfg):
            return False

    # Count-based operators
    def _count(lst: List[G.ConditionBlock]) -> int:
        return sum(1 for c in lst if evaluate_condition_block(c, ctx, base_scopes, scopes_cfg))

    if cond.atLeast and _count(cond.atLeast.conditions) < cond.atLeast.n:
        return False
    if cond.atMost and _count(cond.atMost.conditions) > cond.atMost.n:
        return False
    if cond.exactly and _count(cond.exactly.conditions) != cond.exactly.n:
        return False

    # Default to true if no explicit primitive condition
    # No condition specified and no logical operators → passes by default (permissive null condition)
    if (
        cond.condition is None
        and not (cond.AND or cond.OR or cond.NOT or cond.atLeast or cond.atMost or cond.exactly)
    ):
        return True
    return True


def apply_effect_list(
    effects: List[G._SimpleEffect],
    ctx: EvalContext,
    base_scopes: Dict[str, Any],
    current_scope: Scope,
) -> None:
    scopes_wrapped = {k: _wrap_view(v) for k, v in base_scopes.items()}
    for eff in effects or []:
        if isinstance(eff, str):
            parts = eff.strip().split()
            if not parts:
                continue
            op = parts[0]
            kv: Dict[str, Any] = {}
            for p in parts[1:]:
                if "=" in p:
                    k, v = p.split("=", 1)
                    kv[k] = v
            spec = EFFECT_ACTIONS.get((op, current_scope)) or EFFECT_ACTIONS.get((op, Scope.WORLD))
            if not spec:
                raise ValueError(f"Unknown effect action '{op}' for scope {current_scope}")
            params = spec.params_model.model_validate(kv)
            spec.fn(params, ctx, scopes_wrapped)
        else:
            data = dict(eff)
            op = data.pop("op", None)
            if not op:
                if len(data) == 1:
                    op, single = next(iter(data.items()))
                    kv: Dict[str, Any] = {}
                    for pair in str(single).split(";"):
                        pair = pair.strip()
                        if pair and "=" in pair:
                            k, v = pair.split("=", 1)
                            kv[k] = v
                    data = kv
                else:
                    raise ValueError(f"Effect dict missing 'op': {eff}")
            spec = EFFECT_ACTIONS.get((op, current_scope)) or EFFECT_ACTIONS.get((op, Scope.WORLD))
            if not spec:
                raise ValueError(f"Unknown effect action '{op}' for scope {current_scope}")
            params = spec.params_model.model_validate(data)
            spec.fn(params, ctx, scopes_wrapped)


def apply_effect_block(
    block: G.EffectBlock,
    ctx: EvalContext,
    base_scopes: Dict[str, Any],
    current_scope: Scope,
) -> None:
    # Conditional branches first
    for _, entries in (block.IFS or {}).items():
        for entry in entries:
            if evaluate_condition_block(entry.condition, ctx, base_scopes):
                apply_effect_list(entry.effect, ctx, base_scopes, current_scope)
    # Always apply base effects
    if block.effect:
        apply_effect_list(block.effect, ctx, base_scopes, current_scope)
