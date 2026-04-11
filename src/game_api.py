from __future__ import annotations
from decimal import Decimal
from typing import Optional
from pydantic import BaseModel, Field
import objects as G
from views import WorldView, PersonView, CompanyView, BuildingView
from scripting import condition_getter, scope_selector, effect_action, Scope


# ---------------- Getters -----------------
@condition_getter("person.money", Scope.PERSON, as_property=True)
def person_money(self: PersonView) -> Decimal:
    return self.resources.get("money", Decimal(0))


@condition_getter("company.money", Scope.COMPANY, as_property=True)
def company_money(self: CompanyView) -> Decimal:
    return self.resources.get("money", Decimal(0))


@condition_getter("entity.has_resource", Scope.PERSON, as_property=False)
def person_has_resource(self: PersonView, resource_id: str, at_least: Decimal) -> bool:
    return self.resources.get(resource_id, Decimal(0)) >= at_least


@condition_getter("entity.has_resource", Scope.COMPANY, as_property=False)
def company_has_resource(self: CompanyView, resource_id: str, at_least: Decimal) -> bool:
    return self.resources.get(resource_id, Decimal(0)) >= at_least


@condition_getter("building.free_space", Scope.BUILDING, as_property=True)
def building_free_space(self: BuildingView) -> int:
    return self.remaining_floor_space


# ---------------- Selectors -----------------
@scope_selector("company.of_person", Scope.PERSON, Scope.COMPANY)
def company_of_person(person: PersonView, world: WorldView) -> Optional[CompanyView]:
    if person.employer_id is None:
        return None
    return world.company_by_id(person.employer_id)


@scope_selector("building.by_id", Scope.WORLD, Scope.BUILDING)
def building_by_id(world: WorldView, id: int) -> Optional[BuildingView]:
    return world.building_by_id(id)


# ---------------- Effects -----------------
class TransferParams(BaseModel):
    from_scope: str = Field(description="scope var name (e.g., 'self' or 'company')")
    to_scope: str = Field(description="scope var name (e.g., 'self' or 'company')")
    amount: Decimal


@effect_action("transfer_money", Scope.WORLD, TransferParams)
def op_transfer_money(params: TransferParams, ctx, scopes):
    src = scopes.get(params.from_scope)
    dst = scopes.get(params.to_scope)
    if not src or not dst:
        raise ValueError("transfer_money: invalid from_scope/to_scope")
    src_ref = getattr(src, "_ref", None) or getattr(src, "_p", None) or getattr(src, "_c", None)
    dst_ref = getattr(dst, "_ref", None) or getattr(dst, "_p", None) or getattr(dst, "_c", None)
    if not src_ref or not dst_ref:
        raise ValueError("transfer_money: scopes must be financial entities")
    money = src_ref.resources.get("money", Decimal(0))
    amt = params.amount
    if money < amt:
        raise ValueError("transfer_money: insufficient funds")
    src_ref.resources["money"] = money - amt
    dst_ref.resources["money"] = dst_ref.resources.get("money", Decimal(0)) + amt


class AddResourceParams(BaseModel):
    to_scope: str
    resource_id: str
    quantity: Decimal


@effect_action("add_resource", Scope.WORLD, AddResourceParams)
def op_add_resource(params: AddResourceParams, ctx, scopes):
    tgt = scopes.get(params.to_scope)
    if not tgt:
        raise ValueError("add_resource: invalid to_scope")
    ref = getattr(tgt, "_ref", None) or getattr(tgt, "_p", None) or getattr(tgt, "_c", None)
    if not ref:
        raise ValueError("add_resource: scope must be financial entity")
    qty = ref.resources.get(params.resource_id, Decimal(0))
    ref.resources[params.resource_id] = qty + params.quantity


class PostJobParams(BaseModel):
    company_scope: str
    role: str
    wage: Decimal


@effect_action("post_job", Scope.WORLD, PostJobParams)
def op_post_job(params: PostJobParams, ctx, scopes):
    compv = scopes.get(params.company_scope)
    if not compv:
        raise ValueError("post_job: invalid company_scope")
    comp = getattr(compv, "_c", None)
    if not comp:
        raise ValueError("post_job: scope must be company")
    role = G.JobRole(params.role)
    ctx.world.post_job(comp.instance_id, role, params.wage)
