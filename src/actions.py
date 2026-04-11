"""
actions.py — Registers condition getters, scope selectors, and effect actions
via decorators from scripting.py.

Imported at sim.py startup to populate the registries.
DO NOT import sim.py here (circular). Import objects as G, import scripting.
"""
from __future__ import annotations
from decimal import Decimal
from typing import Optional, TYPE_CHECKING
from pydantic import BaseModel
import objects as G
from scripting import condition_getter, scope_selector, effect_action, Scope

if TYPE_CHECKING:
    from views import PersonView, CompanyView, BuildingView, WorldView


# ---------------------------------------------------------------------------
# Condition Getters — Person scope
# ---------------------------------------------------------------------------

@condition_getter("person.money", Scope.PERSON, as_property=True)
def getter_person_money(person: G._PersonInstance) -> float:
    return float(person.resources.get("money", Decimal(0)))


@condition_getter("person.age", Scope.PERSON, as_property=True)
def getter_person_age(person: G._PersonInstance) -> float:
    return person.age


@condition_getter("person.is_employed", Scope.PERSON, as_property=True)
def getter_person_employed(person: G._PersonInstance) -> bool:
    return person.employer_id is not None


@condition_getter("person.has_home", Scope.PERSON, as_property=True)
def getter_person_has_home(person: G._PersonInstance) -> bool:
    return person.home_unit_id is not None


@condition_getter("person.in_debtors_prison", Scope.PERSON, as_property=True)
def getter_person_debtors_prison(person: G._PersonInstance) -> bool:
    return person.in_debtors_prison


@condition_getter("person.has_resource", Scope.PERSON, as_property=False)
def getter_person_has_resource(person: G._PersonInstance, resource_id: str, at_least: float = 1.0) -> bool:
    return float(person.resources.get(resource_id, Decimal(0))) >= at_least


# ---------------------------------------------------------------------------
# Condition Getters — Company scope
# ---------------------------------------------------------------------------

@condition_getter("company.money", Scope.COMPANY, as_property=True)
def getter_company_money(company: G._CompanyInstance) -> float:
    return float(company.resources.get("money", Decimal(0)))


@condition_getter("company.has_resource", Scope.COMPANY, as_property=False)
def getter_company_has_resource(company: G._CompanyInstance, resource_id: str, at_least: float = 1.0) -> bool:
    return float(company.resources.get(resource_id, Decimal(0))) >= at_least


@condition_getter("company.employee_count", Scope.COMPANY, as_property=True)
def getter_company_employees(company: G._CompanyInstance) -> int:
    return len(company.employees)


@condition_getter("company.is_landlord", Scope.COMPANY, as_property=True)
def getter_company_is_landlord(company: G._CompanyInstance) -> bool:
    return not company.active_conversions and bool(company.buildings)


# ---------------------------------------------------------------------------
# Condition Getters — World scope
# ---------------------------------------------------------------------------

@condition_getter("world.population_count", Scope.WORLD, as_property=True)
def getter_world_population(world: G._WorldState) -> int:
    return len(world.population)


@condition_getter("world.company_count", Scope.WORLD, as_property=True)
def getter_world_companies(world: G._WorldState) -> int:
    return len(world.companies)


# ---------------------------------------------------------------------------
# Scope Selectors
# ---------------------------------------------------------------------------

@scope_selector("employer", Scope.PERSON, Scope.COMPANY)
def selector_employer(person: G._PersonInstance, world: G._WorldState) -> Optional[G._CompanyInstance]:
    if person.employer_id is None:
        return None
    return world.companies.get(person.employer_id)


@scope_selector("owner_company", Scope.BUILDING, Scope.COMPANY)
def selector_building_owner(building: G._BuildingInstance, world: G._WorldState) -> Optional[G._CompanyInstance]:
    return world.companies.get(int(building.owner_company_id))


# ---------------------------------------------------------------------------
# Effect Actions — Params models + implementations
# ---------------------------------------------------------------------------

class TransferMoneyParams(BaseModel):
    from_id: int
    to_id: int
    amount: float
    entity_type: str = "company"  # "company" or "person"


@effect_action("transfer_money", Scope.WORLD, TransferMoneyParams)
def action_transfer_money(params: TransferMoneyParams, ctx, scopes: dict) -> None:
    world: G._WorldState = ctx.world
    amount = Decimal(str(params.amount))
    if params.entity_type == "company":
        frm = world.companies.get(params.from_id)
        to = world.companies.get(params.to_id)
    else:
        frm = world.population.get(params.from_id)
        to = world.population.get(params.to_id)
    if not frm or not to:
        return
    available = frm.resources.get("money", Decimal(0))
    actual = min(amount, available)
    frm.resources["money"] = available - actual
    to.resources["money"] = to.resources.get("money", Decimal(0)) + actual


class AddResourceParams(BaseModel):
    entity_id: int
    resource_id: str
    quantity: float
    entity_type: str = "company"


@effect_action("add_resource", Scope.WORLD, AddResourceParams)
def action_add_resource(params: AddResourceParams, ctx, scopes: dict) -> None:
    world: G._WorldState = ctx.world
    qty = Decimal(str(params.quantity))
    if params.entity_type == "company":
        entity = world.companies.get(params.entity_id)
    else:
        entity = world.population.get(params.entity_id)
    if entity:
        entity.resources[params.resource_id] = entity.resources.get(params.resource_id, Decimal(0)) + qty


class SetFlagParams(BaseModel):
    mod_id: str
    key: str
    value: str  # JSON-serializable string value
    entity_type: str = "company"
    entity_id: int = 0


@effect_action("set_flag", Scope.WORLD, SetFlagParams)
def action_set_flag(params: SetFlagParams, ctx, scopes: dict) -> None:
    world: G._WorldState = ctx.world
    if params.entity_type == "company":
        entity = world.companies.get(params.entity_id)
    elif params.entity_type == "person":
        entity = world.population.get(params.entity_id)
    elif params.entity_type == "building":
        entity = world.buildings.get(params.entity_id)
    else:
        return
    if entity and hasattr(entity, "mod_state"):
        if params.mod_id not in entity.mod_state:
            entity.mod_state[params.mod_id] = {}
        entity.mod_state[params.mod_id][params.key] = params.value
