from __future__ import annotations
from typing import Mapping, Optional
from decimal import Decimal
import objects as G


class WorldView:
    def __init__(self, world: G._WorldState):
        self._w = world

    def company_by_id(self, cid: int) -> Optional["CompanyView"]:
        comp = self._w.companies.get(cid)
        return CompanyView(comp) if comp else None

    def building_by_id(self, bid: int) -> Optional["BuildingView"]:
        b = self._w.buildings.get(bid)
        return BuildingView(b) if b else None

    def person_by_id(self, pid: int) -> Optional["PersonView"]:
        p = self._w.population.get(pid)
        return PersonView(p) if p else None

    @property
    def population_count(self) -> int:
        return len(self._w.population)

    @property
    def company_count(self) -> int:
        return len(self._w.companies)

    @property
    def building_count(self) -> int:
        return len(self._w.buildings)


class FinancialView:
    def __init__(self, ref: G._FinancialEntityInstance):
        self._ref = ref

    @property
    def resources(self) -> Mapping[str, Decimal]:
        return self._ref.resources


class PersonView(FinancialView):
    def __init__(self, p: G._PersonInstance):
        super().__init__(p)
        self._p = p

    @property
    def age(self) -> float:
        return self._p.age

    @property
    def money(self) -> float:
        return float(self._p.resources.get("money", Decimal(0)))

    @property
    def employer_id(self) -> Optional[int]:
        return self._p.employer_id

    @property
    def is_employed(self) -> bool:
        return self._p.employer_id is not None

    @property
    def has_home(self) -> bool:
        return self._p.home_unit_id is not None

    @property
    def education_id(self) -> Optional[str]:
        return self._p.education.id if self._p.education else None

    @property
    def current_role(self) -> Optional[str]:
        return self._p.current_role

    @property
    def in_debtors_prison(self) -> bool:
        return self._p.in_debtors_prison

    def has_resource(self, resource_id: str, at_least: float = 1.0) -> bool:
        return float(self._p.resources.get(resource_id, Decimal(0))) >= at_least

    def brand_loyalty(self, company_id: int) -> float:
        return self._p.brand_loyalty.get(company_id, 0.0)


class CompanyView(FinancialView):
    def __init__(self, c: G._CompanyInstance):
        super().__init__(c)
        self._c = c

    @property
    def money(self) -> float:
        return float(self._c.resources.get("money", Decimal(0)))

    @property
    def total_debt(self) -> float:
        return float(self._c.total_debt)

    @property
    def employee_count(self) -> int:
        return len(self._c.employees)

    @property
    def building_count(self) -> int:
        return len(self._c.buildings)

    @property
    def is_landlord(self) -> bool:
        return not self._c.active_conversions and bool(self._c.buildings)

    def has_resource(self, resource_id: str, at_least: float = 1.0) -> bool:
        return float(self._c.resources.get(resource_id, Decimal(0))) >= at_least

    def produces(self, resource_id: str) -> bool:
        from sim import ConvDefs
        return any(
            resource_id in conv.output_resources
            for cid in self._c.active_conversions
            if (conv := ConvDefs.get(cid))
        )


class BuildingView:
    def __init__(self, b: G._BuildingInstance):
        self._b = b

    @property
    def remaining_floor_space(self) -> int:
        return self._b.remaining_floor_space()

    @property
    def owner_company_id(self) -> str:
        return self._b.owner_company_id

    @property
    def condition(self) -> float:
        return self._b.condition

    @property
    def decay_state(self) -> str:
        return self._b.decay_state.value

    @property
    def is_residential(self) -> bool:
        return getattr(self._b.definition, 'residential', False)

    @property
    def is_public(self) -> bool:
        return self._b.definition.public_access

    @property
    def unit_count(self) -> int:
        return len(self._b.units)

    @property
    def has_tenants(self) -> bool:
        return any(
            u.access_team_ids and any(
                tid != str(u.owner_company_id) for tid in u.access_team_ids
            )
            for u in self._b.units.values()
        )


class MachineView:
    def __init__(self, m: G._MachineInstance):
        self._m = m

    @property
    def is_active(self) -> bool:
        return self._m.is_active

    @property
    def machine_id(self) -> str:
        return self._m.machine.id

    @property
    def recipe(self) -> Optional[str]:
        return self._m.recipe

    @property
    def floor_space(self) -> int:
        return self._m.machine.floor_space
