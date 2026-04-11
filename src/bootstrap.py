from __future__ import annotations

from decimal import Decimal
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import random

import objects as G

# ---------------------------------------------------------------------------
# Footprint helpers
# ---------------------------------------------------------------------------

def _find_buildable_origin(
    world: G._WorldState,
    width: int,
    height: int,
    x_range: range,
    y_range: range,
) -> Optional[Tuple[int, int]]:
    """Return first (x, y) origin where a width x height footprint is entirely WILD_BUILDABLE."""
    for x in x_range:
        for y in y_range:
            if all(
                world.land.get(f"{x+dx}-{y+dy}", G._LandParcel(x=x+dx, y=y+dy, state_id=G.LandState.WILD_UNBUILDABLE)).state_id == G.LandState.WILD_BUILDABLE
                for dx in range(width)
                for dy in range(height)
            ):
                return (x, y)
    return None


def claim_building_footprint(world: G._WorldState, bdef: G.BuildingDefinition, origin: Tuple[int, int], owner_company_id: str, building_id: int) -> None:
    """Mark all tiles in the building footprint as BUILDING."""
    ox, oy = origin
    for dx in range(bdef.floor_width):
        for dy in range(bdef.floor_length):
            parcel = _get_or_create_parcel(world, (ox + dx, oy + dy))
            parcel.state_id = G.LandState.BUILDING
            parcel.owner_company_id = owner_company_id
            parcel.building_id = str(building_id)


def release_building_footprint(world: G._WorldState, building: G._BuildingInstance) -> None:
    """Release all footprint tiles back to WILD_BUILDABLE."""
    ox, oy = building.origin
    bdef = building.definition
    for dx in range(bdef.floor_width):
        for dy in range(bdef.floor_length):
            key = f"{ox+dx}-{oy+dy}"
            parcel = world.land.get(key)
            if parcel and parcel.building_id == str(building.instance_id):
                parcel.state_id = G.LandState.WILD_BUILDABLE
                parcel.owner_company_id = None
                parcel.building_id = None


def validate_footprint(world: G._WorldState, bdef: G.BuildingDefinition, origin: Tuple[int, int]) -> bool:
    """Return True if all footprint tiles are WILD_BUILDABLE."""
    ox, oy = origin
    return all(
        world.land.get(f"{ox+dx}-{oy+dy}", G._LandParcel(x=ox+dx, y=oy+dy, state_id=G.LandState.WILD_UNBUILDABLE)).state_id == G.LandState.WILD_BUILDABLE
        for dx in range(bdef.floor_width)
        for dy in range(bdef.floor_length)
    )

DEFAULT_MAP_WIDTH = 30
DEFAULT_MAP_HEIGHT = 20

# ---------------------------------------------------------------------------
# Content backstops
# ---------------------------------------------------------------------------

def ensure_content_baseline(
    resource_defs: Dict[str, G.Resource],
    demand_defs: Dict[str, G.ResourceDemand],
) -> None:
    """Patch in minimal resources/demands if the content pack omitted them."""
    if "money" not in resource_defs:
        resource_defs["money"] = G.Resource(
            id="money",
            display_name="Money",
            unit_name="dollar",
            cost_per=Decimal("1"),
            quality=Decimal("0"),
            type=G.ResourceType(id="currency"),
            market_behavior="currency",
            fulfills_demand={},
        )
    if "fuel" not in resource_defs:
        resource_defs["fuel"] = G.Resource(
            id="fuel",
            display_name="Fuel",
            unit_name="gallon",
            cost_per=Decimal("3"),
            quality=Decimal("0"),
            type=G.ResourceType(id="liquid"),
            market_behavior="commodity",
            fulfills_demand={},
        )
    if "food" not in resource_defs:
        resource_defs["food"] = G.Resource(
            id="food",
            display_name="Food",
            unit_name="meal",
            cost_per=Decimal("5"),
            quality=Decimal("1"),
            type=G.ResourceType(id="solid"),
            market_behavior="commodity",
            fulfills_demand={"hunger": 1.0},
        )
    if "investment" not in demand_defs:
        demand_defs["investment"] = G.ResourceDemand(
            id="investment",
            is_need=False,
            quantity=1,
            min_time_until_repeat=None,
            demand_can_occur=None,
            chance_per_tick=0.0,
        )
    if "shares" not in resource_defs:
        resource_defs["shares"] = G.Resource(
            id="shares",
            display_name="Shares",
            unit_name="share",
            cost_per=Decimal("1"),
            quality=Decimal("0"),
            type=G.ResourceType(id="security"),
            market_behavior="equity",
            fulfills_demand={"investment": 1},
        )

    if "crude_oil" not in resource_defs:
        resource_defs["crude_oil"] = G.Resource(
            id="crude_oil",
            display_name="Crude Oil",
            unit_name="barrel",
            cost_per=Decimal("2"),
            quality=Decimal("0"),
            type=G.ResourceType(id="liquid"),
            market_behavior="commodity",
            fulfills_demand={},
        )
    if "comfort" not in demand_defs:
        demand_defs["comfort"] = G.ResourceDemand(
            id="comfort",
            is_need=False,
            quantity=1,
            min_time_until_repeat=12,
            demand_can_occur=None,
            chance_per_tick=0.1,
        )

# ---------------------------------------------------------------------------
# Map helpers
# ---------------------------------------------------------------------------

def _parcel_key(pos: Tuple[int, int]) -> str:
    return f"{pos[0]}-{pos[1]}"


def _get_or_create_parcel(world: G._WorldState, pos: Tuple[int, int]) -> G._LandParcel:
    key = _parcel_key(pos)
    parcel = world.land.get(key)
    if parcel is None:
        parcel = G._LandParcel(x=pos[0], y=pos[1])
        world.land[key] = parcel
    return parcel


def _paint_parcels(world: G._WorldState, positions: Iterable[Tuple[int, int]], state: G.LandState) -> None:
    for pos in positions:
        parcel = _get_or_create_parcel(world, pos)
        if parcel.state_id != G.LandState.BUILDING:
            parcel.state_id = state


def ensure_land(world: G._WorldState, width: int = DEFAULT_MAP_WIDTH, height: int = DEFAULT_MAP_HEIGHT) -> None:
    """Create a simple town grid with perimeter wilds and a cross-road."""
    if not world.land:
        for x in range(width):
            for y in range(height):
                state = G.LandState.WILD_BUILDABLE
                if x in (0, width - 1) or y in (0, height - 1):
                    state = G.LandState.WILD_UNBUILDABLE
                parcel = G._LandParcel(x=x, y=y, state_id=state)
                world.land[parcel.id] = parcel
    else:
        width = max(p.x for p in world.land.values()) + 1
        height = max(p.y for p in world.land.values()) + 1

    main_h = height // 2
    main_v = width // 2
    _paint_parcels(world, ((x, main_h) for x in range(width)), G.LandState.ROAD)
    _paint_parcels(world, ((main_v, y) for y in range(height)), G.LandState.ROAD)
    # Spurs toward the industrial block
    spur_y = min(height - 2, main_h + 4)
    _paint_parcels(world, ((x, spur_y) for x in range(3, main_v + 5)), G.LandState.ROAD)

# ---------------------------------------------------------------------------
# Company bootstrap
# ---------------------------------------------------------------------------

CompanyCode = str


SCENARIO_COMPANIES = [
    {
        "code": "FARM",
        "resources": {
            "money": "28000",
            "fuel": "250",
            "shares": "1000000",
            "raw_food": "30"
        },
        "focus": ["raw_food"],
        "planning_horizon": 24 * 30,
        "personality": {
            "risk_tolerance": 0.35,
            "marketing_focus": 0.2,
            "price_competitiveness": 0.75,
            "patience": 120,
            "expansionism": 0.25,
            "integration_bias": 0.8,
            "logistics_focus": 0.6
        },
        "conversions": ["farm_raw_food"],
        "target_inventory": {"raw_food": "140", "fuel": "200"},
        "preferred_suppliers": {"fuel": ["ENERGY"]},
        "buildings": [
            {
                "kind": "Factory",
                "tile": (8, 2),
                "machines": [
                    {"machine_id": "farm", "recipe": "farm_raw_food", "active": True, "count": 6},
                    {"machine_id": "warehouse", "recipe": None, "active": False, "count": 1}
                ]
            }
        ],
        "vehicles": [
            {"vehicle_id": "truck", "tile": (8, 1), "fuel": "120"},
            {"vehicle_id": "truck", "tile": (9, 1), "fuel": "120"}
        ],
        "jobs": [
            {"role": "operator", "wage": "25", "count": 6}
        ]
    },
    {
        "code": "COOK",
        "resources": {
            "money": "22000",
            "fuel": "180",
            "shares": "1000000",
            "raw_food": "12"
        },
        "focus": ["food", "convenience_food"],
        "planning_horizon": 24 * 45,
        "personality": {
            "risk_tolerance": 0.55,
            "marketing_focus": 0.35,
            "price_competitiveness": 0.65,
            "patience": 144,
            "expansionism": 0.55,
            "integration_bias": 0.55,
            "logistics_focus": 0.7
        },
        "conversions": ["cook_food", "prepare_convenience_food"],
        "target_inventory": {"raw_food": "90", "food": "80", "fuel": "160", "convenience_food": "60"},
        "preferred_suppliers": {
            "raw_food": ["FARM"],
            "fuel": ["ENERGY"]
        },
        "buildings": [
            {
                "kind": "Factory",
                "tile": (16, 2),
                "machines": [
                    {"machine_id": "cooker", "recipe": "cook_food", "active": True, "count": 4},
                    {"machine_id": "kitchen", "recipe": "prepare_convenience_food", "active": True, "count": 2},
                    {"machine_id": "warehouse", "recipe": None, "active": False, "count": 2}
                ]
            }
        ],
        "vehicles": [
            {"vehicle_id": "truck", "tile": (16, 1), "fuel": "100"}
        ],
        "jobs": [
            {"role": "operator", "wage": "32", "count": 5}
        ]
    },
    {
        "code": "RETAIL",
        "resources": {
            "money": "16000",
            "fuel": "120",
            "shares": "1000000",
            "food": "40",
            "convenience_food": "30"
        },
        "focus": ["food", "convenience_food"],
        "planning_horizon": 24 * 21,
        "personality": {
            "risk_tolerance": 0.45,
            "marketing_focus": 0.7,
            "price_competitiveness": 0.4,
            "patience": 96,
            "expansionism": 0.4,
            "integration_bias": 0.2,
            "logistics_focus": 0.5
        },
        "conversions": [],
        "target_inventory": {"food": "90", "convenience_food": "80", "fuel": "120"},
        "preferred_suppliers": {
            "food": ["COOK"],
            "convenience_food": ["COOK"]
        },
        "buildings": [
            {
                "kind": "Retail",
                "tile": (21, 6),
                "machines": [
                    {"machine_id": "warehouse", "recipe": None, "active": False, "count": 3}
                ]
            }
        ],
        "vehicles": [
            {"vehicle_id": "truck", "tile": (25, 7), "fuel": "80"}
        ],
        "jobs": [
            {"role": "operator", "wage": "22", "count": 5}
        ]
    },
    {
        "code": "ENERGY",
        "resources": {
            "money": "26000",
            "fuel": "140",
            "shares": "1000000",
            "crude_oil": "60",
            "conveyor_segment": "50",
            "conveyor_segment_overhead": "20"
        },
        "focus": ["fuel", "crude_oil"],
        "planning_horizon": 24 * 40,
        "personality": {
            "risk_tolerance": 0.5,
            "marketing_focus": 0.25,
            "price_competitiveness": 0.6,
            "patience": 120,
            "expansionism": 0.45,
            "integration_bias": 0.9,
            "logistics_focus": 0.8
        },
        "conversions": ["pump_crude_oil", "refine_fuel"],
        "target_inventory": {"crude_oil": "120", "fuel": "220"},
        "preferred_suppliers": {},
        "buildings": [
            {
                "kind": "Factory",
                "tile": (2, 2),
                "machines": [
                    {"machine_id": "oil_well", "recipe": "pump_crude_oil", "active": True, "count": 2},
                    {"machine_id": "refinery", "recipe": "refine_fuel", "active": True, "count": 2},
                    {"machine_id": "warehouse", "recipe": None, "active": False, "count": 1}
                ]
            }
        ],
        "vehicles": [
            {"vehicle_id": "truck", "tile": (2, 1), "fuel": "120"},
            {"vehicle_id": "truck", "tile": (3, 1), "fuel": "120"}
        ],
        "jobs": [
            {"role": "operator", "wage": "28", "count": 4}
        ]
    }
]




def _find_building_def(defs: Sequence[G.BuildingDefinition], kind: str) -> G.BuildingDefinition:
    for definition in defs:
        if definition.building_type.name.lower() == kind.lower():
            return definition
    raise KeyError(f"No building definition for kind '{kind}'")


def assign_unit_tiles(
    building: G._BuildingInstance,
    unit: G._UnitInstance,
    origin_x: int,
    origin_y: int,
    width: int,
    height: int,
    floor: int = 0,
) -> None:
    """Assign a rectangular block of tiles to a unit, populating unit_tile_map."""
    tiles = []
    for dy in range(height):
        for dx in range(width):
            tiles.append((origin_x + dx, origin_y + dy, floor))
    unit.tiles = tiles
    unit.floor_space = len(tiles)
    for x, y, f in tiles:
        building.unit_tile_map[f"{x},{y},{f}"] = unit.instance_id


def _create_building(
    world: G._WorldState,
    company: G._CompanyInstance,
    building_defs: Sequence[G.BuildingDefinition],
    machine_defs: Dict[str, G.MachineDefinition],
    spec: Dict,
) -> G._BuildingInstance:
    bdef = _find_building_def(building_defs, spec["kind"])
    building = G._BuildingInstance(
        land_parcel_id=0,
        definition=bdef,
        owner_company_id=str(company.instance_id),
        units={},
        loading_bay_occupants=[],
        parking_occupants=[],
        origin=tuple(spec["tile"]),
    )
    unit = G._UnitInstance(owner_company_id=str(company.instance_id), floor_space=bdef.floor_area)
    building.add_unit(unit)
    assign_unit_tiles(building, unit, 0, 0, bdef.floor_width, bdef.floor_length)

    for machine_spec in spec.get("machines", []):
        machine_def = machine_defs.get(machine_spec["machine_id"])
        if machine_def is None:
            continue
        count = max(1, int(machine_spec.get("count", 1)))
        for _ in range(count):
            unit.add_machine(machine_def, active=machine_spec.get("active", False))
            machine_inst = unit.machines[-1]
            machine_inst.recipe = machine_spec.get("recipe")
            machine_inst.is_active = machine_spec.get("active", False)
            # Inline auto-placement: assign sequential positions at bootstrap time
            if machine_inst.grid_position is None:
                placed_tiles = sum(
                    m.machine.floor_space for m in unit.machines[:-1] if m.grid_position is not None
                )
                machine_inst.grid_position = (placed_tiles, 0, 0)
                for dx in range(machine_inst.machine.floor_space):
                    building.floor_grid[f"{placed_tiles + dx},0,0"] = f"machine:{machine_inst.instance_id}"

    if not validate_footprint(world, bdef, tuple(spec["tile"])):
        raise ValueError(f"Building footprint at {spec['tile']} overlaps occupied tiles")

    world.buildings[building.instance_id] = building
    company.buildings.append(building.instance_id)
    claim_building_footprint(world, bdef, tuple(spec["tile"]), str(company.instance_id), building.instance_id)
    return building


def _create_vehicle(
    world: G._WorldState,
    company: G._CompanyInstance,
    vehicle_defs: Dict[str, G.VehicleDefinition],
    spec: Dict,
) -> Optional[G._VehicleInstance]:
    vdef = vehicle_defs.get(spec["vehicle_id"])
    if vdef is None:
        return None
    vehicle = G._VehicleInstance(
        vehicle_type=vdef,
        owner_company_id=company.instance_id,
        position=tuple(spec["tile"]),
        destination=None,
        status="idle",
        inventory=G._Inventory(capacity=int(vdef.cargo_inventory_size)),
        fuel_stored=Decimal(str(spec.get("fuel", "0"))),
    )
    world.vehicles[vehicle.instance_id] = vehicle
    _get_or_create_parcel(world, vehicle.position)
    return vehicle


def ensure_companies(
    world: G._WorldState,
    registry: Dict[str, object],
) -> Dict[CompanyCode, G._CompanyInstance]:
    if world.companies:
        return {}

    building_defs: Sequence[G.BuildingDefinition] = registry.get("BuildingDefinition", [])  # type: ignore[assignment]
    machine_defs: Dict[str, G.MachineDefinition] = registry.get("MachineDefinition", {})  # type: ignore[assignment]
    vehicle_defs: Dict[str, G.VehicleDefinition] = registry.get("VehicleDefinition", {})  # type: ignore[assignment]

    created: Dict[CompanyCode, G._CompanyInstance] = {}
    supplier_code_targets: Dict[int, Dict[str, List[str]]] = {}
    for spec in SCENARIO_COMPANIES:
        resources = {rid: Decimal(str(amount)) for rid, amount in spec["resources"].items()}
        company = G._CompanyInstance(
            resources=resources,
            focus_resources=[rid for rid in spec.get("focus", [])],
            planning_horizon=int(spec.get("planning_horizon", 24 * 14)),
            personality=G.BusinessPersonality(**spec.get("personality", {})),
            active_conversions={cid: 0 for cid in spec.get("conversions", [])},
        )
        target_inventory = {rid: Decimal(str(val)) for rid, val in spec.get("target_inventory", {}).items()}
        if target_inventory:
            company.target_inventory.update(target_inventory)
            for rid in target_inventory:
                company.supplier_history.setdefault(rid, {})
        supplier_code_targets[company.instance_id] = {rid: list(codes) for rid, codes in spec.get("preferred_suppliers", {}).items()}
        world.companies[company.instance_id] = company
        created[spec["code"]] = company

        for building_spec in spec.get("buildings", []):
            _create_building(world, company, building_defs, machine_defs, building_spec)

        for vehicle_spec in spec.get("vehicles", []):
            _create_vehicle(world, company, vehicle_defs, vehicle_spec)

        for job_spec in spec.get("jobs", []):
            role_value = job_spec.get("role", "operator").lower()
            try:
                role = G.JobRole(role_value)
            except ValueError:
                role = G.JobRole.operator
            wage = Decimal(str(job_spec.get("wage", "10")))
            count = int(job_spec.get("count", 1))
            for _ in range(max(1, count)):
                world.post_job(company.instance_id, role, wage)

    all_company_ids = list(world.companies.keys())
    for company in world.companies.values():
        company.known_actors = [cid for cid in all_company_ids if cid != company.instance_id]

    # Resolve supplier preference codes into concrete company ids
    for comp_id, pref_map in supplier_code_targets.items():
        comp = world.companies.get(comp_id)
        if comp is None:
            continue
        resolved: Dict[str, List[int]] = {}
        for resource_id, codes in pref_map.items():
            resolved_ids = [created[code].instance_id for code in codes if code in created]
            if not resolved_ids:
                continue
            resolved[resource_id] = resolved_ids
            comp.supplier_history.setdefault(resource_id, {})
            for supplier_id in resolved_ids:
                if supplier_id not in comp.known_actors:
                    comp.known_actors.append(supplier_id)
                supplier = world.companies.get(supplier_id)
                if supplier and comp.instance_id not in supplier.known_actors:
                    supplier.known_actors.append(comp.instance_id)
        if resolved:
            comp.supplier_preferences.update(resolved)

    return created

# ---------------------------------------------------------------------------
# Population bootstrap
# ---------------------------------------------------------------------------

def _decimal_from_float(value: float, places: int = 2) -> Decimal:
    return Decimal(f"{value:.{places}f}")


def ensure_population(
    world: G._WorldState,
    resource_defs: Dict[str, G.Resource],
    companies: Dict[CompanyCode, G._CompanyInstance],
    rng: random.Random,
    size: int = 60,
) -> None:
    if world.population:
        return

    people: List[G._PersonInstance] = []
    company_ids = list(world.companies.keys())
    retail_company = companies.get("RETAIL")
    farm_company = companies.get("FARM")
    energy_company = companies.get("ENERGY")

    for _ in range(size):
        cash = _decimal_from_float(rng.uniform(350, 900))
        person = G._PersonInstance(
            age=rng.uniform(18.0, 62.0),
            resources={'money': cash},
            personal_relationship={},
            personality=G.Personality(
                comfort_value=round(rng.uniform(0.8, 1.3), 3),
                time_value=round(rng.uniform(0.6, 1.4), 3),
                info_sharing=round(rng.uniform(0.2, 0.8), 3),
                marketing_susceptibility=round(rng.uniform(0.1, 0.9), 3),
                risk_tolerance=round(rng.uniform(0.2, 0.9), 3),
                adaptability=round(rng.uniform(0.15, 0.95), 3),
            ),
        )
        # basic awareness
        if company_ids:
            known = rng.sample(company_ids, k=min(len(company_ids), rng.randint(1, 3)))
            person.known_actors.extend(known)
        # intrinsic preferences
        person.resource_preferences["food"] = rng.uniform(0.6, 1.0)
        if "convenience_food" in resource_defs:
            person.resource_preferences["convenience_food"] = rng.uniform(0.3, 1.0)
        if "fuel" in resource_defs:
            person.resource_preferences["fuel"] = rng.uniform(0.2, 0.8)
        if "raw_food" in resource_defs and rng.random() < 0.35:
            person.resource_preferences["raw_food"] = rng.uniform(0.1, 0.4)

        world.population[person.instance_id] = person
        people.append(person)

    # build relationship graph
    for person in people:
        friends = rng.sample([p for p in people if p is not person], k=rng.randint(5, 12))
        for other in friends:
            weight = rng.uniform(0.05, 0.4)
            person.personal_relationship[other.instance_id] = weight
            other.personal_relationship.setdefault(person.instance_id, weight)

    # seed brand loyalty and awareness biases
    for person in people:
        if retail_company:
            person.brand_loyalty[retail_company.instance_id] = rng.uniform(0.15, 0.55)
            if retail_company.instance_id not in person.known_actors:
                person.known_actors.append(retail_company.instance_id)
        if farm_company and rng.random() < 0.25:
            person.brand_loyalty[farm_company.instance_id] = rng.uniform(0.05, 0.2)
        if energy_company and rng.random() < 0.4:
            person.brand_loyalty[energy_company.instance_id] = rng.uniform(0.1, 0.35)
            if energy_company.instance_id not in person.known_actors:
                person.known_actors.append(energy_company.instance_id)
        # rotate awareness so every company has a customer base
        for company_id in company_ids:
            if company_id not in person.known_actors and rng.random() < 0.35:
                person.known_actors.append(company_id)


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def ensure_world_bootstrap(
    world: G._WorldState,
    registry: Dict[str, object],
    resource_defs: Dict[str, G.Resource],
    rng: Optional[random.Random] = None,
) -> None:
    rng = rng or random.Random()
    ensure_land(world)
    companies = ensure_companies(world, registry)
    ensure_population(world, resource_defs, companies, rng)
