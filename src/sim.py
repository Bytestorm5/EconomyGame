import json
from pathlib import Path
from decimal import Decimal, ROUND_UP
from typing import Dict, Optional, List, Tuple, Literal
from math import ceil
import random

from pydantic import BaseModel, Field

from register import register_content, MOD_PATHS, LOCAL_CONTENT
import objects as G
from objects import get_instance_id
import bootstrap
import tick_time
# Ensure decorated API is loaded so registries are populated
import game_api  # noqa: F401
from dsl import EvalContext, evaluate_condition_block
import actions  # noqa: F401 — registers scripting decorators on import
# Constants
WORLD_STATE_PATH = Path("world_state.json")
ALL_SOURCES = [LOCAL_CONTENT] + MOD_PATHS

# Load registry
REGISTRY = register_content(ALL_SOURCES)
ResourceDefs: Dict[str, G.Resource] = REGISTRY.get("Resource", {}) # type: ignore
PresetDefs: Dict[str, G.ResourceMarketPreset] = REGISTRY.get("ResourceMarketPreset", {}) # type: ignore
ConvDefs: Dict[str, G.ResourceConversion] = REGISTRY.get("ResourceConversion", {}) # type: ignore
DemandDefs: Dict[str, G.ResourceDemand] = REGISTRY.get("ResourceDemand", {}) # type: ignore

bootstrap.ensure_content_baseline(ResourceDefs, DemandDefs)

# Building condition constants (1 tick = 1 game-minute)
BUILDING_BASE_RATE = 0.000002   # max condition change per tick (recover or decay)
# At m=1.0: recovers from 0 to 1 in ~500,000 ticks (~347 days)
# At m=0.0: decays from 1 to 0 in ~500,000 ticks (~347 days)
# Equilibrium at m=0.5

DECAY_STATE_DROP_CHANCE = 0.00005  # chance per tick of dropping a decay tier when condition < 0.1 and still declining
# Expected time at worst case: ~20,000 ticks (~13.9 days) of active neglect before tier drop

# Machine efficiency caps by decay state
DECAY_STATE_EFFICIENCY: Dict[str, float] = {
    "normal":      1.0,   # uses condition float directly for slowdown
    "fixer_upper": 0.7,   # max 70% efficiency even at condition=1.0
    "abandoned":   0.2,
    "condemned":   0.0,
    "collapsed":   0.0,
}

# Decay state progression order (for tier drop logic)
DECAY_STATE_ORDER = [
    G.DecayState.NORMAL,
    G.DecayState.FIXER_UPPER,
    G.DecayState.ABANDONED,
    G.DecayState.CONDEMNED,
    G.DecayState.COLLAPSED,
]


def _company_has_public_building(world: G._WorldState, company_id: int) -> bool:
    comp = world.companies.get(company_id)
    if not comp:
        return False
    for bid in comp.buildings:
        b = world.buildings.get(bid)
        if b and b.definition.public_access:
            return True
    return False


# ---------------------------------------------------------------------------
# Conveyor helpers
# ---------------------------------------------------------------------------

_FACE_OFFSETS = {
    "north": (0,  1),
    "south": (0, -1),
    "east":  (1,  0),
    "west":  (-1, 0),
}
_FACE_OPPOSITE = {"north": "south", "south": "north", "east": "west", "west": "east"}


def _port_world_pos(machine_grid_pos: Tuple[int,int,int], port: G.MachinePort) -> Tuple[int,int,int]:
    """Absolute position of a port's tile within building grid."""
    x, y, floor = machine_grid_pos
    return (x + port.local_x, y + port.local_y, floor)


def _port_entry_tile(machine_grid_pos: Tuple[int,int,int], port: G.MachinePort) -> Tuple[int,int,int]:
    """The tile immediately outside the port face — where a conveyor segment attaches."""
    px, py, floor = _port_world_pos(machine_grid_pos, port)
    dx, dy = _FACE_OFFSETS[port.face]
    return (px + dx, py + dy, floor)


def _assign_unit_tiles(
    building: G._BuildingInstance,
    unit: G._UnitInstance,
    origin_x: int,
    origin_y: int,
    width: int,
    height: int,
    floor: int = 0,
) -> None:
    """
    Assign a rectangular block of tiles to a unit.
    Marks each tile as owned by this unit in the building's unit_tile_map.
    """
    tiles = []
    for dy in range(height):
        for dx in range(width):
            tiles.append((origin_x + dx, origin_y + dy, floor))
    unit.tiles = tiles
    unit.floor_space = len(tiles)
    for x, y, f in tiles:
        building.unit_tile_map[f"{x},{y},{f}"] = unit.instance_id


def _auto_place_machine(building: G._BuildingInstance, unit: G._UnitInstance, machine: G._MachineInstance) -> None:
    """Find first free tile(s) within the unit's tile set and assign grid_position."""
    if machine.grid_position is not None:
        return  # already placed
    mw = machine.machine.floor_space  # width in tiles

    # If unit has explicit tiles, search within them
    if unit.tiles:
        unit_tiles_set = unit.tile_set()
        # Group tiles by floor then y, find contiguous runs along x
        from itertools import groupby
        floors = sorted({f for _, _, f in unit.tiles})
        for floor in floors:
            floor_tiles = sorted(
                [(x, y) for x, y, f in unit.tiles if f == floor],
                key=lambda p: (p[1], p[0])
            )
            by_row = {}
            for x, y in floor_tiles:
                by_row.setdefault(y, []).append(x)
            for y in sorted(by_row):
                xs = sorted(by_row[y])
                for i in range(len(xs) - mw + 1):
                    if xs[i:i+mw] == list(range(xs[i], xs[i] + mw)):
                        # Check all tiles are free
                        free = all(
                            f"{xs[i]+dx},{y},{floor}" not in building.floor_grid
                            for dx in range(mw)
                        )
                        if free:
                            machine.grid_position = (xs[i], y, floor)
                            for dx in range(mw):
                                building.floor_grid[f"{xs[i]+dx},{y},{floor}"] = f"machine:{machine.instance_id}"
                            return
    else:
        # Fallback: search entire building floor grid
        bdef = building.definition
        w, h = bdef.floor_width, bdef.floor_length
        floor = 0
        for y in range(h):
            for x in range(w - mw + 1):
                free = all(
                    f"{x+dx},{y},{floor}" not in building.floor_grid
                    for dx in range(mw)
                )
                if free:
                    machine.grid_position = (x, y, floor)
                    for dx in range(mw):
                        building.floor_grid[f"{x+dx},{y},{floor}"] = f"machine:{machine.instance_id}"
                    return
    # No space found — leave grid_position as None


def _plan_conveyor_path_legacy(
    world: G._WorldState,
    building: G._BuildingInstance,
    src_machine: G._MachineInstance,
    src_port: G.MachinePort,
    dst_machine: G._MachineInstance,
    dst_port: G.MachinePort,
    resource_id: str,
) -> Optional[List[Tuple[int,int,int,int]]]:
    """
    Plan a straight-line or L-shaped conveyor path from src_port to dst_port.
    Returns ordered list of (x, y, floor, level) tiles, or None if impossible.
    Level is 1 (ground) unless a tile is blocked, then tries level 3 (overhead).
    Legacy implementation kept as fallback.
    """
    if src_machine.grid_position is None or dst_machine.grid_position is None:
        return None

    start = _port_entry_tile(src_machine.grid_position, src_port)
    end   = _port_entry_tile(dst_machine.grid_position, dst_port)

    sx, sy, sf = start
    ex, ey, ef = end
    if sf != ef:
        return None  # cross-floor conveyors not supported yet

    floor = sf

    def tile_level(x, y):
        key1 = f"{x},{y},{floor}"
        key3 = f"{x},{y},3"
        if key1 not in building.floor_grid:
            return 1
        occupant = building.floor_grid[key1]
        if occupant.startswith("machine:"):
            if key3 not in building.floor_grid:
                return 3
            return None
        if key3 not in building.floor_grid:
            return 3
        return None

    def build_run(xs, ys):
        tiles = []
        for x, y in zip(xs, ys):
            lv = tile_level(x, y)
            if lv is None:
                return None
            tiles.append((x, y, floor, lv))
        return tiles

    if sx == ex:
        step = 1 if ey > sy else -1
        ys = list(range(sy, ey + step, step))
        xs = [sx] * len(ys)
        return build_run(xs, ys)

    if sy == ey:
        step = 1 if ex > sx else -1
        xs = list(range(sx, ex + step, step))
        ys = [sy] * len(xs)
        return build_run(xs, ys)

    def route1():
        step_x = 1 if ex > sx else -1
        step_y = 1 if ey > sy else -1
        xs1 = list(range(sx, ex + step_x, step_x))
        ys1 = [sy] * len(xs1)
        xs2 = [ex] * abs(ey - sy)
        ys2 = list(range(sy + step_y, ey + step_y, step_y))
        run1 = build_run(xs1, ys1)
        run2 = build_run(xs2, ys2)
        if run1 is None or run2 is None:
            return None
        return run1 + run2

    def route2():
        step_x = 1 if ex > sx else -1
        step_y = 1 if ey > sy else -1
        xs1 = [sx] * abs(ey - sy)
        ys1 = list(range(sy + step_y, ey + step_y, step_y))
        xs2 = list(range(sx + step_x, ex + step_x, step_x))
        ys2 = [ey] * len(xs2)
        run1 = build_run(xs1, ys1)
        run2 = build_run(xs2, ys2)
        if run1 is None or run2 is None:
            return None
        return run1 + run2

    r1 = route1()
    r2 = route2()
    if r1 is None and r2 is None:
        return None
    if r1 is None:
        return r2
    if r2 is None:
        return r1
    oh1 = sum(1 for t in r1 if t[3] == 3)
    oh2 = sum(1 for t in r2 if t[3] == 3)
    return r1 if oh1 <= oh2 else r2


def _astar_conveyor_path(
    world: G._WorldState,
    building: G._BuildingInstance,
    unit: G._UnitInstance,
    src_machine: G._MachineInstance,
    src_port: G.MachinePort,
    dst_machine: G._MachineInstance,
    dst_port: G.MachinePort,
    resource_id: str,
) -> Optional[List[Tuple[int, int, int, int]]]:
    """
    A* path from src_port exit tile to dst_port entry tile.
    Returns ordered (x, y, floor, level) list or None.
    Constrained to tiles in unit.tile_set().
    Overhead (level=3) allowed on any tile blocked at ground level by a machine.
    """
    if src_machine.grid_position is None or dst_machine.grid_position is None:
        return None

    start = _port_entry_tile(src_machine.grid_position, src_port)
    end   = _port_entry_tile(dst_machine.grid_position, dst_port)

    sx, sy, sf = start
    ex, ey, ef = end
    if sf != ef:
        return None  # cross-floor not supported

    floor = sf
    allowed = unit.tile_set() if unit.tiles else None

    def in_unit(x, y):
        if allowed is None:
            return True
        return (x, y, floor) in allowed

    def tile_level(x, y):
        key1 = f"{x},{y},{floor}"
        key3 = f"{x},{y},3"
        if key1 not in building.floor_grid:
            return 1
        occupant = building.floor_grid[key1]
        if occupant.startswith("machine:"):
            if key3 not in building.floor_grid:
                return 3
            return None
        if key3 not in building.floor_grid:
            return 3
        return None

    if not in_unit(sx, sy) or not in_unit(ex, ey):
        return None

    import heapq
    def heuristic(x, y):
        return abs(x - ex) + abs(y - ey)

    # state: (cost, x, y)
    open_heap = [(heuristic(sx, sy), 0, sx, sy)]
    came_from: dict = {}
    g_score: dict = {(sx, sy): 0}

    while open_heap:
        f, g, cx, cy = heapq.heappop(open_heap)
        if (cx, cy) == (ex, ey):
            # Reconstruct path
            path = []
            cur = (cx, cy)
            while cur in came_from:
                x, y = cur
                lv = tile_level(x, y)
                path.append((x, y, floor, lv if lv is not None else 3))
                cur = came_from[cur]
            # Add start
            x, y = cur
            lv = tile_level(x, y)
            path.append((x, y, floor, lv if lv is not None else 3))
            path.reverse()
            return path

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = cx + dx, cy + dy
            if not in_unit(nx, ny):
                continue
            lv = tile_level(nx, ny)
            if lv is None:
                continue  # fully blocked
            ng = g + 1
            if (nx, ny) not in g_score or ng < g_score[(nx, ny)]:
                g_score[(nx, ny)] = ng
                came_from[(nx, ny)] = (cx, cy)
                heapq.heappush(open_heap, (ng + heuristic(nx, ny), ng, nx, ny))

    return None


def _plan_conveyor_path(
    world: G._WorldState,
    building: G._BuildingInstance,
    src_machine: G._MachineInstance,
    src_port: G.MachinePort,
    dst_machine: G._MachineInstance,
    dst_port: G.MachinePort,
    resource_id: str,
) -> Optional[List[Tuple[int, int, int, int]]]:
    """Wrapper: try A* if unit context is available, else fall back to legacy."""
    # Find the unit containing src_machine
    unit = None
    for u in building.units.values():
        for m in u.machines:
            if m.instance_id == src_machine.instance_id:
                unit = u
                break
        if unit is not None:
            break

    if unit is not None and unit.tiles:
        result = _astar_conveyor_path(world, building, unit, src_machine, src_port, dst_machine, dst_port, resource_id)
        if result is not None:
            return result
    return _plan_conveyor_path_legacy(world, building, src_machine, src_port, dst_machine, dst_port, resource_id)


def _place_conveyor_path(
    world: G._WorldState,
    building: G._BuildingInstance,
    comp: G._CompanyInstance,
    src_machine: G._MachineInstance,
    src_port: G.MachinePort,
    dst_machine: G._MachineInstance,
    dst_port: G.MachinePort,
    resource_id: str,
    tiles: List[Tuple[int,int,int,int]],
) -> Optional[G._ConveyorPath]:
    """Consume segments from company inventory and instantiate the path."""
    ground_count   = sum(1 for t in tiles if t[3] < 3)
    overhead_count = sum(1 for t in tiles if t[3] >= 3)

    have_ground   = comp.resources.get('conveyor_segment', Decimal(0))
    have_overhead = comp.resources.get('conveyor_segment_overhead', Decimal(0))
    if have_ground < ground_count or have_overhead < overhead_count:
        return None

    comp.resources['conveyor_segment'] = have_ground - Decimal(ground_count)
    comp.resources['conveyor_segment_overhead'] = have_overhead - Decimal(overhead_count)

    segments = []
    for i, (x, y, floor, level) in enumerate(tiles):
        if i == 0:
            dx, dy = _FACE_OFFSETS[src_port.face]
            dir_in = (dx, dy)
        else:
            px, py = tiles[i-1][0], tiles[i-1][1]
            dir_in = (x - px, y - py)

        if i < len(tiles) - 1:
            nx, ny = tiles[i+1][0], tiles[i+1][1]
            dir_out = (nx - x, ny - y)
        else:
            dx, dy = _FACE_OFFSETS[_FACE_OPPOSITE[dst_port.face]]
            dir_out = (dx, dy)

        seg = G._ConveyorSegment(
            building_id=building.instance_id,
            position=(x, y, floor),
            level=level,
            direction_in=dir_in,
            direction_out=dir_out,
        )
        world.conveyor_segments[seg.instance_id] = seg
        segments.append(seg.instance_id)

        grid_key = f"{x},{y},{level if level >= 3 else floor}"
        building.floor_grid[grid_key] = f"conveyor:{seg.instance_id}"

    path = G._ConveyorPath(
        building_id=building.instance_id,
        source_machine_id=src_machine.instance_id,
        source_port_id=src_port.port_id,
        dest_machine_id=dst_machine.instance_id,
        dest_port_id=dst_port.port_id,
        resource_id=resource_id,
        segment_ids=segments,
        active=False,
    )
    world.conveyor_paths[path.instance_id] = path
    _revalidate_path(world, path, src_machine, dst_machine)
    return path


def _revalidate_path(
    world: G._WorldState,
    path: G._ConveyorPath,
    src_machine: G._MachineInstance,
    dst_machine: G._MachineInstance,
) -> None:
    """Set path.active based on port uniqueness and recipe compatibility."""
    src_out_paths = [
        p for p in world.conveyor_paths.values()
        if p.source_machine_id == path.source_machine_id
        and p.source_port_id == path.source_port_id
    ]
    dst_in_paths = [
        p for p in world.conveyor_paths.values()
        if p.dest_machine_id == path.dest_machine_id
        and p.dest_port_id == path.dest_port_id
        and p.resource_id == path.resource_id
    ]
    if len(src_out_paths) != 1 or len(dst_in_paths) != 1:
        path.active = False
        return

    src_outputs = {}
    if src_machine.recipe:
        conv = ConvDefs.get(src_machine.recipe)
        if conv:
            src_outputs = dict(conv.output_resources)

    # Machines with no possible_recipes (e.g. warehouse) accept any resource via conveyor
    dst_is_generic = not dst_machine.machine.possible_recipes
    dst_inputs = {}
    if dst_machine.recipe:
        conv = ConvDefs.get(dst_machine.recipe)
        if conv:
            dst_inputs = dict(conv.input_resources)

    resource_ok = (
        path.resource_id in src_outputs
        and (dst_is_generic or path.resource_id in dst_inputs)
    )

    path.active = (
        resource_ok
        and src_machine.is_active
        and (dst_is_generic or dst_machine.is_active)
    )


def _find_machine_by_id(world: G._WorldState, machine_id: int) -> Optional[G._MachineInstance]:
    for building in world.buildings.values():
        for unit in building.units.values():
            for m in unit.machines:
                if m.instance_id == machine_id:
                    return m
    return None


def _auto_connect_conveyors(
    world: G._WorldState,
    building: G._BuildingInstance,
    new_machine: G._MachineInstance,
    comp: G._CompanyInstance,
) -> None:
    """
    After placing a machine, attempt to connect it to compatible machines in the same building
    via conveyor paths. Updates comp.pending_conveyor_plans if segments are unavailable.
    """
    if new_machine.grid_position is None:
        return
    if not new_machine.machine.output_ports and not new_machine.machine.input_ports:
        return

    all_machines = [
        m for u in building.units.values() for m in u.machines
        if m.instance_id != new_machine.instance_id and m.grid_position is not None
    ]

    def try_connect(src_m, src_port, dst_m, dst_port, res_id):
        exists = any(
            p.source_machine_id == src_m.instance_id
            and p.source_port_id == src_port.port_id
            and p.dest_machine_id == dst_m.instance_id
            and p.dest_port_id == dst_port.port_id
            and p.resource_id == res_id
            for p in world.conveyor_paths.values()
        )
        if exists:
            return

        tiles = _plan_conveyor_path(world, building, src_m, src_port, dst_m, dst_port, res_id)
        if tiles is None:
            return

        ground_needed   = sum(1 for t in tiles if t[3] < 3)
        overhead_needed = sum(1 for t in tiles if t[3] >= 3)

        have_g = comp.resources.get('conveyor_segment', Decimal(0))
        have_o = comp.resources.get('conveyor_segment_overhead', Decimal(0))

        if have_g >= ground_needed and have_o >= overhead_needed:
            _place_conveyor_path(world, building, comp, src_m, src_port, dst_m, dst_port, res_id, tiles)
        else:
            comp.target_inventory['conveyor_segment'] = (
                comp.target_inventory.get('conveyor_segment', Decimal(0)) + Decimal(ground_needed)
            )
            comp.target_inventory['conveyor_segment_overhead'] = (
                comp.target_inventory.get('conveyor_segment_overhead', Decimal(0)) + Decimal(overhead_needed)
            )
            comp.pending_conveyor_plans.append({
                'building_id':    building.instance_id,
                'src_machine_id': src_m.instance_id,
                'src_port_id':    src_port.port_id,
                'dst_machine_id': dst_m.instance_id,
                'dst_port_id':    dst_port.port_id,
                'resource_id':    res_id,
            })

    for src_port in new_machine.machine.output_ports:
        for dst_m in all_machines:
            for dst_port in dst_m.machine.input_ports:
                src_rids = src_port.resource_ids or []
                dst_rids = dst_port.resource_ids or []
                compatible = bool(
                    set(src_rids) & set(dst_rids)
                    or (not src_rids and not dst_rids)
                    or not src_rids
                    or not dst_rids
                )
                if not compatible:
                    continue
                for res_id in (set(src_rids) & set(dst_rids) or set(src_rids) or set(dst_rids) or {''}):
                    if res_id:
                        try_connect(new_machine, src_port, dst_m, dst_port, res_id)

    for dst_port in new_machine.machine.input_ports:
        for src_m in all_machines:
            for src_port in src_m.machine.output_ports:
                src_rids = src_port.resource_ids or []
                dst_rids = dst_port.resource_ids or []
                compatible = bool(
                    set(src_rids) & set(dst_rids)
                    or (not src_rids and not dst_rids)
                    or not src_rids
                    or not dst_rids
                )
                if not compatible:
                    continue
                for res_id in (set(src_rids) & set(dst_rids) or set(src_rids) or set(dst_rids) or {''}):
                    if res_id:
                        try_connect(src_m, src_port, new_machine, dst_port, res_id)


def _update_fair_value(agent: G._FinancialEntityInstance, resource_id: str, market_price: Decimal,
                       rng: random.Random, lr: float, noise_scale: float) -> Decimal:
    """
    Evolve the agent's fair-value estimate toward the observed price,
    with a little zero-mean noise (models imperfect information).
    """
    current = agent.fair_values.get(resource_id, market_price)
    if isinstance(agent, G._PersonInstance):
        comfort_value = ResourceDefs[resource_id].quality * Decimal(agent.personality.comfort_value)
        current = min(comfort_value, current) # Product is worth at least this much to this person
    error   = market_price - current
    noise   = Decimal(str(rng.uniform(-noise_scale, noise_scale))) * market_price
    new_val = current + Decimal(str(lr)) * error + noise        # simple exponential learning
    agent.fair_values[resource_id] = max(Decimal("0.01"), new_val)
    return new_val

def evaluate_condition(
    cond: Optional[G.ConditionBlock],
    world: G._WorldState,
    person: Optional[G._PersonInstance] = None,
    extra_scopes: Optional[Dict[str, object]] = None,
) -> bool:
    ctx = EvalContext(world=world)
    base_scopes: Dict[str, object] = {}
    if person is not None:
        base_scopes["self"] = person
    if extra_scopes:
        base_scopes.update(extra_scopes)
    return evaluate_condition_block(cond, ctx, base_scopes)

# -----------------------------------
# Persistence
# -----------------------------------

def load_world() -> G._WorldState:
    if WORLD_STATE_PATH.exists():
        raw = WORLD_STATE_PATH.read_text(encoding="utf-8")
        world = G._WorldState.model_validate_json(raw)
        world.registry = REGISTRY # type: ignore
        return world
    w = G._WorldState()
    w.registry = REGISTRY # type: ignore
    return w


def save_world(world: G._WorldState) -> None:
    data = world.model_dump()
    data.pop("registry", None)
    WORLD_STATE_PATH.write_text(G._WorldState(**data).model_dump_json(), encoding="utf-8")

# -----------------------------------
# Order Book
# -----------------------------------

class Order(BaseModel):
    order_id: int = Field(default_factory=get_instance_id)
    actor_type: Literal["person", "company"]
    actor_id: int
    resource_id: str
    quantity: Decimal
    limit_price: Decimal
    side: Literal["buy", "sell"]
    timestamp: int


class OrderBook:
    def __init__(self):
        self.orders: List[Order] = []

    def submit(self, order: Order) -> None:
        self.orders.append(order)

class MarketingOrder(BaseModel):
    order_id: int = Field(default_factory=get_instance_id)
    company_id: int
    quantity: int
    limit_price: Decimal
    timestamp: int

class MarketingMarket:
    def __init__(self, world: G._WorldState, base_price: Decimal = Decimal("1")):
        self.world = world
        self.price: Decimal = base_price
        self.order_book: List[MarketingOrder] = []
        self.last_update: int = 0

    def submit(self, order: MarketingOrder) -> None:
        self.order_book.append(order)

    def exec_orders(self, world: G._WorldState, tick: int, rng: random.Random) -> None:
        buys = [order for order in self.order_book if order.timestamp == tick]
        total_qty = sum(order.quantity for order in buys)
        if total_qty == 0:
            return
        self.price = self.price * (Decimal("1") + Decimal(str(total_qty)) / Decimal("100"))
        population = list(world.population.values())
        if not population:
            return
        for order in buys:
            for _ in range(order.quantity):
                person = rng.choice(population)
                self._apply_impression(person, order.company_id, rng)
        self.order_book = [order for order in self.order_book if order.timestamp > tick]

    def update_prices(self, world: G._WorldState, tick: int, rng: random.Random) -> None:
        self.exec_orders(world, tick, rng)

    def _apply_impression(self, person: G._PersonInstance, company_id: int, rng: random.Random) -> None:
        susceptibility = person.personality.marketing_susceptibility
        loyalty_gain = 0.04 * susceptibility
        if company_id not in person.known_actors:
            person.known_actors.append(company_id)
        person.brand_loyalty[company_id] = min(1.0, person.brand_loyalty.get(company_id, 0.0) + loyalty_gain)
        for other_id, weight in person.personal_relationship.items():
            if rng.random() < weight * susceptibility:
                other = self.world.population.get(other_id)
                if not other:
                    continue
                if company_id not in other.known_actors:
                    other.known_actors.append(company_id)
                other.brand_loyalty[company_id] = min(1.0, other.brand_loyalty.get(company_id, 0.0) + loyalty_gain * 0.5)



class Market:
    def __init__(self, world: G._WorldState, resource_id: str, adjustment_rate: Decimal = Decimal("0.05")):
        self.world = world
        self.resource_id = resource_id
        self.adjustment_rate = adjustment_rate
        res_def = ResourceDefs[resource_id]
        self.price: Decimal = res_def.cost_per
        self.preset: Optional[G.ResourceMarketPreset] = PresetDefs.get(res_def.market_behavior)
        self.order_book = OrderBook()
        self.last_update: int = 0
        self.prev_price: Decimal = self.price
        self.markets: Dict[str, "Market"] = {}
        self.volume_weighted_price: Decimal = self.price

    def compute_supply_demand(self, tick: int) -> Tuple[Decimal, Decimal]:
        supply = sum((o.quantity for o in self.order_book.orders if o.side == "sell" and o.timestamp == tick), Decimal(0))
        demand = sum((o.quantity for o in self.order_book.orders if o.side == "buy" and o.timestamp == tick), Decimal(0))
        return Decimal(supply), Decimal(demand)

    def exec_orders(self, world: G._WorldState, tick: int) -> List[Tuple[Order, Order, Decimal, Decimal]]:
        buys = sorted(
            (o for o in self.order_book.orders if o.side == "buy" and o.timestamp == tick),
            key=lambda o: (-o.limit_price, o.timestamp)
        )
        all_sells = [o for o in self.order_book.orders if o.side == "sell" and o.timestamp == tick]
        trades: List[Tuple[Order, Order, Decimal, Decimal]] = []
        for buy in buys:
            if buy.actor_type == "company" and buy.actor_id in world.companies:
                known_actors = world.companies[buy.actor_id].known_actors
            elif buy.actor_type == "person" and buy.actor_id in world.population:
                known_actors = world.population[buy.actor_id].known_actors
            else:
                known_actors = []

            if buy.actor_type == "person":
                buyer = world.population.get(buy.actor_id)
                sells = sorted(
                    [s for s in all_sells
                     if s.actor_id in known_actors
                     and _company_has_public_building(world, s.actor_id)],
                    key=lambda s: (
                        self.total_unit_cost_for_match(buy, s)
                        * (1 - buyer.brand_loyalty.get(s.actor_id, 0.0))
                        # Discount for sellers whose product is explicitly known
                        * (0.9 if self.resource_id in buyer.known_products.get(s.actor_id, []) else 1.0)
                        if buyer else self.total_unit_cost_for_match(buy, s),
                        s.timestamp,
                    )
                )
            else:
                sells = sorted(
                    [s for s in all_sells if s.actor_id in known_actors],
                    key=lambda s: (self.total_unit_cost_for_match(buy, s), s.timestamp)
                )
            for sell in sells:
                if buy.resource_id != sell.resource_id or buy.quantity <= 0 or sell.quantity <= 0:
                    continue
                if buy.limit_price >= sell.limit_price:
                    price = (buy.limit_price + sell.limit_price) / Decimal(2)
                    trade_qty = min(buy.quantity, sell.quantity)
                    trades.append((buy, sell, price, trade_qty))
                    buy.quantity -= trade_qty
                    sell.quantity -= trade_qty

                    buyer_company = world.companies.get(buy.actor_id) if buy.actor_type == "company" else None
                    seller_company = world.companies.get(sell.actor_id) if sell.actor_type == "company" else None
                    buyer_person = world.population.get(buy.actor_id) if buy.actor_type == "person" else None
                    seller_person = world.population.get(sell.actor_id) if sell.actor_type == "person" else None

                    total_cost = price * trade_qty

                    if buyer_company:
                        buyer_company.resources['money'] = buyer_company.resources.get('money', Decimal(0)) - total_cost
                    elif buyer_person:
                        buyer_person.resources['money'] = buyer_person.resources.get('money', Decimal(0)) - total_cost

                    if seller_company:
                        seller_company.resources['money'] = seller_company.resources.get('money', Decimal(0)) + total_cost
                        seller_company.resources[sell.resource_id] = seller_company.resources.get(sell.resource_id, Decimal(0)) - trade_qty
                    elif seller_person:
                        seller_person.resources[sell.resource_id] = seller_person.resources.get(sell.resource_id, Decimal(0)) - trade_qty
                        seller_person.resources['money'] = seller_person.resources.get('money', Decimal(0)) + total_cost

                    if buyer_company and seller_company:
                        self._queue_trade_delivery(world, buyer_company, seller_company, buy.resource_id, trade_qty, price, tick)
                    elif buyer_company:
                        buyer_company.resources[buy.resource_id] = buyer_company.resources.get(buy.resource_id, Decimal(0)) + trade_qty
                        buyer_company.resource_acquired_tick[buy.resource_id] = tick
                    elif buyer_person:
                        buyer_person.resources[buy.resource_id] = buyer_person.resources.get(buy.resource_id, Decimal(0)) + trade_qty

                    buyer_entity = buyer_company or buyer_person
                    seller_entity = seller_company or seller_person
                    if buyer_entity and sell.actor_id not in buyer_entity.known_actors:
                        buyer_entity.known_actors.append(sell.actor_id)
                    if seller_entity and buy.actor_id not in seller_entity.known_actors:
                        seller_entity.known_actors.append(buy.actor_id)

        self.order_book.orders = [o for o in self.order_book.orders if o.quantity > 0]
        return trades

    def _company_origin(self, world: G._WorldState, company_id: int) -> Tuple[int, int]:
        comp = world.companies.get(company_id)
        if not comp or not comp.buildings:
            return (0, 0)
        building = world.buildings.get(comp.buildings[0])
        if building and building.origin:
            return building.origin
        return (0, 0)

    def _queue_trade_delivery(self, world: G._WorldState, buyer: G._CompanyInstance, seller: G._CompanyInstance,
                              resource_id: str, quantity: Decimal, price: Decimal, tick: int) -> None:
        qty_int = int(quantity)
        if qty_int <= 0:
            return
        buyer.inbound_pending[resource_id] = buyer.inbound_pending.get(resource_id, Decimal(0)) + quantity
        seller.outbound_pending[resource_id] = seller.outbound_pending.get(resource_id, Decimal(0)) + quantity

        source_pos = self._company_origin(world, seller.instance_id)
        dest_pos = self._company_origin(world, buyer.instance_id)
        max_batch = max(10, min(50, qty_int))
        remaining = qty_int
        while remaining > 0:
            batch = min(remaining, max_batch)
            task = {
                'id': get_instance_id(),
                'task_kind': 'trade',
                'resource': resource_id,
                'quantity': batch,
                'remaining': batch,
                'payload': 0,
                'phase': 'pending',
                'source': {'type': 'company', 'company_id': seller.instance_id, 'pos': source_pos},
                'destination': {'type': 'company', 'company_id': buyer.instance_id, 'pos': dest_pos},
                'destination_key': ('company', buyer.instance_id, resource_id),
                'price': float(price),
                'seller_id': seller.instance_id,
                'buyer_id': buyer.instance_id,
                'created_tick': tick
            }
            buyer.logistics_queue.append(task)
            remaining -= batch

    def update_prices(self, world: G._WorldState, tick: int, rng: random.Random) -> None:
        if not self.preset or tick - self.last_update < self.preset.price_stickiness:
            self.prev_price = self.price
            return

        supply, demand = self.compute_supply_demand(tick)
        old_price = self.price

        if supply == 0 and demand > 0:
            best_bid = max(
                (o.limit_price for o in self.order_book.orders
                if o.side == "buy" and o.timestamp == tick),
                default=None
            )
            if best_bid is not None:
                self.price += (best_bid - old_price) * self.adjustment_rate
                self.last_update = tick
            self.prev_price = old_price
            return

        if demand == 0 and supply > 0:
            best_ask = min(
                (o.limit_price for o in self.order_book.orders
                if o.side == "sell" and o.timestamp == tick),
                default=None
            )
            if best_ask is not None:
                self.price += (best_ask - old_price) * self.adjustment_rate
                self.last_update = tick
            self.prev_price = old_price
            return
        if supply == 0 and demand == 0:
            self.price = self.prev_price
            return

        trades = self.exec_orders(world, tick)

        if trades:
            tot_qty = sum(qty for *_, qty in trades)
            wtd_sum = sum(price * qty for *_, price, qty in trades)
            tick_vwap = wtd_sum / tot_qty

            alpha = Decimal("0.25")
            self.volume_weighted_price = ((1 - alpha) * self.volume_weighted_price) + (alpha * tick_vwap)
        else:
            self.volume_weighted_price = (self.volume_weighted_price + self.price) / Decimal(2)

        self.prev_price = self.price
        if trades:
            self.price = trades[-1][2]

    def estimate_shipping_cost(self, company_id: int, quantity: Decimal, supplier_id: int) -> Decimal:
        res_def = ResourceDefs.get(self.resource_id)
        if not res_def or not getattr(res_def, 'is_physical', False):
            return Decimal("0")
        vehicles = [
            v
            for v in self.world.vehicles.values()
            if v.owner_company_id == company_id and v.status in {"idle", "parked"}
        ]
        if not vehicles:
            return Decimal("0")
        fuel_market = self.markets.get('fuel')
        fuel_price = fuel_market.price if fuel_market else Decimal("1")
        parcels = [p for p in self.world.land.values() if p.building_id == str(supplier_id)]
        if not parcels:
            return Decimal("0")
        sbx, sby = parcels[0].x, parcels[0].y
        vehicle_info: List[Tuple[G._VehicleInstance, int, Decimal]] = []
        for v in vehicles:
            cap = int(v.vehicle_type.cargo_inventory_size)
            if cap <= 0:
                continue
            vx, vy = v.position
            dist = Decimal(abs(sbx - vx) + abs(sby - vy))
            vehicle_info.append((v, cap, dist))
        if not vehicle_info:
            return Decimal("0")
        vehicle_info.sort(key=lambda x: x[2])
        remaining = quantity
        total_cost = Decimal(0)
        for v, cap, dist in vehicle_info:
            if remaining <= 0:
                break
            take = min(remaining, cap)
            trips = ceil(float(take / cap)) if cap > 0 else 0
            consume = v.vehicle_type.fuel_consumption
            total_cost += dist * Decimal(2) * consume * fuel_price * trips
            remaining -= take
        if remaining > 0 and vehicle_info:
            v, cap, dist = vehicle_info[0]
            trips = ceil(float(remaining / cap)) if cap > 0 else 0
            consume = v.vehicle_type.fuel_consumption
            total_cost += dist * Decimal(2) * consume * fuel_price * trips
        return total_cost / quantity if quantity > 0 else Decimal(0)

    def get_effective_price(self, order: Order, supplier_id: int) -> Decimal:
        base_price = order.limit_price
        shipping = self.estimate_shipping_cost(order.actor_id, order.quantity, supplier_id)
        return base_price + shipping

    def total_unit_cost_for_match(self, buy: Order, sell: Order) -> Decimal:
        shipping = self.estimate_shipping_cost(buy.actor_id, min(buy.quantity, sell.quantity), sell.actor_id)
        return sell.limit_price + shipping

# (min_coverage_ratio, option_A, option_B, option_C)
# Each option is (down_pct, term_months)
LOAN_TIERS = [
    # Tier 1: coverage >= 2.0x
    (2.0,  [(0.05, 84), (0.15, 60), (0.30, 36)]),
    # Tier 2: coverage >= 1.0x
    (1.0,  [(0.10, 72), (0.20, 48), (0.35, 30)]),
    # Tier 3: coverage >= 0.5x
    (0.5,  [(0.15, 60), (0.30, 36), (0.45, 24)]),
    # Tier 4: coverage >= 0.25x
    (0.25, [(0.25, 48), (0.40, 30), (0.55, 18)]),
    # Tier 5: coverage >= 0.1x
    (0.1,  [(0.35, 36), (0.50, 24), (0.65, 12)]),
    # Below 0.1x: refused
]


class CentralBank:
    """
    Stateless utility: assesses and issues loans, provides repayment schedules.
    Not modeled as an agent — no balance sheet, infinite liquidity.
    """

    @staticmethod
    def _net_worth(entity, world: G._WorldState, markets: Dict[str, Market]) -> Decimal:
        """Compute assets - liabilities for a FinancialEntityInstance."""
        # Cash + inventory at market value
        assets = Decimal('0')
        for rid, qty in entity.resources.items():
            if rid == 'money':
                assets += qty
            elif rid in markets:
                assets += qty * markets[rid].price
            # shares valued at market price
        # Buildings: use building cost as proxy for now
        for bid in entity.buildings:
            b = world.buildings.get(bid)
            if b:
                # Condition and decay-state adjust value
                state_factor = {
                    'normal': 1.0, 'fixer_upper': 0.6,
                    'abandoned': 0.3, 'condemned': 0.1, 'collapsed': 0.0
                }.get(b.decay_state.value, 1.0)
                assets += b.definition.cost * Decimal(str(state_factor)) * Decimal(str(b.condition))
        # Liabilities: total_debt
        liabilities = getattr(entity, 'total_debt', Decimal('0'))
        return assets - liabilities

    @staticmethod
    def assess(
        borrower_id: int,
        borrower_type: Literal["company", "person"],
        principal: Decimal,
        world: G._WorldState,
        markets: Dict[str, Market],
    ) -> Optional[List[Tuple[float, int]]]:
        """
        Assess a loan request. Returns list of (down_pct, term_months) options,
        or None if refused.
        Each option is [option_A, option_B, option_C] for the applicable tier.
        """
        if borrower_type == "company":
            entity = world.companies.get(borrower_id)
        else:
            entity = world.population.get(borrower_id)
        if entity is None:
            return None

        net_worth = CentralBank._net_worth(entity, world, markets)
        if net_worth <= 0:
            return None  # negative net worth: auto-refuse

        # Hard gate: cash must cover any down payment option
        # (checked per-option at issuance time, not here)

        coverage = float(net_worth / principal)

        # Find applicable tier
        options = None
        for min_ratio, tier_options in LOAN_TIERS:
            if coverage >= min_ratio:
                options = tier_options
                break

        if options is None:
            return None  # coverage < 0.1x: refused

        return options  # list of (down_pct, term_months)

    @staticmethod
    def issue(
        borrower_id: int,
        borrower_type: Literal["company", "person"],
        principal: Decimal,
        down_pct: float,
        term_months: int,
        current_tick: int,
        world: G._WorldState,
        markets: Dict[str, Market],
    ) -> Optional[G._Loan]:
        """
        Issue a loan after the borrower has chosen their option.
        Validates cash coverage of down payment.
        Returns the Loan instance or None if validation fails.
        """
        if borrower_type == "company":
            entity = world.companies.get(borrower_id)
        else:
            entity = world.population.get(borrower_id)
        if entity is None:
            return None

        down_amount = principal * Decimal(str(down_pct))
        cash = entity.resources.get('money', Decimal('0'))
        if cash < down_amount:
            return None  # can't cover down payment from cash

        financed = principal - down_amount
        monthly = (financed / Decimal(term_months)).quantize(Decimal('0.01'))

        # Deduct down payment
        entity.resources['money'] = cash - down_amount

        # Credit financed amount
        entity.resources['money'] = entity.resources.get('money', Decimal('0')) + financed

        # Update total_debt
        entity.total_debt = getattr(entity, 'total_debt', Decimal('0')) + financed

        loan = G._Loan(
            borrower_type=borrower_type,
            borrower_id=borrower_id,
            principal=principal,
            financed_amount=financed,
            monthly_payment=monthly,
            term_months=term_months,
            next_payment_tick=current_tick + tick_time.LOAN_PAYMENT_PERIOD_TICKS,
            issued_tick=current_tick,
        )
        world.loans[loan.instance_id] = loan
        return loan


class Behavior:
    def tick(
        self,
        world: G._WorldState,
        markets: Dict[str, Market],
        marketing: MarketingMarket,
        tick: int,
        rng: random.Random,
    ) -> None:
        raise NotImplementedError

class MarketBehavior(Behavior):
    def tick(
        self,
        world: G._WorldState,
        markets: Dict[str, Market],
        marketing: MarketingMarket,
        tick: int,
        rng: random.Random,
    ) -> None:
        # first update prices for all markets
        for market in markets.values():
            market.update_prices(world, tick, rng)
            
        # mark new prices as previous for next tick
        for market in markets.values():
            market.prev_price = market.price

class MarketingBehavior(Behavior):
    def tick(
        self,
        world: G._WorldState,
        markets: Dict[str, Market],
        marketing: MarketingMarket,
        tick: int,
        rng: random.Random,
    ) -> None:
        marketing.update_prices(world, tick, rng)

class CompanyBehavior(Behavior):
    def __init__(self) -> None:
        """Placeholder constructor; kept for future expansion."""

    _LR = 0.25
    _NOISE_SCALE = 0.02
    _THRESHOLD_PCT = Decimal("0.03")
    _SPEC_FRAC = Decimal("0.30")
    _UNDERCUT_PCT = Decimal("0.01")
    _SELL_BUFFER = Decimal("5")
    _BUILD_BUFFER = Decimal("20")
    _SUPPLY_BUFFER = Decimal("2")
    _COLLECT_BATCH = 40

    def _compute_unit_economics(self, conv, markets) -> Decimal:
        revenue = sum(markets[rid].price * amt for rid, amt in conv.output_resources.items())
        cost = sum(markets[rid].price * amt for rid, amt in conv.input_resources.items())
        return revenue - cost

    def _inventory_amount(self, inventory: G._Inventory, resource_id: str) -> Decimal:
        return Decimal(sum(stack.amount for stack in inventory.stacks if stack.resource_id == resource_id))

    def _company_origin(self, world: G._WorldState, comp: G._CompanyInstance) -> Tuple[int, int]:
        if not comp.buildings:
            return (0, 0)
        building = world.buildings.get(comp.buildings[0])
        if building and building.origin:
            return building.origin
        return (0, 0)

    def _task_exists(
        self,
        comp: G._CompanyInstance,
        resource_id: str,
        task_kind: str,
        destination_key: Tuple,
        world: G._WorldState,
    ) -> bool:
        for task in comp.logistics_queue:
            if task.get('task_kind') == task_kind and task.get('resource') == resource_id and task.get('destination_key') == destination_key:
                return True
        for vehicle in world.vehicles.values():
            if vehicle.owner_company_id != comp.instance_id or not vehicle.task:
                continue
            vt = vehicle.task
            if vt.get('task_kind') == task_kind and vt.get('resource') == resource_id and vt.get('destination_key') == destination_key:
                return True
        return False

    def _schedule_internal_logistics(self, comp: G._CompanyInstance, world: G._WorldState, tick: int) -> None:
        if not comp.buildings:
            return
        hub_pos = self._company_origin(world, comp)
        for building_id in comp.buildings:
            binst = world.buildings.get(building_id)
            if not binst:
                continue
            for unit in binst.units.values():
                for machine in unit.machines:
                    if not machine.is_active or machine.recipe is None:
                        continue
                    conv = ConvDefs.get(machine.recipe)
                    if conv is None:
                        continue
                    dest_pos = binst.origin
                    for rid, amt in conv.input_resources.items():
                        supply_key = ('machine_supply', machine.instance_id, rid)
                        have = self._inventory_amount(machine.inventory, rid)
                        target = Decimal(amt) * self._SUPPLY_BUFFER
                        shortfall = target - have
                        available = comp.resources.get(rid, Decimal(0))
                        deliver = min(shortfall, available)
                        deliver_int = int(deliver)
                        if deliver_int <= 0:
                            continue
                        if machine.inventory.capacity is not None:
                            capacity_left = machine.inventory.capacity - machine.inventory.total_amount()
                            if capacity_left <= 0:
                                continue
                            deliver_int = min(deliver_int, capacity_left)
                        deliver_int = min(deliver_int, self._COLLECT_BATCH)
                        if deliver_int <= 0:
                            continue
                        if self._task_exists(comp, rid, 'supply', supply_key, world):
                            continue
                        # Skip supply task if an active conveyor already feeds this machine for this resource
                        has_conveyor_supply = any(
                            p.active and p.dest_machine_id == machine.instance_id and p.resource_id == rid
                            for p in world.conveyor_paths.values()
                        )
                        if has_conveyor_supply:
                            continue
                        task = {
                            'id': get_instance_id(),
                            'task_kind': 'supply',
                            'resource': rid,
                            'quantity': deliver_int,
                            'remaining': deliver_int,
                            'payload': 0,
                            'phase': 'pending',
                            'source': {'type': 'company', 'company_id': comp.instance_id, 'pos': hub_pos},
                            'destination': {
                                'type': 'machine',
                                'company_id': comp.instance_id,
                                'building_id': binst.instance_id,
                                'unit_id': unit.instance_id,
                                'machine_id': machine.instance_id,
                                'pos': dest_pos
                            },
                            'destination_key': supply_key,
                            'created_tick': tick
                        }
                        comp.logistics_queue.append(task)
                    for rid, amt in conv.output_resources.items():
                        collect_key = ('machine_collect', machine.instance_id, rid)
                        stored = self._inventory_amount(machine.inventory, rid)
                        collect_int = int(stored)
                        if collect_int <= 0:
                            continue
                        collect_int = min(collect_int, self._COLLECT_BATCH)
                        if self._task_exists(comp, rid, 'collect', collect_key, world):
                            continue
                        # Skip collect task if an active conveyor already moves output from this machine
                        has_conveyor_collect = any(
                            p.active and p.source_machine_id == machine.instance_id and p.resource_id == rid
                            for p in world.conveyor_paths.values()
                        )
                        if has_conveyor_collect:
                            continue
                        task = {
                            'id': get_instance_id(),
                            'task_kind': 'collect',
                            'resource': rid,
                            'quantity': collect_int,
                            'remaining': collect_int,
                            'payload': 0,
                            'phase': 'pending',
                            'source': {
                                'type': 'machine',
                                'company_id': comp.instance_id,
                                'building_id': binst.instance_id,
                                'unit_id': unit.instance_id,
                                'machine_id': machine.instance_id,
                                'pos': dest_pos
                            },
                            'destination': {'type': 'company', 'company_id': comp.instance_id, 'pos': hub_pos},
                            'destination_key': collect_key,
                            'created_tick': tick
                        }
                        comp.logistics_queue.append(task)

        # Refuel vehicles with low fuel
        for vehicle in world.vehicles.values():
            if vehicle.owner_company_id != comp.instance_id:
                continue
            if vehicle.status == 'moving' and vehicle.task and vehicle.task.get('task_kind') == 'refuel':
                continue  # already heading to refuel

            fuel_capacity = vehicle.vehicle_type.fuel_capacity
            fuel_threshold = fuel_capacity * Decimal('0.20')
            if vehicle.fuel_stored > fuel_threshold:
                continue  # fuel is fine

            # Check company has fuel to give
            fuel_available = comp.resources.get('fuel', Decimal('0'))
            if fuel_available <= 0:
                continue

            # Don't double-queue
            already_queued = any(
                t.get('task_kind') == 'refuel' and t.get('vehicle_id') == vehicle.instance_id
                for t in comp.logistics_queue
            )
            if already_queued:
                continue
            if vehicle.task and vehicle.task.get('task_kind') == 'refuel':
                continue

            refuel_amount = min(fuel_capacity - vehicle.fuel_stored, fuel_available)
            if refuel_amount <= 0:
                continue

            task = {
                'id': get_instance_id(),
                'task_kind': 'refuel',
                'resource': 'fuel',
                'quantity': int(refuel_amount),
                'remaining': int(refuel_amount),
                'payload': 0,
                'phase': 'pending',
                'vehicle_id': vehicle.instance_id,
                'source': {'type': 'company', 'company_id': comp.instance_id, 'pos': hub_pos},
                'destination': {'type': 'vehicle_self'},  # vehicle refuels itself at source
                'destination_key': ('refuel', vehicle.instance_id),
                'created_tick': tick
            }
            comp.logistics_queue.append(task)

    def _acquire_resource(
        self,
        comp: G._CompanyInstance,
        resource_id: str,
        amount: Decimal,
        markets: Dict[str, Market],
        tick: int,
        rng: random.Random,
    ) -> None:
        target = comp.target_inventory.get(resource_id, amount)
        desired_total = max(target, amount)
        have = comp.resources.get(resource_id, Decimal(0))
        needed = desired_total - have
        if needed <= 0:
            return

        market = markets[resource_id]
        fair_value = _update_fair_value(comp, resource_id, market.volume_weighted_price, rng, self._LR, self._NOISE_SCALE)
        buy_cost = fair_value * needed

        best_conv: Optional[G.ResourceConversion] = None
        best_cost: Optional[Decimal] = None
        for conv in ConvDefs.values():
            if resource_id not in conv.output_resources:
                continue
            out_amt = conv.output_resources[resource_id]
            cycles = (needed / out_amt).quantize(Decimal("1."), rounding=ROUND_UP)
            inputs_cost = sum(
                markets[rid].price * amt for rid, amt in conv.input_resources.items()
            ) * cycles
            cost = inputs_cost / max(needed, Decimal(1))
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_conv = conv

        produced = Decimal(0)
        prefer_make = comp.personality.integration_bias >= 0.5
        if best_conv and best_cost is not None:
            if prefer_make or best_cost <= buy_cost:
                out_amt = best_conv.output_resources[resource_id]
                max_cycles = int((needed / out_amt).quantize(Decimal("1."), rounding=ROUND_UP))
                for _ in range(max_cycles):
                    if all(comp.resources.get(rid, Decimal(0)) >= amt for rid, amt in best_conv.input_resources.items()):
                        for rid, amt in best_conv.input_resources.items():
                            comp.resources[rid] -= amt
                        for rid, amt in best_conv.output_resources.items():
                            comp.resources[rid] = comp.resources.get(rid, Decimal(0)) + amt
                        produced += out_amt
                        if produced >= needed:
                            break
                    else:
                        break
        if produced >= needed:
            return

        needed = max(Decimal(0), needed - produced)
        if needed <= 0:
            return

        qty = needed.quantize(Decimal("1."), rounding=ROUND_UP)
        qty_int = int(qty)
        if qty_int <= 0:
            return

        prefs = comp.supplier_preferences.get(resource_id, [])
        limit_price = fair_value
        if prefs:
            shipping = market.estimate_shipping_cost(comp.instance_id, Decimal(qty_int), prefs[0])
            limit_price = fair_value + shipping
        else:
            limit_price = fair_value * (Decimal("1.00") + Decimal(str(0.02 * (1 - comp.personality.price_competitiveness))))

        market.order_book.submit(
            Order(
                actor_type="company",
                actor_id=comp.instance_id,
                resource_id=resource_id,
                quantity=Decimal(qty_int),
                limit_price=limit_price,
                side="buy",
                timestamp=tick,
            )
        )

    def _find_best_conversion(self, markets: Dict[str, Market]) -> Optional[G.ResourceConversion]:
        best_conv = None
        best_profit = Decimal(0)
        for conv in ConvDefs.values():
            if not all(rid in markets for rid in list(conv.input_resources) + list(conv.output_resources)):
                continue
            profit = self._compute_unit_economics(conv, markets)
            if profit > best_profit:
                best_profit = profit
                best_conv = conv
        return best_conv

    def _best_competitor_ask(self, mkt: Market, tick: int, exclude_company_id: int) -> Optional[Decimal]:
        asks = [
            o.limit_price for o in mkt.order_book.orders
            if o.side == 'sell' and o.timestamp == tick and (o.actor_id != exclude_company_id)
        ]
        return min(asks) if asks else None

    def _try_build_capacity(self, comp, world, markets) -> None:
        profitable = [conv for conv in ConvDefs.values() if self._compute_unit_economics(conv, markets) > 0]
        if not profitable:
            return
        conv = profitable[0]

        input_backlog = sum(comp.resources.get(rid, Decimal(0)) for rid in conv.input_resources.keys())
        if input_backlog < self._BUILD_BUFFER:
            return

        bdefs = REGISTRY.get("BuildingDefinition", {})
        if not bdefs:
            return
        bdef = next(iter(bdefs.values()))
        cost = bdef.cost
        cash = comp.resources.get("money", Decimal(0))
        if cash < cost:
            return

        origin = bootstrap._find_buildable_origin(world, bdef.floor_width, bdef.floor_length, range(1, 29), range(1, 19))
        if origin is None:
            return

        binst = G._BuildingInstance(
            land_parcel_id=0,
            definition=bdef,
            owner_company_id=str(comp.instance_id),
            units={},
            loading_bay_occupants=[],
            parking_occupants=[],
            origin=origin
        )
        world.buildings[binst.instance_id] = binst
        bootstrap.claim_building_footprint(world, bdef, origin, str(comp.instance_id), binst.instance_id)
        comp.buildings.append(binst.instance_id)
        comp.resources["money"] = cash - cost

        mdefs = REGISTRY.get("MachineDefinition", {})
        if mdefs:
            mdef = mdefs.get("cooker") or next(iter(mdefs.values()))
            unit = G._UnitInstance(owner_company_id=str(comp.instance_id), floor_space=bdef.floor_area)
            binst.add_unit(unit)
            unit.add_machine(mdef, active=True)
            if unit.machines:
                unit.machines[-1].recipe = "cook_food"
                _auto_place_machine(binst, unit, unit.machines[-1])

        vdefs = REGISTRY.get("VehicleDefinition", {})
        if vdefs and not any(v.owner_company_id == comp.instance_id for v in world.vehicles.values()):
            vdef = next(iter(vdefs.values()))
            v = G._VehicleInstance(
                vehicle_type=vdef,
                owner_company_id=comp.instance_id,
                position=origin,
                destination=None,
                status='idle',
                inventory=G._Inventory(capacity=int(vdef.cargo_inventory_size)),
                fuel_stored=Decimal(10)
            )
            world.vehicles[v.instance_id] = v

    def _list_inventory_for_sale(self, comp: G._CompanyInstance, markets: Dict[str, Market], rng: random.Random, tick: int) -> None:
        for rid, qty in list(comp.resources.items()):
            if rid in ("money", "shares"):
                continue
            target_floor = comp.target_inventory.get(rid, Decimal(self._SELL_BUFFER))
            if qty <= target_floor:
                continue
            surplus = qty - target_floor
            mkt = markets[rid]
            fv = _update_fair_value(comp, rid, mkt.volume_weighted_price, rng, self._LR, self._NOISE_SCALE)
            best_ask = self._best_competitor_ask(mkt, tick, comp.instance_id)
            undercut = self._UNDERCUT_PCT * Decimal(str(comp.personality.price_competitiveness))
            target_price = fv if best_ask is None else min(fv, best_ask * (Decimal("1") - undercut))
            sell_qty = surplus * self._SPEC_FRAC * Decimal(str(comp.personality.risk_tolerance + 0.5))
            sell_qty = sell_qty.quantize(Decimal("1."))
            if sell_qty > 0:
                comp.sale_values[rid] = target_price
                mkt.order_book.submit(Order(
                    actor_type="company",
                    actor_id=comp.instance_id,
                    resource_id=rid,
                    quantity=sell_qty,
                    limit_price=target_price,
                    side="sell",
                    timestamp=tick,
                ))

    def _plan_cycles(self, conv: G.ResourceConversion, markets: Dict[str, Market], cash: Decimal) -> int:
        inputs_cost = sum(markets[rid].price * amt for rid, amt in conv.input_resources.items())
        if inputs_cost <= 0:
            return 0
        budget = cash * self._SPEC_FRAC
        max_cycles = int((budget / inputs_cost).quantize(Decimal("1.")))
        return max(0, min(10, max_cycles))

    def _ensure_marketing(self, comp: G._CompanyInstance, marketing: MarketingMarket, tick: int) -> None:
        buffered = Decimal(0)
        for rid, qty in comp.resources.items():
            if rid in ("money", "shares"):
                continue
            target_floor = comp.target_inventory.get(rid, Decimal(self._SELL_BUFFER))
            if qty > target_floor:
                buffered += qty - target_floor
        cash = comp.resources.get("money", Decimal(0))
        if buffered > 0 and cash > 10:
            qty = int(min(10, float(buffered)))
            spend_cap = cash * Decimal(str(0.05 * comp.personality.marketing_focus))
            spend = min(Decimal(qty) * marketing.price, spend_cap)
            if spend > 0:
                marketing.submit(MarketingOrder(company_id=comp.instance_id, quantity=qty, limit_price=marketing.price, timestamp=tick))
                comp.resources["money"] = cash - spend

    def _pay_wages(self, comp: G._CompanyInstance, world: G._WorldState) -> None:
        if not comp.employees:
            return
        cash = comp.resources.get("money", Decimal(0))
        retained: list[int] = []
        for person_id in comp.employees:
            person = world.population.get(person_id)
            job_id = person.job_slot_id if person else None
            job = world.jobs.get(job_id) if job_id is not None else None
            if person is None or job is None or job.company_id != comp.instance_id:
                if job and job.company_id == comp.instance_id:
                    job.person_id = None
                    if job.instance_id not in comp.open_job_ids:
                        comp.open_job_ids.append(job.instance_id)
                if person:
                    person.job_slot_id = None
                    person.employer_id = None
                    person.works_at = None
                continue
            wage = getattr(job, 'wage', Decimal(0))
            if wage <= 0:
                retained.append(person_id)
                continue
            if cash <= 0:
                person.brand_loyalty[comp.instance_id] = max(0.0, person.brand_loyalty.get(comp.instance_id, 0.1) * 0.9)
                retained.append(person_id)
                continue
            pay = min(wage, cash)
            cash -= pay
            comp.resources['money'] = cash
            person.resources['money'] = person.resources.get('money', Decimal(0)) + pay
            if pay < wage:
                person.brand_loyalty[comp.instance_id] = max(0.0, person.brand_loyalty.get(comp.instance_id, 0.1) * 0.9)
            else:
                person.brand_loyalty[comp.instance_id] = min(1.0, person.brand_loyalty.get(comp.instance_id, 0.2) + 0.02)
            retained.append(person_id)
        comp.employees = retained

    def tick(self, world, markets, marketing, tick, rng):
        for comp in world.companies.values():
            for cid in list(comp.active_conversions.keys()):
                conv = ConvDefs.get(cid)
                if not conv:
                    del comp.active_conversions[cid]
                    continue
                profit = self._compute_unit_economics(conv, markets)
                if profit <= 0:
                    comp.active_conversions[cid] += 1
                    if comp.active_conversions[cid] > comp.personality.patience:
                        del comp.active_conversions[cid]
                else:
                    comp.active_conversions[cid] = 0

            if not comp.active_conversions or rng.random() < comp.personality.expansionism:
                conv = self._find_best_conversion(markets)
                if conv and conv.id not in comp.active_conversions:
                    comp.active_conversions[conv.id] = 0

            for conv_id in list(comp.active_conversions.keys()):
                conv = ConvDefs.get(conv_id)
                if not conv:
                    continue
                cash = comp.resources.get("money", Decimal(0))
                target_cycles = self._plan_cycles(conv, markets, cash)
                for rid, amt in conv.input_resources.items():
                    self._acquire_resource(comp, rid, amt * Decimal(target_cycles or 1), markets, tick, rng)

                can_loop = True
                loop_guard = 0
                while can_loop and loop_guard < max(1, target_cycles):
                    loop_guard += 1
                    can_run = all(comp.resources.get(rid, Decimal(0)) >= amt for rid, amt in conv.input_resources.items())
                    if not can_run:
                        break
                    for rid, amt in conv.input_resources.items():
                        comp.resources[rid] -= amt
                    for rid, amt in conv.output_resources.items():
                        comp.resources[rid] = comp.resources.get(rid, Decimal(0)) + amt
                        mkt = markets[rid]
                        fv = _update_fair_value(comp, rid, mkt.volume_weighted_price, rng, self._LR, self._NOISE_SCALE)
                        best_ask = self._best_competitor_ask(mkt, tick, comp.instance_id)
                        undercut = self._UNDERCUT_PCT * Decimal(str(comp.personality.price_competitiveness))
                        ask_price = fv if best_ask is None else min(fv, best_ask * (Decimal("1") - undercut))
                        mkt.order_book.submit(Order(
                            actor_type="company",
                            actor_id=comp.instance_id,
                            resource_id=rid,
                            quantity=amt,
                            limit_price=ask_price,
                            side="sell",
                            timestamp=tick,
                        ))

            cash = comp.resources.get("money", Decimal(0))
            spec_frac = self._SPEC_FRAC * Decimal(str(comp.personality.risk_tolerance + 0.5))
            for rid, mkt in markets.items():
                fv = _update_fair_value(comp, rid, mkt.volume_weighted_price, rng, self._LR, self._NOISE_SCALE)
                mis_pct = (fv - mkt.price) / mkt.price
                if mis_pct > self._THRESHOLD_PCT and cash > 0:
                    budget = cash * spec_frac
                    qty = (budget / mkt.price).quantize(Decimal("1."))
                    if qty > 0:
                        mkt.order_book.submit(Order(
                            actor_type="company",
                            actor_id=comp.instance_id,
                            resource_id=rid,
                            quantity=qty,
                            limit_price=fv,
                            side="buy",
                            timestamp=tick
                        ))
                elif mis_pct < -self._THRESHOLD_PCT:
                    stock = comp.resources.get(rid, Decimal(0))
                    qty = (stock * spec_frac).quantize(Decimal("1."))
                    if qty > 0:
                        mkt.order_book.submit(Order(
                            actor_type="company",
                            actor_id=comp.instance_id,
                            resource_id=rid,
                            quantity=qty,
                            limit_price=fv,
                            side="sell",
                            timestamp=tick
                        ))

            try:
                self._try_build_capacity(comp, world, markets)
            except Exception:
                pass

            self._schedule_internal_logistics(comp, world, tick)
            self._attempt_pending_conveyors(comp, world)
            self._list_inventory_for_sale(comp, markets, rng, tick)
            self._ensure_marketing(comp, marketing, tick)
            self._pay_wages(comp, world)
            # NPC loan consideration: roughly once per ~17 game-hours on average
            if rng.random() < 0.001:
                self._consider_loan(comp, world, markets, tick)

            # Revalidate conveyor paths when conversions change
            for path in world.conveyor_paths.values():
                if path.building_id not in comp.buildings:
                    continue
                src_m = _find_machine_by_id(world, path.source_machine_id)
                dst_m = _find_machine_by_id(world, path.dest_machine_id)
                if src_m and dst_m:
                    _revalidate_path(world, path, src_m, dst_m)

    def _attempt_pending_conveyors(self, comp: G._CompanyInstance, world: G._WorldState) -> None:
        if not comp.pending_conveyor_plans:
            return
        still_pending = []
        for plan in comp.pending_conveyor_plans:
            building = world.buildings.get(plan['building_id'])
            if not building:
                continue
            src_m = dst_m = None
            src_port = dst_port = None
            for u in building.units.values():
                for m in u.machines:
                    if m.instance_id == plan['src_machine_id']:
                        src_m = m
                        for p in m.machine.output_ports:
                            if p.port_id == plan['src_port_id']:
                                src_port = p
                    if m.instance_id == plan['dst_machine_id']:
                        dst_m = m
                        for p in m.machine.input_ports:
                            if p.port_id == plan['dst_port_id']:
                                dst_port = p
            if not src_m or not dst_m or not src_port or not dst_port:
                continue
            exists = any(
                p.source_machine_id == src_m.instance_id and p.dest_machine_id == dst_m.instance_id
                and p.resource_id == plan['resource_id']
                for p in world.conveyor_paths.values()
            )
            if exists:
                continue
            tiles = _plan_conveyor_path(world, building, src_m, src_port, dst_m, dst_port, plan['resource_id'])
            if tiles is None:
                still_pending.append(plan)
                continue
            ground_needed   = sum(1 for t in tiles if t[3] < 3)
            overhead_needed = sum(1 for t in tiles if t[3] >= 3)
            have_g = comp.resources.get('conveyor_segment', Decimal(0))
            have_o = comp.resources.get('conveyor_segment_overhead', Decimal(0))
            if have_g >= ground_needed and have_o >= overhead_needed:
                _place_conveyor_path(world, building, comp, src_m, src_port, dst_m, dst_port, plan['resource_id'], tiles)
            else:
                still_pending.append(plan)
        comp.pending_conveyor_plans = still_pending

    def _consider_loan(self, comp, world, markets, tick):
        # Don't take loans if already in debt beyond 50% of asset value
        nw = CentralBank._net_worth(comp, world, markets)
        if comp.total_debt > nw * Decimal('0.5'):
            return
        cash = comp.resources.get('money', Decimal('0'))
        target_value = sum(
            markets[rid].price * Decimal(str(qty))
            for rid, qty in comp.target_inventory.items()
            if rid in markets
        )
        if target_value <= 0 or cash >= target_value * Decimal('0.2'):
            return  # enough cash, no need for loan
        # Request a modest loan: 2x target inventory value
        principal = target_value * Decimal('2')
        options = CentralBank.assess(comp.instance_id, 'company', principal, world, markets)
        if not options:
            return
        # NPCs always take option A (lowest payment, longest term)
        down_pct, term_months = options[0]
        CentralBank.issue(comp.instance_id, 'company', principal, down_pct, term_months, tick, world, markets)

class LogisticsBehavior(Behavior):
    def _get_machine(self, world: G._WorldState, building_id: Optional[int], unit_id: Optional[int], machine_id: Optional[int]) -> Optional[G._MachineInstance]:
        if building_id is None or unit_id is None or machine_id is None:
            return None
        building = world.buildings.get(building_id)
        if not building:
            return None
        unit = building.units.get(unit_id)
        if not unit:
            return None
        for machine in unit.machines:
            if machine.instance_id == machine_id:
                return machine
        return None

    def _can_service_task(self, comp: G._CompanyInstance, task: dict, world: G._WorldState) -> bool:
        resource_id = task.get('resource')
        qty = int(task.get('remaining', task.get('quantity', 0)))
        if qty <= 0 or not resource_id:
            return False
        kind = task.get('task_kind')
        if kind == 'supply':
            available = comp.resources.get(resource_id, Decimal(0))
            return available >= Decimal(qty)
        if kind == 'trade':
            seller_id = task.get('seller_id')
            seller = world.companies.get(seller_id)
            if not seller:
                return False
            available = seller.outbound_pending.get(resource_id, Decimal(0))
            return available >= Decimal(qty)
        if kind == 'collect':
            source = task.get('source', {})
            machine = self._get_machine(world, source.get('building_id'), source.get('unit_id'), source.get('machine_id'))
            if not machine:
                return False
            stored = sum(stack.amount for stack in machine.inventory.stacks if stack.resource_id == resource_id)
            return stored >= qty
        if kind == 'refuel':
            owner = world.companies.get(task.get('source', {}).get('company_id'))
            if not owner:
                return False
            return owner.resources.get('fuel', Decimal(0)) > 0
        return True

    def _start_task(self, vehicle: G._VehicleInstance, task: dict) -> None:
        task['phase'] = 'to_source'
        task['remaining'] = task.get('remaining', task.get('quantity', 0))
        task['payload'] = task.get('payload', 0)
        task['assigned_vehicle'] = vehicle.instance_id
        vehicle.task = task
        vehicle.destination = tuple(task['source']['pos'])
        vehicle.status = 'moving'
        vehicle.current_speed = vehicle.vehicle_type.max_speed

    def tick(
        self,
        world: G._WorldState,
        markets: Dict[str, Market],
        marketing: MarketingMarket,
        tick: int,
        rng: random.Random,
    ) -> None:
        for comp in world.companies.values():
            if not comp.logistics_queue:
                continue

            # Handle refuel tasks — assigned to specific vehicles, not generic idle pool
            refuel_tasks = [t for t in comp.logistics_queue if t.get('task_kind') == 'refuel']
            remaining_queue = [t for t in comp.logistics_queue if t.get('task_kind') != 'refuel']
            for task in refuel_tasks:
                vid = task.get('vehicle_id')
                vehicle = world.vehicles.get(vid)
                if vehicle is None:
                    continue  # vehicle gone
                if vehicle.task is not None:
                    # Vehicle is busy — keep in queue
                    remaining_queue.append(task)
                    continue
                # Assign refuel task to this specific vehicle
                task['phase'] = 'to_source'
                task['assigned_tick'] = tick
                vehicle.task = task
                vehicle.destination = tuple(task['source']['pos'])
                vehicle.status = 'moving'
                vehicle.current_speed = vehicle.vehicle_type.max_speed
            comp.logistics_queue = remaining_queue

            idle = [
                v for v in world.vehicles.values()
                if v.owner_company_id == comp.instance_id and v.task is None and v.status in ('idle', 'parked')
            ]
            if not idle:
                continue
            queue = comp.logistics_queue
            new_queue: List[dict] = []
            while queue and idle:
                task = queue.pop(0)
                qty = int(task.get('quantity', 0))
                if qty <= 0:
                    continue
                if not self._can_service_task(comp, task, world):
                    new_queue.append(task)
                    continue
                vehicle = idle.pop(0)
                task['assigned_tick'] = tick
                self._start_task(vehicle, task)
            new_queue.extend(queue)
            comp.logistics_queue = new_queue



def _handle_person_death(world: G._WorldState, person: G._PersonInstance) -> None:
    if person.job_slot_id is not None:
        job = world.jobs.get(person.job_slot_id)
        if job:
            job.person_id = None
            employer = world.companies.get(person.employer_id) if person.employer_id else None
            if employer and job.instance_id not in employer.open_job_ids:
                employer.open_job_ids.append(job.instance_id)
    if person.owner_company_id is not None:
        comp = world.companies.get(person.owner_company_id)
        if comp:
            comp.owner_person_id = None
    for other in world.population.values():
        other.personal_relationship.pop(person.instance_id, None)
    del world.population[person.instance_id]


class PersonBehavior(Behavior):
    def __init__(self) -> None:
        ...

    _LR            = 0.05
    _NOISE_SCALE   = 0.10      # +/-10 % observational noise
    _THRESHOLD_PCT = Decimal("0.250")   # need big gap to speculate
    _SPEC_FRAC     = Decimal("0.10")

    def _seek_employment(self, person: G._PersonInstance, world: G._WorldState, rng: random.Random) -> None:
        if person.job_slot_id is not None:
            return
        open_slots = [slot for slot in world.jobs.values() if slot.person_id is None]
        if not open_slots:
            return
        scored: List[Tuple[float, G._JobSlot, Optional[G._CompanyInstance]]] = []
        for slot in open_slots:
            company = world.companies.get(slot.company_id) if slot.company_id is not None else None
            wage = getattr(slot, 'wage', Decimal(0))
            if company is None or wage <= 0:
                continue
            loyalty = person.brand_loyalty.get(company.instance_id, 0.0)
            familiarity = 1.0 if company.instance_id in person.known_actors else 0.6
            score = float(wage) * (0.6 + person.personality.time_value * 0.4) * (0.8 + loyalty * 0.4) * familiarity
            scored.append((score, slot, company))
        if not scored:
            return
        scored.sort(key=lambda item: item[0], reverse=True)
        top_score, slot, company = scored[0]
        threshold = 25.0 * (0.5 + rng.random())
        if top_score < threshold:
            return
        slot.person_id = person.instance_id
        person.job_slot_id = slot.instance_id
        person.employer_id = company.instance_id
        if slot.instance_id in company.open_job_ids:
            company.open_job_ids.remove(slot.instance_id)
        if person.instance_id not in company.employees:
            company.employees.append(person.instance_id)
        if company.instance_id not in person.known_actors:
            person.known_actors.append(company.instance_id)
        if company.buildings:
            # Assign to building with most unfilled operator demand
            best_building_id = None
            best_deficit = -1
            for bid in company.buildings:
                b = world.buildings.get(bid)
                if not b:
                    continue
                required = sum(
                    len(m.machine.operator_requirements)
                    for u in b.units.values()
                    for m in u.machines
                    if m.is_active and m.recipe is not None
                )
                assigned = sum(
                    1 for p2 in world.population.values()
                    if p2.employer_id == company.instance_id and p2.works_at == bid
                )
                deficit = required - assigned
                if deficit > best_deficit:
                    best_deficit = deficit
                    best_building_id = bid
            person.works_at = best_building_id if best_building_id is not None else (company.buildings[0] if company.buildings else None)
        person.brand_loyalty[company.instance_id] = min(1.0, person.brand_loyalty.get(company.instance_id, 0.3) + 0.1)
    def tick(self, world, markets, marketing, tick, rng):
        import math
        for p in list(world.population.values()):
            p.age += tick_time.YEARS_PER_TICK  # 1 tick = 1 game-minute; 525,600 minutes/year
            if p.age > 18.0:
                annual_death_prob = 0.00015 * math.exp(0.085 * p.age)
                tick_death_prob = 1.0 - (1.0 - annual_death_prob) ** tick_time.YEARS_PER_TICK
                if rng.random() < tick_death_prob:
                    _handle_person_death(world, p)
                    continue
            if p.employer_id:
                company = world.companies.get(p.employer_id)
                if company:
                    for conv_id in company.active_conversions:
                        if conv_id not in p.known_conversions:
                            # ~2% chance per hour = 0.02/60 per minute tick
                            learn_chance = (0.02 / 60.0) * (p.effective_adaptability / 0.5)
                            if rng.random() < learn_chance:
                                p.known_conversions.append(conv_id)
            cash = p.resources.get("money", Decimal(0))
            self._seek_employment(p, world, rng)
            # initialise preferences once
            if not p.resource_preferences:
                for rid in ResourceDefs.keys():
                    p.resource_preferences[rid] = rng.random()
            # loyalty decay
            for cid in list(p.brand_loyalty.keys()):
                p.brand_loyalty[cid] *= 0.99
            # surface new demands - chance influenced by personality
            for demand in DemandDefs.values():
                if demand.id in p.active_demands:
                    continue
                if tick < p.demand_cooldowns.get(demand.id, 0):
                    continue
                chance = demand.chance_per_tick * (0.5 + p.personality.time_value / 2)
                if evaluate_condition(demand.demand_can_occur, world, p) and rng.random() <= chance:
                    p.active_demands[demand.id] = int(demand.quantity)

            for rid, mkt in markets.items():
                fv = _update_fair_value(p, rid, mkt.price, rng,
                                        self._LR, self._NOISE_SCALE)
                mis_pct = (fv - mkt.price) / mkt.price
                if mis_pct > self._THRESHOLD_PCT and cash > 0:
                    budget = cash * self._SPEC_FRAC
                    qty = (budget / mkt.price).quantize(Decimal("1."))
                    if qty > 0:
                        mkt.order_book.submit(Order(
                            actor_type="person",
                            actor_id=p.instance_id,
                            resource_id=rid,
                            quantity=qty,
                            limit_price=fv,
                            side="buy",
                            timestamp=tick
                        ))
                elif mis_pct < -self._THRESHOLD_PCT:
                    stock = p.resources.get(rid, Decimal(0))
                    qty = (stock * self._SPEC_FRAC).quantize(Decimal("1."))
                    if qty > 0:
                        mkt.order_book.submit(Order(
                            actor_type="person",
                            actor_id=p.instance_id,
                            resource_id=rid,
                            quantity=qty,
                            limit_price=fv,
                            side="sell",
                            timestamp=tick
                        ))

            # attempt to satisfy active demands
            for d_id, qty in list(p.active_demands.items()):
                options = [res for res in ResourceDefs.values() if d_id in res.fulfills_demand]
                if not options:
                    continue
                res = min(
                    options,
                    key=lambda r: (
                        float(markets[r.id].price) / float(r.fulfills_demand[d_id])
                    ) / (
                        p.resource_preferences.get(r.id, 0.5) + 0.5
                    ) / (
                        # Bonus if person knows a seller of this resource (reduces effective price by 20%)
                        1.2 if any(
                            r.id in p.known_products.get(cid, [])
                            for cid in p.known_actors
                            if cid in world.companies
                        ) else 1.0
                    ),
                )
                limit = markets[res.id].price
                if cash >= limit * qty:
                    markets[res.id].order_book.submit(Order(
                        actor_type="person",
                        actor_id=p.instance_id,
                        resource_id=res.id,
                        quantity=Decimal(qty),
                        limit_price=limit,
                        side="buy",
                        timestamp=tick
                    ))
                    cash -= limit * qty
                    del p.active_demands[d_id]
                    d_def = DemandDefs.get(d_id)
                    if d_def and d_def.min_time_until_repeat:
                        p.demand_cooldowns[d_id] = tick + d_def.min_time_until_repeat

            # share known actors with relationships
            for actor_id in p.known_actors:
                # Only spread awareness of companies with public presence (don't spread factory locations)
                if actor_id in world.companies and not _company_has_public_building(world, actor_id):
                    continue
                for other_id, rel in p.personal_relationship.items():
                    if rng.random() < rel * p.personality.info_sharing:
                        other = world.population.get(other_id)
                        if other and actor_id not in other.known_actors:
                            other.known_actors.append(actor_id)
class LifecycleBehavior(Behavior):
    """Handles births, education progression, and career advancement."""

    def tick(self, world, markets, marketing, tick, rng):
        import tick_time as tt
        if tick % tt.BIRTH_CHECK_INTERVAL_TICKS != 0:
            return
        self._maybe_spawn_citizen(world, markets, tick, rng)
        self._update_education(world, tick, rng)
        self._update_career(world, tick, rng)

    def _maybe_spawn_citizen(self, world, markets, tick, rng):
        import tick_time as tt
        if len(world.population) >= tt.MAX_POPULATION:
            return
        if rng.random() > tt.BASE_BIRTH_CHANCE_PER_DAY:
            return

        # New citizen: young, high adaptability and risk tolerance
        cash = Decimal(str(round(rng.uniform(50, 300), 2)))

        personality = G.Personality(
            comfort_value=round(rng.uniform(0.8, 1.3), 3),
            time_value=round(rng.uniform(0.6, 1.4), 3),
            info_sharing=round(rng.uniform(0.2, 0.8), 3),
            marketing_susceptibility=round(rng.uniform(0.1, 0.9), 3),
            # Young people skew toward high risk tolerance and adaptability
            risk_tolerance=round(rng.uniform(0.5, 1.0), 3),
            adaptability=round(rng.uniform(0.6, 1.0), 3),
        )

        person = G._PersonInstance(
            age=rng.uniform(18.0, 25.0),
            resources={'money': cash},
            personal_relationship={},
            personality=personality,
        )

        # Give new citizen awareness of a few existing companies (they grow up in this world)
        company_ids = list(world.companies.keys())
        if company_ids:
            known_count = min(len(company_ids), rng.randint(1, 3))
            person.known_actors = rng.sample(company_ids, known_count)

        # Connect to a few existing citizens via relationship graph (bidirectional)
        existing = list(world.population.values())
        if existing:
            friend_count = min(len(existing), rng.randint(3, 8))
            friends = rng.sample(existing, friend_count)
            for friend in friends:
                weight = round(rng.uniform(0.05, 0.4), 3)
                person.personal_relationship[friend.instance_id] = weight
                friend.personal_relationship.setdefault(person.instance_id, weight)

        world.population[person.instance_id] = person

    def _update_education(self, world, tick, rng):
        import tick_time as tt
        ed_registry = REGISTRY.get('EducationCategory', {})
        if not ed_registry:
            return

        for person in world.population.values():
            if not person.employer_id:
                continue
            company = world.companies.get(person.employer_id)
            if not company:
                continue

            # Accumulate ticks in each domain this company operates
            for conv_id in company.active_conversions:
                conv = ConvDefs.get(conv_id)
                if not conv or not conv.domain:
                    continue
                person.education_ticks[conv.domain] = person.education_ticks.get(conv.domain, 0) + 1

            # Check for education promotion
            for domain_id, domain_ticks in list(person.education_ticks.items()):
                if domain_ticks < tt.EDUCATION_PROMOTION_TICKS:
                    continue
                ed_cat = ed_registry.get(domain_id)
                if ed_cat is None:
                    continue
                # Only promote if person doesn't already have this or a more specific education
                if person.education is None:
                    person.education = ed_cat
                    person.education_ticks[domain_id] = 0  # reset counter
                elif person.education.id == domain_id:
                    pass  # already have it
                else:
                    # Promote to this domain if it's more specific or a different branch
                    current_dist = person.education.degrees_of_separation(ed_cat)
                    if current_dist is None:
                        # Different branch — switch if they've spent more time here
                        other_ticks = person.education_ticks.get(person.education.id, 0)
                        if domain_ticks > other_ticks:
                            person.education = ed_cat
                            person.education_ticks[domain_id] = 0

    def _update_career(self, world, tick, rng):
        import tick_time as tt
        if tick % tt.CAREER_EVAL_INTERVAL_TICKS != 0:
            return

        ROLE_ORDER = ['operator', 'lead', 'manager', 'director', 'executive']

        for person in world.population.values():
            if not person.employer_id or not person.job_slot_id:
                continue

            job = world.jobs.get(person.job_slot_id)
            if not job:
                continue

            # Increment employment ticks
            person.employment_ticks += tt.CAREER_EVAL_INTERVAL_TICKS
            person.current_role = job.role.value

            if person.employment_ticks < tt.PROMOTION_MIN_TICKS:
                continue

            current_role_idx = ROLE_ORDER.index(job.role.value) if job.role.value in ROLE_ORDER else 0
            if current_role_idx >= len(ROLE_ORDER) - 1:
                continue  # already at top

            # Education required: must have some domain education
            if person.education is None:
                continue

            company = world.companies.get(person.employer_id)
            if not company:
                continue

            # Promotion probability: ~5% per week for a qualified candidate
            if rng.random() > 0.05:
                continue

            next_role_str = ROLE_ORDER[current_role_idx + 1]
            try:
                next_role = G.JobRole(next_role_str)
            except ValueError:
                continue

            # Wage scales up with role
            wage_multipliers = {'lead': 1.3, 'manager': 1.7, 'director': 2.2, 'executive': 3.0}
            new_wage = job.wage * Decimal(str(wage_multipliers.get(next_role_str, 1.0)))

            # Vacate old job slot
            job.person_id = None
            if job.instance_id not in company.open_job_ids:
                company.open_job_ids.append(job.instance_id)
            person.job_slot_id = None

            # Create and fill new slot
            new_slot_id = world.post_job(company.instance_id, next_role, new_wage)
            new_slot = world.jobs.get(new_slot_id)
            if new_slot:
                new_slot.person_id = person.instance_id
                person.job_slot_id = new_slot_id
                if new_slot_id in company.open_job_ids:
                    company.open_job_ids.remove(new_slot_id)
                person.current_role = next_role_str


class DiscoveryBehavior(Behavior):
    """
    Citizens discover what products companies sell by:
    1. Visiting a public-access building (transacting = discovery)
    2. Word-of-mouth from friends who have transacted
    3. Marketing impressions (already handled in MarketingBehavior)
    """

    def tick(self, world, markets, marketing, tick, rng):
        if tick % 60 != 0:   # every hour of game-time
            return
        self._update_from_transactions(world, markets, tick)
        self._propagate_via_social(world, rng)

    def _update_from_transactions(self, world, markets, tick):
        """
        For each market, companies with public buildings that have sell orders
        get their products discovered by interested persons who know them.
        """
        for rid, market in markets.items():
            sellers_with_public = [
                o.actor_id for o in market.order_book.orders
                if o.side == 'sell'
                and o.actor_type == 'company'
                and _company_has_public_building(world, o.actor_id)
            ]
            if not sellers_with_public:
                continue

            for person in world.population.values():
                # Only people who have interest in this resource notice it
                if person.resource_preferences.get(rid, 0) < 0.3:
                    continue
                for company_id in sellers_with_public:
                    if company_id in person.known_actors:
                        # Learn their product
                        if company_id not in person.known_products:
                            person.known_products[company_id] = []
                        if rid not in person.known_products[company_id]:
                            person.known_products[company_id].append(rid)

    def _propagate_via_social(self, world, rng):
        """People share product knowledge with friends at a rate governed by info_sharing."""
        for person in world.population.values():
            if not person.known_products:
                continue
            for friend_id, rel_strength in person.personal_relationship.items():
                if rng.random() > rel_strength * person.personality.info_sharing:
                    continue
                friend = world.population.get(friend_id)
                if not friend:
                    continue
                for company_id, product_list in person.known_products.items():
                    if company_id not in friend.known_actors:
                        friend.known_actors.append(company_id)
                    if company_id not in friend.known_products:
                        friend.known_products[company_id] = []
                    for rid in product_list:
                        if rid not in friend.known_products[company_id]:
                            friend.known_products[company_id].append(rid)


class AnalyticsBehavior(Behavior):
    def __init__(self, max_history: int = 360) -> None:
        self.max_history = max_history

    def _trim_list(self, values: List) -> None:
        while len(values) > self.max_history:
            values.pop(0)

    def _trim_mapping(self, mapping: Dict[str, List]) -> None:
        for values in mapping.values():
            self._trim_list(values)

    def tick(
        self,
        world: G._WorldState,
        markets: Dict[str, Market],
        marketing: MarketingMarket,
        tick: int,
        rng: random.Random,
    ) -> None:
        analytics = world.analytics
        analytics.ticks.append(tick)

        population = list(world.population.values())
        pop_count = len(population)
        employed = sum(1 for person in population if person.employer_id is not None)
        employment_rate = (employed / pop_count) if pop_count else 0.0
        analytics.employment_rate.append(employment_rate)

        if pop_count:
            loyalty_sum = 0.0
            for person in population:
                loyalty_sum += max(person.brand_loyalty.values(), default=0.0)
            analytics.avg_brand_loyalty.append(loyalty_sum / pop_count)
        else:
            analytics.avg_brand_loyalty.append(0.0)

        for resource_id in ResourceDefs.keys():
            market = markets.get(resource_id)
            if not market:
                continue
            price = float(market.price)
            analytics.resource_prices.setdefault(resource_id, []).append(price)

            supply = Decimal(0)
            for comp in world.companies.values():
                supply += comp.resources.get(resource_id, Decimal(0))
            for building in world.buildings.values():
                for unit in building.units.values():
                    for machine in unit.machines:
                        supply += Decimal(sum(stack.amount for stack in machine.inventory.stacks if stack.resource_id == resource_id))
            for person in population:
                supply += person.resources.get(resource_id, Decimal(0))
            analytics.resource_supply.setdefault(resource_id, []).append(float(supply))

            demand_total = Decimal(0)
            fulfills = ResourceDefs[resource_id].fulfills_demand
            for person in population:
                for demand_id, qty in person.active_demands.items():
                    ratio = fulfills.get(demand_id)
                    if ratio:
                        demand_total += Decimal(qty) * Decimal(str(ratio))
            analytics.resource_demand.setdefault(resource_id, []).append(float(demand_total))

        self._trim_list(analytics.ticks)
        self._trim_list(analytics.employment_rate)
        self._trim_list(analytics.avg_brand_loyalty)
        self._trim_mapping(analytics.resource_prices)
        self._trim_mapping(analytics.resource_supply)
        self._trim_mapping(analytics.resource_demand)



class BankBehavior(Behavior):
    """
    Runs the central bank each tick:
    - Collects monthly loan payments
    - Triggers liquidation windows on missed payments
    - Seizes assets after liquidation window expires
    - Dissolves insolvent companies
    - Flags persons with no assets as in debtor's prison
    Also checks for zero-net-worth bankruptcy (no loans required).
    """

    def tick(self, world, markets, marketing, tick, rng):
        self._collect_payments(world, markets, tick)
        self._process_liquidations(world, markets, tick, rng)
        self._check_zero_networth_bankruptcy(world, markets)

    def _get_entity(self, world, loan):
        if loan.borrower_type == "company":
            return world.companies.get(loan.borrower_id)
        return world.population.get(loan.borrower_id)

    def _collect_payments(self, world, markets, tick):
        for loan in list(world.loans.values()):
            if tick < loan.next_payment_tick:
                continue
            if loan.payments_made >= loan.term_months:
                # Loan fully repaid
                entity = self._get_entity(world, loan)
                if entity:
                    entity.total_debt = max(Decimal('0'), entity.total_debt - loan.financed_amount)
                del world.loans[loan.instance_id]
                continue

            entity = self._get_entity(world, loan)
            if entity is None:
                del world.loans[loan.instance_id]
                continue

            cash = entity.resources.get('money', Decimal('0'))
            if cash >= loan.monthly_payment:
                entity.resources['money'] = cash - loan.monthly_payment
                entity.total_debt = max(Decimal('0'), entity.total_debt - loan.monthly_payment)
                loan.payments_made += 1
                loan.payments_missed = 0
                loan.liquidation_deadline_tick = None
                loan.next_payment_tick = tick + tick_time.LOAN_PAYMENT_PERIOD_TICKS
            else:
                # Missed payment
                loan.payments_missed += 1
                loan.next_payment_tick = tick + tick_time.LOAN_PAYMENT_PERIOD_TICKS
                if loan.liquidation_deadline_tick is None:
                    loan.liquidation_deadline_tick = tick + tick_time.LOAN_LIQUIDATION_WINDOW_TICKS

    def _process_liquidations(self, world, markets, tick, rng):
        for loan in list(world.loans.values()):
            if loan.liquidation_deadline_tick is None:
                continue
            if tick < loan.liquidation_deadline_tick:
                # Still in window — try to sell assets to make payment
                self._attempt_voluntary_liquidation(world, markets, loan, tick)
            else:
                # Window expired — bank seizes
                self._seize_assets(world, markets, loan, tick, rng)

    def _attempt_voluntary_liquidation(self, world, markets, loan, tick):
        """Entity tries to raise cash by selling inventory at market price."""
        entity = self._get_entity(world, loan)
        if entity is None:
            return
        cash = entity.resources.get('money', Decimal('0'))
        needed = loan.monthly_payment * Decimal(loan.payments_missed)
        if cash >= needed:
            return  # already have enough, payment will succeed next cycle
        # Liquidate non-money, non-shares resources at market price
        for rid in list(entity.resources.keys()):
            if rid in ('money', 'shares'):
                continue
            qty = entity.resources.get(rid, Decimal('0'))
            if qty <= 0:
                continue
            mkt = markets.get(rid)
            if not mkt:
                continue
            sell_value = qty * mkt.price
            entity.resources['money'] = entity.resources.get('money', Decimal('0')) + sell_value
            entity.resources[rid] = Decimal('0')
            if entity.resources.get('money', Decimal('0')) >= needed:
                break

    def _seize_assets(self, world, markets, loan, tick, rng):
        """Bank takes assets to cover outstanding debt after window expires."""
        entity = self._get_entity(world, loan)
        remaining_debt = loan.financed_amount - (loan.monthly_payment * loan.payments_made)

        if entity:
            # Seize cash first
            cash = entity.resources.get('money', Decimal('0'))
            seized = min(cash, remaining_debt)
            entity.resources['money'] = cash - seized
            remaining_debt -= seized

            # Seize inventory
            if remaining_debt > 0:
                for rid in list(entity.resources.keys()):
                    if rid in ('money', 'shares') or remaining_debt <= 0:
                        continue
                    qty = entity.resources.get(rid, Decimal('0'))
                    mkt = markets.get(rid)
                    if not mkt or qty <= 0:
                        continue
                    value = qty * mkt.price
                    if value >= remaining_debt:
                        sell_qty = (remaining_debt / mkt.price).quantize(Decimal('1'), rounding=ROUND_UP)
                        entity.resources[rid] = max(Decimal('0'), qty - sell_qty)
                        remaining_debt = Decimal('0')
                    else:
                        entity.resources[rid] = Decimal('0')
                        remaining_debt -= value

            # Seize buildings (transfer to bank = just remove from entity, free the land)
            if remaining_debt > 0 and entity.buildings:
                for bid in list(entity.buildings):
                    if remaining_debt <= 0:
                        break
                    b = world.buildings.get(bid)
                    if not b:
                        continue
                    state_factor = {
                        'normal': 1.0, 'fixer_upper': 0.6, 'abandoned': 0.3,
                        'condemned': 0.1, 'collapsed': 0.0
                    }.get(b.decay_state.value, 1.0)
                    bval = b.definition.cost * Decimal(str(state_factor)) * Decimal(str(b.condition))
                    remaining_debt -= bval
                    # Release building: free land, remove from world
                    bootstrap.release_building_footprint(world, b)
                    del world.buildings[bid]
                    entity.buildings.remove(bid)

        # Handle entity-type-specific post-seizure
        if loan.borrower_type == "company":
            comp = world.companies.get(loan.borrower_id)
            if comp:
                # Check if company is effectively insolvent
                nw = CentralBank._net_worth(comp, world, markets)
                if nw <= 0 or (not comp.buildings and not comp.resources):
                    self._dissolve_company(world, markets, comp, remaining_debt)
        else:
            person = world.population.get(loan.borrower_id)
            if person:
                nw = CentralBank._net_worth(person, world, markets)
                if nw <= 0:
                    person.in_debtors_prison = True

        # Close or update the loan
        if remaining_debt <= 0:
            entity = self._get_entity(world, loan)
            if entity:
                entity.total_debt = max(Decimal('0'), entity.total_debt - loan.financed_amount)
            if loan.instance_id in world.loans:
                del world.loans[loan.instance_id]
        else:
            # Partial seizure — loan partially remains
            loan.financed_amount = remaining_debt
            entity = self._get_entity(world, loan)
            if entity:
                entity.total_debt = max(Decimal('0'), remaining_debt)
            loan.payments_missed = 0
            loan.liquidation_deadline_tick = None

    def _dissolve_company(self, world, markets, comp, remaining_debt):
        """
        Dissolve an insolvent company:
        - Fire all employees
        - Cancel open jobs
        - Any remaining assets distributed to shareholders (pro-rata by shares held)
        - Remaining debt distributed to shareholders (pro-rata)
        - Remove company from world
        """
        # Fire employees
        for person_id in list(comp.employees):
            person = world.population.get(person_id)
            if person:
                if person.job_slot_id is not None:
                    job = world.jobs.get(person.job_slot_id)
                    if job:
                        del world.jobs[job.instance_id]
                person.job_slot_id = None
                person.employer_id = None
                person.works_at = None
                # Brand loyalty hit
                person.brand_loyalty.pop(comp.instance_id, None)
        comp.employees = []

        # Cancel open jobs
        for jid in list(comp.open_job_ids):
            world.jobs.pop(jid, None)
        comp.open_job_ids = []

        # Clear real_estate_listings for units owned by this company
        for lid in [lid for lid, lst in world.real_estate_listings.items() if lst.listed_by_company_id == comp.instance_id]:
            del world.real_estate_listings[lid]

        # Remove company from rented units (access_team_ids)
        for building in world.buildings.values():
            for unit in building.units.values():
                tid_str = str(comp.instance_id)
                if tid_str in unit.access_team_ids:
                    unit.access_team_ids = [t for t in unit.access_team_ids if t != tid_str]

        # Release buildings
        for bid in list(comp.buildings):
            b = world.buildings.get(bid)
            if b:
                # Also clear listings for units in these buildings
                for uid in list(b.units.keys()):
                    world.remove_listing(bid, uid)
                bootstrap.release_building_footprint(world, b)
                del world.buildings[bid]
        comp.buildings = []

        # Cancel vehicles
        for vid in list(world.vehicles.keys()):
            v = world.vehicles[vid]
            if v.owner_company_id == comp.instance_id:
                del world.vehicles[vid]

        # Distribute remaining debt to shareholders (persons who own this company)
        shareholders = [p for p in world.population.values() if getattr(p, 'owner_company_id', None) == comp.instance_id]
        if shareholders and remaining_debt > 0:
            share = remaining_debt / Decimal(len(shareholders))
            for sh in shareholders:
                sh.total_debt = getattr(sh, 'total_debt', Decimal('0')) + share
                # Check if shareholder is now insolvent
                nw = CentralBank._net_worth(sh, world, {})
                if nw <= 0:
                    sh.in_debtors_prison = True
                # Disown the company
                sh.owner_company_id = None

        # Remove company
        if comp.instance_id in world.companies:
            del world.companies[comp.instance_id]

    def _check_zero_networth_bankruptcy(self, world, markets):
        """
        Zero-net-worth failure condition: entities with no loans but also no assets.
        For companies: dissolve. For persons: debtor's prison flag.
        This is the non-loan bankruptcy path (just ran out of money and assets).
        """
        for comp in list(world.companies.values()):
            if comp.owner_person_id == -1:
                continue  # never auto-dissolve player company via this path
            nw = CentralBank._net_worth(comp, world, markets)
            # Has any active loans? If so, BankBehavior handles it
            has_loans = any(l.borrower_id == comp.instance_id and l.borrower_type == 'company' for l in world.loans.values())
            if has_loans:
                continue
            # Check insolvent: negative NW AND no meaningful assets
            cash = comp.resources.get('money', Decimal('0'))
            if nw < Decimal('-100') and cash <= 0 and not comp.buildings:
                self._dissolve_company(world, markets, comp, Decimal('0'))

        for person in list(world.population.values()):
            if person.in_debtors_prison:
                continue
            nw = CentralBank._net_worth(person, world, markets)
            has_loans = any(l.borrower_id == person.instance_id and l.borrower_type == 'person' for l in world.loans.values())
            if has_loans:
                continue
            if nw <= 0 and person.resources.get('money', Decimal('0')) <= 0 and not person.owner_company_id:
                person.in_debtors_prison = True


def _building_staffing_ratio(world: G._WorldState, building: G._BuildingInstance) -> float:
    """
    Returns a 0.0–1.0 ratio of (assigned workers) / (required operators across all active machines).
    Returns 1.0 if no operator requirements exist (machine runs unstaffed by design).
    """
    owner_id = int(building.owner_company_id)

    total_required = sum(
        len(m.machine.operator_requirements)
        for u in building.units.values()
        for m in u.machines
        if m.is_active and m.recipe is not None
    )

    if total_required == 0:
        return 1.0  # no staffing needed

    assigned = sum(
        1 for p in world.population.values()
        if p.employer_id == owner_id and p.works_at == building.instance_id
    )

    return min(1.0, assigned / total_required)


class MachineBehavior(Behavior):
    """
    Handles production: consumes inputs and adds outputs to each active machine's inventory
    """
    def tick(
        self,
        world: G._WorldState,
        markets: Dict[str, Market],
        marketing: MarketingMarket,
        tick: int,
        rng: random.Random,
    ) -> None:
        for building in world.buildings.values():
            for unit in building.units.values():
                for m in unit.machines:
                    if not m.is_active or m.recipe is None:
                        continue
                    conv = ConvDefs.get(m.recipe)
                    if conv is None:
                        continue
                    # determine cycle time with building decay state and condition scaling
                    cycle = m.machine.possible_recipes.get(m.recipe) or conv.default_time_taken
                    # Get building efficiency for this machine
                    building_efficiency = 1.0
                    if building is not None:
                        state_key = building.decay_state.value
                        state_cap = DECAY_STATE_EFFICIENCY.get(state_key, 0.0)
                        if state_cap == 0.0:
                            continue  # building is condemned/collapsed, no production
                        # Within NORMAL state: condition directly affects speed
                        if building.decay_state == G.DecayState.NORMAL:
                            building_efficiency = max(0.1, building.condition) * state_cap
                        else:
                            # Non-normal decay states: condition within tier * state cap
                            building_efficiency = max(0.05, building.condition) * state_cap
                    staffing_ratio = _building_staffing_ratio(world, building) if building is not None else 1.0

                    # Combine decay efficiency with staffing: both must be satisfied
                    if staffing_ratio <= 0.0:
                        continue  # no staff, no production

                    combined_efficiency = building_efficiency * staffing_ratio
                    # Apply efficiency to cycle time: lower efficiency = longer cycles
                    effective_cycle = int(int(cycle) / max(0.05, combined_efficiency))
                    # only run at multiples of cycle
                    if tick < m.wait_for_tick:
                        continue
                    # attempt to consume all inputs
                    can_run = True
                    for res_id, amt in conv.input_resources.items():
                        demand = G._ResourceStack(amount=int(amt), cost=float(markets[res_id].price * amt), resource_id=res_id)
                        taken  = m.inventory.take(demand, partial=False)
                        if taken is None or taken.amount < demand.amount:
                            can_run = False
                            break
                    if not can_run:
                        continue
                    m.wait_for_tick = tick + effective_cycle
                    # produce outputs
                    for res_id, amt in conv.output_resources.items():
                        out_stack = G._ResourceStack(amount=int(amt), cost=1.0, resource_id=res_id, created_tick=tick)

                        # Check for active conveyor path from this machine for this resource
                        active_path = next(
                            (p for p in world.conveyor_paths.values()
                             if p.source_machine_id == m.instance_id
                             and p.resource_id == res_id
                             and p.active),
                            None
                        )

                        if active_path:
                            dst_machine = _find_machine_by_id(world, active_path.dest_machine_id)
                            if dst_machine is not None:
                                dst_machine.inventory.put(out_stack, fill=True)
                            else:
                                active_path.active = False
                                m.inventory.put(out_stack, fill=True)
                        else:
                            remainder = m.inventory.put(out_stack, fill=False)
                            if remainder is not None:
                                m.inventory.put(out_stack, fill=True)

class ExpirationBehavior(Behavior):
    """
    Enforces goods expiration. Removes stacks from inventories that have exceeded
    their resource's expires_in tick count.
    Runs every hour of game-time to avoid per-tick overhead.
    """

    def tick(self, world, markets, marketing, tick, rng):
        if tick % 60 != 0:  # every game-hour
            return
        self._expire_company_resources(world, tick)
        self._expire_machine_inventories(world, tick)
        self._expire_person_inventories(world, tick)

    def _is_expired(self, stack: G._ResourceStack, tick: int) -> bool:
        res_def = ResourceDefs.get(stack.resource_id)
        if res_def is None or res_def.expires_in is None:
            return False
        if stack.created_tick == 0:
            return False  # legacy stacks with no timestamp: don't expire
        return (tick - stack.created_tick) >= res_def.expires_in

    def _expire_company_resources(self, world: G._WorldState, tick: int) -> None:
        for comp in world.companies.values():
            for rid, qty in list(comp.resources.items()):
                if rid in ('money', 'shares') or qty <= 0:
                    continue
                res_def = ResourceDefs.get(rid)
                if res_def is None or res_def.expires_in is None:
                    continue
                acquired_tick = comp.resource_acquired_tick.get(rid, tick)
                age = tick - acquired_tick
                if age < res_def.expires_in:
                    continue
                # Past expiry: decay at 1% per hour proportionally
                decay_fraction = Decimal('0.01')  # 1% per hour
                spoiled = (qty * decay_fraction).quantize(Decimal('0.01'))
                comp.resources[rid] = max(Decimal('0'), qty - spoiled)
                if comp.resources[rid] <= 0:
                    comp.resources[rid] = Decimal('0')

    def _expire_machine_inventories(self, world: G._WorldState, tick: int) -> None:
        """Machine inventories use proper _Inventory with _ResourceStack — expire by stack."""
        for building in world.buildings.values():
            for unit in building.units.values():
                for machine in unit.machines:
                    expired_stacks = [s for s in machine.inventory.stacks if self._is_expired(s, tick)]
                    for s in expired_stacks:
                        machine.inventory.stacks.remove(s)

    def _expire_person_inventories(self, world: G._WorldState, tick: int) -> None:
        """Person inventories use _Inventory with stacks."""
        for person in world.population.values():
            expired_stacks = [s for s in person.inventory.stacks if self._is_expired(s, tick)]
            for s in expired_stacks:
                person.inventory.stacks.remove(s)


class VehicleBehavior(Behavior):
    """Moves vehicles, consumes fuel, and handles loading/unloading."""

    def _get_company(self, world: G._WorldState, company_id: Optional[int]) -> Optional[G._CompanyInstance]:
        if company_id is None:
            return None
        return world.companies.get(company_id)

    def _get_machine(self, world: G._WorldState, building_id: Optional[int], unit_id: Optional[int], machine_id: Optional[int]) -> Optional[G._MachineInstance]:
        if building_id is None or unit_id is None or machine_id is None:
            return None
        building = world.buildings.get(building_id)
        if not building:
            return None
        unit = building.units.get(unit_id)
        if not unit:
            return None
        for machine in unit.machines:
            if machine.instance_id == machine_id:
                return machine
        return None

    def _process_loading(self, vehicle: G._VehicleInstance, world: G._WorldState, task: dict) -> None:
        resource = task.get('resource')
        if not resource:
            vehicle.status = 'idle'
            vehicle.task = None
            return
        remaining = int(task.get('remaining', 0))
        if remaining <= 0:
            task['phase'] = 'to_destination'
            vehicle.destination = tuple(task['destination']['pos'])
            vehicle.status = 'moving'
            vehicle.current_speed = vehicle.vehicle_type.max_speed
            return
        capacity_left = vehicle.inventory.capacity - vehicle.inventory.total_amount()
        if capacity_left <= 0:
            task['phase'] = 'to_destination'
            vehicle.destination = tuple(task['destination']['pos'])
            vehicle.status = 'moving'
            vehicle.current_speed = vehicle.vehicle_type.max_speed
            return
        load_amount = min(remaining, capacity_left)
        kind = task.get('task_kind')
        loaded = 0
        if kind == 'trade':
            seller = self._get_company(world, task.get('seller_id'))
            if seller:
                available = seller.outbound_pending.get(resource, Decimal(0))
                to_take = min(load_amount, int(available))
                if to_take > 0:
                    seller.outbound_pending[resource] = available - Decimal(to_take)
                    loaded = to_take
        elif kind == 'supply':
            owner = self._get_company(world, vehicle.owner_company_id)
            if owner:
                available = owner.resources.get(resource, Decimal(0))
                to_take = min(load_amount, int(available))
                if to_take > 0:
                    owner.resources[resource] = available - Decimal(to_take)
                    loaded = to_take
        elif kind == 'collect':
            source = task.get('source', {})
            machine = self._get_machine(world, source.get('building_id'), source.get('unit_id'), source.get('machine_id'))
            if machine:
                stack = G._ResourceStack(amount=load_amount, cost=0.0, resource_id=resource)
                taken = machine.inventory.take(stack, partial=True)
                loaded = taken.amount if taken else 0
        elif kind == 'refuel':
            owner = self._get_company(world, vehicle.owner_company_id)
            if owner:
                fuel_capacity = vehicle.vehicle_type.fuel_capacity
                needed = fuel_capacity - vehicle.fuel_stored
                available = owner.resources.get('fuel', Decimal(0))
                to_add = min(needed, available)
                if to_add > 0:
                    owner.resources['fuel'] = available - to_add
                    vehicle.fuel_stored = min(fuel_capacity, vehicle.fuel_stored + to_add)
            # Refueling complete — no destination needed, go idle
            vehicle.task = None
            vehicle.status = 'idle'
            vehicle.destination = None
            return
        if loaded <= 0:
            return

        stack = G._ResourceStack(amount=loaded, cost=float(loaded), resource_id=resource)
        vehicle.inventory.put(stack, fill=False)
        task['remaining'] = int(task.get('remaining', loaded) - loaded)
        task['payload'] = int(task.get('payload', 0) + loaded)

        if task['remaining'] <= 0 or vehicle.inventory.total_amount() >= vehicle.inventory.capacity:
            task['phase'] = 'to_destination'
            vehicle.destination = tuple(task['destination']['pos'])
            vehicle.status = 'moving'
            vehicle.current_speed = vehicle.vehicle_type.max_speed

    def _process_unloading(self, vehicle: G._VehicleInstance, world: G._WorldState, task: dict) -> None:
        resource = task.get('resource')
        payload = int(task.get('payload', 0))
        if not resource:
            vehicle.task = None
            vehicle.status = 'idle'
            return
        if payload <= 0:
            if int(task.get('remaining', 0)) > 0:
                task['phase'] = 'to_source'
                vehicle.destination = tuple(task['source']['pos'])
                vehicle.status = 'moving'
                vehicle.current_speed = vehicle.vehicle_type.max_speed
            else:
                vehicle.task = None
                vehicle.status = 'idle'
            return
        drop_amount = min(payload, vehicle.inventory.total_amount())
        if drop_amount <= 0:
            return
        stack = vehicle.inventory.take(G._ResourceStack(amount=drop_amount, cost=0.0, resource_id=resource), partial=True)
        delivered = stack.amount if stack else 0
        if delivered <= 0:
            return
        destination = task.get('destination', {})
        dest_type = destination.get('type')
        if dest_type == 'company':
            comp = self._get_company(world, destination.get('company_id'))
            if comp:
                comp.resources[resource] = comp.resources.get(resource, Decimal(0)) + Decimal(delivered)
                pending = comp.inbound_pending.get(resource)
                if pending is not None:
                    comp.inbound_pending[resource] = max(Decimal(0), pending - Decimal(delivered))
                # Track acquisition tick for expiry
                comp.resource_acquired_tick[resource] = task.get('created_tick', 0)
        elif dest_type == 'machine':
            machine = self._get_machine(world, destination.get('building_id'), destination.get('unit_id'), destination.get('machine_id'))
            if machine:
                put_stack = G._ResourceStack(amount=delivered, cost=0.0, resource_id=resource)
                remainder = machine.inventory.put(put_stack, fill=True)
                if remainder and remainder.amount > 0:
                    delivered -= remainder.amount
                    vehicle.inventory.put(remainder, fill=False)
        task['payload'] = max(0, payload - delivered)
        if task.get('remaining', 0) <= 0 and task['payload'] <= 0:
            vehicle.task = None
            vehicle.status = 'idle'
            vehicle.destination = None
        elif task['payload'] <= 0 and task.get('remaining', 0) > 0:
            task['phase'] = 'to_source'
            vehicle.destination = tuple(task['source']['pos'])
            vehicle.status = 'moving'
            vehicle.current_speed = vehicle.vehicle_type.max_speed

    def tick(
        self,
        world: G._WorldState,
        markets: Dict[str, Market],
        marketing: MarketingMarket,
        tick: int,
        rng: random.Random,
    ) -> None:
        for v in world.vehicles.values():
            task = v.task
            if v.status == "moving" and v.destination:
                speed = int(v.current_speed or v.vehicle_type.max_speed)
                x, y = v.position
                dx, dy = v.destination
                for _ in range(speed):
                    if (x, y) == (dx, dy):
                        break
                    if x < dx:
                        x += 1
                    elif x > dx:
                        x -= 1
                    elif y < dy:
                        y += 1
                    elif y > dy:
                        y -= 1
                    v.fuel_stored -= v.vehicle_type.fuel_consumption
                    if v.fuel_stored <= 0:
                        v.fuel_stored = Decimal(0)
                        v.current_speed = Decimal(0)
                        v.destination = None
                        v.status = "idle"
                        v.task = None  # allow refuel scheduling on next logistics tick
                        break
                v.position = (x, y)
                if v.destination and (x, y) == v.destination:
                    v.destination = None
                    if task:
                        phase = task.get('phase')
                        if phase == 'to_source':
                            v.status = 'loading'
                        elif phase == 'to_destination':
                            v.status = 'unloading'
                        else:
                            v.status = 'idle'
                    else:
                        v.status = 'idle'
            elif v.status == 'loading':
                if task:
                    self._process_loading(v, world, task)
                else:
                    v.status = 'idle'
            elif v.status == 'unloading':
                if task:
                    self._process_unloading(v, world, task)
                else:
                    v.status = 'idle'

def estimate_startup_cost(conv: G.ResourceConversion, markets: Dict[str, Market], registry: dict) -> Decimal:
    input_cost = sum(markets[rid].price * amt * 2 for rid, amt in conv.input_resources.items() if rid in markets)
    machine_cost = Decimal('500')
    mdefs = registry.get('MachineDefinition', {})
    for mdef in mdefs.values():
        if conv.id in mdef.possible_recipes:
            machine_cost = Decimal('200')
            break
    bdefs_raw = registry.get('BuildingDefinition', [])
    bdefs = list(bdefs_raw.values()) if hasattr(bdefs_raw, 'values') else list(bdefs_raw)
    building_cost = min((b.cost for b in bdefs), default=Decimal('1000'))
    return input_cost + machine_cost + building_cost


def estimate_rental_startup_cost(conv: G.ResourceConversion, markets: Dict[str, Market], registry: dict) -> Decimal:
    """
    Startup cost if agent rents space instead of buying.
    3 months deposit on cheapest building's maintenance cost + machine cost only (no building purchase).
    """
    machine_cost = Decimal('500')
    mdefs = registry.get('MachineDefinition', {})
    for mdef in mdefs.values():
        if conv.id in mdef.possible_recipes:
            machine_cost = Decimal('200')
            break
    bdefs_raw = registry.get('BuildingDefinition', {})
    bdefs = list(bdefs_raw.values()) if hasattr(bdefs_raw, 'values') else list(bdefs_raw)
    if bdefs:
        min_maint = min(b.max_maintenance_cost_per_tick for b in bdefs)
        # 3 months deposit = 3 * 30 * 1440 ticks * max_maintenance_cost_per_tick
        deposit = min_maint * Decimal('129600')
    else:
        deposit = Decimal('500')
    return machine_cost + deposit


def estimate_rental_ongoing_cost(registry: dict) -> Decimal:
    """Per-tick ongoing cost if renting (maintenance paid by landlord, but rent charged to tenant)."""
    bdefs_raw = registry.get('BuildingDefinition', {})
    bdefs = list(bdefs_raw.values()) if hasattr(bdefs_raw, 'values') else list(bdefs_raw)
    if bdefs:
        return min(b.max_maintenance_cost_per_tick for b in bdefs) * Decimal('1.5')
    return Decimal('0.1')


def opportunity_score_landlord(agent_capital: Decimal, registry: dict, markets: Dict[str, Market]) -> Tuple[float, Optional[str]]:
    """
    Score the opportunity of buying a building to rent out.
    Returns (score, building_def_id).
    """
    bdefs_raw = registry.get('BuildingDefinition', {})
    if not bdefs_raw:
        return 0.0, None
    bdefs = bdefs_raw if hasattr(bdefs_raw, 'items') else {b.building_type.name: b for b in bdefs_raw}

    best_score = 0.0
    best_bdef_id = None

    for bid, bdef in bdefs.items():
        if agent_capital < bdef.cost:
            continue
        # Expected monthly rent = 2x maintenance (landlord keeps profit margin)
        monthly_ticks = 43200  # 30 days * 1440
        monthly_rent_income = bdef.max_maintenance_cost_per_tick * Decimal('2') * monthly_ticks
        monthly_maintenance = bdef.max_maintenance_cost_per_tick * monthly_ticks
        monthly_profit = monthly_rent_income - monthly_maintenance

        # Simple payback period score: higher = faster ROI
        if monthly_profit <= 0:
            continue
        payback_months = float(bdef.cost / monthly_profit)
        if payback_months <= 0:
            continue

        # Score inversely proportional to payback period, capital-feasibility gated
        score = (1.0 / payback_months) * min(1.0, float(agent_capital / bdef.cost) - 1.0 + 0.5)
        if score > best_score:
            best_score = score
            best_bdef_id = bid

    return best_score, best_bdef_id


def _spawn_landlord_company(world: G._WorldState, person: G._PersonInstance, bdef_id: str, registry: dict, tick: int) -> None:
    """Person starts a real estate business: buys a building, lists units for rent."""
    bdefs_raw = registry.get('BuildingDefinition', {})
    bdefs = bdefs_raw if hasattr(bdefs_raw, 'get') else {b.building_type.name: b for b in bdefs_raw}
    bdef = bdefs.get(bdef_id)
    if not bdef:
        return

    person_capital = person.resources.get('money', Decimal(0))
    if person_capital < bdef.cost:
        return

    # Find placement
    origin = bootstrap._find_buildable_origin(world, bdef.floor_width, bdef.floor_length, range(1, 29), range(1, 19))
    if not origin:
        return

    # Create landlord company
    biz_personality = G.BusinessPersonality(
        risk_tolerance=person.personality.risk_tolerance * 0.7,
        marketing_focus=0.3,
        price_competitiveness=0.4,
        patience=int(24 * 60 * 6),  # 6 months patience
        expansionism=person.personality.adaptability * 0.3,
        integration_bias=0.2,
        logistics_focus=0.1,
    )

    comp = G._CompanyInstance(
        resources={'money': person_capital - bdef.cost, 'shares': Decimal('100000')},
        personality=biz_personality,
        active_conversions={},
        owner_person_id=person.instance_id,
    )
    world.companies[comp.instance_id] = comp
    person.resources['money'] = Decimal('0')
    person.owner_company_id = comp.instance_id

    if person.job_slot_id is not None:
        job = world.jobs.get(person.job_slot_id)
        if job:
            job.person_id = None
            employer = world.companies.get(person.employer_id) if person.employer_id else None
            if employer and job.instance_id not in employer.open_job_ids:
                employer.open_job_ids.append(job.instance_id)
        person.job_slot_id = None
        person.employer_id = None
        person.works_at = None

    # Build the building
    building = G._BuildingInstance(
        land_parcel_id=0,
        definition=bdef,
        owner_company_id=str(comp.instance_id),
        units={},
        loading_bay_occupants=[],
        parking_occupants=[],
        origin=origin,
    )
    # Create units for rent (split into 2-4 equal units)
    total_area = bdef.floor_area * bdef.floors
    n_units = min(4, max(1, total_area // 8))
    unit_w = bdef.floor_width
    unit_h = bdef.floor_length // n_units if n_units > 1 else bdef.floor_length
    for i in range(n_units):
        unit = G._UnitInstance(
            owner_company_id=str(comp.instance_id),
            floor_space=unit_w * unit_h,
            rent_price_per_tick=bdef.max_maintenance_cost_per_tick * Decimal('2'),
        )
        building.units[unit.instance_id] = unit
        _assign_unit_tiles(building, unit, 0, i * unit_h, unit_w, unit_h, 0)

    world.buildings[building.instance_id] = building
    comp.buildings.append(building.instance_id)
    bootstrap.claim_building_footprint(world, bdef, origin, str(comp.instance_id), building.instance_id)

    # List all units for rent
    for unit in building.units.values():
        world.list_unit_for_rent(building.instance_id, unit.instance_id, comp.instance_id,
                                  unit.rent_price_per_tick or bdef.max_maintenance_cost_per_tick * Decimal('2'), tick)

    person.venture_cooldown = tick + tick_time.VENTURE_COOLDOWN_TICKS


def opportunity_score_developer(agent_capital: Decimal, registry: dict, world: G._WorldState) -> Tuple[float, Optional[str]]:
    """
    Score the opportunity of building residential property.
    Returns (score, building_def_id).
    Good when: unsatisfied shelter demand exists among population, capital allows.
    """
    unhoused_count = sum(1 for p in world.population.values() if p.home_unit_id is None)
    if unhoused_count == 0:
        return 0.0, None

    bdefs_raw = registry.get('BuildingDefinition', {})
    bdefs = bdefs_raw if hasattr(bdefs_raw, 'items') else {b.building_type.name: b for b in bdefs_raw}

    best_score = 0.0
    best_id = None
    for bid, bdef in bdefs.items():
        if not getattr(bdef, 'residential', False):
            continue
        if agent_capital < bdef.cost:
            continue
        demand_signal = min(1.0, unhoused_count / max(1, len(world.population)))
        headroom = min(1.0, float(agent_capital / bdef.cost) - 1.0 + 0.5)
        score = demand_signal * headroom
        if score > best_score:
            best_score = score
            best_id = bid
    return best_score, best_id


def _spawn_developer_company(world: G._WorldState, person: G._PersonInstance, bdef_id: str, registry: dict, tick: int) -> None:
    """Person starts a residential developer company: builds and rents/sells to citizens."""
    bdefs_raw = registry.get('BuildingDefinition', {})
    bdefs = bdefs_raw if hasattr(bdefs_raw, 'get') else {b.building_type.name: b for b in bdefs_raw}
    bdef = bdefs.get(bdef_id)
    if not bdef or not getattr(bdef, 'residential', False):
        return

    person_capital = person.resources.get('money', Decimal(0))
    if person_capital < bdef.cost:
        return

    origin = bootstrap._find_buildable_origin(world, bdef.floor_width, bdef.floor_length, range(1, 29), range(1, 19))
    if not origin:
        return

    biz_personality = G.BusinessPersonality(
        risk_tolerance=person.personality.risk_tolerance * 0.6,
        marketing_focus=0.3,
        price_competitiveness=0.5,
        patience=int(24 * 60 * 6),
        expansionism=person.personality.adaptability * 0.4,
        integration_bias=0.1,
        logistics_focus=0.05,
    )

    comp = G._CompanyInstance(
        resources={'money': person_capital - bdef.cost, 'shares': Decimal('100000')},
        personality=biz_personality,
        active_conversions={},
        owner_person_id=person.instance_id,
    )
    world.companies[comp.instance_id] = comp
    person.resources['money'] = Decimal('0')
    person.owner_company_id = comp.instance_id

    if person.job_slot_id is not None:
        job = world.jobs.get(person.job_slot_id)
        if job:
            job.person_id = None
            employer = world.companies.get(person.employer_id) if person.employer_id else None
            if employer and job.instance_id not in employer.open_job_ids:
                employer.open_job_ids.append(job.instance_id)
        person.job_slot_id = None
        person.employer_id = None
        person.works_at = None

    building = G._BuildingInstance(
        land_parcel_id=0,
        definition=bdef,
        owner_company_id=str(comp.instance_id),
        units={},
        loading_bay_occupants=[],
        parking_occupants=[],
        origin=origin,
    )

    # Determine number of residential units
    total_area = bdef.floor_area * bdef.floors
    # house -> 1 unit per floor, apartment -> 4-6 units per floor
    if bdef.floor_area <= 9:  # small building (house-sized)
        n_units = bdef.floors
    else:
        n_units = min(6, max(4, bdef.floors * 2))

    unit_h = max(1, bdef.floor_length // (n_units // bdef.floors if bdef.floors > 0 else n_units))
    unit_w = bdef.floor_width
    rent_per_tick = bdef.max_maintenance_cost_per_tick * Decimal('2')

    units_per_floor = n_units // bdef.floors if bdef.floors > 0 else n_units
    for floor_idx in range(bdef.floors):
        for u_idx in range(units_per_floor):
            unit = G._UnitInstance(
                owner_company_id=str(comp.instance_id),
                floor_space=unit_w * unit_h,
                rent_price_per_tick=rent_per_tick,
                unit_purpose="residential",
            )
            building.units[unit.instance_id] = unit
            _assign_unit_tiles(building, unit, 0, u_idx * unit_h, unit_w, unit_h, floor_idx)

    world.buildings[building.instance_id] = building
    comp.buildings.append(building.instance_id)
    bootstrap.claim_building_footprint(world, bdef, origin, str(comp.instance_id), building.instance_id)

    # List all units for rent
    for unit in building.units.values():
        world.list_unit_for_rent(building.instance_id, unit.instance_id, comp.instance_id, rent_per_tick, tick)

    person.venture_cooldown = tick + tick_time.VENTURE_COOLDOWN_TICKS


def domain_confidence_person(person: G._PersonInstance, conv: G.ResourceConversion, ed_registry: dict) -> float:
    target_domain_id = conv.domain
    if not target_domain_id:
        return 0.4
    if conv.id in person.known_conversions:
        return 0.9
    for known_id in person.known_conversions:
        known_conv = ConvDefs.get(known_id)
        if known_conv and known_conv.domain == target_domain_id:
            return 0.65
    if person.education:
        target_ed = ed_registry.get(target_domain_id)
        if target_ed:
            dist = person.education.degrees_of_separation(target_ed)
            if dist is not None:
                return max(0.3, 1.0 - dist * 0.25)
            dist_rev = target_ed.degrees_of_separation(person.education)
            if dist_rev is not None:
                return max(0.4, 1.0 - dist_rev * 0.15)
    return 0.3


def domain_confidence_company(comp: G._CompanyInstance, candidate_conv: G.ResourceConversion) -> float:
    if candidate_conv.id in comp.active_conversions:
        return 0.0
    active_convs = [ConvDefs[cid] for cid in comp.active_conversions if cid in ConvDefs]
    uses_own_output = any(
        rid in comp.resources and comp.resources.get(rid, Decimal(0)) > comp.target_inventory.get(rid, Decimal(0))
        for rid in candidate_conv.input_resources
    )
    if uses_own_output:
        return 0.95
    shared = any(
        set(candidate_conv.input_resources) & set(c.input_resources) or
        set(candidate_conv.output_resources) & set(c.output_resources)
        for c in active_convs
    )
    if shared:
        return 0.75
    if candidate_conv.domain and any(c.domain == candidate_conv.domain for c in active_convs):
        return 0.55
    return 0.35


def opportunity_score_person(person, conv, markets, registry, ed_registry) -> float:
    all_rids = list(conv.input_resources) + list(conv.output_resources)
    if not all(rid in markets for rid in all_rids):
        return 0.0
    revenue = sum(float(markets[rid].volume_weighted_price) * float(amt) for rid, amt in conv.output_resources.items())
    cost = sum(float(markets[rid].volume_weighted_price) * float(amt) for rid, amt in conv.input_resources.items())
    raw_margin = revenue - cost
    startup = float(estimate_startup_cost(conv, markets, registry))
    margin_signal = raw_margin / startup if startup > 0 else 0.0
    frustration_signal = 0.0
    for rid in conv.output_resources:
        res_def = ResourceDefs.get(rid)
        if res_def:
            for demand_id in res_def.fulfills_demand:
                frustration_signal = max(frustration_signal, person.venture_frustration.get(demand_id, 0.0) / 20.0)
    combined = margin_signal * 0.6 + frustration_signal * 0.4
    domain_conf = domain_confidence_person(person, conv, ed_registry)
    capital = float(person.resources.get('money', Decimal(0)))

    # Try both buy and rent startup costs, use whichever person can afford
    buy_cost = float(estimate_startup_cost(conv, markets, registry))
    rent_cost = float(estimate_rental_startup_cost(conv, markets, registry))

    if capital >= buy_cost:
        startup_cost = buy_cost
    elif capital >= rent_cost:
        startup_cost = rent_cost
        # Renting increases break-even; reduce score slightly
        ongoing_penalty = float(estimate_rental_ongoing_cost(registry)) * tick_time.TICKS_PER_YEAR
        if revenue <= 0 or ongoing_penalty / (revenue * tick_time.TICKS_PER_YEAR) > 0.5:
            return 0.0
    else:
        return 0.0

    headroom = capital / startup_cost
    cap_feasibility = min(1.0, (headroom - 1.0) / 3.0 + 0.5)
    return combined * domain_conf * cap_feasibility


def opportunity_score_company(comp, conv, markets, registry) -> float:
    all_rids = list(conv.input_resources) + list(conv.output_resources)
    if not all(rid in markets for rid in all_rids):
        return 0.0
    revenue = sum(float(markets[rid].volume_weighted_price) * float(amt) for rid, amt in conv.output_resources.items())
    cost = sum(float(markets[rid].volume_weighted_price) * float(amt) for rid, amt in conv.input_resources.items())
    raw_margin = revenue - cost
    startup = float(estimate_startup_cost(conv, markets, registry))
    margin_signal = raw_margin / startup if startup > 0 else 0.0
    frustration_signal = 0.0
    for rid in conv.input_resources:
        frustration_signal = max(frustration_signal, float(comp.input_frustration.get(rid, 0.0)) / 20.0)
    combined = margin_signal * 0.8 + frustration_signal * 0.2
    domain_conf = domain_confidence_company(comp, conv)
    reserve = sum(comp.target_inventory.get(rid, Decimal(0)) * markets[rid].price for rid in comp.target_inventory if rid in markets)
    available_capital = float(comp.resources.get('money', Decimal(0)) - reserve)
    startup_cost = float(estimate_startup_cost(conv, markets, registry))
    if available_capital < startup_cost:
        return 0.0
    headroom = available_capital / startup_cost
    cap_feasibility = min(1.0, (headroom - 1.0) / 3.0 + 0.5)
    return combined * domain_conf * cap_feasibility


def person_entry_threshold(person: G._PersonInstance) -> float:
    base = 0.4
    threshold = base * (2.0 - person.effective_risk_tolerance)
    if person.employer_id is not None:
        threshold *= 1.2
    return threshold


def company_entry_threshold(comp: G._CompanyInstance) -> float:
    return 0.4 * (2.0 - comp.personality.risk_tolerance)


def _spawn_person_company(world, person, conv, markets, registry, tick):
    bdefs_raw = registry.get('BuildingDefinition', [])
    bdefs = list(bdefs_raw.values()) if hasattr(bdefs_raw, 'values') else list(bdefs_raw)
    # Filter out residential buildings for commercial company spawning
    bdefs = [b for b in bdefs if not getattr(b, 'residential', False)]
    mdefs = registry.get('MachineDefinition', {})
    vdefs = registry.get('VehicleDefinition', {})
    if not mdefs:
        return

    person_capital = person.resources.get('money', Decimal(0))
    startup_cost = estimate_startup_cost(conv, markets, registry)
    rental_cost = estimate_rental_startup_cost(conv, markets, registry)

    machine_def = next((m for m in mdefs.values() if conv.id in m.possible_recipes), None)
    if machine_def is None:
        return

    can_buy = bdefs and person_capital >= startup_cost
    can_rent = not can_buy and person_capital >= rental_cost

    if not can_buy and not can_rent:
        return

    biz_personality = G.BusinessPersonality(
        risk_tolerance=person.personality.risk_tolerance,
        marketing_focus=min(1.0, person.personality.comfort_value * 0.5),
        price_competitiveness=max(0.1, 1.0 - person.personality.comfort_value * 0.5),
        patience=int(24 * (1 + person.personality.risk_tolerance * 3)),
        expansionism=min(1.0, person.personality.adaptability * 0.6),
        integration_bias=0.5,
        logistics_focus=0.4,
    )

    comp = G._CompanyInstance(
        resources={'money': Decimal('0'), 'shares': Decimal('100000')},
        personality=biz_personality,
        active_conversions={conv.id: 0},
        owner_person_id=person.instance_id,
    )

    if can_rent:
        # Rental path: find available commercial unit with enough space
        available = world.available_rentals()
        suitable = [
            lst for lst in available
            if world.buildings.get(lst.building_id) is not None
            and world.buildings[lst.building_id].units.get(lst.unit_id) is not None
            and world.buildings[lst.building_id].units[lst.unit_id].free_floor_space() >= machine_def.floor_space
        ]
        if not suitable:
            return
        best_listing = min(suitable, key=lambda l: l.rent_per_tick or Decimal('999999'))
        rent_per_tick = best_listing.rent_per_tick or Decimal('0')
        deposit = rent_per_tick * Decimal('129600')  # 3 months
        if person_capital < deposit:
            return

        world.companies[comp.instance_id] = comp
        comp.resources['money'] = person_capital - deposit
        person.resources['money'] = Decimal('0')
        person.owner_company_id = comp.instance_id

        if person.job_slot_id is not None:
            job = world.jobs.get(person.job_slot_id)
            if job:
                job.person_id = None
                employer = world.companies.get(person.employer_id) if person.employer_id else None
                if employer and job.instance_id not in employer.open_job_ids:
                    employer.open_job_ids.append(job.instance_id)
            person.job_slot_id = None
            person.employer_id = None
            person.works_at = None

        # Deposit goes to landlord
        landlord = world.companies.get(best_listing.listed_by_company_id)
        if landlord:
            landlord.resources['money'] = landlord.resources.get('money', Decimal(0)) + deposit

        rented_building = world.buildings[best_listing.building_id]
        rented_unit = rented_building.units[best_listing.unit_id]
        world.rent_unit(best_listing.building_id, best_listing.unit_id, str(comp.instance_id), rent_per_tick)
        world.remove_listing(best_listing.building_id, best_listing.unit_id)
        comp.renting_unit_ids.append(best_listing.unit_id)

        rented_unit.add_machine(machine_def, active=True)
        rented_unit.machines[-1].recipe = conv.id
        rented_unit.machines[-1].is_active = True
        _auto_place_machine(rented_building, rented_unit, rented_unit.machines[-1])

        for other_id in person.personal_relationship:
            other = world.population.get(other_id)
            if other and comp.instance_id not in other.known_actors:
                other.known_actors.append(comp.instance_id)

        person.venture_cooldown = tick + tick_time.VENTURE_COOLDOWN_TICKS
        return

    # Buy path (existing)
    bdef = bdefs[0]
    origin = bootstrap._find_buildable_origin(world, bdef.floor_width, bdef.floor_length, range(1, 29), range(1, 19))
    if origin is None:
        return

    world.companies[comp.instance_id] = comp
    comp.resources['money'] = person_capital - startup_cost
    person.resources['money'] = Decimal('0')
    person.owner_company_id = comp.instance_id

    if person.job_slot_id is not None:
        job = world.jobs.get(person.job_slot_id)
        if job:
            job.person_id = None
            employer = world.companies.get(person.employer_id) if person.employer_id else None
            if employer and job.instance_id not in employer.open_job_ids:
                employer.open_job_ids.append(job.instance_id)
        person.job_slot_id = None
        person.employer_id = None
        person.works_at = None

    building = G._BuildingInstance(
        land_parcel_id=0,
        definition=bdef,
        owner_company_id=str(comp.instance_id),
        units={},
        loading_bay_occupants=[],
        parking_occupants=[],
        origin=origin,
    )
    unit = G._UnitInstance(owner_company_id=str(comp.instance_id), floor_space=bdef.floor_area)
    building.add_unit(unit)
    _assign_unit_tiles(building, unit, 0, 0, bdef.floor_width, bdef.floor_length)
    unit.add_machine(machine_def, active=True)
    unit.machines[-1].recipe = conv.id
    unit.machines[-1].is_active = True
    _auto_place_machine(building, unit, unit.machines[-1])

    world.buildings[building.instance_id] = building
    comp.buildings.append(building.instance_id)
    bootstrap.claim_building_footprint(world, bdef, origin, str(comp.instance_id), building.instance_id)

    if vdefs:
        vdef = next(iter(vdefs.values()))
        v = G._VehicleInstance(
            vehicle_type=vdef,
            owner_company_id=comp.instance_id,
            position=origin,
            destination=None,
            status='idle',
            inventory=G._Inventory(capacity=int(vdef.cargo_inventory_size)),
            fuel_stored=Decimal('30'),
        )
        world.vehicles[v.instance_id] = v

    for other_id in person.personal_relationship:
        other = world.population.get(other_id)
        if other and comp.instance_id not in other.known_actors:
            other.known_actors.append(comp.instance_id)

    person.venture_cooldown = tick + tick_time.VENTURE_COOLDOWN_TICKS


def _company_expand_into(world, comp, conv, markets, registry, rng):
    mdefs = registry.get('MachineDefinition', {})
    bdefs_raw = registry.get('BuildingDefinition', [])
    bdefs = list(bdefs_raw.values()) if hasattr(bdefs_raw, 'values') else list(bdefs_raw)
    vdefs = registry.get('VehicleDefinition', {})

    machine_def = next((m for m in mdefs.values() if conv.id in m.possible_recipes), None)
    if machine_def is None:
        return

    startup_cost = estimate_startup_cost(conv, markets, registry)
    reserve = sum(comp.target_inventory.get(rid, Decimal(0)) * markets[rid].price for rid in comp.target_inventory if rid in markets)
    available = comp.resources.get('money', Decimal(0)) - reserve
    if available < startup_cost:
        return

    # Try to fit in existing building first
    for building_id in comp.buildings:
        building = world.buildings.get(building_id)
        if not building:
            continue
        for unit in building.units.values():
            if unit.can_add_machine(machine_def):
                unit.add_machine(machine_def, active=True)
                unit.machines[-1].recipe = conv.id
                unit.machines[-1].is_active = True
                _auto_place_machine(building, unit, unit.machines[-1])
                comp.active_conversions[conv.id] = 0
                comp.resources['money'] = comp.resources.get('money', Decimal(0)) - (startup_cost * Decimal('0.3'))
                return

    # Build new building
    if not bdefs:
        return
    bdef = bdefs[0]
    origin = bootstrap._find_buildable_origin(world, bdef.floor_width, bdef.floor_length, range(1, 29), range(1, 19))
    if origin is None:
        return

    building = G._BuildingInstance(
        land_parcel_id=0,
        definition=bdef,
        owner_company_id=str(comp.instance_id),
        units={},
        loading_bay_occupants=[],
        parking_occupants=[],
        origin=origin,
    )
    unit = G._UnitInstance(owner_company_id=str(comp.instance_id), floor_space=bdef.floor_area)
    building.add_unit(unit)
    _assign_unit_tiles(building, unit, 0, 0, bdef.floor_width, bdef.floor_length)
    unit.add_machine(machine_def, active=True)
    unit.machines[-1].recipe = conv.id
    unit.machines[-1].is_active = True
    _auto_place_machine(building, unit, unit.machines[-1])

    world.buildings[building.instance_id] = building
    comp.buildings.append(building.instance_id)
    bootstrap.claim_building_footprint(world, bdef, origin, str(comp.instance_id), building.instance_id)
    comp.active_conversions[conv.id] = 0
    comp.resources['money'] = comp.resources.get('money', Decimal(0)) - startup_cost

    if vdefs and not any(v.owner_company_id == comp.instance_id for v in world.vehicles.values()):
        vdef = next(iter(vdefs.values()))
        v = G._VehicleInstance(
            vehicle_type=vdef,
            owner_company_id=comp.instance_id,
            position=origin,
            destination=None,
            status='idle',
            inventory=G._Inventory(capacity=int(vdef.cargo_inventory_size)),
            fuel_stored=Decimal('30'),
        )
        world.vehicles[v.instance_id] = v


class EntrepreneurshipBehavior(Behavior):
    EVAL_INTERVAL = tick_time.ENTREPRENEURSHIP_EVAL_TICKS  # evaluate once per game-day, staggered by instance_id

    def _update_frustration(self, world, markets):
        for person in world.population.values():
            for demand_id in person.active_demands:
                person.venture_frustration[demand_id] = (
                    person.venture_frustration.get(demand_id, 0.0) * 0.97 + 1.0
                )
            for demand_id in list(person.venture_frustration.keys()):
                if demand_id not in person.active_demands:
                    person.venture_frustration[demand_id] *= 0.95
                    if person.venture_frustration[demand_id] < 0.01:
                        del person.venture_frustration[demand_id]
        for comp in world.companies.values():
            for rid, market in markets.items():
                unfilled = any(
                    o.actor_type == 'company' and o.actor_id == comp.instance_id and o.side == 'buy'
                    for o in market.order_book.orders
                )
                if unfilled:
                    comp.input_frustration[rid] = comp.input_frustration.get(rid, 0.0) * 0.97 + 1.0
                else:
                    if rid in comp.input_frustration:
                        comp.input_frustration[rid] *= 0.95

    def tick(self, world, markets, marketing, tick, rng):
        self._update_frustration(world, markets)
        ed_registry = REGISTRY.get('EducationCategory', {})

        for person in list(world.population.values()):
            if (tick + person.instance_id) % self.EVAL_INTERVAL != 0:
                continue
            if tick < person.venture_cooldown:
                continue
            if person.owner_company_id is not None:
                continue
            best_conv = None
            best_score = person_entry_threshold(person)
            for conv in ConvDefs.values():
                score = opportunity_score_person(person, conv, markets, REGISTRY, ed_registry)
                if score > best_score:
                    best_score = score
                    best_conv = conv
            if best_conv is not None:
                _spawn_person_company(world, person, best_conv, markets, REGISTRY, tick)

            # Also evaluate landlord venture
            if person.owner_company_id is None and tick >= person.venture_cooldown:
                landlord_score, bdef_id = opportunity_score_landlord(
                    person.resources.get('money', Decimal(0)), REGISTRY, markets
                )
                landlord_threshold = person_entry_threshold(person) * 0.8
                if landlord_score > landlord_threshold and bdef_id:
                    if person.personality.adaptability < 0.6 and rng.random() < 0.3:
                        _spawn_landlord_company(world, person, bdef_id, REGISTRY, tick)

            # Also evaluate developer venture (residential)
            if person.owner_company_id is None and tick >= person.venture_cooldown:
                dev_score, dev_bdef_id = opportunity_score_developer(
                    person.resources.get('money', Decimal(0)), REGISTRY, world
                )
                dev_threshold = person_entry_threshold(person) * 0.7
                if dev_score > dev_threshold and dev_bdef_id:
                    if rng.random() < 0.25:
                        _spawn_developer_company(world, person, dev_bdef_id, REGISTRY, tick)

        for comp in list(world.companies.values()):
            if comp.owner_person_id == -1:
                continue
            if (tick + comp.instance_id) % self.EVAL_INTERVAL != 0:
                continue
            best_conv = None
            best_score = company_entry_threshold(comp)
            for conv in ConvDefs.values():
                score = opportunity_score_company(comp, conv, markets, REGISTRY)
                if score > best_score:
                    best_score = score
                    best_conv = conv
            if best_conv is not None:
                _company_expand_into(world, comp, best_conv, markets, REGISTRY, rng)


def repair_building(world: G._WorldState, building: G._BuildingInstance, paying_company_id: int) -> bool:
    """
    Attempt to repair building up one decay tier.
    Returns True if repair succeeded.
    Cannot repair NORMAL or COLLAPSED buildings this way.
    COLLAPSED buildings can only be demolished.
    """
    if building.decay_state == G.DecayState.NORMAL:
        return False  # nothing to repair
    if building.decay_state == G.DecayState.COLLAPSED:
        return False  # must demolish instead

    bdef = building.definition
    sqft = bdef.floor_area * bdef.floors
    state_key = building.decay_state.value
    cost_per_sqft = bdef.repair_cost_per_sqft.get(state_key)
    if cost_per_sqft is None:
        return False  # no repair cost defined for this state

    total_cost = cost_per_sqft * Decimal(sqft)
    payer = world.companies.get(paying_company_id)
    if not payer:
        return False
    if payer.resources.get('money', Decimal(0)) < total_cost:
        return False  # can't afford it

    payer.resources['money'] -= total_cost

    # Move up one decay tier
    current_idx = DECAY_STATE_ORDER.index(building.decay_state)
    if current_idx > 0:
        building.decay_state = DECAY_STATE_ORDER[current_idx - 1]
        building.condition = 0.5  # partially restored
    return True


def demolish_building(world: G._WorldState, building: G._BuildingInstance) -> None:
    """
    Demolish a building, freeing land and recovering minimal salvage value.
    Only valid for COLLAPSED buildings or as an explicit owner choice.
    """
    bdef = building.definition
    sqft = bdef.floor_area * bdef.floors
    # Salvage: $5/sqft for collapsed, proportional fraction of cost for others
    if building.decay_state == G.DecayState.COLLAPSED:
        salvage = Decimal(sqft) * Decimal('5')
    else:
        state_idx = DECAY_STATE_ORDER.index(building.decay_state)
        fraction = Decimal(str(max(0.0, 1.0 - state_idx * 0.2)))
        salvage = bdef.cost * fraction * Decimal('0.1')  # 10% of residual value

    owner = world.companies.get(int(building.owner_company_id))
    if owner:
        owner.resources['money'] = owner.resources.get('money', Decimal(0)) + salvage

    # Release footprint
    bootstrap.release_building_footprint(world, building)

    # Remove from world and company
    if building.instance_id in world.buildings:
        del world.buildings[building.instance_id]
    if owner and building.instance_id in owner.buildings:
        owner.buildings.remove(building.instance_id)


class RealEstateBehavior(Behavior):
    def tick(self, world, markets, marketing, tick, rng):
        self._collect_rent(world, tick)
        self._update_condition(world, rng)
        self._consider_repairs(world)
        self._consider_demolition(world)
        if tick % tick_time.TICKS_PER_DAY == 0:
            self._adjust_maintenance_levels(world)
            self._manage_listings(world, tick)

    def _adjust_maintenance_levels(self, world: G._WorldState) -> None:
        """
        Adjust maintenance_level per building based on company cash and building productivity.
        Called daily (tick % TICKS_PER_DAY == 0).
        """
        for building in world.buildings.values():
            owner = world.companies.get(int(building.owner_company_id))
            if not owner:
                continue

            cash = owner.resources.get('money', Decimal('0'))
            max_maint = building.definition.max_maintenance_cost_per_tick
            if max_maint <= 0:
                continue

            # Daily cost at full maintenance
            daily_full_cost = max_maint * Decimal('1440')

            # Is this building productive?
            has_active_machines = any(
                m.is_active and m.recipe is not None
                for u in building.units.values()
                for m in u.machines
            )
            has_tenants = any(
                unit.rent_price_per_tick and unit.rent_price_per_tick > 0 and unit.access_team_ids
                for unit in building.units.values()
            )
            is_productive = has_active_machines or has_tenants

            # Is company a landlord (no active conversions)?
            is_landlord = not owner.active_conversions and owner.buildings

            if is_landlord:
                if building.condition < 0.3:
                    target = 0.8  # recovering a degraded building
                elif building.condition > 0.8:
                    target = 0.55  # stable, optimize for profit
                else:
                    target = 0.65  # mid-range
            elif is_productive:
                cash_days = float(cash / daily_full_cost) if daily_full_cost > 0 else 999
                if cash_days > 30:
                    target = 1.0
                elif cash_days > 10:
                    target = 0.7
                elif cash_days > 3:
                    target = 0.5
                else:
                    target = 0.3
            else:
                if building.condition < 0.5 or building.decay_state.value != 'normal':
                    target = 0.6  # recover it
                else:
                    target = 0.2  # let it slowly drift, minimal cost

            # Smooth adjustment toward target (move 10% toward target per day)
            current = building.maintenance_level
            delta = (target - current) * 0.1
            building.maintenance_level = max(0.0, min(1.0, current + delta))

    def _collect_rent(self, world: G._WorldState, tick: int = 0) -> None:
        for building in world.buildings.values():
            for unit in building.units.values():
                rent = unit.rent_price_per_tick
                if not rent or rent <= 0:
                    continue
                landlord = world.companies.get(int(unit.owner_company_id))
                for tenant_id_str in unit.access_team_ids:
                    try:
                        tenant_id = int(tenant_id_str)
                    except (ValueError, TypeError):
                        continue
                    if int(unit.owner_company_id) == tenant_id:
                        continue
                    # Try company tenant first
                    tenant_comp = world.companies.get(tenant_id)
                    if tenant_comp:
                        available = tenant_comp.resources.get('money', Decimal(0))
                        payment = min(rent, available)
                        tenant_comp.resources['money'] = available - payment
                        if landlord:
                            landlord.resources['money'] = landlord.resources.get('money', Decimal(0)) + payment
                        continue
                    # Try person tenant (residential units)
                    tenant_person = world.population.get(tenant_id)
                    if tenant_person:
                        available = tenant_person.resources.get('money', Decimal(0))
                        payment = min(rent, available)
                        tenant_person.resources['money'] = available - payment
                        if landlord:
                            landlord.resources['money'] = landlord.resources.get('money', Decimal(0)) + payment
                        # If person can't pay, evict them
                        if available < rent:
                            unit.access_team_ids = [
                                tid for tid in unit.access_team_ids if tid != tenant_id_str
                            ]
                            p = world.population.get(tenant_id)
                            if p and p.home_unit_id == unit.instance_id:
                                p.home_building_id = None
                                p.home_unit_id = None
                                # Re-list the unit
                                world.list_unit_for_rent(building.instance_id, unit.instance_id,
                                                          int(unit.owner_company_id),
                                                          unit.rent_price_per_tick or Decimal('0'), tick)

    def _manage_listings(self, world: G._WorldState, tick: int) -> None:
        """Manage real estate listings for landlord companies and distressed sellers."""
        seven_days = tick_time.TICKS_PER_DAY * 7
        fourteen_days = tick_time.TICKS_PER_DAY * 14
        thirty_days = tick_time.TICKS_PER_DAY * 30

        for building in world.buildings.values():
            owner = world.companies.get(int(building.owner_company_id))
            if not owner:
                continue

            is_landlord = not owner.active_conversions and owner.buildings

            # Landlord listing management
            if is_landlord:
                for unit in building.units.values():
                    has_non_owner_tenant = any(
                        tid != str(unit.owner_company_id) for tid in unit.access_team_ids
                    )
                    # Also check person tenants for residential units
                    if not has_non_owner_tenant and unit.unit_purpose == "residential":
                        # check if any person has this as home
                        has_non_owner_tenant = any(
                            p.home_unit_id == unit.instance_id for p in world.population.values()
                        )

                    if not has_non_owner_tenant:
                        # Find existing listing
                        existing = next(
                            (lst for lst in world.real_estate_listings.values()
                             if lst.building_id == building.instance_id and lst.unit_id == unit.instance_id),
                            None
                        )
                        base_rent = building.definition.max_maintenance_cost_per_tick * Decimal('2')
                        floor_rent = building.definition.max_maintenance_cost_per_tick

                        if existing is None:
                            world.list_unit_for_rent(building.instance_id, unit.instance_id,
                                                      owner.instance_id, base_rent, tick)
                        elif tick - existing.listed_at_tick >= seven_days:
                            # No tenant after 7 days: reduce rent by 10%, floor at 1x maintenance
                            new_rent = max(floor_rent, (existing.rent_per_tick or base_rent) * Decimal('0.9'))
                            world.list_unit_for_rent(building.instance_id, unit.instance_id,
                                                      owner.instance_id, new_rent, tick)
                            unit.rent_price_per_tick = new_rent
                    else:
                        # Has active tenant: remove listing
                        world.remove_listing(building.instance_id, unit.instance_id)

        # Distressed sale: companies with financial stress and surplus buildings
        for comp in world.companies.values():
            if not comp.buildings or len(comp.buildings) <= 1:
                continue
            # Calculate daily wage + maintenance cost
            daily_wage = sum(
                world.jobs[jid].wage * Decimal('1440')
                for jid in comp.open_job_ids
                if jid in world.jobs
            ) + sum(
                world.jobs[jid].wage * Decimal('1440')
                for jid in (comp.employees or [])
                if jid in world.jobs  # jid here is person_id; use open_job_ids only
            )
            daily_cost = daily_wage + sum(
                world.buildings[bid].definition.max_maintenance_cost_per_tick * Decimal('1440')
                for bid in comp.buildings if bid in world.buildings
            )
            cash = comp.resources.get('money', Decimal(0))
            financially_stressed = cash < daily_cost * Decimal('14') and len(comp.buildings) > 1

            if financially_stressed:
                # List all buildings beyond the first
                for bid in comp.buildings[1:]:
                    b = world.buildings.get(bid)
                    if not b:
                        continue
                    asking = b.definition.cost * Decimal('0.75')
                    for unit in b.units.values():
                        existing = next(
                            (lst for lst in world.real_estate_listings.values()
                             if lst.building_id == bid and lst.unit_id == unit.instance_id
                             and lst.listing_type == G.ListingType.FOR_SALE),
                            None
                        )
                        if existing is None:
                            world.list_unit_for_sale(bid, unit.instance_id, comp.instance_id, asking, tick)
                        elif tick - existing.listed_at_tick >= fourteen_days:
                            # No buyer after 14 days: reduce by 10%, floor at 0.4x cost
                            floor_price = b.definition.cost * Decimal('0.4')
                            new_price = max(floor_price, (existing.asking_price or asking) * Decimal('0.9'))
                            world.list_unit_for_sale(bid, unit.instance_id, comp.instance_id, new_price, tick)

        # Stale listing cleanup: listings older than 30 days
        for lid, lst in list(world.real_estate_listings.items()):
            if tick - lst.listed_at_tick >= thirty_days:
                building = world.buildings.get(lst.building_id)
                if not building:
                    del world.real_estate_listings[lid]
                    continue
                unit = building.units.get(lst.unit_id)
                if not unit:
                    del world.real_estate_listings[lid]

    def _update_condition(self, world: G._WorldState, rng: random.Random) -> None:
        for building in world.buildings.values():
            if building.decay_state == G.DecayState.COLLAPSED:
                continue  # collapsed buildings don't maintain or further decay

            owner = world.companies.get(int(building.owner_company_id))
            m = building.maintenance_level  # 0.0–1.0

            # Pay maintenance cost proportional to maintenance_level
            cost = building.definition.max_maintenance_cost_per_tick * Decimal(str(m))
            if cost > 0 and owner:
                available = owner.resources.get('money', Decimal(0))
                payment = min(cost, available)
                owner.resources['money'] = available - payment
                # If company can't afford full maintenance, reduce effective level
                if cost > 0:
                    effective_m = float(payment / cost) * m
                else:
                    effective_m = m
            else:
                effective_m = m

            # Condition change: linear between full decay (m=0) and full recovery (m=1)
            # delta = (effective_m * 2 - 1) * BASE_RATE
            # At m=1.0: +BASE_RATE; at m=0.5: 0; at m=0.0: -BASE_RATE
            delta = (effective_m * 2.0 - 1.0) * BUILDING_BASE_RATE
            building.condition = max(0.0, min(1.0, building.condition + delta))

            # Decay state drop check: only when condition < 0.1 AND still declining
            if building.condition < 0.1 and delta < 0 and building.decay_state != G.DecayState.COLLAPSED:
                if rng.random() < DECAY_STATE_DROP_CHANCE:
                    current_idx = DECAY_STATE_ORDER.index(building.decay_state)
                    next_idx = current_idx + 1
                    if next_idx < len(DECAY_STATE_ORDER):
                        building.decay_state = DECAY_STATE_ORDER[next_idx]
                        building.condition = 1.0  # reset condition within new tier

    def _consider_repairs(self, world: G._WorldState) -> None:
        """Companies with degraded buildings consider paying for repairs."""
        for building in list(world.buildings.values()):
            if building.decay_state in (G.DecayState.NORMAL, G.DecayState.COLLAPSED):
                continue
            owner = world.companies.get(int(building.owner_company_id))
            if not owner:
                continue
            # Only repair if building is worth keeping (has active machines or active conversions)
            has_productive_machines = any(
                m.is_active and m.recipe is not None
                for u in building.units.values()
                for m in u.machines
            )
            is_landlord_building = not owner.active_conversions and owner.buildings
            if not has_productive_machines and not is_landlord_building:
                continue
            # Attempt repair if cash is comfortable
            bdef = building.definition
            sqft = bdef.floor_area * bdef.floors
            state_key = building.decay_state.value
            cost_per_sqft = bdef.repair_cost_per_sqft.get(state_key, Decimal('0'))
            total_cost = cost_per_sqft * Decimal(sqft)
            cash = owner.resources.get('money', Decimal(0))
            # Repair if cash > 3x the repair cost (don't drain working capital)
            if cash > total_cost * 3:
                repair_building(world, building, owner.instance_id)

    def _consider_demolition(self, world: G._WorldState) -> None:
        """Companies demolish collapsed buildings to free land."""
        for building in list(world.buildings.values()):
            if building.decay_state != G.DecayState.COLLAPSED:
                continue
            demolish_building(world, building)


class HousingBehavior(Behavior):
    """Handles residential housing search, shelter satisfaction, and rent for persons."""

    def tick(self, world, markets, marketing, tick, rng):
        if tick % tick_time.TICKS_PER_DAY != 0:
            return
        self._satisfy_shelter(world, tick)
        self._housing_search(world, tick, rng)

    def _satisfy_shelter(self, world: G._WorldState, tick: int) -> None:
        for person in world.population.values():
            if person.home_unit_id is None:
                continue
            # Verify unit still exists and person is in access_team_ids
            building = world.buildings.get(person.home_building_id) if person.home_building_id else None
            if building:
                unit = building.units.get(person.home_unit_id)
                if unit and str(person.instance_id) in unit.access_team_ids:
                    # Shelter satisfied: remove demand and reset cooldown
                    person.active_demands.pop('shelter', None)
                    d_def = DemandDefs.get('shelter')
                    if d_def and d_def.min_time_until_repeat:
                        person.demand_cooldowns['shelter'] = tick + d_def.min_time_until_repeat
                    continue
            # Unit gone or evicted — clear home
            person.home_building_id = None
            person.home_unit_id = None

    def _housing_search(self, world: G._WorldState, tick: int, rng: random.Random) -> None:
        for person in world.population.values():
            if person.home_unit_id is not None:
                continue
            if 'shelter' not in person.active_demands:
                continue
            # Find residential rental listings
            residential_listings = [
                lst for lst in world.real_estate_listings.values()
                if lst.listing_type == G.ListingType.FOR_RENT
                and world.buildings.get(lst.building_id) is not None
                and world.buildings[lst.building_id].units.get(lst.unit_id) is not None
                and world.buildings[lst.building_id].units[lst.unit_id].unit_purpose == "residential"
            ]
            if not residential_listings:
                continue
            # Pick cheapest
            best = min(residential_listings, key=lambda l: l.rent_per_tick or Decimal('999999'))
            rent_per_tick = best.rent_per_tick or Decimal('0')
            # 1 month deposit = 30 * 1440 * rent_per_tick
            deposit = rent_per_tick * Decimal('43200')
            cash = person.resources.get('money', Decimal(0))
            if cash < deposit:
                continue
            # Pay deposit to landlord
            landlord = world.companies.get(best.listed_by_company_id)
            if landlord:
                landlord.resources['money'] = landlord.resources.get('money', Decimal(0)) + deposit
            person.resources['money'] = cash - deposit
            # Rent the unit
            b = world.buildings[best.building_id]
            unit = b.units[best.unit_id]
            world.rent_unit(best.building_id, best.unit_id, str(person.instance_id), rent_per_tick)
            world.remove_listing(best.building_id, best.unit_id)
            person.home_building_id = best.building_id
            person.home_unit_id = best.unit_id
            person.active_demands.pop('shelter', None)
            d_def = DemandDefs.get('shelter')
            if d_def and d_def.min_time_until_repeat:
                person.demand_cooldowns['shelter'] = tick + d_def.min_time_until_repeat


class BehaviorManager:
    """Coordinates all Behaviors and maintains a deterministic random source."""

    def __init__(self, world: G._WorldState, markets: Dict[str, Market], marketing: MarketingMarket, seed: int = 0):
        self.world = world
        self.markets = markets
        self.marketing = marketing
        self.behaviors: List[Behavior] = []
        self.tick_count: int = 0
        self.random = random.Random(seed)

    def register(self, behavior: Behavior) -> None:
        self.behaviors.append(behavior)

    def tick(self) -> None:
        self.tick_count += 1
        for behavior in self.behaviors:
            behavior.tick(self.world, self.markets, self.marketing, self.tick_count, self.random)

# -----------------------------------
# Main Simulation Loop
# -----------------------------------

def main(ticks: int = 1, seed: int = 0) -> None:
    """Run the simulation for a number of ticks using the given RNG seed."""

    world = load_world()
    bootstrap.ensure_world_bootstrap(world, REGISTRY, ResourceDefs, random.Random(seed))
    markets = {res_id: Market(world, res_id) for res_id in ResourceDefs}
    for m in markets.values():
        m.markets = markets
    marketing_market = MarketingMarket(world)
    bm = BehaviorManager(world, markets, marketing_market, seed=seed)

    bm.register(MarketBehavior())
    bm.register(CompanyBehavior())
    bm.register(LogisticsBehavior())
    bm.register(VehicleBehavior())
    bm.register(MachineBehavior())
    bm.register(ExpirationBehavior())
    bm.register(RealEstateBehavior())
    bm.register(HousingBehavior())
    bm.register(PersonBehavior())
    bm.register(LifecycleBehavior())
    bm.register(DiscoveryBehavior())
    bm.register(MarketingBehavior())
    bm.register(EntrepreneurshipBehavior())
    bm.register(AnalyticsBehavior())
    bm.register(BankBehavior())

    for _ in range(ticks):
        bm.tick()
    save_world(world)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run economic simulation")
    parser.add_argument("--ticks", type=int, default=10, help="Number of ticks to run")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for deterministic runs")
    args = parser.parse_args()

    main(ticks=args.ticks, seed=args.seed)
