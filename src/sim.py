import json
from pathlib import Path
from decimal import Decimal
from typing import Dict, Optional, List, Tuple, Literal
from math import ceil

from pydantic import BaseModel, Field

from register import register_content, MOD_PATHS, LOCAL_CONTENT
import objects as G
from objects import get_instance_id

# Constants
WORLD_STATE_PATH = Path("world_state.json")
ALL_SOURCES = [LOCAL_CONTENT] + MOD_PATHS

# Load registry
REGISTRY = register_content(ALL_SOURCES)
ResourceDefs: Dict[str, G.Resource] = REGISTRY.get("Resource", {})
PresetDefs: Dict[str, G.ResourceMarketPreset] = REGISTRY.get("ResourceMarketPreset", {})
ConvDefs: Dict[str, G.ResourceConversion] = REGISTRY.get("ResourceConversion", {})

# -----------------------------------
# Persistence
# -----------------------------------

def load_world() -> G._WorldState:
    if WORLD_STATE_PATH.exists():
        raw = WORLD_STATE_PATH.read_text(encoding="utf-8")
        return G._WorldState.model_validate_json(raw)
    return G._WorldState()


def save_world(world: G._WorldState) -> None:
    WORLD_STATE_PATH.write_text(world.model_dump_json(), encoding="utf-8")

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

    def match(self, tick: int) -> List[Tuple[Order, Order, Decimal, Decimal]]:
        buys = sorted(
            (o for o in self.orders if o.side == "buy" and o.timestamp == tick),
            key=lambda o: (-o.limit_price, o.timestamp)
        )
        sells = sorted(
            (o for o in self.orders if o.side == "sell" and o.timestamp == tick),
            key=lambda o: (o.limit_price, o.timestamp)
        )
        trades: List[Tuple[Order, Order, Decimal, Decimal]] = []
        for buy in buys:
            for sell in sells:
                if buy.resource_id != sell.resource_id or buy.quantity <= 0 or sell.quantity <= 0:
                    continue
                if buy.limit_price >= sell.limit_price:
                    price = (buy.limit_price + sell.limit_price) / Decimal(2)
                    trade_qty = min(buy.quantity, sell.quantity)
                    trades.append((buy, sell, price, trade_qty))
                    buy.quantity -= trade_qty
                    sell.quantity -= trade_qty
        self.orders = [o for o in self.orders if o.quantity > 0]
        return trades

# -----------------------------------
# Market (Single Resource)
# -----------------------------------

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
        # Will be set externally to access other markets (e.g., fuel)
        self.markets: Dict[str, Market] = {}

    def compute_supply_demand(self, tick: int) -> Tuple[Decimal, Decimal]:
        supply = sum(
            o.quantity for o in self.order_book.orders
            if o.timestamp == tick and o.side == "sell"
        )
        demand = sum(
            o.quantity for o in self.order_book.orders
            if o.timestamp == tick and o.side == "buy"
        )
        return supply, demand

    def update_prices(self, tick: int) -> None:
        if not self.preset or tick - self.last_update < self.preset.price_stickiness:
            return
        s, d = self.compute_supply_demand(tick)
        old_price = self.price
        if s > 0:
            delta = (d - s) / s
            change = (old_price * (delta ** (Decimal(1) / self.preset.elasticity))) - old_price
            if abs(change) < self.preset.menu_cost:
                return
            new_price = old_price + change * self.adjustment_rate
            self.price = max(Decimal("0.01"), new_price)
            self.last_update = tick

    def get_available_supply(self, actor_type: str) -> Decimal:
        total = Decimal(0)
        for b in self.world.buildings.values():
            btype = getattr(b.definition, 'building_type', None)
            if actor_type == 'person' and btype != 'Retail':
                continue
            inv = getattr(b, 'inventory', {})
            total += inv.get(self.resource_id, Decimal(0))
        return total

    def estimate_shipping_cost(self, company_id: int, quantity: Decimal, supplier_id: int) -> Decimal:
        """
        Calculate per-unit shipping cost for buying `quantity` from a specific supplier.
        """
        res_def = ResourceDefs.get(self.resource_id)
        if not res_def or not getattr(res_def, 'is_physical', False):
            return Decimal("0")
        # Gather parked vehicles belonging to company
        vehicles = [v for v in self.world.vehicles.values()
                    if v.owner_company_id == company_id and getattr(v, 'is_parked', False)]
        if not vehicles:
            return Decimal("0")
        # Pre-calculate fuel price
        fuel_market = self.markets.get('fuel')
        fuel_price = fuel_market.price if fuel_market else Decimal("1")
        # Determine supplier building location
        supplier_buildings = [b for b in self.world.buildings.values() if b.instance_id == supplier_id]
        if not supplier_buildings:
            return Decimal("0")
        parcels = [p for p in self.world.land.values() if p.building_id == supplier_id]
        if not parcels:
            return Decimal("0")
        sbx, sby = parcels[0].x, parcels[0].y
        # Collect vehicles with (capacity, distance)
        vehicle_info: List[Tuple[G.Vehicle, int, Decimal]] = []
        for v in vehicles:
            cap = v.cargo_inventory_size
            vx, vy = v.position
            dist = Decimal(abs(sbx - vx) + abs(sby - vy))
            vehicle_info.append((v, cap, dist))
        if not vehicle_info:
            return Decimal("0")
        # Sort by proximity
        vehicle_info.sort(key=lambda x: x[2])
        remaining = quantity
        total_cost = Decimal(0)
        # Greedy fill
        for v, cap, dist in vehicle_info:
            if remaining <= 0:
                break
            take = min(remaining, cap)
            trips = ceil(float(take / cap)) if cap > 0 else 0
            consume = v.vehicle_type.fuel_consumption
            total_cost += dist * Decimal(2) * consume * fuel_price * trips
            remaining -= take
        # Extra trips if needed
        if remaining > 0:
            v, cap, dist = vehicle_info[0]
            trips = ceil(float(remaining / cap)) if cap > 0 else 0
            consume = v.vehicle_type.fuel_consumption
            total_cost += dist * Decimal(2) * consume * fuel_price * trips
        return total_cost / quantity if quantity > 0 else Decimal(0)

    def get_effective_price(self, order: Order, supplier_id: int) -> Decimal:
        """
        Returns the total per-unit cost including shipping for a buy order from a given supplier.
        """
        base_price = order.limit_price
        shipping = self.estimate_shipping_cost(order.actor_id, order.quantity, supplier_id)
        return base_price + shipping

# -----------------------------------
# AI Behavior Hooks
# -----------------------------------

class Behavior:
    def tick(self, world: G._WorldState, markets: Dict[str, Market], tick: int):
        raise NotImplementedError

class MarketBehavior(Behavior):
    def tick(self, world: G._WorldState, markets: Dict[str, Market], tick: int):
        for market in markets.values():
            market.update_prices(tick)
            trades = market.order_book.match(tick)
            for buy, sell, price, qty in trades:
                if buy.actor_type == 'company':
                    comp = world.companies.get(buy.actor_id)
                    if comp:
                        comp.resources[buy.resource_id] = comp.resources.get(buy.resource_id, Decimal(0)) + qty
                        comp.resources['money'] = comp.resources.get('money', Decimal(0)) - price * qty
                if sell.actor_type == 'company':
                    comp = world.companies.get(sell.actor_id)
                    if comp:
                        comp.resources[sell.resource_id] = comp.resources.get(sell.resource_id, Decimal(0)) - qty
                        comp.resources['money'] = comp.resources.get('money', Decimal(0)) + price * qty

class CompanyBehavior(Behavior):
    def tick(self, world: G._WorldState, markets: Dict[str, Market], tick: int):
        for comp in world.companies.values():
            for conv in ConvDefs.values():
                for res_id, amt in conv.input_resources.items():
                    market = markets[res_id]
                    # When placing orders, base limit_price on fair price (excluding specific supplier)
                    price = market.price
                    order = Order(
                        actor_type='company', actor_id=comp.instance_id,
                        resource_id=res_id, quantity=amt,
                        limit_price=price, side='buy', timestamp=tick
                    )
                    market.order_book.submit(order)

class PersonBehavior(Behavior):
    def tick(self, world: G._WorldState, markets: Dict[str, Market], tick: int):
        for person in world.population.values():
            for res_id, market in markets.items():
                res_def = ResourceDefs.get(res_id)
                if not getattr(res_def, 'is_physical', False):
                    continue
                if market.get_available_supply('person') >= Decimal(1):
                    order = Order(
                        actor_type='person', actor_id=person.instance_id,
                        resource_id=res_id, quantity=Decimal(1),
                        limit_price=market.price, side='buy', timestamp=tick
                    )
                    market.order_book.submit(order)

class MachineBehavior(Behavior):
    """
    Handles production: consumes inputs and adds outputs to each active machine's inventory
    """
    def tick(self, world: G._WorldState, markets: Dict[str, Market], tick: int):
        for building in world.buildings.values():
            for unit in building.units.values():
                for m in unit.machines:
                    if not m.is_active:
                        continue
                    conv = ConvDefs.get(m.recipe)
                    # determine cycle time
                    cycle = m.machine.possible_recipes.get(m.recipe) or conv.default_time_taken
                    # only run at multiples of cycle
                    if m.wait_for_tick < tick:
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
                    m.wait_for_tick = tick + cycle
                    # produce outputs
                    for res_id, amt in conv.output_resources.items():
                        out_stack = G._ResourceStack(amount=int(amt), cost=0.0, resource_id=res_id)
                        remainder = m.inventory.put(out_stack, fill=False)
                        if remainder is not None:
                            # if full put failed, fill partially
                            m.inventory.put(out_stack, fill=True)

class BehaviorManager:
    def __init__(self, world: G._WorldState, markets: Dict[str, Market]):
        self.world = world
        self.markets = markets
        self.behaviors: List[Behavior] = []
        self.tick_count: int = 0

    def register(self, behavior: Behavior) -> None:
        self.behaviors.append(behavior)

    def tick(self) -> None:
        self.tick_count += 1
        for behavior in self.behaviors:
            behavior.tick(self.world, self.markets, self.tick_count)

# -----------------------------------
# Main Simulation Loop
# -----------------------------------

def main(ticks: int = 1) -> None:
    world = load_world()
    markets = {res_id: Market(world, res_id) for res_id in ResourceDefs}
    for m in markets.values():
        m.markets = markets
    bm = BehaviorManager(world, markets)

    bm.register(MarketBehavior())
    bm.register(CompanyBehavior())
    bm.register(PersonBehavior())

    for _ in range(ticks):
        bm.tick()
    save_world(world)

if __name__ == "__main__":
    main(ticks=10)
