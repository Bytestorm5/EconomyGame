import json
from pathlib import Path
from decimal import Decimal
from typing import Dict, Optional, List, Tuple, Literal
from math import ceil
import random

from pydantic import BaseModel, Field

from register import register_content, MOD_PATHS, LOCAL_CONTENT
import objects as G
from objects import get_instance_id
# Constants
WORLD_STATE_PATH = Path("world_state.json")
ALL_SOURCES = [LOCAL_CONTENT] + MOD_PATHS

# Load registry
REGISTRY = register_content(ALL_SOURCES)
ResourceDefs: Dict[str, G.Resource] = REGISTRY.get("Resource", {}) # type: ignore
PresetDefs: Dict[str, G.ResourceMarketPreset] = REGISTRY.get("ResourceMarketPreset", {}) # type: ignore
ConvDefs: Dict[str, G.ResourceConversion] = REGISTRY.get("ResourceConversion", {}) # type: ignore
DemandDefs: Dict[str, G.ResourceDemand] = REGISTRY.get("ResourceDemand", {}) # type: ignore

def _update_fair_value(agent: G._FinancialEntityInstance, resource_id: str, market_price: Decimal,
                       rng: random.Random, lr: float, noise_scale: float) -> Decimal:
    """
    Evolve the agent’s fair-value estimate toward the observed price,
    with a little zero-mean noise (models imperfect information).
    """
    current = agent.fair_values.get(resource_id, market_price)
    error   = market_price - current
    noise   = Decimal(str(rng.uniform(-noise_scale, noise_scale))) * market_price
    new_val = current + Decimal(str(lr)) * error + noise        # simple exponential learning
    agent.fair_values[resource_id] = max(Decimal("0.01"), new_val)
    return new_val

def evaluate_condition(cond: Optional[G.ConditionBlock], world: G._WorldState, person: Optional[G._PersonInstance] = None) -> bool:
    # Placeholder for future DSL evaluation
    return True

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

    def exec_orders(self, tick: int, rng: random.Random) -> None:
        buys = [o for o in self.order_book if o.timestamp == tick]
        total_qty = sum(o.quantity for o in buys)
        if total_qty == 0:
            return
        # simple dynamic pricing: price increases with demand
        self.price = self.price * (Decimal("1") + Decimal(str(total_qty)) / Decimal("100"))
        population = list(self.world.population.values())
        for order in buys:
            for _ in range(order.quantity):
                if not population:
                    break
                person = rng.choice(population)
                self._apply_impression(person, order.company_id, rng)
        self.order_book = [o for o in self.order_book if o.timestamp > tick]

    def update_prices(self, world: G._WorldState, tick: int, rng: random.Random) -> None:
        self.exec_orders(tick, rng)

    def _apply_impression(self, person: G._PersonInstance, company_id: int, rng: random.Random) -> None:
        roll = rng.random()
        if roll < 0.15 * person.personality.marketing_susceptibility:
            if company_id not in person.known_actors:
                person.known_actors.append(company_id)
        elif roll < 0.20 * person.personality.marketing_susceptibility:
            if company_id not in person.known_actors:
                person.known_actors.append(company_id)
            self._share_information(person, company_id, rng)
        elif roll > 0.98:
            if company_id in person.known_actors:
                person.known_actors.remove(company_id)
            self._share_information(person, company_id, rng, negative=True)

    def _share_information(self, person: G._PersonInstance, company_id: int, rng: random.Random, negative: bool = False) -> None:
        for other_id, rel in person.personal_relationship.items():
            if rng.random() < rel * person.personality.info_sharing:
                other = self.world.population.get(other_id)
                if not other:
                    continue
                if negative:
                    if company_id in other.known_actors:
                        other.known_actors.remove(company_id)
                else:
                    if company_id not in other.known_actors:
                        other.known_actors.append(company_id)

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
        # price at start of current tick (used for cross-price effects)
        self.prev_price: Decimal = self.price
        # Will be set externally to access other markets (e.g., fuel)
        self.markets: Dict[str, Market] = {}
        self.volume_weighted_price: Decimal = self.price        # <── NEW

    def compute_supply_demand(self, tick: int) -> Tuple[Decimal, Decimal]:
        supply = sum(
            o.quantity for o in self.order_book.orders
            if o.timestamp == tick and o.side == "sell"
        )
        demand = sum(
            o.quantity for o in self.order_book.orders
            if o.timestamp == tick and o.side == "buy"
        )
        return Decimal(supply), Decimal(demand)

    def exec_orders(self, world: G._WorldState, tick: int) -> List[Tuple[Order, Order, Decimal, Decimal]]:
        buys = sorted(
            (o for o in self.order_book.orders if o.side == "buy" and o.timestamp == tick),
            key=lambda o: (-o.limit_price, o.timestamp)
        )
        all_sells = [o for o in self.order_book.orders if o.side == "sell" and o.timestamp == tick]        
        trades: List[Tuple[Order, Order, Decimal, Decimal]] = []
        for buy in buys:
            known_actors = []
            if buy.actor_id in world.companies:
                known_actors =  world.companies[buy.actor_id].known_actors
            else:
                known_actors =  world.population[buy.actor_id].known_actors
            # Limit this buyer's market to sellers that they know about
            sells = sorted(
                [s for s in all_sells if s.actor_id in known_actors],
                key=lambda o: (self.get_effective_price(buy, o.actor_id), o.timestamp)
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
                    # share knowledge of trading partner
                    buyer = world.companies.get(buy.actor_id) if buy.actor_type == 'company' else world.population.get(buy.actor_id)
                    seller = world.companies.get(sell.actor_id) if sell.actor_type == 'company' else world.population.get(sell.actor_id)
                    if buyer and sell.actor_id not in buyer.known_actors:
                        buyer.known_actors.append(sell.actor_id)
                    if seller and buy.actor_id not in seller.known_actors:
                        seller.known_actors.append(buy.actor_id)
        self.order_book.orders = [o for o in self.order_book.orders if o.quantity > 0]
        return trades
    
    def update_prices(self, world: G._WorldState, tick: int, rng: random.Random) -> None:
        """
        Drop-in replacement that eliminates the downward price drift.

        Key change: use the demand-to-supply ratio (1.0 = balance) instead of the
        old (d − s)/s difference term.  Cross-elasticities, menu cost checks, and
        price-stickiness all behave exactly as before.
        """
        # Respect price-stickiness
        if not self.preset or tick - self.last_update < self.preset.price_stickiness:
            self.prev_price = self.price
            return

        supply, demand = self.compute_supply_demand(tick)
        old_price = self.price

        # ----------------------------------------------------------
        # 0-supply or 0-demand special-cases
        # ----------------------------------------------------------
        if supply == 0 and demand > 0:
            # Walk upward toward the highest bid (limit-up style)
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
            # Walk downward toward the lowest ask (limit-down style)
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
            # Market with NUTHIN?
            self.price = self.prev_price
            return

        trades = self.exec_orders(world, tick)
        
        if trades:
            tot_qty  = sum(qty for *_ , qty in trades)
            wtd_sum  = sum(price * qty for *_ , price, qty in trades)
            tick_vwap = wtd_sum / tot_qty

            α = Decimal("0.25")                  # 25 % weight on the latest tick; tweak as you like
            self.volume_weighted_price = ((1-α) * self.volume_weighted_price) + (α * tick_vwap)
        else:
            # no trades this tick → nudge very slowly toward the quoted spot
            self.volume_weighted_price = (self.volume_weighted_price + self.price) / Decimal(2)
            
        self.prev_price = self.price
        for buy, sell, price, qty in trades:
            # Buy Side
            if buy.actor_type == 'company':
                comp = world.companies.get(buy.actor_id)
            else:
                comp = world.population.get(buy.actor_id)
            if comp:
                comp.resources[buy.resource_id] = comp.resources.get(buy.resource_id, Decimal(0)) + qty
                comp.resources['money'] = comp.resources.get('money', Decimal(0)) - price * qty
            self.price = price
            
            # Sell Side
            if sell.actor_type == 'company':
                comp = world.companies.get(buy.actor_id)
            else:
                comp = world.population.get(buy.actor_id)
            if comp:
                comp.resources[sell.resource_id] = comp.resources.get(sell.resource_id, Decimal(0)) - qty
                comp.resources['money'] = comp.resources.get('money', Decimal(0)) + price * qty
            self.price = price
    
    # def get_available_supply(self, actor_type: str) -> Decimal:
    #     total = Decimal(0)
    #     for b in self.world.buildings.values():
    #         btype = getattr(b.definition, 'building_type', None)
    #         if actor_type == 'person' and btype != 'Retail':
    #             continue
    #         inv = getattr(b, 'inventory', {})
    #         total += inv.get(self.resource_id, Decimal(0))
    #     return total

    def estimate_shipping_cost(self, company_id: int, quantity: Decimal, supplier_id: int) -> Decimal:
        """
        Calculate per-unit shipping cost for buying `quantity` from a specific supplier.
        """
        res_def = ResourceDefs.get(self.resource_id)
        if not res_def or not getattr(res_def, 'is_physical', False):
            return Decimal("0")
        # Gather idle/parked vehicles belonging to company
        vehicles = [
            v
            for v in self.world.vehicles.values()
            if v.owner_company_id == company_id and v.status in {"idle", "parked"}
        ]
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
        vehicle_info: List[Tuple[G._VehicleInstance, int, Decimal]] = []
        for v in vehicles:
            cap = int(v.vehicle_type.cargo_inventory_size)
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
        ...

    _LR            = 0.25      # learning-rate toward market
    _NOISE_SCALE   = 0.02      # ±2 % random observation error
    _THRESHOLD_PCT = Decimal("0.03")   # 3 % mis-pricing before acting
    _SPEC_FRAC     = Decimal("0.30")   # deploy up to 30 % of cash / stock

    def tick(self, world, markets, marketing, tick, rng):
        for comp in world.companies.values():
            cash = comp.resources.get("money", Decimal(0))
            for rid, mkt in markets.items():
                fv = _update_fair_value(comp, rid, mkt.volume_weighted_price, rng,
                                        self._LR, self._NOISE_SCALE)
                mis_pct = (fv - mkt.price) / mkt.price
                if mis_pct > self._THRESHOLD_PCT and cash > 0:           # buy low
                    budget   = cash * self._SPEC_FRAC
                    qty      = (budget / mkt.price).quantize(Decimal("1."))
                    if qty > 0:
                        mkt.order_book.submit(Order(actor_type="company",
                                                    actor_id=comp.instance_id,
                                                    resource_id=rid,
                                                    quantity=qty,
                                                    limit_price=fv,
                                                    side="buy",
                                                    timestamp=tick))
                elif mis_pct < -self._THRESHOLD_PCT:                     # sell high
                    stock = comp.resources.get(rid, Decimal(0))
                    qty   = (stock * self._SPEC_FRAC).quantize(Decimal("1."))
                    if qty > 0:
                        mkt.order_book.submit(Order(actor_type="company",
                                                    actor_id=comp.instance_id,
                                                    resource_id=rid,
                                                    quantity=qty,
                                                    limit_price=fv,
                                                    side="sell",
                                                    timestamp=tick))

class PersonBehavior(Behavior):
    def __init__(self) -> None:
        ...

    _LR            = 0.05
    _NOISE_SCALE   = 0.10      # ±10 % observational noise
    _THRESHOLD_PCT = Decimal("0.10")   # need big gap to speculate
    _SPEC_FRAC     = Decimal("0.10")

    def tick(self, world, markets, marketing, tick, rng):
        for p in world.population.values():
            cash = p.resources.get("money", Decimal(0))
            # surface new demands
            for demand in DemandDefs.values():
                if demand.id in p.active_demands:
                    continue
                if tick < p.demand_cooldowns.get(demand.id, 0):
                    continue
                if evaluate_condition(demand.demand_can_occur, world, p) and rng.random() <= demand.chance_per_tick:
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
                res = min(options, key=lambda r: markets[r.id].price / Decimal(str(r.fulfills_demand[d_id])))
                price_adj = Decimal(1) + Decimal(str(p.personality.comfort_value * 0.05 + p.personality.time_value * 0.05))
                limit = markets[res.id].price * price_adj
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
                for other_id, rel in p.personal_relationship.items():
                    if rng.random() < rel * p.personality.info_sharing:
                        other = world.population.get(other_id)
                        if other and actor_id not in other.known_actors:
                            other.known_actors.append(actor_id)

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
                    m.wait_for_tick = tick + int(cycle)
                    # produce outputs
                    for res_id, amt in conv.output_resources.items():
                        out_stack = G._ResourceStack(amount=int(amt), cost=0.0, resource_id=res_id)
                        remainder = m.inventory.put(out_stack, fill=False)
                        if remainder is not None:
                            # if full put failed, fill partially
                            m.inventory.put(out_stack, fill=True)

class VehicleBehavior(Behavior):
    """Moves vehicles, consumes fuel, and handles loading/unloading."""

    def _find_building_at(self, world: G._WorldState, pos: Tuple[int, int]) -> Optional[G._BuildingInstance]:
        parcel = next((p for p in world.land.values() if p.x == pos[0] and p.y == pos[1]), None)
        if parcel and parcel.building_id is not None:
            return world.buildings.get(int(parcel.building_id))
        return None

    def _transfer_all(self, src: G._Inventory, dst: G._Inventory) -> bool:
        moved = False
        for stack in list(src.stacks):
            src.stacks.remove(stack)
            remainder = dst.put(stack, fill=True)
            if remainder and remainder.amount > 0:
                src.stacks.append(remainder)
            moved = True
        return moved

    def tick(
        self,
        world: G._WorldState,
        markets: Dict[str, Market],
        marketing: MarketingMarket,
        tick: int,
        rng: random.Random,
    ) -> None:
        for v in world.vehicles.values():
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
                        break
                v.position = (x, y)
                if v.destination and (x, y) == v.destination:
                    v.destination = None
                    v.status = "idle"

            elif v.status in {"loading", "unloading"}:
                bld = self._find_building_at(world, v.position)
                if not bld:
                    v.status = "idle"
                    continue
                container = None
                for unit in bld.units.values():
                    for m in unit.machines:
                        if m.machine.is_container:
                            container = m.inventory
                            break
                    if container:
                        break
                if not container:
                    v.status = "idle"
                    continue
                if v.status == "loading":
                    moved = self._transfer_all(container, v.inventory)
                else:
                    moved = self._transfer_all(v.inventory, container)
                if not moved:
                    v.status = "idle"

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
    markets = {res_id: Market(world, res_id) for res_id in ResourceDefs}
    for m in markets.values():
        m.markets = markets
    marketing_market = MarketingMarket(world)
    bm = BehaviorManager(world, markets, marketing_market, seed=seed)

    bm.register(MarketBehavior())
    bm.register(CompanyBehavior())
    bm.register(PersonBehavior())
    bm.register(MarketingBehavior())
    bm.register(VehicleBehavior())

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
