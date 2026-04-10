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
                    [s for s in all_sells if s.actor_id in known_actors],
                    key=lambda s: (
                        self.total_unit_cost_for_match(buy, s) * (1 - buyer.brand_loyalty.get(s.actor_id, 0.0))
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

        free = [p for p in world.land.values() if p.state_id == G.LandState.WILD_BUILDABLE and p.building_id is None]
        if not free:
            return
        parcel = random.choice(free)

        binst = G._BuildingInstance(
            land_parcel_id=int(parcel.id.split('-')[0]) if '-' in parcel.id else 0,
            definition=bdef,
            owner_company_id=str(comp.instance_id),
            units={},
            loading_bay_occupants=[],
            parking_occupants=[],
            origin=parcel.pos_tuple()
        )
        world.buildings[binst.instance_id] = binst
        parcel.state_id = G.LandState.BUILDING
        parcel.building_id = str(binst.instance_id)
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

        vdefs = REGISTRY.get("VehicleDefinition", {})
        if vdefs and not any(v.owner_company_id == comp.instance_id for v in world.vehicles.values()):
            vdef = next(iter(vdefs.values()))
            v = G._VehicleInstance(
                vehicle_type=vdef,
                owner_company_id=comp.instance_id,
                position=parcel.pos_tuple(),
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
            self._list_inventory_for_sale(comp, markets, rng, tick)
            self._ensure_marketing(comp, marketing, tick)
            self._pay_wages(comp, world)
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
            person.works_at = company.buildings[0]
        person.brand_loyalty[company.instance_id] = min(1.0, person.brand_loyalty.get(company.instance_id, 0.3) + 0.1)
    def tick(self, world, markets, marketing, tick, rng):
        for p in world.population.values():
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
                        float(markets[r.id].price)
                        / float(r.fulfills_demand[d_id])
                    )
                    / (p.resource_preferences.get(r.id, 0.5) + 0.5),
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
                for other_id, rel in p.personal_relationship.items():
                    if rng.random() < rel * p.personality.info_sharing:
                        other = world.population.get(other_id)
                        if other and actor_id not in other.known_actors:
                            other.known_actors.append(actor_id)
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
                    m.wait_for_tick = tick + int(cycle)
                    # produce outputs
                    for res_id, amt in conv.output_resources.items():
                        out_stack = G._ResourceStack(amount=int(amt), cost=1.0, resource_id=res_id)
                        remainder = m.inventory.put(out_stack, fill=False)
                        if remainder is not None:
                            # if full put failed, fill partially
                            m.inventory.put(out_stack, fill=True)

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
    bm.register(PersonBehavior())
    bm.register(MarketingBehavior())
    bm.register(AnalyticsBehavior())

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
