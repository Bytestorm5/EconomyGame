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
    # optional building context for building-level orders
    building_id: Optional[int] = None
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
        # Historical prices for trend analysis (speculation)
        self.history: List[Decimal] = [self.price]

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
        # Adjust price if preset allows (respecting stickiness)
        if self.preset and tick - self.last_update >= self.preset.price_stickiness:
            s, d = self.compute_supply_demand(tick)
            old_price = self.price
            if s > 0:
                delta = (d - s) / s
                change = (old_price * (delta ** (Decimal(1) / self.preset.elasticity))) - old_price
                if abs(change) >= self.preset.menu_cost:
                    new_price = old_price + change * self.adjustment_rate
                    self.price = max(Decimal("0.01"), new_price)
                    self.last_update = tick
        # Record price history for speculative behavior
        self.history.append(self.price)

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
    
    def max_buy_price(self, actor_type: str, actor_id: int) -> Decimal:
        """
        Maximum price a buyer is willing to pay based on need tiers (for persons) or base price (for companies).
        """
        base = self.price
        if actor_type == 'person':
            person = self.world.population.get(actor_id)
            if not person:
                return base
            # current inventory amount of this resource
            current = sum(s.amount for s in person.inventory.stacks if s.resource_id == self.resource_id)
            # thresholds for needs
            thresholds = ResourceDefs[self.resource_id].needs
            # base tier multiplier
            if current < thresholds.get('survival', Decimal(0)):
                multiplier = Decimal('2')
            elif current < thresholds.get('comfort', Decimal(0)):
                multiplier = Decimal('1.2')
            elif current < thresholds.get('luxury', Decimal(0)):
                multiplier = Decimal('1.0')
            else:
                multiplier = Decimal('0.5')
            # adjust based on personality traits: emotionality, openness, conscientiousness
            emo = Decimal(str(person.personality.emotionality))
            opn = Decimal(str(person.personality.openness))
            conc = Decimal(str(person.personality.conscientiousness))
            # increase willingness to pay if anxious or open to experience, decrease if conscientious
            adj = Decimal('1') + emo * Decimal('0.1') + opn * Decimal('0.05') - conc * Decimal('0.05')
            if adj <= 0:
                adj = Decimal('0.1')
            multiplier *= adj
            return base * multiplier
        # companies pay base price
        return base

    def min_sell_price(self, actor_type: str, actor_id: int) -> Decimal:
        """
        Minimum price a seller is willing to accept, based on cost of production (for companies)
        or surplus (for persons).
        """
        base = self.price
        if actor_type == 'company':
            # estimate production cost per unit for this resource
            candidates = [conv for conv in ConvDefs.values() if self.resource_id in conv.output_resources]
            if not candidates:
                return base
            best_cost = None
            for conv in candidates:
                out_qty = conv.output_resources[self.resource_id]
                # sum input costs at current market prices
                input_cost = sum(
                    amt * self.markets[input_res].price
                    for input_res, amt in conv.input_resources.items()
                )
                unit_cost = input_cost / out_qty if out_qty > 0 else Decimal(0)
                if best_cost is None or unit_cost < best_cost:
                    best_cost = unit_cost
            static = self.preset.transaction_cost if self.preset else Decimal(0)
            base_price = (best_cost or Decimal(0)) + static
            # adjust by recent profitability
            comp = self.world.companies.get(actor_id)
            if comp and comp.profit_history:
                recent = comp.profit_history[-10:]
                avg_profit = sum(recent) / Decimal(len(recent))
                # if profitable, add 5% margin; if not, reduce by 5%
                factor = Decimal('1.05') if avg_profit > 0 else Decimal('0.95')
                return base_price * factor
            return base_price
        if actor_type == 'person':
            person = self.world.population.get(actor_id)
            if not person:
                return base
            current = sum(s.amount for s in person.inventory.stacks if s.resource_id == self.resource_id)
            thresholds = ResourceDefs[self.resource_id].needs
            # only sell surplus above comfort
            if current > thresholds.get('comfort', Decimal(0)):
                return base
            # otherwise, discourage selling
            return base * Decimal('1.5')
        return base

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
                # Company buy: update resources at building or company, and record loss
                if buy.actor_type == 'company':
                    comp = world.companies.get(buy.actor_id)
                    if comp:
                        if buy.building_id is not None:
                            bld = world.buildings.get(buy.building_id)
                            if bld:
                                bld.resources[buy.resource_id] = bld.resources.get(buy.resource_id, Decimal(0)) + qty
                        else:
                            comp.resources[buy.resource_id] = comp.resources.get(buy.resource_id, Decimal(0)) + qty
                        # deduct money and record loss
                        comp.resources['money'] = comp.resources.get('money', Decimal(0)) - price * qty
                        comp.profit_history.append(-price * qty)
                # Company sell: update resources and record profit
                if sell.actor_type == 'company':
                    comp = world.companies.get(sell.actor_id)
                    if comp:
                        comp.resources[sell.resource_id] = comp.resources.get(sell.resource_id, Decimal(0)) - qty
                        comp.resources['money'] = comp.resources.get('money', Decimal(0)) + price * qty
                        comp.profit_history.append(price * qty)
                # Person buy: add to inventory, deduct money, update brand impressions
                if buy.actor_type == 'person':
                    person = world.population.get(buy.actor_id)
                    if person:
                        # add purchased resources to inventory
                        stack = G._ResourceStack(amount=int(qty), cost=float(price * qty), resource_id=buy.resource_id)
                        person.inventory.put(stack, fill=False)
                        # adjust money resource
                        person.resources['money'] = person.resources.get('money', Decimal(0)) - price * qty
                        # brand update if buying from a company
                        if sell.actor_type == 'company':
                            seller_id = sell.actor_id
                            # positive experience if price at or below market price
                            market = markets.get(buy.resource_id)
                            delta = Decimal('0.1') if market and price <= market.price else Decimal('-0.1')
                            prev = person.brand_impressions.get(seller_id, Decimal(0))
                            person.brand_impressions[seller_id] = prev + delta
                # Person sell: remove from inventory, add money
                if sell.actor_type == 'person':
                    person = world.population.get(sell.actor_id)
                    if person:
                        demand = G._ResourceStack(amount=int(qty), cost=0.0, resource_id=sell.resource_id)
                        person.inventory.take(demand, partial=False)
                        person.resources[sell.resource_id] = person.resources.get(sell.resource_id, Decimal(0)) - qty
                        person.resources['money'] = person.resources.get('money', Decimal(0)) + price * qty
                # Create shipments for physical goods
                res_def = ResourceDefs.get(buy.resource_id)
                if res_def and getattr(res_def, 'is_physical', False) and sell.actor_type == 'company' and buy.actor_type in ('person', 'company'):
                    shipment = G._Shipment(
                        resource_id=buy.resource_id,
                        quantity=qty,
                        origin_building_id=sell.building_id,
                        dest_building_id=buy.building_id if buy.actor_type == 'company' else None,
                        dest_person_id=buy.actor_id if buy.actor_type == 'person' else None,
                    )
                    world.shipments.append(shipment)

class DeliveryBehavior(Behavior):
    """
    Handles physical shipments by assigning vehicles to transport goods.
    """
    def tick(self, world: G._WorldState, markets: Dict[str, Market], tick: int):
        for shipment in list(world.shipments):
            if shipment.status != "pending":
                continue
            vehicle = None
            # Determine vehicle for delivery
            if shipment.dest_person_id is not None:
                person = world.population.get(shipment.dest_person_id)
                if not person or not person.personal_vehicle_id:
                    continue
                vehicle = world.vehicles.get(person.personal_vehicle_id)
            else:
                origin = world.buildings.get(shipment.origin_building_id)
                if not origin:
                    continue
                # find an idle truck belonging to the company
                trucks = [v for v in world.vehicles.values()
                          if v.owner_company_id == origin.owner_company_id
                          and "truck" in v.vehicle_type.id.lower()
                          and v.status == 'idle']
                if not trucks:
                    continue
                vehicle = trucks[0]
            if not vehicle:
                continue
            # Remove from origin building stock
            if shipment.origin_building_id is not None:
                bsrc = world.buildings.get(shipment.origin_building_id)
                if bsrc:
                    bsrc.resources[shipment.resource_id] = bsrc.resources.get(shipment.resource_id, Decimal(0)) - shipment.quantity
            # Add to destination
            if shipment.dest_building_id is not None:
                bd = world.buildings.get(shipment.dest_building_id)
                if bd:
                    bd.resources[shipment.resource_id] = bd.resources.get(shipment.resource_id, Decimal(0)) + shipment.quantity
            elif shipment.dest_person_id is not None:
                per = world.population.get(shipment.dest_person_id)
                if per:
                    per.inventory.put(G._ResourceStack(amount=int(shipment.quantity), cost=0.0, resource_id=shipment.resource_id), fill=False)
            shipment.status = 'delivered'

class ConstructionBehavior(Behavior):
    """
    Progresses building construction using available labor.
    """
    def tick(self, world: G._WorldState, markets: Dict[str, Market], tick: int):
        # count available workers (unassigned persons)
        idle_workers = [p for p in world.population.values() if p.works_at is None]
        for building in world.buildings.values():
            if not getattr(building, 'under_construction', False):
                continue
            defn = building.definition
            max_workers = getattr(defn, 'max_construction_workers', 1)
            assigned = min(len(idle_workers), max_workers)
            if assigned <= 0:
                continue
            building.remaining_construction_time -= assigned
            if building.remaining_construction_time <= 0:
                building.under_construction = False
                building.remaining_construction_time = 0
        # Regenerate real estate listings each tick
        world.listings.clear()
        for bld in world.buildings.values():
            defn = bld.definition
            total_space = defn.floor_width * defn.floor_length * defn.floors
            # single-family homes sold as whole building
            if getattr(defn, 'is_single_family', False):
                price = defn.cost * getattr(defn, 'sale_multiplier', Decimal('1'))
                world.listings.append(G.RealEstateListing(
                    building_id=bld.instance_id,
                    is_sale=True,
                    sale_price=price
                ))
            else:
                # list each unit for sale and lease
                for unit in bld.units.values():
                    # sale price pro-rated by floor_space
                    unit_price = defn.cost * (Decimal(unit.floor_space) / Decimal(total_space)) * getattr(defn, 'sale_multiplier', Decimal('1'))
                    lease_price = unit_price * getattr(defn, 'rent_yield_per_tick', Decimal('0.01'))
                    world.listings.append(G.RealEstateListing(
                        building_id=bld.instance_id,
                        unit_id=unit.instance_id,
                        is_sale=True,
                        sale_price=unit_price,
                        lease_price=None
                    ))
                    world.listings.append(G.RealEstateListing(
                        building_id=bld.instance_id,
                        unit_id=unit.instance_id,
                        is_sale=False,
                        sale_price=unit_price,
                        lease_price=lease_price
                    ))
        
class RealEstateBehavior(Behavior):
    """
    Matches buyers/tenants to real estate listings, processes sales and leases.
    """
    def tick(self, world: G._WorldState, markets: Dict[str, Market], tick: int):
        # Process listings
        for listing in list(world.listings):
            # Find interested individuals or companies
            # Residential leases: persons
            if not listing.is_sale and listing.lease_price is not None:
                for person in world.population.values():
                    if person.home_building_id is not None:
                        continue
                    # choose first affordable lease
                    if person.resources.get('money', Decimal(0)) >= listing.lease_price:
                        # assign lease
                        person.resources['money'] -= listing.lease_price
                        owner = world.buildings[listing.building_id].owner_company_id
                        comp = world.companies.get(owner)
                        if comp:
                            comp.resources['money'] = comp.resources.get('money', Decimal(0)) + listing.lease_price
                        # set occupant
                        if listing.unit_id is not None:
                            unit = world.buildings[listing.building_id].units.get(listing.unit_id)
                            unit.occupant_person_id = person.instance_id
                        person.home_building_id = listing.building_id
                        break
            # Residential sales (single-family)
            if listing.is_sale and listing.unit_id is None:
                defn = world.buildings[listing.building_id].definition
                if not getattr(defn, 'is_single_family', False):
                    continue
                for person in world.population.values():
                    if person.home_building_id is not None:
                        continue
                    price = listing.sale_price
                    down = price * Decimal('0.2')
                    if person.resources.get('money', Decimal(0)) >= down:
                        # process sale
                        person.resources['money'] -= down
                        loan_amt = price - down
                        # create loan
                        term = getattr(defn, 'mortgage_term', 360)
                        rate = getattr(defn, 'mortgage_rate', Decimal('0.005'))
                        principal_payment = loan_amt / Decimal(term)
                        payment = principal_payment + rate * loan_amt
                        loan = G.Loan(
                            principal=loan_amt,
                            remaining_balance=loan_amt,
                            rate=rate,
                            term=term,
                            remaining_term=term,
                            payment_per_tick=payment
                        )
                        person.loans.append(loan)
                        # assign ownership
                        world.buildings[listing.building_id].owner_person_id = person.instance_id
                        person.home_building_id = listing.building_id
                        # remove listing
                        world.listings.remove(listing)
                        break

class LoanBehavior(Behavior):
    """
    Processes loan repayments each tick.
    """
    def tick(self, world: G._WorldState, markets: Dict[str, Market], tick: int):
        # persons
        for person in world.population.values():
            for loan in list(person.loans):
                pay = loan.payment_per_tick
                bal = person.resources.get('money', Decimal(0))
                if bal < pay:
                    continue
                person.resources['money'] = bal - pay
                # principal reduction
                principal_portion = loan.principal / Decimal(loan.term)
                loan.remaining_balance -= principal_portion
                loan.remaining_term -= 1
                if loan.remaining_term <= 0:
                    person.loans.remove(loan)
        # companies similarly
        for comp in world.companies.values():
            for loan in list(comp.loans):
                pay = loan.payment_per_tick
                bal = comp.resources.get('money', Decimal(0))
                if bal < pay:
                    continue
                comp.resources['money'] = bal - pay
                principal_portion = loan.principal / Decimal(loan.term)
                loan.remaining_balance -= principal_portion
                loan.remaining_term -= 1
                if loan.remaining_term <= 0:
                    comp.loans.remove(loan)

class CompanyBehavior(Behavior):
    def tick(self, world: G._WorldState, markets: Dict[str, Market], tick: int):
        for comp in world.companies.values():
            # Place buy orders for inputs, using max acceptable prices
            for conv in ConvDefs.values():
                for res_id, amt in conv.input_resources.items():
                    market = markets.get(res_id)
                    if not market:
                        continue
                    price = market.max_buy_price('company', comp.instance_id)
                    order = Order(
                        actor_type='company', actor_id=comp.instance_id,
                        resource_id=res_id, quantity=amt,
                        limit_price=price, side='buy', timestamp=tick
                    )
                    market.order_book.submit(order)
            # Place sell orders for any held resources, at or above minimum acceptable prices
            for res_id, qty in list(comp.resources.items()):
                if res_id == 'money' or qty <= 0:
                    continue
                market = markets.get(res_id)
                if not market:
                    continue
                min_price = market.min_sell_price('company', comp.instance_id)
                order = Order(
                    actor_type='company', actor_id=comp.instance_id,
                    resource_id=res_id, quantity=qty,
                    limit_price=min_price, side='sell', timestamp=tick
                )
                market.order_book.submit(order)

class PersonBehavior(Behavior):
    def tick(self, world: G._WorldState, markets: Dict[str, Market], tick: int):
        for person in world.population.values():
            # Determine purchase and selling based on personal need thresholds
            for res_id, market in markets.items():
                res_def = ResourceDefs.get(res_id)
                # skip non-physical resources
                if not getattr(res_def, 'is_physical', False):
                    continue
                # calculate current inventory for this resource
                current = sum(s.amount for s in person.inventory.stacks if s.resource_id == res_id)
                # load need thresholds
                thresholds = res_def.needs
                # determine amount needed to reach next tier
                needed = Decimal(0)
                if current < thresholds.get('survival', Decimal(0)):
                    needed = thresholds['survival'] - current
                elif current < thresholds.get('comfort', Decimal(0)):
                    needed = thresholds['comfort'] - current
                elif current < thresholds.get('luxury', Decimal(0)):
                    needed = thresholds['luxury'] - current
                # place buy order if needed
                if needed > 0:
                    # find nearby retail stores
                    stores = [b for b in world.buildings.values()
                              if any(m.machine.is_retail for u in b.units.values() for m in u.machines)
                              and not getattr(b, 'under_construction', False)]
                    if stores:
                        store = stores[0]
                        # compute personal shipping cost (round-trip)
                        shipping_cost = Decimal(0)
                        if person.personal_vehicle_id:
                            veh = world.vehicles.get(person.personal_vehicle_id)
                            # find parcel coords
                            home = world.buildings.get(person.home_building_id) if person.home_building_id else None
                            home_parcels = [p for p in world.land.values() if home and p.building_id == home.instance_id]
                            store_parcels = [p for p in world.land.values() if p.building_id == store.instance_id]
                            if veh and home_parcels and store_parcels:
                                hx, hy = home_parcels[0].x, home_parcels[0].y
                                sx, sy = store_parcels[0].x, store_parcels[0].y
                                dist = Decimal(abs(hx - sx) + abs(hy - sy))
                                fuel_price = markets.get('fuel').price if markets.get('fuel') else Decimal(1)
                                shipping_cost = dist * Decimal(2) * veh.vehicle_type.fuel_consumption * fuel_price
                        base_price = market.max_buy_price('person', person.instance_id)
                        price = base_price + shipping_cost
                        order = Order(
                            actor_type='person', actor_id=person.instance_id,
                            building_id=store.instance_id,
                            resource_id=res_id, quantity=needed,
                            limit_price=price, side='buy', timestamp=tick
                        )
                        market.order_book.submit(order)
                    else:
                        # no storefront, fallback to direct purchase
                        price = market.max_buy_price('person', person.instance_id)
                        order = Order(
                            actor_type='person', actor_id=person.instance_id,
                            resource_id=res_id, quantity=needed,
                            limit_price=price, side='buy', timestamp=tick
                        )
                        market.order_book.submit(order)
                # place sell order for surplus above comfort
                surplus = current - thresholds.get('comfort', Decimal(0))
                if surplus > 0:
                    sell_price = market.min_sell_price('person', person.instance_id)
                    sell_order = Order(
                        actor_type='person', actor_id=person.instance_id,
                        resource_id=res_id, quantity=surplus,
                        limit_price=sell_price, side='sell', timestamp=tick
                    )
                    market.order_book.submit(sell_order)

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
                    # attempt to consume all inputs, with fallback to building stock
                    can_run = True
                    # track if any consumption failed
                    for res_id, amt in conv.input_resources.items():
                        required = int(amt)
                        # try machine inventory (partial)
                        taken = m.inventory.take(G._ResourceStack(amount=required, cost=0.0, resource_id=res_id), partial=True)
                        inv_taken = taken.amount if taken else 0
                        remainder = required - inv_taken
                        # fallback to building-level resources
                        if remainder > 0:
                            avail = building.resources.get(res_id, Decimal(0))
                            if avail >= remainder:
                                building.resources[res_id] = avail - remainder
                            else:
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

class BuildingBehavior(Behavior):
    """
    Forecasts resource needs per building and stockpiles accordingly.
    Each building first transfers from company holdings, then submits market orders for shortages.
    """
    def tick(self, world: G._WorldState, markets: Dict[str, Market], tick: int):
        for building in world.buildings.values():
            comp_id = building.owner_company_id
            comp = world.companies.get(comp_id)
            if not comp:
                continue
            # aggregate per-tick consumption from machines
            needs: Dict[str, Decimal] = {}
            for unit in building.units.values():
                for m in unit.machines:
                    profile = m.machine.consumes_active if m.is_active else m.machine.consumes_idle
                    for res_id, rate in profile.items():
                        needs[res_id] = needs.get(res_id, Decimal(0)) + rate
            # plan for a horizon = price_stickiness or default 1
            for res_id, rate in needs.items():
                preset = PresetDefs.get(ResourceDefs[res_id].market_behavior)
                horizon = preset.price_stickiness if preset else 1
                desired = rate * Decimal(horizon)
                current = building.resources.get(res_id, Decimal(0))
                if current >= desired:
                    continue
                to_get = desired - current
                # internal transfer from other buildings in the same company
                for bsrc in (b for b in world.buildings.values()
                             if b.owner_company_id == comp_id and b.instance_id != building.instance_id):
                    b_avail = bsrc.resources.get(res_id, Decimal(0))
                    if b_avail > 0 and to_get > 0:
                        tx = min(b_avail, to_get)
                        bsrc.resources[res_id] = b_avail - tx
                        building.resources[res_id] = building.resources.get(res_id, Decimal(0)) + tx
                        to_get -= tx
                        if to_get <= 0:
                            break
                # fallback to company pool
                if to_get > 0:
                    avail = comp.resources.get(res_id, Decimal(0))
                    transfer = min(avail, to_get)
                    if transfer > 0:
                        comp.resources[res_id] = avail - transfer
                        building.resources[res_id] = building.resources.get(res_id, Decimal(0)) + transfer
                        to_get -= transfer
                # market order for remaining
                if to_get > 0:
                    market = markets.get(res_id)
                    if not market:
                        continue
                    price = market.max_buy_price('company', comp_id)
                    order = Order(
                        actor_type='company', actor_id=comp_id,
                        building_id=building.instance_id,
                        resource_id=res_id, quantity=to_get,
                        limit_price=price, side='buy', timestamp=tick
                    )
                    market.order_book.submit(order)
        
class SpeculatorBehavior(Behavior):
    """
    Handles speculative trading: agents buy when prices are rising and sell when falling.
    Applies only to resources marked is_speculative.
    """
    def tick(self, world: G._WorldState, markets: Dict[str, Market], tick: int):
        for res_id, market in markets.items():
            # skip non-speculative assets
            res_def = ResourceDefs.get(res_id)
            if not getattr(res_def, 'is_speculative', False):
                continue
            # need at least two price points to detect trend
            if len(market.history) < 2:
                continue
            prev_price, curr_price = market.history[-2], market.history[-1]
            trending_up = curr_price > prev_price
            # Persons speculate: buy 1 unit if rising, sell 1 if falling
            for person in world.population.values():
                # only speculative if open to new experiences
                if person.personality.openness < 0:
                    continue
                money = person.resources.get('money', Decimal(0))
                if trending_up:
                    price = market.max_buy_price('person', person.instance_id)
                    if money >= price:
                        order = Order(
                            actor_type='person', actor_id=person.instance_id,
                            resource_id=res_id, quantity=Decimal(1),
                            limit_price=price, side='buy', timestamp=tick
                        )
                        market.order_book.submit(order)
                else:
                    held = sum(s.amount for s in person.inventory.stacks if s.resource_id == res_id)
                    if held >= 1:
                        sell_price = market.min_sell_price('person', person.instance_id)
                        sell_order = Order(
                            actor_type='person', actor_id=person.instance_id,
                            resource_id=res_id, quantity=Decimal(1),
                            limit_price=sell_price, side='sell', timestamp=tick
                        )
                        market.order_book.submit(sell_order)
            # Companies speculate: buy 10 units if rising, sell 10 if falling
            for comp in world.companies.values():
                spec_qty = Decimal(10)
                if trending_up:
                    price = market.max_buy_price('company', comp.instance_id)
                    cost = price * spec_qty
                    if comp.resources.get('money', Decimal(0)) >= cost:
                        order = Order(
                            actor_type='company', actor_id=comp.instance_id,
                            resource_id=res_id, quantity=spec_qty,
                            limit_price=price, side='buy', timestamp=tick
                        )
                        market.order_book.submit(order)
                else:
                    avail = comp.resources.get(res_id, Decimal(0))
                    if avail >= spec_qty:
                        sell_price = market.min_sell_price('company', comp.instance_id)
                        sell_order = Order(
                            actor_type='company', actor_id=comp.instance_id,
                            resource_id=res_id, quantity=spec_qty,
                            limit_price=sell_price, side='sell', timestamp=tick
                        )
                        market.order_book.submit(sell_order)
        
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

    # Core market updates and trade matching
    bm.register(MarketBehavior())
    # Physical delivery of goods
    bm.register(DeliveryBehavior())
    # Progress construction sites
    bm.register(ConstructionBehavior())
    # Real estate sales and leasing
    bm.register(RealEstateBehavior())
    # Mortgage and loan repayments
    bm.register(LoanBehavior())
    # Stockpile resources at building-level before company orders
    bm.register(BuildingBehavior())
    # Business procurement and sales
    bm.register(CompanyBehavior())
    # Consumer purchasing and leasing
    bm.register(PersonBehavior())
    # Speculative trading behavior for marked assets
    bm.register(SpeculatorBehavior())

    for _ in range(ticks):
        bm.tick()
    save_world(world)

if __name__ == "__main__":
    main(ticks=10)
