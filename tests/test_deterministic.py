import sys
from pathlib import Path
from decimal import Decimal
import random

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import sim  # type: ignore
import objects as G  # type: ignore


def setup_market():
    sim.ResourceDefs.clear()
    sim.PresetDefs.clear()
    water_type = G.ResourceType(id="raw")
    preset = G.ResourceMarketPreset(id="basic", elasticity=Decimal("1"))
    res = G.Resource(
        id="water",
        display_name="Water",
        cost_per=Decimal("1"),
        type=water_type,
        market_behavior="basic",
    )
    sim.ResourceDefs["water"] = res
    sim.PresetDefs["basic"] = preset
    world = G._WorldState()
    world.registry = {}
    market = sim.Market(world, "water")
    market.markets = {"water": market}
    # create orders to ensure supply/demand
    sell = sim.Order(
        actor_type="company",
        actor_id=1,
        resource_id="water",
        quantity=Decimal("1"),
        limit_price=Decimal("1"),
        side="sell",
        timestamp=1,
    )
    buy = sim.Order(
        actor_type="company",
        actor_id=2,
        resource_id="water",
        quantity=Decimal("1"),
        limit_price=Decimal("2"),
        side="buy",
        timestamp=1,
    )
    market.order_book.submit(sell)
    market.order_book.submit(buy)
    return market


def test_same_seed_reproducible():
    m1 = setup_market()
    rng1 = random.Random(42)
    m2 = setup_market()
    rng2 = random.Random(42)
    m1.update_prices(1, rng1)
    m2.update_prices(1, rng2)
    assert m1.price == m2.price


def test_different_seeds_diverge():
    m1 = setup_market()
    rng1 = random.Random(1)
    m2 = setup_market()
    rng2 = random.Random(2)
    m1.update_prices(1, rng1)
    m2.update_prices(1, rng2)
    assert m1.price != m2.price

