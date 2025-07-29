from decimal import Decimal
from pathlib import Path
import sys
import random

# Make src package discoverable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import sim  # type: ignore
import objects as G  # type: ignore


def setup_simple_world():
    """Create a minimal world with a wheat market."""
    sim.ResourceDefs.clear()
    sim.PresetDefs.clear()

    raw = G.ResourceType(id="raw")
    preset = G.ResourceMarketPreset(id="basic", elasticity=Decimal("1"))
    sim.PresetDefs[preset.id] = preset

    wheat = G.Resource(id="wheat", display_name="Wheat", cost_per=Decimal("1"), type=raw, market_behavior="basic")
    sim.ResourceDefs[wheat.id] = wheat

    world = G._WorldState()
    world.registry = {}

    market = sim.Market(world, wheat.id)
    market.markets = {wheat.id: market}

    return world, {wheat.id: market}


def run_demo(ticks: int = 5, seed: int = 42) -> None:
    world, markets = setup_simple_world()

    bm = sim.BehaviorManager(world, markets, seed=seed)
    bm.register(sim.MarketBehavior())

    rng = bm.random

    for t in range(ticks):
        # Simple matching order each tick
        markets["wheat"].order_book.submit(sim.Order(
            actor_type="company",
            actor_id=1,
            resource_id="wheat",
            quantity=Decimal("10"),
            limit_price=Decimal("1"),
            side="sell",
            timestamp=t + 1,
        ))
        markets["wheat"].order_book.submit(sim.Order(
            actor_type="company",
            actor_id=2,
            resource_id="wheat",
            quantity=Decimal("10"),
            limit_price=Decimal("1.5"),
            side="buy",
            timestamp=t + 1,
        ))
        bm.tick()
        print(f"Tick {t + 1}: wheat price = {markets['wheat'].price}")


if __name__ == "__main__":
    run_demo()
