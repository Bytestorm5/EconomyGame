from decimal import Decimal
from pathlib import Path
import sys
import random

# Make src package discoverable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import sim  # type: ignore
import objects as G  # type: ignore

# --- New import for live plotting ---
import matplotlib.pyplot as plt


def setup_world():
    """Create a world with multiple resources and markets."""
    sim.ResourceDefs.clear()
    sim.PresetDefs.clear()

    raw = G.ResourceType(id="raw")
    preset = G.ResourceMarketPreset(id="basic", elasticity=Decimal("1"))
    sim.PresetDefs[preset.id] = preset

    resources = [
        G.Resource(id="wheat", display_name="Wheat", cost_per=Decimal("1"), type=raw, market_behavior="basic"),
        G.Resource(id="corn", display_name="Corn", cost_per=Decimal("0.8"), type=raw, market_behavior="basic"),
        G.Resource(id="steel", display_name="Steel", cost_per=Decimal("5"), type=raw, market_behavior="basic"),
    ]

    for r in resources:
        sim.ResourceDefs[r.id] = r

    world = G._WorldState()
    world.registry = {}

    markets = {r.id: sim.Market(world, r.id) for r in resources}

    # Cross-register markets so behaviors can observe the full environment
    for market in markets.values():
        market.markets = markets

    # Create two cohorts of companies so we can control who buys/sells
    for i in range(100, 1000):
        company = G._CompanyInstance(instance_id=i)
        company.resources["money"] = Decimal(10000)
        for resource in random.choices(resources, k=random.randint(0, len(resources))):
            company.resources[resource.id] = Decimal(random.randint(0, 10))
        world.companies[i] = company

    for i in range(1000, 2000):
        company = G._CompanyInstance(instance_id=i)
        company.resources["money"] = Decimal(10000)
        for resource in random.choices(resources, k=random.randint(0, len(resources))):
            company.resources[resource.id] = Decimal(random.randint(0, 10))
        world.companies[i] = company

    return world, markets, {r.id: r.cost_per for r in resources}


def submit_random_orders(world: G._WorldState, market: sim.Market, cost, rng, tick):
    """Create a small random batch of buy & sell orders around the cost price."""
    # Sellers – offer at up to ±99 % around their fair value (but never below cost)
    for _ in range(rng.randint(1, 3)):
        actor = world.companies[rng.randint(100, 999)]
        quantity = Decimal(str(rng.randint(5, 20)))
        quantity = min(actor.resources.get(market.resource_id, 0), quantity)
        if quantity == 0:                # <-- short-circuit
            continue
        limit_price = actor.fair_values.get(market.resource_id, market.price) * Decimal(str(1 + rng.uniform(-0.1, 0.4)))
        limit_price = max(limit_price, cost)
        market.order_book.submit(
            sim.Order(
                actor_type="company",
                actor_id=actor.instance_id,
                resource_id=market.resource_id,
                quantity=quantity,
                limit_price=limit_price,
                side="sell",
                timestamp=tick,
            )
        )

    # Buyers – willing to pay up to ±99 % around their fair value (but never more money than they have)
    for _ in range(rng.randint(1, 3)):
        actor = world.companies[rng.randint(1000, 1999)]
        quantity = Decimal(str(rng.randint(5, 20)))
        limit_price = actor.fair_values.get(market.resource_id, market.price) * Decimal(str(1 + rng.uniform(-0.8, 0.2)))
        limit_price = min(limit_price, actor.resources["money"] / quantity)
        market.order_book.submit(
            sim.Order(
                actor_type="company",
                actor_id=actor.instance_id,
                resource_id=market.resource_id,
                quantity=quantity,
                limit_price=limit_price,
                side="buy",
                timestamp=tick,
            )
        )


def run_simulation(ticks: int = 50, seed: int = 42) -> None:
    """Run a more complex multi‑commodity market simulation with live graphing.

    Besides the per‑resource price chart, this function now also plots:
    1. The total amount of money held by all companies.
    2. The average ``fair_value`` that companies assign to each resource.
    """

    world, markets, costs = setup_world()

    bm = sim.BehaviorManager(world, markets, seed=seed)
    bm.register(sim.MarketBehavior())
    bm.register(sim.CompanyBehavior())

    rng = bm.random

    # ---------------------------------------------------------------------
    # Price plot setup (figure 1)
    # ---------------------------------------------------------------------
    plt.ion()  # Enable interactive mode so the GUI updates continuously

    fig_prices, ax_prices = plt.subplots()
    price_history = {rid: [] for rid in markets}
    price_lines = {}
    for rid in markets:
        (line,) = ax_prices.plot([], [], label=rid)
        price_lines[rid] = line
    ax_prices.set_xlabel("Tick")
    ax_prices.set_ylabel("Price")
    ax_prices.set_title("Live Market Prices")
    ax_prices.legend()

    # ---------------------------------------------------------------------
    # Money + fair‑value plot setup (figure 2)
    # ---------------------------------------------------------------------
    fig_money, ax_money = plt.subplots()

    # Total money line
    (line_total_money,) = ax_money.plot([], [], label="total_money")
    money_history: list[float] = []

    # Average fair‑value lines for each resource
    fair_value_history = {rid: [] for rid in markets}
    fair_value_lines = {}
    for rid in markets:
        (line,) = ax_money.plot([], [], label=f"avg_fair_value_{rid}")
        fair_value_lines[rid] = line

    ax_money.set_xlabel("Tick")
    ax_money.set_ylabel("Value")
    ax_money.set_title("Total Money & Average Fair Values")
    ax_money.legend()

    # ---------------------------------------------------------------------
    # Main simulation loop
    # ---------------------------------------------------------------------
    for t in range(1, ticks + 1):
        # 1. Generate supply & demand for every market
        for rid, market in markets.items():
            submit_random_orders(world, market, costs[rid], rng, t)

        # 2. Advance simulation: match orders / update prices / company states
        bm.tick()

        # 3. Snapshot current prices for the price chart
        for rid in markets:
            price_history[rid].append(float(markets[rid].price))
            price_lines[rid].set_data(range(1, t + 1), price_history[rid])

        # 4. Compute economic indicators for the second chart
        total_money = max(float(c.resources["money"]) for c in world.companies.values())
        money_history.append(total_money)
        line_total_money.set_data(range(1, t + 1), money_history)

        for rid in markets:
            avg_fv = sum(
                float(c.fair_values.get(rid, markets[rid].price)) for c in world.companies.values()
            ) / len(world.companies)
            fair_value_history[rid].append(avg_fv)
            fair_value_lines[rid].set_data(range(1, t + 1), fair_value_history[rid])

        # 5. Console log for quick inspection
        price_snapshot = ", ".join(f"{rid}: {markets[rid].price}" for rid in markets)
        print(f"Tick {t:>3}: {price_snapshot} | Total $: {total_money:.0f}")

        # 6. Rescale & repaint both figures
        ax_prices.relim()
        ax_prices.autoscale_view()

        ax_money.relim()
        ax_money.autoscale_view()

        plt.pause(0.001)  # Allow the GUI event loop to process events

    # ---------------------------------------------------------------------
    # End of simulation – freeze figures so they stay visible
    # ---------------------------------------------------------------------
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    run_simulation(ticks=10000)
