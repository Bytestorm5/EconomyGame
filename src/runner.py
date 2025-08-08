import pygame
import random
from decimal import Decimal
import sim  # your existing simulation module
import objects as G

TILE_SIZE = 24
BOTTOM_UI_HEIGHT = 40
FPS = 30

# ──────────────────────────────────────────────────────────────────────────────
# Bootstrap helpers – resources, agents, world geometry
# ──────────────────────────────────────────────────────────────────────────────

def _ensure_default_resources() -> None:
    """Guarantee there is at least a minimal usable resource set."""
    # Money ────────────────────────────────────────────────────────────────────
    if "money" not in sim.ResourceDefs:
        currency_type = G.ResourceType(id="currency")
        sim.ResourceDefs["money"] = G.Resource(
            id="money",
            display_name="Money",
            unit_name="dollar",
            cost_per=Decimal("1"),
            quality=Decimal("0"),
            type=currency_type,
            market_behavior="currency",
            fulfills_demand={},
        )
    # Fuel ─────────────────────────────────────────────────────────────────────
    if "fuel" not in sim.ResourceDefs:
        fuel_type = G.ResourceType(id="liquid")
        sim.ResourceDefs["fuel"] = G.Resource(
            id="fuel",
            display_name="Fuel",
            unit_name="gallon",
            cost_per=Decimal("3"),
            quality=Decimal("0"),
            type=fuel_type,
            market_behavior="commodity",
            fulfills_demand={},
        )
    # Food – satisfies generic hunger demand if such a demand is later defined
    if "food" not in sim.ResourceDefs:
        food_type = G.ResourceType(id="solid")
        sim.ResourceDefs["food"] = G.Resource(
            id="food",
            display_name="Food",
            unit_name="meal",
            cost_per=Decimal("5"),
            quality=Decimal("0"),
            type=food_type,
            market_behavior="commodity",
            fulfills_demand={},
        )

    # Investment demand used for company shares
    if "investment" not in sim.DemandDefs:
        sim.DemandDefs["investment"] = G.ResourceDemand(
            id="investment",
            is_need=False,
            quantity=1,
            min_time_until_repeat=None,
            demand_can_occur=None,
            chance_per_tick=0.0,
        )

    # Generic company shares resource
    if "shares" not in sim.ResourceDefs:
        share_type = G.ResourceType(id="security")
        sim.ResourceDefs["shares"] = G.Resource(
            id="shares",
            display_name="Shares",
            unit_name="share",
            cost_per=Decimal("1"),
            quality=Decimal("0"),
            type=share_type,
            market_behavior="equity",
            fulfills_demand={"investment": 1},
        )


def _ensure_default_agents(world: G._WorldState, rng: random.Random) -> None:
    """Populate the world with a starter company and some citizens so the sim has actors."""
    vehicle_def = G.VehicleDefinition(
        id='truck',
        display_name="Truck",
        max_speed=Decimal(20),
        fuel_consumption=Decimal(1),
        cargo_inventory_size=Decimal(100),
        carry_capacity=Decimal(100),
        fuel_capacity=Decimal(100)
    )
    # ----------------------------------------------------------------– Company
    if not world.companies:
        for i in range(10):
            starter = G._CompanyInstance(resources={
                "money": Decimal("10000"),
                "shares": Decimal("1000000000"),
            })
            if rng.random() > 0.8:
                starter.resources['food'] = Decimal(rng.randint(1, 100))
            if rng.random() > 0.3:
                starter.resources['fuel'] = Decimal(rng.randint(10, 300))

            available = [rid for rid in sim.ResourceDefs.keys() if rid not in ("money", "shares")]
            if available:
                starter.focus_resources = rng.sample(available, k=min(len(available), rng.randint(1, 2)))
            starter.planning_horizon = rng.randint(24, 24 * 90)

            vehicle_def = next(iter(sim.REGISTRY.get("VehicleDefinition", {}).values()), None)
            vehicle_count = rng.randint(1, 3)
            for _ in range(vehicle_count):
                if vehicle_def is None:
                    break
                v = G._VehicleInstance(
                    vehicle_type=vehicle_def,
                    owner_company_id=starter.instance_id,
                    position=world.land[random.choice(list(world.land.keys()))].pos_tuple(),
                    destination=None,
                    status='idle',
                    inventory=G._Inventory(capacity=30),
                    fuel_stored=Decimal(10)
                )
                world.vehicles[v.instance_id] = v
            world.companies[starter.instance_id] = starter

    # ----------------------------------------------------------------– People
    if not world.population:
        people: list[G._PersonInstance] = []
        for _ in range(20):
            p = G._PersonInstance(
                resources={"money": Decimal("500")},
                personal_relationship={},
                inventory=G._Inventory(capacity=30)
            )
            p.known_actors = random.choices(list(world.companies.keys()), k=random.randint(1, 3))
            people.append(p)
            world.population[p.instance_id] = p

        # Simple symmetric relationship network
        for a in people:
            for b in people:
                if a is b:
                    continue
                a.personal_relationship[b.instance_id] = rng.uniform(0.05, 0.25)


def _ensure_default_land(world: G._WorldState, width: int = 20, height: int = 20) -> None:
    """Populate a blank world with a simple buildable grid so we have something to draw."""
    if world.land:
        return
    for x in range(width):
        for y in range(height):
            parcel = G._LandParcel(x=x, y=y)
            world.land[parcel.id] = parcel


def _init_simulation(seed: int = 0) -> tuple[G._WorldState, sim.BehaviorManager]:
    rng = random.Random(seed)

    world = sim.load_world()
    _ensure_default_land(world)
    _ensure_default_agents(world, rng)
    
    markets = {rid: sim.Market(world, rid) for rid in sim.ResourceDefs}
    for m in markets.values():
        m.markets = markets

    marketing_market = sim.MarketingMarket(world)

    bm = sim.BehaviorManager(world, markets, marketing_market, seed=seed)
    bm.register(sim.MarketBehavior())
    bm.register(sim.CompanyBehavior())
    bm.register(sim.PersonBehavior())
    bm.register(sim.MarketingBehavior())
    bm.register(sim.VehicleBehavior())
    bm.register(sim.MachineBehavior())
    return world, bm

# ──────────────────────────────────────────────────────────────────────────────
# Rendering helpers
# ──────────────────────────────────────────────────────────────────────────────

def _color_for_land(state: G.LandState) -> tuple[int, int, int]:
    return {
        G.LandState.WILD_UNBUILDABLE: (60, 60, 60),
        G.LandState.WILD_BUILDABLE:   (34, 139, 34),
        G.LandState.ROAD:             (120, 120, 120),
        G.LandState.BUILDING:         (139, 69, 19),
    }.get(state, (0, 0, 0))
    


def _compute_bounds(world: G._WorldState) -> tuple[int, int]:
    if not world.land:
        return (20, 20)
    max_x = max(p.x for p in world.land.values()) + 1
    max_y = max(p.y for p in world.land.values()) + 1
    return max_x, max_y

# ──────────────────────────────────────────────────────────────────────────────
# Main UI loop
# ──────────────────────────────────────────────────────────────────────────────

def run(seed: int = 0) -> None:
    pygame.init()
    pygame.font.init()

    world, bm = _init_simulation(seed)
    max_x, max_y = _compute_bounds(world)

    width  = max_x * TILE_SIZE
    height = max_y * TILE_SIZE + BOTTOM_UI_HEIGHT

    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Economy‑Sim Runner")

    font = pygame.font.SysFont("Arial", 14)
    clock = pygame.time.Clock()

    paused   = False
    running  = True
    tick_num = 0

    while running:
        # ── Event handling ────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused               # toggle pause / play
                elif event.key == pygame.K_RIGHT:      # single‑step when paused
                    if paused:
                        bm.tick()
                        tick_num += 1

        # ── Advance sim ───────────────────────────────────────────────────────
        if not paused:
            bm.tick()
            tick_num += 1

        # ── Drawing ───────────────────────────────────────────────────────────
        screen.fill((0, 0, 0))

        # Land tiles (origin bottom‑left)
        for parcel in world.land.values():
            color = _color_for_land(parcel.state_id)
            draw_x = parcel.x * TILE_SIZE
            draw_y = (max_y - 1 - parcel.y) * TILE_SIZE
            pygame.draw.rect(screen, color, (draw_x, draw_y, TILE_SIZE, TILE_SIZE))

        # Vehicles
        for v in world.vehicles.values():
            vx, vy = v.position
            px = vx * TILE_SIZE + TILE_SIZE // 2
            py = (max_y - 1 - vy) * TILE_SIZE + TILE_SIZE // 2
            pygame.draw.circle(screen, (0, 0, 255), (px, py), TILE_SIZE // 4)

        # UI overlay
        pygame.draw.rect(screen, (30, 30, 30), (0, height - BOTTOM_UI_HEIGHT, width, BOTTOM_UI_HEIGHT))
        status = "Paused" if paused else "Running"
        text = font.render(f"{status} | Tick: {tick_num}  (SPACE toggle, → step, ESC quit)", True, (255, 255, 255))
        screen.blit(text, (8, height - BOTTOM_UI_HEIGHT + 10))

        pygame.display.flip()
        clock.tick(FPS)

    # ── Shutdown ─────────────────────────────────────────────────────────────
    sim.save_world(world)
    pygame.quit()


if __name__ == "__main__":
    run()
