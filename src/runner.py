import pygame
import random

import bootstrap
import sim
import objects as G

TILE_SIZE = 24
BOTTOM_UI_HEIGHT = 40
FPS = 30
GRAPH_PANEL_WIDTH = 360
GRAPH_MARGIN = 12
MAX_GRAPH_POINTS = 240

# ------------------------------------------------------------------------------
# Simulation bootstrap
# ------------------------------------------------------------------------------

def _init_simulation(seed: int = 0) -> tuple[G._WorldState, sim.BehaviorManager]:
    rng = random.Random(seed)

    world = sim.load_world()
    bootstrap.ensure_world_bootstrap(world, sim.REGISTRY, sim.ResourceDefs, rng)

    markets = {rid: sim.Market(world, rid) for rid in sim.ResourceDefs}
    for market in markets.values():
        market.markets = markets

    marketing_market = sim.MarketingMarket(world)

    bm = sim.BehaviorManager(world, markets, marketing_market, seed=seed)
    bm.register(sim.MarketBehavior())
    bm.register(sim.CompanyBehavior())
    bm.register(sim.LogisticsBehavior())
    bm.register(sim.VehicleBehavior())
    bm.register(sim.MachineBehavior())
    bm.register(sim.ExpirationBehavior())
    bm.register(sim.RealEstateBehavior())
    bm.register(sim.PersonBehavior())
    bm.register(sim.LifecycleBehavior())
    bm.register(sim.DiscoveryBehavior())
    bm.register(sim.MarketingBehavior())
    bm.register(sim.EntrepreneurshipBehavior())
    bm.register(sim.AnalyticsBehavior())
    bm.register(sim.BankBehavior())
    return world, bm

# ------------------------------------------------------------------------------
# Rendering helpers
# ------------------------------------------------------------------------------

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

def _color_for_resource_line(resource_id: str) -> tuple[int, int, int]:
    seed = abs(hash(resource_id))
    r = 80 + (seed & 0x7F)
    g = 80 + ((seed >> 7) & 0x7F)
    b = 80 + ((seed >> 14) & 0x7F)
    return (r % 256, g % 256, b % 256)


def _draw_graphs(screen, analytics: G._AnalyticsSeries, panel_rect: pygame.Rect, font: pygame.font.Font) -> None:
    graphs = [
        ("Price", analytics.resource_prices),
        ("Supply", analytics.resource_supply),
        ("Demand", analytics.resource_demand),
    ]
    if not analytics.ticks or panel_rect.width <= 0 or panel_rect.height <= 0:
        pygame.draw.rect(screen, (25, 25, 35), panel_rect)
        pygame.draw.rect(screen, (70, 70, 70), panel_rect, 1)
        return

    sections = len(graphs)
    section_height = panel_rect.height // sections if sections else panel_rect.height
    resources = sorted(analytics.resource_prices.keys())
    legend_font = font
    for idx, (label, data_map) in enumerate(graphs):
        top = panel_rect.y + idx * section_height
        rect = pygame.Rect(panel_rect.x, top, panel_rect.width, section_height)
        pygame.draw.rect(screen, (25, 25, 35), rect)
        pygame.draw.rect(screen, (60, 60, 60), rect, 1)
        title_surf = font.render(label, True, (200, 200, 200))
        screen.blit(title_surf, (rect.x + 8, rect.y + 4))
        if not resources:
            continue
        graph_width = rect.width - 40
        graph_height = rect.height - 30
        if graph_width <= 0 or graph_height <= 0:
            continue
        legend_x = rect.x + rect.width - 100
        legend_y = rect.y + 4
        for rid in resources[:6]:
            color = _color_for_resource_line(rid)
            pygame.draw.rect(screen, color, pygame.Rect(legend_x, legend_y, 8, 8))
            label_surf = legend_font.render(rid, True, (200, 200, 200))
            screen.blit(label_surf, (legend_x + 12, legend_y - 2))
            legend_y += 12
        for rid in resources:
            series = data_map.get(rid)
            if not series:
                continue
            values = series[-min(len(series), MAX_GRAPH_POINTS):]
            if len(values) < 2:
                continue
            min_val = min(values)
            max_val = max(values)
            denom = max_val - min_val if max_val != min_val else 1.0
            points = []
            for i, value in enumerate(values):
                norm = (value - min_val) / denom if denom else 0.5
                x = rect.x + 30 + int(i * (graph_width / max(1, len(values) - 1)))
                y = rect.y + rect.height - 10 - int(norm * graph_height)
                points.append((x, y))
            if len(points) >= 2:
                pygame.draw.lines(screen, _color_for_resource_line(rid), False, points, 2)



# ------------------------------------------------------------------------------
# Main UI loop
# ------------------------------------------------------------------------------

def run(seed: int = 0) -> None:
    pygame.init()
    pygame.font.init()

    world, bm = _init_simulation(seed)
    max_x, max_y = _compute_bounds(world)

    map_pixels = max_x * TILE_SIZE
    width = map_pixels + GRAPH_PANEL_WIDTH
    height = max_y * TILE_SIZE + BOTTOM_UI_HEIGHT

    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Economy-Sim Runner")

    font = pygame.font.SysFont("Arial", 14)
    clock = pygame.time.Clock()

    paused   = False
    running  = True
    tick_num = 0

    while running:
        # -- Event handling ----------------------------------------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused               # toggle pause / play
                elif event.key == pygame.K_RIGHT:      # single-step when paused
                    if paused:
                        bm.tick()
                        tick_num += 1

        # -- Advance sim -------------------------------------------------------
        if not paused:
            bm.tick()
            tick_num += 1

        # -- Drawing -----------------------------------------------------------
        screen.fill((0, 0, 0))

        # Land tiles (origin bottom-left)
        for parcel in world.land.values():
            color = _color_for_land(parcel.state_id)
            draw_x = parcel.x * TILE_SIZE
            draw_y = (max_y - 1 - parcel.y) * TILE_SIZE
            pygame.draw.rect(screen, color, (draw_x, draw_y, TILE_SIZE, TILE_SIZE))

        # Vehicles
        for vehicle in world.vehicles.values():
            vx, vy = vehicle.position
            px = vx * TILE_SIZE + TILE_SIZE // 2
            py = (max_y - 1 - vy) * TILE_SIZE + TILE_SIZE // 2
            pygame.draw.circle(screen, (0, 0, 255), (px, py), TILE_SIZE // 4)

        panel_rect = pygame.Rect(map_pixels, 0, GRAPH_PANEL_WIDTH, height - BOTTOM_UI_HEIGHT)
        _draw_graphs(screen, world.analytics, panel_rect, font)

        # UI overlay
        pygame.draw.rect(screen, (30, 30, 30), (0, height - BOTTOM_UI_HEIGHT, width, BOTTOM_UI_HEIGHT))
        status = "Paused" if paused else "Running"
        employment = world.analytics.employment_rate[-1] if world.analytics.employment_rate else 0.0
        loyalty = world.analytics.avg_brand_loyalty[-1] if world.analytics.avg_brand_loyalty else 0.0
        employment_pct = employment * 100.0
        hud_text = f"{status} | Tick: {tick_num}  (SPACE toggle, -> step, ESC quit) | Employment: {employment_pct:.1f}% | Brand loyalty: {loyalty:.2f}"
        text = font.render(hud_text, True, (255, 255, 255))
        screen.blit(text, (8, height - BOTTOM_UI_HEIGHT + 10))

        pygame.display.flip()
        clock.tick(FPS)

    # -- Shutdown -------------------------------------------------------------
    sim.save_world(world)
    pygame.quit()


if __name__ == "__main__":
    run()
