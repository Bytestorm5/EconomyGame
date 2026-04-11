# EconomyGame

This repository contains a simulation sandbox for modelling regional economies.
The engine composes deterministic tick updates, content-driven data (Resources,
Machines, Buildings, etc.), and AI behaviours for companies and citizens. The
current build focuses on a starter scenario that showcases vertical supply
chains (farm -> processor -> retail), job markets, marketing dynamics, and
brand loyalty feedback loops.

## What is included

- **Content driven data** loaded from `content/`, including resources,
  conversions, machine definitions, and buildings. New content files can be
  dropped in without code changes.
- **World bootstrap** that creates land, three themed companies, a population
  of citizens with personalities, vehicles, and pre-configured supply links.
- **Company AI** capable of planning production, sourcing inputs (via preferred
  suppliers or in-house integration), listing inventory for sale, running
  marketing campaigns, building capacity, and paying wages.
- **Citizen AI** that surfaces demands (e.g. hunger), speculates on markets,
  responds to marketing, forms and shares brand loyalty, and now searches for
  employment based on wage offers and personal preferences.
- **Market mechanics** with limit order books per resource, VWAP price updates,
  supplier shipping cost estimates, and equity/currency handling.
- **Runner UI** (`src/runner.py`) powered by `pygame` for a bird's-eye view of
  the map with pausable ticks.

## Running the simulation

1. Ensure dependencies from `requirements.txt` are installed in your
   environment (or activate the provided `venv`).
2. To run the headless simulation for _N_ ticks:  
   `python src/sim.py --ticks 10 --seed 1`
3. To launch the visual runner:  
   `python src/runner.py`
   - `SPACE` toggles pause/play
   - `RIGHT` steps a single tick while paused
   - `ESC` exits and persists the world state to `world_state.json`

## Extending the sandbox

- Add new content JSONs under the existing folders (e.g. new machines under
  `content/MachineDefinition/`) and they will be auto-registered on start.
- Adjust the bootstrap scenario in `src/bootstrap.py` to seed different
  companies, personalities, or map layouts.
- Company and person behaviours live in `src/sim.py`; these can be extended to
  model additional economic mechanics (e.g. credit, taxation, or utilities).

## Current focus

The current milestone emphasises believable behaviour over raw scale: companies
form relationships, bias towards vertical integration or outsourcing based on
personality, citizens discover jobs and receive wages, and marketing feeds back
into purchasing choices. This provides a foundation for future layers such as
logistics with physical routing, research/technology trees, or governance
systems.
