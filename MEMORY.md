# Project Memory

This document captures the key design goals that have guided the development of our agent-based economic simulation framework, as well as a roadmap of future enhancements and areas where the system can evolve.

## 1. General Design Goals

1. Moddable, Data-Driven Architecture
   - All core entities (Resources, Machines, Buildings, Vehicles, Conversions, Technologies, etc.) are defined with Pydantic models and loaded from JSON/YAML “content” folders.  
   - The DSL for `ConditionBlock` and `EffectBlock` allows designers to express game logic—visibility rules, unlock effects, on-build or on-destroy triggers—in declarative files, without touching Python code.  
   - A clear separation between static definitions (content) and dynamic instances (world state) lets mods override or extend game data effortlessly.

2. Agent-Based Simulation
   - Individuals, Companies, and Buildings each have their own state and behavior.  
   - Behaviors are implemented as independent `Behavior` classes (MarketBehavior, PersonBehavior, CompanyBehavior, MachineBehavior, SpeculatorBehavior, etc.), managed by a central `BehaviorManager`.  
   - This modular approach makes it easy to add, remove, or reorder behaviors without invasive changes.

3. Economic Realism
   - A multi-resource order-book model tracks buy and sell orders per tick, matches them, and updates inventories and money balances.  
   - Market dynamics include price stickiness, menu costs, own-price elasticity, cross-elasticities, and per-unit transaction costs.  
   - Transportation costs (fuel, distance, vehicle capacity) are baked into effective prices, enabling logistic considerations in procurement.

4. Tiered Needs and Behavioral Diversity
   - Resources are associated with per-tick “need” tiers (survival, comfort, luxury), driving consumer purchase priorities.  
   - Personalities (HEXACO-inspired traits) influence individual willingness to pay, speculative appetite, and loyalty formation.  
   - Companies adapt pricing based on recent profit history, enabling rudimentary learning and dynamic margins.

5. Supply Chains and Vertical Integration
   - Buildings forecast their own consumption and stockpile inputs ahead of time, first tapping intra-company storage before open markets.  
   - Machines consume local inventories, then building stock, reflecting multi-tier production flows.  
   - Construction sites and delivery logistics are managed with dedicated behaviors, allowing emergent supply-chain interactions.

6. Real Estate and Financing
   - Buildings are sub-divided into “Units” (apartments, office suites), each of which can be leased or sold.  
   - Single-family homes are sold as whole buildings, with down payments and amortized mortgage loans.  
   - A RealEstateBehavior generates dynamic listings each tick; a LoanBehavior handles per-tick repayments and amortization.

## 2. Areas for Future Work

1. Rich Spatial and Transportation Modeling
   - Introduce a true map grid with road networks, pathfinding, congestion, and multi-tick travel events.  
   - Model specialized vehicle fleets (trucks, trains, ships) with schedules, capacities, and depot management.  
   - Simulate parking lots, garages, and curbside logistics in greater detail, affecting vehicle dispatch and idle times.

2. Enhanced Agent Decision Making
   - Integrate stronger optimization or learning algorithms (genetic, reinforcement learning) for companies to plan production, pricing, and expansion.  
   - Implement more nuanced consumer choice models: brand switching costs, social influence, multi-period satisfaction tracking.  
   - Add labor market dynamics: matching education categories to job postings, wage negotiation, unemployment, and job search behaviors.

3. Advanced Financial Instruments
   - Support bonds, equity issuance, corporate finance, interest rate markets, and banking institutions.  
   - Model default risk, credit ratings, and financial contagion between companies and banks.  
   - Enable derivatives or futures contracts for hedging input price volatility.

4. Expanded DSL and Scripting
   - Extend `ConditionBlock` to support pattern matching, set operations, or time-based triggers (e.g. “has worked ≥ 10 ticks”).  
   - Add higher-order operations in `EffectBlock`: generate new entities, schedule delayed events, or run custom Python callbacks.  
   - Build a user-facing editor or REPL for testing condition/effect scripts interactively.

5. Performance, Persistence, and Scale
   - Profile and optimize the tick loop for large populations, thousands of buildings, and deep supply chains.  
   - Implement incremental save/load, world snapshots, and rollback capability for scenario testing.  
   - Explore parallel or spatial partitioning to enable city-scale or nation-scale simulations.

6. Visualization and UI Integration
   - Connect the simulation to a real-time dashboard or map viewer to inspect agent flows, price heatmaps, and network graphs.  
   - Provide hooks for game engines (Unity, Unreal) or web front-ends (D3, React) to drive interactive experiences.  
   - Integrate scenario scripting and story-driven events for narrative gameplay layers.

---
_This memory document will evolve with the project as new features are added, and as real-world use cases highlight fresh opportunities for expansion._