from __future__ import annotations
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, Field, PositiveInt, field_validator, model_validator, root_validator, validator, PositiveFloat

MAX_MANAGED_WORKERS = 8
import itertools
_id_counter = itertools.count()

def get_instance_id():
    return next(_id_counter)

# ────────────────────────────────────────────────────────────────────────────
# Utility: Condition / Effect (unchanged stubs – keep until DSL is defined)
# ────────────────────────────────────────────────────────────────────────────

_SimpleCond = str | Dict[str, str]

class _AtCountParams(BaseModel):
    """Parameters for atLeast, atMost, exactly operations"""

    n: int = Field(..., ge=0)
    conditions: List[ConditionBlock]

class ConditionBlock(BaseModel):
    """Recursive condition block supporting logical ops and counts"""

    condition: Optional[_SimpleCond] = None
    AND: Optional[List["ConditionBlock"]] = None
    OR: Optional[List["ConditionBlock"]] = None
    NOT: Optional["ConditionBlock"] = None
    atLeast: Optional[_AtCountParams] = None
    atMost: Optional[_AtCountParams] = None
    exactly: Optional[_AtCountParams] = None

ConditionBlock.model_rebuild()

_SimpleEffect = str | Dict[str, str]

class _IFSEntry(BaseModel):
    """Conditional branch applying effects when 'condition' holds"""

    condition: ConditionBlock
    effect: List[_SimpleEffect]

class EffectBlock(BaseModel):
    """Defines base effects and optional named IFS branches"""

    effect: List[_SimpleEffect] = Field(default_factory=list)
    IFS: Dict[str, List[_IFSEntry]] = Field(default_factory=dict)

# ────────────────────────────────────────────────────────────────────────────
# Resources
# ────────────────────────────────────────────────────────────────────────────

class ResourceType(BaseModel):
    id: str

class ResourceMarketPreset(BaseModel):
    # We define some "presets" for how the market for different resources work
    # e.g. we may define a preset for how "luxury" assets work vs. how "fuel" assets work
    id: str
    elasticity: Decimal = Field(..., description="Own-price elasticity")
    cross_elasticities: Dict[str, Decimal] = Field(default_factory=dict, description="Cross-price elasticities")
    transaction_cost: Decimal = Field(default=Decimal("0"), description="Per-unit transaction cost")
    initial_liquidity: Decimal = Field(default=Decimal("1000"), description="Initial market depth")
    price_stickiness: int = Field(default=1, description="Ticks between allowed repricings")
    carry_cost: Decimal = Field(default=Decimal("0"), description="Inventory carrying cost per unit per tick")
    menu_cost: Decimal = Field(default=Decimal("0"), description="Fixed cost to change price")

class Resource(BaseModel):
    id: str
    display_name: str
    unit_name: str = "unit" # What do we call 1 unit of this resource? e.g. "gallon" for water, "bushel" for corn
    cost_per: Decimal = Field(..., description="Cost per 1 unit of this resource")
    type: ResourceType
    market_behavior: str

    # allow int/float literals in JSON/YAML configs
    @validator("cost_per", pre=True)
    def _decimize(cls, v):  # noqa: N805 – pydantic validator name convention
        return Decimal(str(v))
    
class _ResourceStack(BaseModel):
    amount: PositiveInt
    cost: PositiveFloat
    resource_id: str
    
    @property
    def cost_per_unit(self):
        return self.cost / self.amount
    
class _Inventory(BaseModel):
    capacity: PositiveInt
    stacks: List[_ResourceStack] = Field(default_factory=list)

    def total_amount(self) -> int:
        return sum(stack.amount for stack in self.stacks)

    def can_put(self, stack: _ResourceStack, fill: bool = False) -> bool:
        available = self.capacity - self.total_amount()
        if fill:
            # can put any positive amount if there's space
            return available > 0 and stack.amount > 0
        return stack.amount <= available

    def put(self, stack: _ResourceStack, fill: bool = False) -> Optional[_ResourceStack]:
        available = self.capacity - self.total_amount()
        if fill:
            to_add = min(available, stack.amount)
            if to_add <= 0:
                # no space, return original stack
                return stack
            cpu = stack.cost / stack.amount
            added_cost = cpu * to_add
            # append only if >0
            if to_add > 0:
                self.stacks.append(_ResourceStack(
                    amount=to_add,
                    cost=added_cost,
                    resource_id=stack.resource_id
                ))
            remainder_amount = stack.amount - to_add
            remainder_cost = stack.cost - added_cost
            return _ResourceStack(
                amount=remainder_amount,
                cost=remainder_cost,
                resource_id=stack.resource_id
            )
        else:
            if stack.amount > available:
                return None
            self.stacks.append(stack)
            # indicate fully consumed
            return _ResourceStack(amount=0, cost=0.0, resource_id=stack.resource_id)

    def can_take(self, demand: _ResourceStack, partial: bool = False) -> bool:
        total_avail = sum(s.amount for s in self.stacks if s.resource_id == demand.resource_id)
        if partial:
            return total_avail > 0
        return total_avail >= demand.amount

    def take(self, demand: _ResourceStack, partial: bool = False) -> Optional[_ResourceStack]:
        total_avail = sum(s.amount for s in self.stacks if s.resource_id == demand.resource_id)
        if not self.can_take(demand, partial):
            return None
        to_take = demand.amount if not partial else min(demand.amount, total_avail)
        taken_amount = to_take
        taken_cost = 0.0
        new_stacks: List[_ResourceStack] = []
        for s in self.stacks:
            if s.resource_id != demand.resource_id or taken_amount <= 0:
                new_stacks.append(s)
                continue
            if s.amount <= taken_amount:
                # take entire stack
                taken_cost += s.cost
                taken_amount -= s.amount
            else:
                # take a portion of this stack
                fraction = taken_amount / s.amount
                cost_taken = s.cost * fraction
                taken_cost += cost_taken
                # reduce existing stack
                remaining_amt = s.amount - taken_amount
                remaining_cost = s.cost - cost_taken
                new_stacks.append(_ResourceStack(
                    amount=remaining_amt,
                    cost=remaining_cost,
                    resource_id=s.resource_id
                ))
                taken_amount = 0
        self.stacks = new_stacks
        return _ResourceStack(
            amount=to_take,
            cost=taken_cost,
            resource_id=demand.resource_id
        )

# Shorthand mapping used throughout models
_ResourceAmounts = Dict[str, Decimal]

# ────────────────────────────────────────────────────────────────────────────
# Machines & Containers  (Container = Machine with inventory)
# ────────────────────────────────────────────────────────────────────────────

class MachineDefinition(BaseModel):
    """Blueprint for any interactive object occupying floor space."""

    id: str
    display_name: str
    floor_space: PositiveInt = Field(..., description="Tiles consumed inside a unit")

    # conditions
    visible: ConditionBlock
    can_build: ConditionBlock
    # Conditions that must be true about a Person to operate the machine
    # The machine is only considered "in operation" when all operator slots are filled
    # Display name -> Conditions to fill slot
    operator_requirements: Dict[str, ConditionBlock]
    
    # resource consumption profiles per tick
    consumes_idle: _ResourceAmounts = Field(default_factory=dict)
    consumes_active: _ResourceAmounts = Field(default_factory=dict)

    # inventory (optional – makes the machine act like a Container)
    inventory_capacity: Optional[PositiveInt] = Field(
        None, description="Total units of resource that can be stored inside"
    )
    inventory_resource_type: Optional[ResourceType] = Field(
        None, description="Only resources of this type may be stored if capacity > 0"
    )
    # Some advanced machines can have conveyors to others
    conveyor_in_capacity: int = 0
    conveyor_out_capacity: int = 0
    
    # What it takes to change the recipe of a machine
    # Typically only money
    change_recipe_cost: _ResourceAmounts = Field(default_factory=dict)
    # Maps recipe ids to timesteps taken to execute the recipe
    # If None, use the default defined in the conversion
    possible_recipes: Dict[str, int | None]

    # coercion to Decimal for resource dictionaries
    @validator("consumes_idle", "consumes_active", pre=True)
    def _decimize_map(cls, v):  # noqa: N805
        return {k: Decimal(str(val)) for k, val in v.items()}

    # convenience helpers ----------------------------------------------------
    @property
    def is_container(self) -> bool:
        return self.inventory_capacity is not None

    def can_store(self, resource: Resource) -> bool:
        return (
            self.is_container
            and resource.type == self.inventory_resource_type  # type: ignore[compare]
        )

class _MachineInstance(BaseModel):
    """Concrete machine placed in a Unit."""
    instance_id: int = get_instance_id()
    machine: MachineDefinition
    is_active: bool = False
    stored_resources: Dict[str, Decimal] = Field(default_factory=dict)

    # We only define where conveyors are going- the machine's inventory will be
    # updated with inbound conveyor items without this instance having to to anything.
    conveyors_out: List[str]
    # id of the recipe that this machine is set to
    recipe: str
    # the tick at which the currently-running recipie will be completed
    wait_for_tick: int = 0
    
    inventory: _Inventory

    @model_validator(mode="before")
    def init_inventory(cls, values):
        machine_def = values.get('machine')
        cap = machine_def.inventory_capacity or 0
        values['inventory'] = _Inventory(capacity=cap)
        return values
    
    def available_capacity(self) -> int:
        if not self.machine.is_container:
            return 0
        used = int(sum(self.stored_resources.values()))
        return int(self.machine.inventory_capacity) - used  # type: ignore[arg-type]

    def store(self, resource: Resource, amount: Decimal) -> None:
        if amount <= 0:
            raise ValueError("Amount must be positive")
        if not self.machine.can_store(resource):
            raise ValueError(f"{self.machine.display_name} cannot store this resource type")
        if amount > self.available_capacity():
            raise ValueError("Not enough capacity")
        self.stored_resources[resource.id] = self.stored_resources.get(resource.id, Decimal(0)) + amount

# ────────────────────────────────────────────────────────────────────────────
# Units
# ────────────────────────────────────────────────────────────────────────────

class _UnitInstance(BaseModel):
    """Sellable / rentable space inside a building."""

    instance_id: int = get_instance_id()
    owner_company_id: str
    floor_space: PositiveInt = Field(..., description="Maximum usable space in tiles")
    machines: List[_MachineInstance] = Field(default_factory=list)

    # financials (optional – extend as needed)
    purchase_price: Optional[Decimal] = None
    rent_price_per_tick: Optional[Decimal] = None

    # ── Helpers ────────────────────────────────────────────────────────────
    def used_floor_space(self) -> int:
        return sum(m.machine.floor_space for m in self.machines)

    def free_floor_space(self) -> int:
        return self.floor_space - self.used_floor_space()

    def can_add_machine(self, machine: MachineDefinition) -> bool:
        return self.free_floor_space() >= machine.floor_space

    def add_machine(self, machine_def: MachineDefinition, *, active: bool = False) -> str:
        if not self.can_add_machine(machine_def):
            raise ValueError("Not enough free floor space in unit")
        inst = _MachineInstance(machine=machine_def, is_active=active)
        self.machines.append(inst)
        return inst.machine.id

# ────────────────────────────────────────────────────────────────────────────
# Buildings
# ────────────────────────────────────────────────────────────────────────────

class BuildingType(BaseModel):
    name: str

class BuildingDefinition(BaseModel):
    """Static blueprint for a building type."""

    visible: ConditionBlock
    can_build: ConditionBlock
    can_destroy: ConditionBlock
    on_build: EffectBlock
    on_destroy: EffectBlock

    cost: Decimal
    floor_width: PositiveInt
    floor_length: PositiveInt
    floors: PositiveInt
    
    loading_bays: PositiveInt
    parking_spots: PositiveInt
    
    building_type: BuildingType

    @validator("cost", pre=True)
    def _decimize(cls, v):  # noqa: N805
        return Decimal(str(v))

    # derived ---------------------------------------------------------------
    @property
    def floor_area(self) -> int:
        return self.floor_width * self.floor_length

class _BuildingInstance(BaseModel):
    """A concrete building that exists in game‑world."""

    instance_id: int = get_instance_id()
    land_parcel_id: int
    definition: BuildingDefinition
    owner_company_id: str
    # units keyed by their IDs
    units: Dict[str, _UnitInstance] = Field(default_factory=dict)
    
    loading_bay_occupants: List[str]
    parking_occupants: List[str]

    # ── Convenience --------------------------------------------------------
    def add_unit(self, unit: _UnitInstance) -> None:
        total_area = self.definition.floor_area * self.definition.floors
        occupied = sum(u.floor_space for u in self.units.values())
        if occupied + unit.floor_space > total_area:
            raise ValueError("Adding this unit would exceed the building's area")
        self.units[unit.instance_id] = unit

    def remaining_floor_space(self) -> int:
        total_area = self.definition.floor_area * self.definition.floors
        used_area = sum(u.floor_space for u in self.units.values())
        return total_area - used_area
    
    def all_machines(self, company_id = None) -> List[_MachineInstance]:
        """Return every machine in every unit, pre‑aggregated for O(n) access."""
        if company_id == None:
            return [m for u in self.units.values() for m in u.machines]
        else:
            return [m for u in self.units.values() for m in u.machines if u.owner_company_id == company_id]

# ────────────────────────────────────────────────────────────────────────────
# Land
# ────────────────────────────────────────────────────────────────────────────

class LandState(BaseModel):
    """Dummy class describing the *type* of land – allows a simple tree."""

    id: str
    parent_id: Optional[str] = None
    
class _LandParcel(BaseModel):
    """A rectangular parcel on the map."""

    # geometry (origin at bottom‑left corner of map; units == tiles/metres)
    x: int
    y: int
    # size is always a single grid square

    # dynamic data
    state_id: str = Field(default="wild")  # key into LAND_STATE_REGISTRY
    owner_company_id: Optional[str] = None

    # Optional link to a building (if state is "building")
    building_id: Optional[str] = None
    
    @property
    def id(self):
        return f"{self.x}-{self.y}"

# ────────────────────────────────────────────────────────────────────────────
# Vehicles
# ────────────────────────────────────────────────────────────────────────────

class VehicleDefinition(BaseModel):
    id: str
    display_name: str

    # movement
    max_speed: Decimal = Field(..., description="Tiles per tick at full throttle")
    fuel_consumption: Decimal = Field(..., description="Fuel units per tile moved")

    # on‑board capacities
    fuel_capacity: Decimal = Field(..., description="How many units of fuel it can hold")
    cargo_inventory_size: Decimal

    @validator("max_speed", "fuel_consumption", "fuel_capacity", pre=True)
    def to_decimal(cls, v):
        return Decimal(str(v))

class _VehicleInstance(BaseModel):
    instance_id: int = get_instance_id()
    vehicle_type: VehicleDefinition
    owner_company_id: str

    # where am I on the grid?
    position: Tuple[int, int]               # current tile
    destination: Optional[Tuple[int, int]]  # target tile (or None if idle)
    current_speed: Decimal = Decimal(0)     # ≤ vehicle_type.max_speed

    # inventories (by resource id)
    status: Literal["idle","moving","loading","unloading","parked"]
    
    inventory: _Inventory
    fuel_stored: Decimal

    @model_validator(mode="before")
    def init_inventory(cls, values):
        vehicle_def: VehicleDefinition = values.get('vehicle_type')
        cap = vehicle_def.cargo_inventory_size or 0
        values['inventory'] = _Inventory(capacity=cap)
        return values

# ────────────────────────────────────────────────────────────────────────────
# Company & Economy misc. (left mostly untouched)
# ────────────────────────────────────────────────────────────────────────────

class TechnologyDefinition(BaseModel):
    id: str
    display_name: str
    prerequisites: List[str]
    visible: ConditionBlock
    on_unlock: EffectBlock

class _FinancialEntityInstance(BaseModel):
    instance_id: int = get_instance_id()
    resources: _ResourceAmounts
    # Building ids
    buildings: List[int]
    land_owned: List[int]

class _CompanyInstance(_FinancialEntityInstance):
    techs: List[str]
    domain: Optional[str] = None
    executive_slots: List[_JobSlot] = Field(default_factory=list)
    segments: Dict[str, _BusinessSegment] = Field(default_factory=dict)

# Simple resource‑conversion recipe (updated to reference machine *IDs*)
class ResourceConversion(BaseModel):
    id: str
    display_name: str
    visible: ConditionBlock
    parent: ResourceConversion
    input_resources: _ResourceAmounts
    output_resources: _ResourceAmounts
    default_time_taken: Decimal = Field(default_factory=lambda: Decimal(0))

    @validator("input_resources", "output_resources", pre=True)
    def _decimize_map(cls, v):  # noqa: N805
        return {k: Decimal(str(val)) for k, val in v.items()}

# ────────────────────────────────────────────────────────────────────────────
# Individuals
# ────────────────────────────────────────────────────────────────────────────

class _PersonalityTrait(BaseModel):
    """Personality trait definition for LLM prompts (instance object)."""
    id: str
    description: str
    coincidence: Dict[str, float] = Field(default_factory=dict)


class PersonMemory(BaseModel):
    """Short and long term memory for individuals."""
    short_term: List[str] = Field(default_factory=list)
    long_term: Dict[str, str] = Field(default_factory=dict)

    def add_short(self, entry: str) -> None:
        self.short_term.append(entry)
        if len(self.short_term) > 30:
            self.short_term.pop(0)


class EmailMessage(BaseModel):
    """Simple email object for inter-agent communication."""

    message_id: int = Field(default_factory=get_instance_id)
    sender_type: Literal["person", "company"]
    sender_id: int
    to_people: List[int] = Field(default_factory=list)
    to_companies: List[int] = Field(default_factory=list)
    subject: str
    body: str
    timestamp: int


class JobRole(str, Enum):
    """Hierarchy of job roles."""
    operator = "operator"
    lead = "lead"
    manager = "manager"
    director = "director"
    executive = "executive"


class _JobSlot(BaseModel):
    """Dynamic assignment of a job role within a company."""
    instance_id: int = get_instance_id()
    role: JobRole
    person_id: Optional[int] = None
    team_id: Optional[str] = None
    department_id: Optional[str] = None
    segment_id: Optional[str] = None


class _Team(BaseModel):
    id: str
    name: str
    unit_ids: List[str] = Field(default_factory=list)
    lead_slot: _JobSlot = Field(default_factory=lambda: _JobSlot(role=JobRole.lead))
    operator_slots: List[_JobSlot] = Field(default_factory=list)


class _Department(BaseModel):
    id: str
    name: str
    manager_slots: List[_JobSlot] = Field(default_factory=list)
    teams: Dict[str, _Team] = Field(default_factory=dict)


class _BusinessSegment(BaseModel):
    id: str
    name: str
    director_slots: List[_JobSlot] = Field(default_factory=list)
    departments: Dict[str, _Department] = Field(default_factory=dict)
    inventory_for_sale: Dict[str, Decimal] = Field(default_factory=dict)

class EducationCategory(BaseModel):
    id: str
    display_name: str
    to_learn: ConditionBlock
    # The "parent" category of this one. This is used to determine job matching:
    # - workers will try to find jobs close to their specialization, and
    #   will not be able to access jobs more specialized than them
    parent: Optional[EducationCategory]
            
    def __eq__(self, other):
        if not isinstance(other, EducationCategory):
            return False
        current = self.parent
        while current:
            if current.id == other.id:
                return True
            current = current.parent
        return False

    def degrees_of_separation(self, other: EducationCategory) -> Optional[int]:
        """
        Returns the number of steps from `self` to `other` if `other` is an ancestor of `self`.
        Returns None if there is no such relationship.
        """
        current = self.parent
        distance = 1
        while current:
            if current.id == other.id:
                return distance
            current = current.parent
            distance += 1
        return None
    
class _PersonInstance(_FinancialEntityInstance):
    personality_traits: List[str] = Field(default_factory=list)
    memory: PersonMemory = Field(default_factory=PersonMemory)
    education: Optional[EducationCategory]

    job_slot_id: Optional[int] = None

    domain: Optional[str] = None
    
    employer_id: Optional[int]
    works_at: Optional[int]
    
    inventory: _Inventory

    @model_validator(mode="before")
    def init_inventory(cls, values):
        cap = 5
        values['inventory'] = _Inventory(capacity=cap)
        return values

class _WorldState(BaseModel):
    land: Dict[str, _LandParcel] = Field(default_factory=dict)
    buildings: Dict[int, _BuildingInstance] = Field(default_factory=dict)
    companies: Dict[int, _CompanyInstance] = Field(default_factory=dict)
    population: Dict[int, _PersonInstance] = Field(default_factory=dict)
    vehicles: Dict[int, _VehicleInstance] = Field(default_factory=dict)
    registry: Dict[str, dict] = Field(default_factory=dict)
    emails: List[EmailMessage] = Field(default_factory=list)
    internet: Dict[str, str] = Field(default_factory=dict)

    jobs: Dict[int, _JobSlot] = Field(default_factory=dict)

    # ── Helper methods -----------------------------------------------------
