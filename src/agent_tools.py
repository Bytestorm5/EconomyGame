from __future__ import annotations
from typing import List, Optional, Dict
from enum import Enum
from pydantic import BaseModel, Field
from langchain.tools import tool
import objects as G

class OrgEntityType(str, Enum):
    PERSON = "person"
    TEAM = "team"
    DEPARTMENT = "department"
    SEGMENT = "segment"
    COMPANY = "company"


class Period(str, Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


class MemoryScope(str, Enum):
    SHORT = "short"
    LONG = "long"


class MemoryWriteMode(str, Enum):
    APPEND = "append"
    OVERWRITE = "overwrite"


class StaffAction(str, Enum):
    APPLY = "apply"
    HIRE = "hire"
    FIRE = "fire"
    PROMOTE = "promote"
    TRANSFER = "transfer"


class StructureAction(str, Enum):
    CREATE_TEAM = "create_team"
    MERGE_TEAMS = "merge_teams"
    SPLIT_TEAM = "split_team"
    CREATE_DEPARTMENT = "create_department"
    MERGE_DEPARTMENTS = "merge_departments"
    SPLIT_DEPARTMENT = "split_department"


class AssetAction(str, Enum):
    ALLOCATE_UNIT = "allocate_unit"
    REVOKE_UNIT = "revoke_unit"
    ADD_MACHINE = "add_machine"
    REMOVE_MACHINE = "remove_machine"
    SET_BUDGET = "set_budget"


class OrgQueryInput(BaseModel):
    entity_type: OrgEntityType
    entity_id: int
    scope: str = Field("self")
    depth: int = Field(1, ge=1, le=5)


class AssetQueryType(str, Enum):
    BUILDING = "building"
    UNIT = "unit"
    MACHINE = "machine"


class AssetQueryInput(BaseModel):
    type: AssetQueryType = Field(..., description="What asset level to query")
    id: Optional[int] = Field(None, description="Building/Unit/Machine id")


class KPIQueryInput(BaseModel):
    entity_type: OrgEntityType
    entity_id: int
    metrics: List[str]
    period: Period = Period.MONTHLY


class JobBoardInput(BaseModel):
    company_id: Optional[int] = None
    department_id: Optional[int] = None
    role: Optional[str] = None


class MemoryQueryInput(BaseModel):
    person_id: int
    scope: MemoryScope = MemoryScope.SHORT
    key: Optional[str] = None


class StaffActionInput(BaseModel):
    actor_id: int
    action: StaffAction
    target_person_id: Optional[int] = None
    target_role: Optional[str] = None
    target_team_id: Optional[str] = None
    salary: Optional[int] = None
    notes: str = ""


class StructureActionInput(BaseModel):
    actor_id: int
    action: StructureAction
    subject_ids: List[int]
    parent_id: Optional[int] = None
    new_name: Optional[str] = None
    budget: Optional[int] = None
    notes: str = ""


class AssetActionInput(BaseModel):
    actor_id: int
    action: AssetAction
    unit_id: Optional[int] = None
    machine_type: Optional[str] = None
    quantity: Optional[int] = None
    cost: Optional[int] = None
    team_id: Optional[str] = None
    department_id: Optional[str] = None
    notes: str = ""


class MemoryWriteInput(BaseModel):
    person_id: int
    key: str
    content: str
    mode: MemoryWriteMode = MemoryWriteMode.APPEND

class Toolset:
    def __init__(self, world: G._WorldState, tick_provider):
        self.world = world
        self.tick = tick_provider

    def _role_of(self, person: G._PersonInstance) -> Optional[G.JobRole]:
        slot = self.world.jobs.get(person.job_slot_id) if person.job_slot_id else None
        return slot.role if slot else None

    # ---- Query Tools ----
    @tool("org_query")
    def org_query(self, data: OrgQueryInput) -> str:
        """Return a slice of the org chart as JSON."""
        # naive implementation
        import json
        result = {}
        if data.entity_type == OrgEntityType.PERSON:
            person = self.world.population.get(data.entity_id)
            if person:
                result = person.model_dump()
        elif data.entity_type == OrgEntityType.COMPANY:
            comp = self.world.companies.get(data.entity_id)
            if comp:
                result = comp.model_dump()
        return json.dumps(result)

    @tool("asset_query")
    def asset_query(self, data: AssetQueryInput) -> str:
        """Return building, unit, or machine details as JSON."""
        import json
        if data.type == AssetQueryType.BUILDING:
            if data.id is None:
                return json.dumps({})
            b = self.world.buildings.get(data.id)
            return json.dumps(b.model_dump() if b else {})
        if data.type == AssetQueryType.UNIT:
            if data.id is None:
                return json.dumps({})
            for b in self.world.buildings.values():
                u = b.units.get(data.id)
                if u:
                    return json.dumps(u.model_dump())
            return json.dumps({})
        if data.type == AssetQueryType.MACHINE:
            if data.id is None:
                return json.dumps({})
            for b in self.world.buildings.values():
                for u in b.units.values():
                    for m in u.machines:
                        if m.instance_id == data.id:
                            return json.dumps(m.model_dump())
            return json.dumps({})
        return json.dumps({})

    @tool("kpi_query")
    def kpi_query(self, data: KPIQueryInput) -> str:
        """Return fake KPI values."""
        return "{}"

    @tool("job_board")
    def job_board(self, data: JobBoardInput) -> str:
        """List open job slots."""
        jobs = []
        for slot in self.world.jobs.values():
            if slot.person_id is not None:
                continue
            if data.company_id and slot.segment_id:
                pass
            jobs.append(f"{slot.instance_id}:{slot.role.value}")
        return "\n".join(jobs)

    @tool("memory_query")
    def memory_query(self, data: MemoryQueryInput) -> str:
        """Read short or long term memory."""
        person = self.world.population.get(data.person_id)
        if not person:
            return """"""
        if data.scope == MemoryScope.SHORT:
            return "\n".join(person.memory.short_term)
        if data.key:
            return person.memory.long_term.get(data.key, "")
        return ",".join(person.memory.long_term.keys())

    # ---- State-changing Action Tools ----
    @tool("staff_action")
    def staff_action(self, data: StaffActionInput) -> str:
        """Perform a staffing action."""
        actor = self.world.population.get(data.actor_id)
        if not actor:
            return "invalid_actor"
        role = self._role_of(actor)
        if data.action == StaffAction.APPLY:
            slot = self.world.jobs.get(int(data.target_team_id or 0))
            if slot and slot.person_id is None:
                slot.person_id = actor.instance_id
                actor.job_slot_id = slot.instance_id
                actor.memory.add_short(f"Applied and placed in slot {slot.instance_id}")
                return f"apply:{slot.instance_id}"
            return "invalid_slot"
        if data.action == StaffAction.HIRE:
            if role not in [G.JobRole.manager, G.JobRole.director, G.JobRole.executive]:
                return "forbidden"
            person = self.world.population.get(data.target_person_id or -1)
            if not person:
                return "no_person"
            slot = self.world.jobs.get(int(data.target_team_id or 0))
            if not slot or slot.person_id is not None:
                return "invalid_slot"
            slot.person_id = person.instance_id
            person.job_slot_id = slot.instance_id
            person.employer_id = actor.employer_id
            actor.memory.add_short(f"Hired {person.instance_id} into slot {slot.instance_id}")
            return f"hire:{slot.instance_id}"
        if data.action == StaffAction.FIRE:
            if role not in [G.JobRole.manager, G.JobRole.director, G.JobRole.executive]:
                return "forbidden"
            person = self.world.population.get(data.target_person_id or -1)
            if not person or person.job_slot_id is None:
                return "no_person"
            slot = self.world.jobs.get(person.job_slot_id)
            if slot:
                slot.person_id = None
            person.job_slot_id = None
            actor.memory.add_short(f"Fired {person.instance_id}")
            return "fire"
        if data.action == StaffAction.PROMOTE:
            if role not in [G.JobRole.manager, G.JobRole.director, G.JobRole.executive]:
                return "forbidden"
            person = self.world.population.get(data.target_person_id or -1)
            if not person or person.job_slot_id is None:
                return "no_person"
            slot = self.world.jobs.get(person.job_slot_id)
            if not slot:
                return "no_slot"
            new_role = G.JobRole(data.target_role) if data.target_role else slot.role
            slot.role = new_role
            actor.memory.add_short(f"Promoted {person.instance_id} to {new_role.value}")
            return "promote"
        if data.action == StaffAction.TRANSFER:
            if role not in [G.JobRole.manager, G.JobRole.director, G.JobRole.executive]:
                return "forbidden"
            person = self.world.population.get(data.target_person_id or -1)
            if not person or person.job_slot_id is None:
                return "no_person"
            new_slot = self.world.jobs.get(int(data.target_team_id or 0))
            if not new_slot or new_slot.person_id is not None:
                return "invalid_slot"
            old_slot = self.world.jobs.get(person.job_slot_id)
            if old_slot:
                old_slot.person_id = None
            new_slot.person_id = person.instance_id
            person.job_slot_id = new_slot.instance_id
            actor.memory.add_short(f"Transferred {person.instance_id} to slot {new_slot.instance_id}")
            return "transfer"
        actor.memory.add_short(f"Staff action {data.action}")
        return "noop"

    @tool("structure_action")
    def structure_action(self, data: StructureActionInput) -> str:
        """Modify organizational structure such as teams and departments."""
        actor = self.world.population.get(data.actor_id)
        if not actor:
            return "invalid_actor"
        role = self._role_of(actor)
        company = self.world.companies.get(actor.employer_id or -1)
        if not company:
            return "no_company"

        def find_department(dept_id: int):
            for seg in company.segments.values():
                if dept_id in seg.departments:
                    return seg.departments[dept_id]
            return None

        if data.action == StructureAction.CREATE_TEAM:
            if role not in [G.JobRole.manager, G.JobRole.director, G.JobRole.executive]:
                return "forbidden"
            dept = find_department(data.parent_id) if data.parent_id is not None else None
            if not dept:
                return "no_department"
            team_id = f"team{len(dept.teams)+1}"
            dept.teams[team_id] = G._Team(id=team_id, name=data.new_name or team_id)
            actor.memory.add_short(f"Created team {team_id}")
            return f"create_team:{team_id}"
        actor.memory.add_short(f"structure:{data.action}")
        return "ok"

    @tool("asset_action")
    def asset_action(self, data: AssetActionInput) -> str:
        """Allocate or modify physical assets."""
        actor = self.world.population.get(data.actor_id)
        if not actor:
            return "invalid_actor"
        role = self._role_of(actor)
        if role not in [G.JobRole.lead, G.JobRole.manager, G.JobRole.director, G.JobRole.executive]:
            return "forbidden"
        if data.action == AssetAction.ADD_MACHINE:
            if data.unit_id is None or not data.machine_type:
                return "missing_params"
            machine_def = self.world.registry.get("MachineDefinition", {}).get(data.machine_type)
            if not machine_def:
                return "bad_machine"
            for b in self.world.buildings.values():
                unit = b.units.get(data.unit_id)
                if unit:
                    unit.add_machine(machine_def, active=False)
                    actor.memory.add_short(f"Added machine {data.machine_type} to unit {data.unit_id}")
                    return "add_machine"
            return "no_unit"
        actor.memory.add_short(f"asset:{data.action}")
        return "ok"

    # ---- Memory Tools ----
    @tool("memory_write")
    def memory_write(self, data: MemoryWriteInput) -> str:
        """Write to a person's long-term memory."""
        person = self.world.population.get(data.person_id)
        if not person:
            return "invalid_person"
        if data.mode == MemoryWriteMode.APPEND:
            person.memory.long_term[data.key] = person.memory.long_term.get(data.key, "") + data.content
        else:
            person.memory.long_term[data.key] = data.content
        return "ok"

    # ---- Email ----
    @tool("send_email")
    def send_email(self, sender_type: str, sender_id: int, to_people: List[int] | None = None, to_companies: List[int] | None = None, subject: str = "", body: str = "") -> str:
        """Send an email to specific recipients."""
        msg = G.EmailMessage(
            sender_type=sender_type,
            sender_id=sender_id,
            to_people=to_people or [],
            to_companies=to_companies or [],
            subject=subject,
            body=body,
            timestamp=self.tick(),
        )
        self.world.emails.append(msg)
        return f"sent:{msg.message_id}"

    @tool("inbox")
    def inbox(self, person_id: int | None = None, company_id: int | None = None) -> str:
        """List email subjects for the given actor."""
        msgs = []
        for m in self.world.emails:
            if person_id is not None and person_id in m.to_people:
                msgs.append(f"{m.message_id}:{m.subject}")
            if company_id is not None and company_id in m.to_companies:
                msgs.append(f"{m.message_id}:{m.subject}")
        return "\n".join(msgs)

    @tool("read_email")
    def read_email(self, email_id: int) -> str:
        """Read an email by id."""
        for m in self.world.emails:
            if m.message_id == email_id:
                return f"From {m.sender_type} {m.sender_id}: {m.subject}\n{m.body}"
        return "not found"

    @tool("search_inbox")
    def search_inbox(self, query: str, person_id: int | None = None, company_id: int | None = None) -> str:
        """Search email subjects/bodies containing the query for the actor."""
        q = query.lower()
        matches = []
        for m in self.world.emails:
            if person_id is not None and person_id not in m.to_people:
                continue
            if company_id is not None and company_id not in m.to_companies:
                continue
            if q in m.subject.lower() or q in m.body.lower():
                matches.append(f"{m.message_id}:{m.subject}")
        return "\n".join(matches)

    # ---- Internet ----
    @tool("write_domain")
    def write_domain(self, domain: str, path: str, content: str) -> str:
        """Write content to a domain path."""
        key = f"{domain}/{path}" if path else domain
        self.world.internet[key] = content
        return "ok"

    @tool("read_domain")
    def read_domain(self, domain: str, path: str = "") -> str:
        """Read content from a domain path."""
        key = f"{domain}/{path}" if path else domain
        return self.world.internet.get(key, "")

    @tool("search_internet")
    def search_internet(self, query: str) -> str:
        """Return domain paths containing the query."""
        q = query.lower()
        res = [k for k,v in self.world.internet.items() if q in v.lower()]
        return "\n".join(res)

    def as_tools(self):
        return [
            self.org_query,
            self.asset_query,
            self.kpi_query,
            self.job_board,
            self.memory_query,
            self.staff_action,
            self.structure_action,
            self.asset_action,
            self.memory_write,
            self.send_email,
            self.inbox,
            self.read_email,
            self.search_inbox,
            self.write_domain,
            self.read_domain,
            self.search_internet,
        ]
