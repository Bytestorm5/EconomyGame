from __future__ import annotations
from typing import List, Optional, Dict
from decimal import Decimal
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
    categories: Optional[List[str]] = None


class MemoryQueryInput(BaseModel):
    person_id: int
    scope: MemoryScope = MemoryScope.SHORT
    key: Optional[str] = None


class StaffActionInput(BaseModel):
    actor_id: int
    action: StaffAction
    target_person_id: Optional[int] = None
    target_role: Optional[str] = None
    target_slot_id: Optional[int] = None
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
    segment_id: Optional[str] = None
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
        slots = []
        for slot in self.world.jobs.values():
            if slot.person_id is not None:
                continue
            if data.role and slot.role.value != data.role:
                continue
            slots.append(slot)

        def distance(slot: G._JobSlot) -> int:
            if not data.categories:
                return 999
            best = 999
            for req_id in slot.required_education:
                req = self.world.registry.get("EducationCategory", {}).get(req_id)
                if not req:
                    continue
                for cat_id in data.categories:
                    cat = self.world.registry.get("EducationCategory", {}).get(cat_id)
                    if not cat:
                        continue
                    if cat == req:
                        return 0
                    d = req.degrees_of_separation(cat) or cat.degrees_of_separation(req)
                    if d is not None and d < best:
                        best = d
            return best

        if data.categories:
            slots.sort(key=distance)

        return "\n".join(f"{s.instance_id}:{s.role.value}" for s in slots)

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
            slot = self.world.jobs.get(int(data.target_slot_id or 0))
            if not slot or slot.person_id is not None:
                return "invalid_slot"
            if actor.instance_id in slot.applicant_ids:
                return "already_applied"
            slot.applicant_ids.append(actor.instance_id)
            # notify department managers
            mgrs: List[int] = []
            for comp in self.world.companies.values():
                for seg in comp.segments.values():
                    dept = seg.departments.get(slot.department_id or "")
                    if not dept:
                        continue
                    for mslot in dept.manager_slots:
                        if mslot.person_id:
                            mgrs.append(mslot.person_id)
                    break
            if mgrs:
                self.send_email(
                    "person",
                    actor.instance_id,
                    to_people=mgrs,
                    subject="Job Application",
                    body=f"Applicant {actor.instance_id} for slot {slot.instance_id}",
                )
            actor.memory.add_short(f"Applied for slot {slot.instance_id}")
            return "applied"
        if data.action == StaffAction.HIRE:
            if role not in [G.JobRole.manager, G.JobRole.director, G.JobRole.executive]:
                return "forbidden"
            person = self.world.population.get(data.target_person_id or -1)
            if not person:
                return "no_person"
            slot = self.world.jobs.get(int(data.target_slot_id or 0))
            if not slot or slot.person_id is not None:
                return "invalid_slot"
            if person.instance_id not in slot.applicant_ids:
                return "not_applied"
            slot.person_id = person.instance_id
            slot.applicant_ids.remove(person.instance_id)
            person.job_slot_id = slot.instance_id
            person.employer_id = actor.employer_id
            actor.memory.add_short(f"Hired {person.instance_id} into slot {slot.instance_id}")
            self.send_email(
                "person",
                actor.instance_id,
                to_people=[person.instance_id],
                subject="Job Offer",
                body=f"You have been hired into slot {slot.instance_id}",
            )
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
            new_slot = self.world.jobs.get(int(data.target_slot_id or 0))
            if not new_slot or new_slot.person_id is not None:
                return "invalid_slot"
            self.send_email(
                "person",
                actor.instance_id,
                to_people=[person.instance_id],
                subject="Transfer Request",
                body=f"Transfer to slot {new_slot.instance_id}?", 
            )
            actor.memory.add_short(f"Requested transfer of {person.instance_id} to slot {new_slot.instance_id}")
            return "transfer_requested"
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
            if role != G.JobRole.manager:
                return "forbidden"
            dept = find_department(data.parent_id) if data.parent_id is not None else None
            if not dept:
                return "no_department"
            team_id = f"team{len(dept.teams)+1}"
            dept.teams[team_id] = G._Team(id=team_id, name=data.new_name or team_id)
            actor.memory.add_short(f"Created team {team_id}")
            return f"create_team:{team_id}"
        if data.action == StructureAction.MERGE_TEAMS:
            if role != G.JobRole.manager:
                return "forbidden"
            dept = find_department(data.parent_id) if data.parent_id is not None else None
            if not dept or len(data.subject_ids) < 2:
                return "no_department"
            t1 = dept.teams.get(str(data.subject_ids[0]))
            t2 = dept.teams.get(str(data.subject_ids[1]))
            if not t1 or not t2:
                return "no_team"
            t1.operator_slots.extend(t2.operator_slots)
            if not t1.lead_slot.person_id:
                t1.lead_slot = t2.lead_slot
            del dept.teams[str(data.subject_ids[1])]
            actor.memory.add_short("Merged teams")
            return "merge_teams"
        if data.action == StructureAction.SPLIT_TEAM:
            if role != G.JobRole.manager:
                return "forbidden"
            dept = find_department(data.parent_id) if data.parent_id is not None else None
            if not dept or len(data.subject_ids) < 1:
                return "no_department"
            old = dept.teams.get(str(data.subject_ids[0]))
            if not old:
                return "no_team"
            new_id = f"team{len(dept.teams)+1}"
            new_team = G._Team(id=new_id, name=data.new_name or new_id)
            half = len(old.operator_slots) // 2
            new_team.operator_slots = old.operator_slots[half:]
            old.operator_slots = old.operator_slots[:half]
            dept.teams[new_id] = new_team
            actor.memory.add_short("Split team")
            return "split_team"
        if data.action == StructureAction.CREATE_DEPARTMENT:
            if role not in [G.JobRole.director, G.JobRole.executive]:
                return "forbidden"
            for seg in company.segments.values():
                if seg.id == str(data.parent_id):
                    dep_id = f"dept{len(seg.departments)+1}"
                    seg.departments[dep_id] = G._Department(id=dep_id, name=data.new_name or dep_id)
                    actor.memory.add_short("Created department")
                    return "create_department"
            return "no_segment"
        if data.action == StructureAction.MERGE_DEPARTMENTS:
            if role not in [G.JobRole.director, G.JobRole.executive]:
                return "forbidden"
            for seg in company.segments.values():
                d1 = seg.departments.get(str(data.subject_ids[0]))
                d2 = seg.departments.get(str(data.subject_ids[1]))
                if d1 and d2:
                    d1.teams.update(d2.teams)
                    d1.manager_slots.extend(d2.manager_slots)
                    del seg.departments[str(data.subject_ids[1])]
                    actor.memory.add_short("Merged departments")
                    return "merge_departments"
            return "no_department"
        if data.action == StructureAction.SPLIT_DEPARTMENT:
            if role not in [G.JobRole.director, G.JobRole.executive]:
                return "forbidden"
            for seg in company.segments.values():
                dep = seg.departments.get(str(data.subject_ids[0]))
                if dep:
                    new_id = f"dept{len(seg.departments)+1}"
                    new_dep = G._Department(id=new_id, name=data.new_name or new_id)
                    half = len(dep.teams) // 2
                    for i, (tid, team) in enumerate(list(dep.teams.items())):
                        if i >= half:
                            new_dep.teams[tid] = team
                            del dep.teams[tid]
                    seg.departments[new_id] = new_dep
                    actor.memory.add_short("Split department")
                    return "split_department"
            return "no_department"
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
            company = self.world.companies.get(actor.employer_id or -1)
            if not company:
                return "no_company"
            cost = data.cost or 100
            funds = company.resources.get("money", Decimal(0))
            if funds < Decimal(cost):
                return "no_funds"
            for b in self.world.buildings.values():
                unit = b.units.get(data.unit_id)
                if unit:
                    company.resources["money"] = funds - Decimal(cost)
                    unit.add_machine(machine_def, active=False)
                    actor.memory.add_short(
                        f"Purchased {data.machine_type} for unit {data.unit_id} at {cost}"
                    )
                    return "add_machine"
            return "no_unit"
        if data.action == AssetAction.REMOVE_MACHINE:
            if data.unit_id is None:
                return "missing_params"
            for b in self.world.buildings.values():
                unit = b.units.get(data.unit_id)
                if unit and unit.machines:
                    unit.machines.pop()
                    actor.memory.add_short("Removed machine")
                    return "remove_machine"
            return "no_unit"
        if data.action == AssetAction.ALLOCATE_UNIT:
            if data.unit_id is None:
                return "missing_params"
            for b in self.world.buildings.values():
                unit = b.units.get(data.unit_id)
                if unit:
                    unit.owner_company_id = str(actor.employer_id)
                    if data.team_id and data.team_id not in unit.access_team_ids:
                        unit.access_team_ids.append(data.team_id)
                    actor.memory.add_short("Allocated unit")
                    return "allocate_unit"
            return "no_unit"
        if data.action == AssetAction.REVOKE_UNIT:
            if data.unit_id is None:
                return "missing_params"
            for b in self.world.buildings.values():
                unit = b.units.get(data.unit_id)
                if unit:
                    if data.team_id and data.team_id in unit.access_team_ids:
                        unit.access_team_ids.remove(data.team_id)
                    else:
                        unit.owner_company_id = ""
                    actor.memory.add_short("Revoked unit")
                    return "revoke_unit"
            return "no_unit"
        if data.action == AssetAction.SET_BUDGET:
            company = self.world.companies.get(actor.employer_id or -1)
            if not company:
                return "no_company"
            if data.team_id:
                for seg in company.segments.values():
                    for dept in seg.departments.values():
                        team = dept.teams.get(str(data.team_id))
                        if team:
                            team.budget = Decimal(data.cost or 0)
                            actor.memory.add_short(f"Set team {data.team_id} budget to {data.cost}")
                            return "set_budget"
            if data.department_id:
                for seg in company.segments.values():
                    dept = seg.departments.get(str(data.department_id))
                    if dept:
                        dept.budget = Decimal(data.cost or 0)
                        actor.memory.add_short(f"Set department {data.department_id} budget to {data.cost}")
                        return "set_budget"
            if data.segment_id:
                seg = company.segments.get(str(data.segment_id))
                if seg:
                    seg.budget = Decimal(data.cost or 0)
                    actor.memory.add_short(f"Set segment {data.segment_id} budget to {data.cost}")
                    return "set_budget"
            actor.memory.add_short("budget_failed")
            return "no_target"
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
