from __future__ import annotations
from typing import List
from langchain.tools import tool
import objects as G

class Toolset:
    def __init__(self, world: G._WorldState, tick_provider):
        self.world = world
        self.tick = tick_provider

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
            self.send_email,
            self.inbox,
            self.read_email,
            self.search_inbox,
            self.write_domain,
            self.read_domain,
            self.search_internet,
        ]
