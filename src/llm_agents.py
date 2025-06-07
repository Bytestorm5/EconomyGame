from __future__ import annotations
import os
from typing import List

from langchain.agents import AgentType, initialize_agent
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from agent_tools import Toolset

import objects as G


class DummyLLM:
    """Fallback LLM that returns a constant response."""

    def invoke(self, messages):
        return type("Dummy", (), {"content": "noop"})()


class BaseAgent:
    def __init__(self, temperature: float = 0.3):
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.llm = ChatOpenAI(temperature=temperature, openai_api_key=api_key)
        else:
            self.llm = DummyLLM()

    def run(self, messages: List[str]) -> str:
        chat_messages = [SystemMessage(content=messages[0])] + [HumanMessage(content=m) for m in messages[1:]]
        result = self.llm.invoke(chat_messages)
        return getattr(result, "content", "noop")


class PersonAgent(BaseAgent):
    def act(self, person: G._PersonInstance, world: G._WorldState, tick: int) -> str:
        traits = []
        for tid in person.personality_traits:
            trait = world.registry.get("PersonalityTrait", {}).get(tid)
            if trait:
                traits.append(f"- {trait.id}: {trait.description}")
        system = "You control a person in a business simulation. Use the tools to act."
        short_mem = "\n".join(person.memory.short_term)
        long_mem = person.memory.long_term.get("general", "")
        tools = Toolset(world, lambda: tick)
        if isinstance(self.llm, DummyLLM):
            return "noop"
        agent = initialize_agent(tools.as_tools(), self.llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=False)
        prompt = f"Traits:\n" + "\n".join(traits) + f"\nRecent:\n{short_mem}\nLong:\n{long_mem}"
        return agent.run(prompt)


class CompanyAgent(BaseAgent):
    def act(self, company: G._CompanyInstance, world: G._WorldState, tick: int) -> str:
        system = "You control a company in a business simulation. Use the tools to act."
        tools = Toolset(world, lambda: tick)
        if isinstance(self.llm, DummyLLM):
            return "noop"
        agent = initialize_agent(tools.as_tools(), self.llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=False)
        prompt = f"Resources: {company.resources}"
        return agent.run(prompt)
