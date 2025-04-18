# src/objects.py

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Union, Dict, Optional

# --- Example Game Object Classes ---

# --- Condition Model ---

# A simple condition can be a string expression or a dict of named expressions
_SimpleCond = Union[str, Dict[str, str]]

class _AtCountParams(BaseModel):
    """Parameters for atLeast, atMost, exactly operations"""
    n: int = Field(..., ge=0)
    conditions: List[ConditionBlock]  # forward reference

class ConditionBlock(BaseModel):
    """Recursive condition block supporting logical ops and counts"""
    condition: Optional[_SimpleCond] = None
    AND: Optional[List[ConditionBlock]] = None
    OR: Optional[List[ConditionBlock]] = None
    NOT: Optional[ConditionBlock] = None
    atLeast: Optional[_AtCountParams] = None
    atMost: Optional[_AtCountParams] = None
    exactly: Optional[_AtCountParams] = None

# Resolve forward references for recursive fields
ConditionBlock.model_rebuild()

# --- Effect Model ---

# A simple effect can reference a key or be a dict of parameters
SimpleEffect = Union[str, Dict[str, str]]

class _IFSEntry(BaseModel):
    """Conditional branch applying effects when 'condition' holds"""
    condition: ConditionBlock
    effect: List[SimpleEffect]

class EffectBlock(BaseModel):
    """Defines base effects and optional named IFS branches"""
    effect: List[SimpleEffect] = Field(default_factory=list)
    IFS: Dict[str, List[_IFSEntry]] = Field(default_factory=dict)
