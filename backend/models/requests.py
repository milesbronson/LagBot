"""
Pydantic request models for API endpoints.
"""
from pydantic import BaseModel, Field
from typing import Optional


class NewGameRequest(BaseModel):
    """Request to create a new game session."""
    num_opponents: int = Field(default=2, ge=1, le=9, description="Number of bot opponents")
    opponent_type: str = Field(default="trained", description="Type of opponents: trained, call, random, mixed")
    starting_stack: int = Field(default=1000, gt=0, description="Starting chip stack")
    small_blind: int = Field(default=5, gt=0, description="Small blind amount")
    big_blind: int = Field(default=10, gt=0, description="Big blind amount")


class ActionRequest(BaseModel):
    """Request to submit a player action."""
    action_type: int = Field(..., ge=0, description="Action type: 0=fold, 1=call, 2+=raise")
    raise_amount: Optional[int] = Field(None, ge=0, description="Raise amount for raise actions")
