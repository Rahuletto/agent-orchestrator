from pydantic import BaseModel, Field
from typing import List, Literal, Optional


class AgentConfig(BaseModel):
    id: int = Field(..., description="Unique identifier for the agent.")
    type: str = Field(
        ...,
        description="Type of the agent, e.g., 'Research Agent - SaaS Content Marketing'",
    )
    usecase: str = Field(
        ..., description="What this agent is responsible for researching or solving."
    )
    system: str = Field(
        ..., description="System instructions defining the agent's role and behavior."
    )
    prompt: str = Field(
        ..., description="Prompt used to query the model within the agent."
    )
    model: Optional[str] = Field(
        None,
        description="The specific model used by the agent, e.g., 'gemini-2.5-flash'",
    )
    relies_on: Optional[List[int]] = Field(
        None, description="List of agent IDs that this agent relies on for its task."
    )


class MasterAgentConfig(BaseModel):
    type: Literal["Master Agent"] = Field(
        "Master Agent", description="Denotes this is a Master Agent configuration."
    )
    query: str = Field(..., description="Main user request or task to be solved.")
    intent_analysis: str = Field(
        ...,
        description="Detailed analysis of the user's intent, including explicit and implicit needs.",
    )
    response: str = Field(
        ..., description="General description of how the strategy will be delivered."
    )
    agents: List[AgentConfig] = Field(
        ...,
        description="List of all agent configurations involved in fulfilling the task.",
    )
