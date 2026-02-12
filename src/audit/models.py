"""
Pydantic models for the Audit System.

Defines the data structures for Sessions, Interactions, and Tool Executions.
"""

from datetime import datetime
from typing import Any, Optional
from enum import Enum
import uuid

from pydantic import BaseModel, Field


class SessionStatus(str, Enum):
    """Status of an audit session."""
    ACTIVE = "active"
    COMPLETED = "completed"
    ERROR = "error"


class ToolExecutionStatus(str, Enum):
    """Status of a tool execution."""
    STARTED = "started"
    COMPLETED = "completed"
    ERROR = "error"


class UserInfo(BaseModel):
    """Information about the user running the session."""
    username: str = Field(..., description="OS username")
    hostname: str = Field(..., description="Machine hostname")
    # Additional fields can be added for Azure AD identity, etc.


class SessionInfo(BaseModel):
    """
    High-level session metadata.
    Stored in the 'Sessions' container.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique session ID")
    user_info: UserInfo = Field(..., description="User details")
    start_time: datetime = Field(default_factory=datetime.utcnow, description="Session start timestamp")
    end_time: Optional[datetime] = Field(None, description="Session end timestamp")
    status: SessionStatus = Field(default=SessionStatus.ACTIVE, description="Current session status")
    agent_config: Optional[dict[str, Any]] = Field(None, description="Agent configuration used")
    
    # Partition key for Cosmos DB
    @property
    def partition_key(self) -> str:
        return self.user_info.username


class ToolExecution(BaseModel):
    """
    Detailed log for a single tool execution.
    Stored in the 'ToolExecutions' container.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique tool execution ID")
    session_id: str = Field(..., description="Parent session ID")
    interaction_id: Optional[str] = Field(None, description="Parent interaction ID")
    tool_name: str = Field(..., description="Name of the tool executed")
    arguments: Optional[dict[str, Any]] = Field(None, description="Tool input arguments")
    result: Optional[Any] = Field(None, description="Tool execution result")
    status: ToolExecutionStatus = Field(default=ToolExecutionStatus.STARTED, description="Execution status")
    start_time: datetime = Field(default_factory=datetime.utcnow, description="Execution start timestamp")
    end_time: Optional[datetime] = Field(None, description="Execution end timestamp")
    error_message: Optional[str] = Field(None, description="Error message if execution failed")
    
    # Partition key for Cosmos DB
    @property
    def partition_key(self) -> str:
        return self.session_id


class Interaction(BaseModel):
    """
    A single user <-> agent interaction turn.
    Stored in the 'Interactions' container.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique interaction ID")
    session_id: str = Field(..., description="Parent session ID")
    user_query: str = Field(..., description="User's input prompt")
    copilot_response: Optional[str] = Field(None, description="Agent's final response")
    tool_execution_ids: list[str] = Field(default_factory=list, description="IDs of tool executions in this interaction")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Interaction timestamp")
    reasoning: Optional[str] = Field(None, description="Agent's reasoning (if captured)")
    
    # Partition key for Cosmos DB
    @property
    def partition_key(self) -> str:
        return self.session_id