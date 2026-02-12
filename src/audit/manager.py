"""
Audit Manager - High-level interface for the Audit System.

Provides a simple API for tracking sessions, interactions, and tool executions.
This is the main entry point for integrating auditing into client_v2.py.
"""

import getpass
import logging
import platform
from datetime import datetime
from typing import Any, Optional

from .db import AuditCosmosClient
from .models import (
    Interaction,
    SessionInfo,
    SessionStatus,
    ToolExecution,
    ToolExecutionStatus,
    UserInfo,
)

logger = logging.getLogger(__name__)


class AuditManager:
    """
    High-level manager for audit operations.
    
    Provides a simple interface to:
    - Start/end sessions
    - Begin/complete interactions (user query -> agent response turns)
    - Log tool executions
    
    Usage:
        audit = AuditManager()
        audit.initialize()
        
        # Start a session
        session = audit.start_session(agent_config={"model": "gpt-4"})
        
        # For each user turn:
        interaction = audit.start_interaction(user_query="What is the weather?")
        
        # Log tool calls as they happen
        tool_id = audit.log_tool_start(tool_name="get_weather", arguments={"city": "Dublin"})
        audit.log_tool_complete(tool_id=tool_id, result={"temp": 15})
        
        # Complete interaction when agent responds
        audit.complete_interaction(copilot_response="The weather in Dublin is 15°C.")
        
        # End session when done
        audit.end_session()
    """
    
    def __init__(self, cosmos_endpoint: Optional[str] = None):
        """
        Initialize the Audit Manager.
        
        Args:
            cosmos_endpoint: Optional custom Cosmos DB endpoint. If not provided,
                           uses the default configured endpoint.
        """
        if cosmos_endpoint:
            self._db = AuditCosmosClient(endpoint=cosmos_endpoint)
        else:
            self._db = AuditCosmosClient()
        
        self._current_session: Optional[SessionInfo] = None
        self._current_interaction: Optional[Interaction] = None
        self._pending_tool_executions: dict[str, ToolExecution] = {}
        self._initialized = False
    
    def initialize(self) -> None:
        """
        Initialize the audit system.
        
        Creates the Cosmos DB database and containers if they don't exist.
        Must be called before any other operations.
        """
        if not self._initialized:
            self._db.initialize()
            self._initialized = True
            logger.info("Audit Manager initialized.")
    
    def _get_user_info(self) -> UserInfo:
        """Get current user information from the OS."""
        return UserInfo(
            username=getpass.getuser(),
            hostname=platform.node()
        )
    
    # ============ Session Management ============
    
    def start_session(self, agent_config: Optional[dict[str, Any]] = None) -> SessionInfo:
        """
        Start a new audit session.
        
        Args:
            agent_config: Optional configuration dictionary for the agent.
            
        Returns:
            The created SessionInfo object.
        """
        if not self._initialized:
            self.initialize()
        
        user_info = self._get_user_info()
        session = SessionInfo(
            user_info=user_info,
            agent_config=agent_config,
            status=SessionStatus.ACTIVE
        )
        
        # Persist to Cosmos DB
        session_data = session.model_dump(mode="json")
        self._db.create_session(session_data)
        
        self._current_session = session
        logger.info(f"Started audit session: {session.id}")
        
        return session
    
    def end_session(self, status: SessionStatus = SessionStatus.COMPLETED) -> Optional[SessionInfo]:
        """
        End the current audit session.
        
        Args:
            status: Final status of the session (COMPLETED or ERROR).
            
        Returns:
            The updated SessionInfo object, or None if no active session.
        """
        if self._current_session is None:
            logger.warning("No active session to end.")
            return None
        
        # Complete any pending interaction
        if self._current_interaction is not None:
            self.complete_interaction(copilot_response="[Session ended]")
        
        self._current_session.end_time = datetime.utcnow()
        self._current_session.status = status
        
        # Update in Cosmos DB
        session_data = self._current_session.model_dump(mode="json")
        self._db.update_session(
            session_id=self._current_session.id,
            partition_key=self._current_session.partition_key,
            session_data=session_data
        )
        
        logger.info(f"Ended audit session: {self._current_session.id} with status: {status.value}")
        
        ended_session = self._current_session
        self._current_session = None
        
        return ended_session
    
    @property
    def session_id(self) -> Optional[str]:
        """Get the current session ID, if any."""
        return self._current_session.id if self._current_session else None
    
    # ============ Interaction Management ============
    
    def start_interaction(self, user_query: str) -> Interaction:
        """
        Start a new interaction (user turn).
        
        Args:
            user_query: The user's input prompt.
            
        Returns:
            The created Interaction object.
            
        Raises:
            RuntimeError: If no active session exists.
        """
        if self._current_session is None:
            raise RuntimeError("Cannot start interaction without an active session. Call start_session() first.")
        
        # Complete any previous interaction that wasn't explicitly completed
        if self._current_interaction is not None:
            self.complete_interaction(copilot_response="[Interrupted by new query]")
        
        interaction = Interaction(
            session_id=self._current_session.id,
            user_query=user_query
        )
        
        # Persist to Cosmos DB immediately (will be updated when completed)
        interaction_data = interaction.model_dump(mode="json")
        self._db.create_interaction(interaction_data)
        
        self._current_interaction = interaction
        logger.debug(f"Started interaction: {interaction.id} for query: {user_query[:50]}...")
        
        return interaction
    
    def complete_interaction(
        self,
        copilot_response: Optional[str] = None,
        reasoning: Optional[str] = None
    ) -> Optional[Interaction]:
        """
        Complete the current interaction with the agent's response.
        
        Args:
            copilot_response: The agent's final response text.
            reasoning: Optional reasoning/thinking captured during the turn.
            
        Returns:
            The updated Interaction object, or None if no active interaction.
        """
        if self._current_interaction is None:
            logger.warning("No active interaction to complete.")
            return None
        
        self._current_interaction.copilot_response = copilot_response
        self._current_interaction.reasoning = reasoning
        
        # Update in Cosmos DB
        interaction_data = self._current_interaction.model_dump(mode="json")
        self._db.update_interaction(
            interaction_id=self._current_interaction.id,
            partition_key=self._current_interaction.session_id,
            interaction_data=interaction_data
        )
        
        logger.debug(f"Completed interaction: {self._current_interaction.id}")
        
        completed_interaction = self._current_interaction
        self._current_interaction = None
        
        return completed_interaction
    
    @property
    def interaction_id(self) -> Optional[str]:
        """Get the current interaction ID, if any."""
        return self._current_interaction.id if self._current_interaction else None
    
    # ============ Tool Execution Logging ============
    
    def log_tool_start(
        self,
        tool_name: str,
        arguments: Optional[dict[str, Any]] = None
    ) -> str:
        """
        Log the start of a tool execution.
        
        Args:
            tool_name: Name of the tool being executed.
            arguments: Input arguments to the tool.
            
        Returns:
            The tool execution ID (use this to call log_tool_complete later).
            
        Raises:
            RuntimeError: If no active session exists.
        """
        if self._current_session is None:
            raise RuntimeError("Cannot log tool execution without an active session.")
        
        tool_execution = ToolExecution(
            session_id=self._current_session.id,
            interaction_id=self._current_interaction.id if self._current_interaction else None,
            tool_name=tool_name,
            arguments=arguments,
            status=ToolExecutionStatus.STARTED
        )
        
        # Persist to Cosmos DB
        tool_data = tool_execution.model_dump(mode="json")
        self._db.create_tool_execution(tool_data)
        
        # Track pending execution
        self._pending_tool_executions[tool_execution.id] = tool_execution
        
        # Add to current interaction's tool list
        if self._current_interaction:
            self._current_interaction.tool_execution_ids.append(tool_execution.id)
        
        logger.debug(f"Tool execution started: {tool_execution.id} - {tool_name}")
        
        return tool_execution.id
    
    def log_tool_complete(
        self,
        tool_id: str,
        result: Optional[Any] = None,
        error_message: Optional[str] = None
    ) -> Optional[ToolExecution]:
        """
        Log the completion of a tool execution.
        
        Args:
            tool_id: The tool execution ID returned from log_tool_start.
            result: The result returned by the tool.
            error_message: Error message if the tool failed.
            
        Returns:
            The updated ToolExecution object, or None if tool_id not found.
        """
        tool_execution = self._pending_tool_executions.pop(tool_id, None)
        
        if tool_execution is None:
            logger.warning(f"Tool execution not found: {tool_id}")
            return None
        
        tool_execution.end_time = datetime.utcnow()
        tool_execution.result = result
        tool_execution.error_message = error_message
        tool_execution.status = (
            ToolExecutionStatus.ERROR if error_message 
            else ToolExecutionStatus.COMPLETED
        )
        
        # Update in Cosmos DB
        tool_data = tool_execution.model_dump(mode="json")
        self._db.update_tool_execution(
            tool_id=tool_execution.id,
            partition_key=tool_execution.session_id,
            tool_data=tool_data
        )
        
        logger.debug(f"Tool execution completed: {tool_id} - status: {tool_execution.status.value}")
        
        return tool_execution
    
    def log_tool_execution(
        self,
        tool_name: str,
        arguments: Optional[dict[str, Any]] = None,
        result: Optional[Any] = None,
        error_message: Optional[str] = None
    ) -> ToolExecution:
        """
        Log a complete tool execution in one call (start + complete).
        
        Convenience method for when you have all the information at once.
        
        Args:
            tool_name: Name of the tool executed.
            arguments: Input arguments to the tool.
            result: The result returned by the tool.
            error_message: Error message if the tool failed.
            
        Returns:
            The ToolExecution object.
        """
        tool_id = self.log_tool_start(tool_name=tool_name, arguments=arguments)
        return self.log_tool_complete(tool_id=tool_id, result=result, error_message=error_message)