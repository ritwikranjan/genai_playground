"""
Cosmos DB Client for the Audit System.

Handles connection, authentication, and CRUD operations for audit data.
Uses Azure Identity (DefaultAzureCredential) for authentication.
"""

import logging
from typing import Any, Optional

from azure.cosmos import CosmosClient, PartitionKey
from azure.cosmos.exceptions import CosmosResourceExistsError
from azure.identity import DefaultAzureCredential

logger = logging.getLogger(__name__)

# Cosmos DB Configuration
COSMOS_ACCOUNT_NAME = "ai-agent-audits"
COSMOS_ENDPOINT = f"https://{COSMOS_ACCOUNT_NAME}.documents.azure.com:443/"
DATABASE_NAME = "AuditDB"

# Container names
SESSIONS_CONTAINER = "Sessions"
INTERACTIONS_CONTAINER = "Interactions"
TOOL_EXECUTIONS_CONTAINER = "ToolExecutions"


class AuditCosmosClient:
    """
    Cosmos DB client wrapper for audit operations.
    
    Manages database and container creation, and provides
    methods for CRUD operations on audit records.
    """
    
    def __init__(self, endpoint: str = COSMOS_ENDPOINT):
        """
        Initialize the Cosmos DB client with Azure Identity authentication.
        
        Args:
            endpoint: Cosmos DB account endpoint URL.
        """
        self.endpoint = endpoint
        self._credential = DefaultAzureCredential()
        self._client: Optional[CosmosClient] = None
        self._database = None
        self._containers: dict[str, Any] = {}
        
    def _ensure_client(self) -> None:
        """Ensure the Cosmos client is initialized."""
        if self._client is None:
            self._client = CosmosClient(self.endpoint, credential=self._credential)
            logger.info(f"Connected to Cosmos DB at {self.endpoint}")
    
    def _ensure_database(self) -> None:
        """Ensure the database exists, creating it if necessary."""
        self._ensure_client()
        if self._database is None:
            try:
                self._database = self._client.create_database(DATABASE_NAME)
                logger.info(f"Created database: {DATABASE_NAME}")
            except CosmosResourceExistsError:
                self._database = self._client.get_database_client(DATABASE_NAME)
                logger.debug(f"Using existing database: {DATABASE_NAME}")
    
    def _ensure_container(self, container_name: str, partition_key_path: str) -> Any:
        """
        Ensure a container exists, creating it if necessary.
        
        Args:
            container_name: Name of the container.
            partition_key_path: Path to the partition key field (e.g., "/session_id").
            
        Returns:
            Container client.
        """
        self._ensure_database()
        
        if container_name not in self._containers:
            try:
                container = self._database.create_container(
                    id=container_name,
                    partition_key=PartitionKey(path=partition_key_path)
                )
                logger.info(f"Created container: {container_name} with partition key: {partition_key_path}")
            except CosmosResourceExistsError:
                container = self._database.get_container_client(container_name)
                logger.debug(f"Using existing container: {container_name}")
            
            self._containers[container_name] = container
        
        return self._containers[container_name]
    
    def initialize(self) -> None:
        """
        Initialize all required containers for the audit system.
        
        Creates the database and all containers if they don't exist.
        """
        logger.info("Initializing Audit Cosmos DB...")
        
        # Sessions container - partitioned by username for user-centric queries
        self._ensure_container(SESSIONS_CONTAINER, "/user_info/username")
        
        # Interactions container - partitioned by session_id for session-centric queries
        self._ensure_container(INTERACTIONS_CONTAINER, "/session_id")
        
        # ToolExecutions container - partitioned by session_id for session-centric queries
        self._ensure_container(TOOL_EXECUTIONS_CONTAINER, "/session_id")
        
        logger.info("Audit Cosmos DB initialization complete.")
    
    # ============ Sessions Operations ============
    
    def create_session(self, session_data: dict[str, Any]) -> dict[str, Any]:
        """
        Create a new session record.
        
        Args:
            session_data: Session data dictionary (must include 'id' and 'user_info.username').
            
        Returns:
            Created session document.
        """
        container = self._ensure_container(SESSIONS_CONTAINER, "/user_info/username")
        result = container.create_item(body=session_data)
        logger.debug(f"Created session: {session_data.get('id')}")
        return result
    
    def update_session(self, session_id: str, partition_key: str, session_data: dict[str, Any]) -> dict[str, Any]:
        """
        Update an existing session record.
        
        Args:
            session_id: The session ID.
            partition_key: The partition key (username).
            session_data: Updated session data.
            
        Returns:
            Updated session document.
        """
        container = self._ensure_container(SESSIONS_CONTAINER, "/user_info/username")
        result = container.replace_item(item=session_id, body=session_data)
        logger.debug(f"Updated session: {session_id}")
        return result
    
    def get_session(self, session_id: str, partition_key: str) -> Optional[dict[str, Any]]:
        """
        Get a session by ID.
        
        Args:
            session_id: The session ID.
            partition_key: The partition key (username).
            
        Returns:
            Session document or None if not found.
        """
        container = self._ensure_container(SESSIONS_CONTAINER, "/user_info/username")
        try:
            return container.read_item(item=session_id, partition_key=partition_key)
        except Exception as e:
            logger.warning(f"Session not found: {session_id}, error: {e}")
            return None
    
    # ============ Interactions Operations ============
    
    def create_interaction(self, interaction_data: dict[str, Any]) -> dict[str, Any]:
        """
        Create a new interaction record.
        
        Args:
            interaction_data: Interaction data dictionary.
            
        Returns:
            Created interaction document.
        """
        container = self._ensure_container(INTERACTIONS_CONTAINER, "/session_id")
        result = container.create_item(body=interaction_data)
        logger.debug(f"Created interaction: {interaction_data.get('id')}")
        return result
    
    def update_interaction(self, interaction_id: str, partition_key: str, interaction_data: dict[str, Any]) -> dict[str, Any]:
        """
        Update an existing interaction record.
        
        Args:
            interaction_id: The interaction ID.
            partition_key: The partition key (session_id).
            interaction_data: Updated interaction data.
            
        Returns:
            Updated interaction document.
        """
        container = self._ensure_container(INTERACTIONS_CONTAINER, "/session_id")
        result = container.replace_item(item=interaction_id, body=interaction_data)
        logger.debug(f"Updated interaction: {interaction_id}")
        return result
    
    def get_interactions_by_session(self, session_id: str) -> list[dict[str, Any]]:
        """
        Get all interactions for a session.
        
        Args:
            session_id: The session ID.
            
        Returns:
            List of interaction documents.
        """
        container = self._ensure_container(INTERACTIONS_CONTAINER, "/session_id")
        query = "SELECT * FROM c WHERE c.session_id = @session_id ORDER BY c.timestamp"
        parameters = [{"name": "@session_id", "value": session_id}]
        
        items = list(container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=False,
            partition_key=session_id
        ))
        return items
    
    # ============ Tool Executions Operations ============
    
    def create_tool_execution(self, tool_data: dict[str, Any]) -> dict[str, Any]:
        """
        Create a new tool execution record.
        
        Args:
            tool_data: Tool execution data dictionary.
            
        Returns:
            Created tool execution document.
        """
        container = self._ensure_container(TOOL_EXECUTIONS_CONTAINER, "/session_id")
        result = container.create_item(body=tool_data)
        logger.debug(f"Created tool execution: {tool_data.get('id')}")
        return result
    
    def update_tool_execution(self, tool_id: str, partition_key: str, tool_data: dict[str, Any]) -> dict[str, Any]:
        """
        Update an existing tool execution record.
        
        Args:
            tool_id: The tool execution ID.
            partition_key: The partition key (session_id).
            tool_data: Updated tool execution data.
            
        Returns:
            Updated tool execution document.
        """
        container = self._ensure_container(TOOL_EXECUTIONS_CONTAINER, "/session_id")
        result = container.replace_item(item=tool_id, body=tool_data)
        logger.debug(f"Updated tool execution: {tool_id}")
        return result
    
    def get_tool_executions_by_session(self, session_id: str) -> list[dict[str, Any]]:
        """
        Get all tool executions for a session.
        
        Args:
            session_id: The session ID.
            
        Returns:
            List of tool execution documents.
        """
        container = self._ensure_container(TOOL_EXECUTIONS_CONTAINER, "/session_id")
        query = "SELECT * FROM c WHERE c.session_id = @session_id ORDER BY c.start_time"
        parameters = [{"name": "@session_id", "value": session_id}]
        
        items = list(container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=False,
            partition_key=session_id
        ))
        return items
    
    def get_tool_executions_by_interaction(self, session_id: str, interaction_id: str) -> list[dict[str, Any]]:
        """
        Get all tool executions for a specific interaction.
        
        Args:
            session_id: The session ID (partition key).
            interaction_id: The interaction ID.
            
        Returns:
            List of tool execution documents.
        """
        container = self._ensure_container(TOOL_EXECUTIONS_CONTAINER, "/session_id")
        query = "SELECT * FROM c WHERE c.session_id = @session_id AND c.interaction_id = @interaction_id ORDER BY c.start_time"
        parameters = [
            {"name": "@session_id", "value": session_id},
            {"name": "@interaction_id", "value": interaction_id}
        ]
        
        items = list(container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=False,
            partition_key=session_id
        ))
        return items