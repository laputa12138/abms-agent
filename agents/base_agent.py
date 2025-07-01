import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

from core.llm_service import LLMService
from core.embedding_service import EmbeddingService
from core.reranker_service import RerankerService
from core.vector_store import VectorStore
# Import settings to allow agents to access global configs if necessary,
# though direct dependency on specific model names should be via service initialization.
from config import settings

# Configure logging for the agents module
logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """
    Abstract base class for all agents in the RAG system.
    It provides a common structure and can hold shared resources
    like service instances.
    """

    def __init__(self,
                 agent_name: str,
                 llm_service: Optional[LLMService] = None,
                 embedding_service: Optional[EmbeddingService] = None,
                 reranker_service: Optional[RerankerService] = None,
                 vector_store: Optional[VectorStore] = None,
                 config: Optional[dict] = None):
        """
        Initializes the BaseAgent.

        Args:
            agent_name (str): The name of the agent, for logging and identification.
            llm_service (Optional[LLMService]): An instance of LLMService.
            embedding_service (Optional[EmbeddingService]): An instance of EmbeddingService.
            reranker_service (Optional[RerankerService]): An instance of RerankerService.
            vector_store (Optional[VectorStore]): An instance of VectorStore.
            config (Optional[dict]): Agent-specific configuration dictionary.
        """
        self.agent_name = agent_name
        self.llm_service = llm_service
        self.embedding_service = embedding_service
        self.reranker_service = reranker_service
        self.vector_store = vector_store
        self.config = config or {}

        # Global settings can be accessed via the imported `settings` module if needed,
        # e.g., settings.DEFAULT_LLM_MODEL_NAME, but typically services should handle these.
        # self.global_settings = settings

        logger.info(f"Agent '{self.agent_name}' initialized.")

    # @abstractmethod # Removed as direct run() is no longer the primary execution path
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """
        Legacy main execution method for the agent.
        This method is deprecated in favor of execute_task() within the orchestrated workflow.
        Subclasses might still implement this for standalone testing or specific use cases
        not fitting the task-driven workflow.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Any: The result of the agent's execution.
        """
        logger.warning(
            f"Agent '{self.agent_name}' legacy run() method was called directly. "
            f"The primary execution path is now via execute_task() within the orchestrated workflow."
        )
        # raise NotImplementedError(
        #     f"Direct call to run() for agent '{self.agent_name}' is deprecated or not implemented. "
        #     f"Use execute_task() within the orchestrated workflow."
        # )
        pass # Default behavior is to do nothing if not overridden

    def _log_input(self, *args: Any, **kwargs: Any):
        """Helper method to log input parameters."""
        # Truncate long inputs for cleaner logs
        truncated_args = [str(arg)[:200] + '...' if len(str(arg)) > 200 else str(arg) for arg in args]
        truncated_kwargs = {k: str(v)[:200] + '...' if len(str(v)) > 200 else str(v) for k, v in kwargs.items()}
        logger.debug(f"Agent '{self.agent_name}' received input - Args: {truncated_args}, Kwargs: {truncated_kwargs}")

    def _log_output(self, output: Any):
        """Helper method to log output."""
        truncated_output = str(output)[:500] + '...' if len(str(output)) > 500 else str(output)
        logger.debug(f"Agent '{self.agent_name}' produced output: {truncated_output}")

    def _get_config_value(self, key: str, default: Any = None) -> Any:
        """
        Retrieves a value from the agent's configuration.

        Args:
            key (str): The configuration key.
            default (Any, optional): The default value if key is not found. Defaults to None.

        Returns:
            Any: The configuration value or default.
        """
        return self.config.get(key, default)

# Example of a concrete agent (for demonstration, will be moved/refined later)
# class MyDummyAgent(BaseAgent):
#     def __init__(self, llm_service: LLMService, config: Optional[dict]=None):
#         super().__init__(agent_name="MyDummyAgent", llm_service=llm_service, config=config)

#     # This agent would now primarily implement execute_task if used in the new workflow
#     def execute_task(self, workflow_state: Any, task_payload: Dict) -> None:
#         logger.info(f"MyDummyAgent executing task with payload: {task_payload}")
#         # ... dummy logic ...
#         query = task_payload.get("user_query", "default query")
#         self._log_input(user_query=query)

#         if not self.llm_service:
#             logger.error(f"Agent '{self.agent_name}' requires LLMService.")
#             # Update workflow_state with error, or let orchestrator handle exception
#             return

#         prompt = f"User asked: {query}. Respond briefly as a dummy."
#         try:
#             response = self.llm_service.chat(prompt, system_prompt="You are a dummy agent.")
#             self._log_output(response)
#             # In a real scenario, update workflow_state with this response
#             # workflow_state.some_update_method(task_payload.get('chapter_key'), response)
#             # And add next task
#             # workflow_state.add_task("NEXT_DUMMY_TASK", {"source_output": response})
#         except Exception as e:
#             logger.error(f"Agent '{self.agent_name}' encountered an error: {e}")
#             # Update workflow_state with error

if __name__ == '__main__':
    # This example primarily shows BaseAgent's structure.
    # For agent execution, see individual agent files or pipeline/orchestrator examples.
    print("BaseAgent Definition Example")

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Direct instantiation of BaseAgent itself is not very useful as its run() is a placeholder.
    # base_agent_instance = BaseAgent(agent_name="TestBaseAgent")
    # base_agent_instance.run() # This will now log a warning.

    # To test a concrete agent, you'd instantiate that, e.g., TopicAnalyzerAgent,
    # and call its execute_task method with a mock WorkflowState and payload.
    # See individual agent files for their specific __main__ test blocks.

    print("\nBaseAgent example finished. See individual agent files for execution examples with WorkflowState.")
