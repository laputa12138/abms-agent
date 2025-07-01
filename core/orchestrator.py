import logging
from typing import Dict, Any, Optional

from core.workflow_state import WorkflowState, TASK_TYPE_ANALYZE_TOPIC, \
    TASK_TYPE_GENERATE_OUTLINE, TASK_TYPE_PROCESS_CHAPTER, TASK_TYPE_RETRIEVE_FOR_CHAPTER, \
    TASK_TYPE_WRITE_CHAPTER, TASK_TYPE_EVALUATE_CHAPTER, TASK_TYPE_REFINE_CHAPTER, \
    TASK_TYPE_COMPILE_REPORT, STATUS_COMPLETED, STATUS_ERROR
# Import all agent classes
from agents.topic_analyzer_agent import TopicAnalyzerAgent
from agents.outline_generator_agent import OutlineGeneratorAgent
from agents.content_retriever_agent import ContentRetrieverAgent
from agents.chapter_writer_agent import ChapterWriterAgent
from agents.evaluator_agent import EvaluatorAgent
from agents.refiner_agent import RefinerAgent
from agents.report_compiler_agent import ReportCompilerAgent

logger = logging.getLogger(__name__)

class OrchestratorError(Exception):
    """Custom exception for Orchestrator errors."""
    pass

class Orchestrator:
    """
    Drives the report generation workflow by managing tasks from WorkflowState
    and dispatching them to appropriate agents.
    """
    def __init__(self,
                 workflow_state: WorkflowState,
                 topic_analyzer: TopicAnalyzerAgent,
                 outline_generator: OutlineGeneratorAgent,
                 content_retriever: ContentRetrieverAgent, # This is the agent, not the service
                 chapter_writer: ChapterWriterAgent,
                 evaluator: EvaluatorAgent,
                 refiner: RefinerAgent,
                 report_compiler: ReportCompilerAgent,
                 master_control_agent: Optional['MasterControlAgent'] = None, # Added MasterControlAgent
                 max_workflow_iterations: int = 50
                ):
        self.workflow_state = workflow_state
        self.master_control_agent = master_control_agent # Store the MasterControlAgent
        self.agents = { # These are "worker" agents now
            TASK_TYPE_ANALYZE_TOPIC: topic_analyzer,
            TASK_TYPE_GENERATE_OUTLINE: outline_generator,
            TASK_TYPE_RETRIEVE_FOR_CHAPTER: content_retriever,
            TASK_TYPE_WRITE_CHAPTER: chapter_writer,
            TASK_TYPE_EVALUATE_CHAPTER: evaluator,
            TASK_TYPE_REFINE_CHAPTER: refiner,
            TASK_TYPE_COMPILE_REPORT: report_compiler,
            # Note: MasterControlAgent is not in this dict as it's a special controller
        }
        self.max_workflow_iterations = max_workflow_iterations
        logger.info("Orchestrator initialized with agents, workflow state, and MasterControlAgent.")

    def _execute_task_type(self, task: Dict[str, Any]):
        """Executes a specific task by calling the appropriate agent."""
        task_type = task['type']
        task_id = task['id']
        payload = task.get('payload', {})

        agent = self.agents.get(task_type)

        if agent:
            try:
                self.workflow_state.log_event(f"Orchestrator dispatching task '{task_type}' to agent '{agent.agent_name}'.",
                                             {"task_id": task_id, "payload": payload})
                # Assuming all agents now have an 'execute_task' method
                agent.execute_task(self.workflow_state, payload)
                # Agent's execute_task is responsible for adding next tasks and completing current one via workflow_state
                # So, we don't call workflow_state.complete_task here in orchestrator for agent-handled tasks.
                # Agent should call it. If agent raises error, it's caught below.
                # However, for tasks handled *directly* by orchestrator (like PROCESS_CHAPTER), we need to complete it.

                # If an agent's execute_task doesn't call complete_task itself, then orchestrator should.
                # For now, let's assume agents will call workflow_state.complete_task(task_id, ...)
                # This needs to be consistently implemented in all agents.
                # Let's refine: if agent.execute_task doesn't raise, we assume it handled completion.
                # This is a bit implicit. A better way: agent returns a status or next actions.
                # For now, let's assume agent calls complete_task.

            except Exception as e: # Catch errors from agent execution
                logger.error(f"Error executing task {task_type} ({task_id}) with agent {getattr(agent, 'agent_name', 'UnknownAgent')}: {e}", exc_info=True)
                self.workflow_state.log_event(f"Agent execution error for task {task_type} ({task_id})",
                                             {"error": str(e), "agent": getattr(agent, 'agent_name', 'UnknownAgent')},
                                             level="CRITICAL")
                self.workflow_state.complete_task(task_id, f"Agent failed: {e}", status='failed')
                if 'chapter_key' in payload:
                    self.workflow_state.add_chapter_error(payload['chapter_key'], f"Agent {task_type} failed: {e}")

        elif task_type == TASK_TYPE_PROCESS_CHAPTER: # Meta-task handled by orchestrator
            # This task initiates the sequence for a chapter: retrieve -> write (-> eval -> refine)*
            chapter_key = payload['chapter_key']
            self.workflow_state.update_chapter_status(chapter_key, STATUS_PENDING) # Reset/confirm status
            # Add retrieval task as the first concrete step for this chapter
            self.workflow_state.add_task(TASK_TYPE_RETRIEVE_FOR_CHAPTER,
                                         payload=payload, # Pass on chapter_key, chapter_title
                                         priority=task.get('priority', 3))
            self.workflow_state.complete_task(task_id, f"PROCESS_CHAPTER task for '{payload.get('chapter_title')}' initiated retrieval.")

        # TODO: Handle TASK_TYPE_SUGGEST_OUTLINE_REFINEMENT, TASK_TYPE_APPLY_OUTLINE_REFINEMENT here
        # These might involve more complex logic, potentially calling OutlineGeneratorAgent again or LLM.

        else:
            logger.warning(f"No agent or direct handler registered for task type: {task_type} (task_id: {task_id}).")
            self.workflow_state.log_event(f"Unknown task type: {task_type}", {"task_id": task_id}, level="ERROR")
            self.workflow_state.complete_task(task_id, f"Unknown task type {task_type}", status='failed')


    def coordinate_workflow(self) -> None:
        """
        Main loop to coordinate the workflow.
        It now incorporates MasterControlAgent for decision making if available.
        """
        self.workflow_state.log_event("Orchestrator starting workflow coordination.")
        iteration_count = 0

        while not self.workflow_state.get_flag('report_generation_complete', False):
            if iteration_count >= self.max_workflow_iterations:
                self.workflow_state.log_event("Max workflow iterations reached by Orchestrator. Halting.", level="ERROR")
                self.workflow_state.set_flag('report_generation_complete', True)
                break

            # --- MasterControlAgent Decision Point ---
            if self.master_control_agent:
                if not self.workflow_state.pending_tasks or iteration_count % 5 == 0 : # Decide if queue is empty or periodically
                    self.workflow_state.log_event("Invoking MasterControlAgent for next actions.")
                    try:
                        new_tasks = self.master_control_agent.decide_next_actions(self.workflow_state)
                        if new_tasks:
                            for task_to_add in new_tasks: # Add tasks returned by MCA
                                self.workflow_state.add_task(
                                    task_type=task_to_add['type'],
                                    payload=task_to_add.get('payload'),
                                    priority=task_to_add.get('priority', 10) # Default priority for MCA tasks
                                )
                            self.workflow_state.log_event(f"MasterControlAgent added {len(new_tasks)} tasks.", {"tasks": new_tasks})
                        else:
                            self.workflow_state.log_event("MasterControlAgent decided no new tasks are needed at this moment.")
                    except Exception as mca_e:
                        logger.error(f"MasterControlAgent failed to decide next actions: {mca_e}", exc_info=True)
                        self.workflow_state.log_event("MasterControlAgent decision error.", {"error": str(mca_e)}, level="CRITICAL")
                        # Potentially add a default recovery task or halt. For now, continue to see if existing tasks can proceed.


            # --- Task Execution ---
            task = self.workflow_state.get_next_task() # Pops task and marks 'in_progress'

            if not task: # No tasks in queue (even after MCA potentially added some)
                if self.workflow_state.are_all_chapters_completed() and \
                   self.workflow_state.get_flag('outline_finalized', False) and \
                   not self.workflow_state.get_flag('report_compilation_requested', False):

                    self.workflow_state.add_task(TASK_TYPE_COMPILE_REPORT, priority=100)
                    self.workflow_state.set_flag('report_compilation_requested', True)
                    self.workflow_state.log_event("Auto-triggered COMPILE_REPORT as conditions met.")
                    # Loop will continue to pick up this new task in the next iteration.

                elif self.workflow_state.get_flag('report_generation_complete'):
                    break # Already marked complete by a task (e.g., compile_report)

                else: # No tasks, and not ready for compilation or already complete
                    # This might be a natural pause if MCA decided no tasks, or a stall.
                    self.workflow_state.log_event("Task queue empty. Report not yet complete. MCA might add tasks next cycle or workflow stalling.", level="INFO")
                    # Simple stall detection: if queue is empty for too many iterations.
                    # This needs a counter that resets when tasks are added or MCA runs.
                    # For now, the main iteration_count and max_workflow_iterations will handle hard stop.
                    if iteration_count > 5 and not self.master_control_agent : # If no MCA to add tasks, and queue empty > 5 times
                         logger.warning("Orchestrator (no MCA): Potential stall. Halting.")
                         self.workflow_state.set_flag('report_generation_complete', True) # Force stop
                         break
            else: # Task found in queue
                self._execute_task_type(task)

            iteration_count += 1
            self.workflow_state.log_event(f"Orchestrator: Workflow iteration {iteration_count} complete.")

        self.workflow_state.log_event("Orchestrator finished workflow coordination.",
                                     {"total_iterations": iteration_count,
                                      "final_status_complete": self.workflow_state.get_flag('report_generation_complete')})

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
    logger.info("Orchestrator Example Start")

    # --- Mock Agents and WorkflowState for testing Orchestrator ---
    from agents.base_agent import BaseAgent # For MockAgent
    from agents.master_control_agent import MasterControlAgent # For type hint and mock
    from core.llm_service import LLMService # For mock MasterControlAgent

    class MockMasterControlAgent(BaseAgent): # Inherits BaseAgent for name, not functionality
        def __init__(self):
            super().__init__(agent_name="MockMasterControlAgent")
            self.decision_log = []
        def decide_next_actions(self, workflow_state: WorkflowState) -> List[Dict[str, Any]]:
            logger.info(f"MockMasterControlAgent deciding actions. Pending tasks: {len(workflow_state.pending_tasks)}")
            self.decision_log.append(workflow_state.get_flag('current_iteration_for_mca_decision', 0)) # Log iteration
            # Simple mock: if no tasks, and not analyzed, analyze. If analyzed, generate outline. etc.
            # This mock doesn't use LLM but simulates some basic rule-based decisions.
            if not workflow_state.get_flag('topic_analyzed'):
                return [{"type": TASK_TYPE_ANALYZE_TOPIC, "payload": {"user_topic": workflow_state.user_topic}}]
            if not workflow_state.get_flag('outline_generated'):
                 # Check if topic_analysis_results exists before accessing it
                if workflow_state.topic_analysis_results:
                    return [{"type": TASK_TYPE_GENERATE_OUTLINE, "payload": {"topic_details": workflow_state.topic_analysis_results}}]
                else: # Should not happen if topic_analyzed is true and it worked
                    logger.warning("MCA: Topic analyzed but no results found in state. Cannot generate outline task.")
                    return []


            # If outline is generated, find pending chapters and add PROCESS_CHAPTER tasks
            if workflow_state.parsed_outline and not workflow_state.get_flag('all_chapters_processed_flag', False): # Add a new flag
                pending_chapters = False
                for item in workflow_state.parsed_outline:
                    ch_data = workflow_state.get_chapter_data(item['id'])
                    if not ch_data or ch_data.get('status', STATUS_PENDING) not in [STATUS_COMPLETED, STATUS_ERROR, "writing_needed_after_mock_retrieve"]: # Check if not done
                        # Add task only if not already in pending_tasks or recently processed by specific type
                        # This check is complex, for mock, let's assume we add if not completed
                        # A real MCA LLM would look at pending tasks too.
                        # For this mock, let's just add one PROCESS_CHAPTER task if any chapter is not COMPLETED.
                        if ch_data.get('status', STATUS_PENDING) != STATUS_COMPLETED:
                             logger.info(f"MCA: Found chapter {item['id']} not completed. Adding PROCESS_CHAPTER.")
                             workflow_state.set_flag('all_chapters_processed_flag', False) # Reset if we add one
                             return [{"type": TASK_TYPE_PROCESS_CHAPTER, "payload": {"chapter_key": item['id'], "chapter_title": item['title']}}]
                if not pending_chapters: # If loop finishes and no pending found
                     workflow_state.set_flag('all_chapters_processed_flag', True)


            if workflow_state.are_all_chapters_completed() and workflow_state.get_flag('outline_finalized'):
                return [{"type": TASK_TYPE_COMPILE_REPORT, "payload": {}}]

            return [] # Default to no new tasks if conditions above aren't met


    class MockWorkerAgent(BaseAgent): # Generic worker agent for testing orchestrator
        def __init__(self, agent_name, task_type_it_handles, next_task_type_to_trigger_mca_for=None):
            super().__init__(agent_name=agent_name)
            self.task_type = task_type_it_handles
            self.next_task_for_mca = next_task_type_to_trigger_mca_for

        def execute_task(self, workflow_state: WorkflowState, task_payload: Dict):
            task_id = workflow_state.current_processing_task_id
            logger.info(f"MockWorkerAgent '{self.agent_name}' executing task '{self.task_type}' for task_id '{task_id}'.")

            # Simulate work
            if self.task_type == TASK_TYPE_ANALYZE_TOPIC:
                workflow_state.update_topic_analysis({"analysis_by": self.agent_name})
                workflow_state.set_flag('topic_analyzed', True)
            elif self.task_type == TASK_TYPE_GENERATE_OUTLINE:
                workflow_state.update_outline("- MockCh1\n- MockCh2", [{'id':'mc1','title':'MockCh1','level':1},{'id':'mc2','title':'MockCh2','level':1}])
                workflow_state.set_flag('outline_generated', True)
            elif self.task_type == TASK_TYPE_RETRIEVE_FOR_CHAPTER:
                key = task_payload['chapter_key']
                entry = workflow_state._get_chapter_entry(key, True)
                entry['retrieved_docs'] = ["doc for "+key]
                workflow_state.update_chapter_status(key, STATUS_WRITING_NEEDED)
                 # This agent now directly adds the next logical step in its own domain
                workflow_state.add_task(TASK_TYPE_WRITE_CHAPTER, payload=task_payload)
            elif self.task_type == TASK_TYPE_WRITE_CHAPTER:
                key = task_payload['chapter_key']
                workflow_state.update_chapter_content(key, "written content for "+key)
                workflow_state.update_chapter_status(key, STATUS_EVALUATION_NEEDED)
                workflow_state.add_task(TASK_TYPE_EVALUATE_CHAPTER, payload=task_payload)
            elif self.task_type == TASK_TYPE_EVALUATE_CHAPTER:
                key = task_payload['chapter_key']
                workflow_state.add_chapter_evaluation(key, {"score": 90}) # Assume good score
                workflow_state.update_chapter_status(key, STATUS_COMPLETED)
            elif self.task_type == TASK_TYPE_COMPILE_REPORT:
                workflow_state.set_flag('final_report_md', "## Final Mock Report by Orchestrator")

            workflow_state.complete_task(task_id, f"{self.agent_name} completed {self.task_type}")

            # This mock agent doesn't add macro tasks itself; relies on MCA via Orchestrator.
            # Exception: chapter processing sub-tasks like retrieve->write->eval can be chained by worker agents.

    mock_wf_state_orch = WorkflowState(user_topic="Orchestrator with MCA Test")
    mock_llm_svc_for_mca = LLMService(api_url="mock://llm_mca", model_name="mock_mca_model") # Mock this if MCA uses LLM

    # In a real scenario, MasterControlAgent would be an LLM-based agent.
    # For this test, we use a rule-based MockMasterControlAgent.
    mock_mca = MockMasterControlAgent()


    orchestrator = Orchestrator(
        workflow_state=mock_wf_state_orch,
        master_control_agent=mock_mca, # Pass the MCA
        topic_analyzer=MockWorkerAgent("TpcAnlyzr", TASK_TYPE_ANALYZE_TOPIC),
        outline_generator=MockWorkerAgent("OutlnGen", TASK_TYPE_GENERATE_OUTLINE),
        content_retriever=MockWorkerAgent("Retrvr", TASK_TYPE_RETRIEVE_FOR_CHAPTER),
        chapter_writer=MockWorkerAgent("ChWritr", TASK_TYPE_WRITE_CHAPTER),
        evaluator=MockWorkerAgent("Evaltr", TASK_TYPE_EVALUATE_CHAPTER),
        refiner=MockWorkerAgent("Refinr", TASK_TYPE_REFINE_CHAPTER), # Not hit in this simple mock flow
        report_compiler=MockWorkerAgent("Compilr", TASK_TYPE_COMPILE_REPORT),
        max_workflow_iterations=30 # Increased limit for multi-step mock
    )

    # Initial task needs to be added by the pipeline before calling orchestrator usually
    # For this direct test of orchestrator, let's assume pipeline would do this:
    # mock_wf_state_orch.add_task(TASK_TYPE_ANALYZE_TOPIC, {"user_topic": "Orchestrator with MCA Test"})
    # No, the MCA should decide the first task. So, start with empty queue.
    mock_wf_state_orch.set_flag('outline_finalized', True) # For simplicity in test

    logger.info("\n--- Starting Orchestrator.coordinate_workflow() with MockMCA ---")
    try:
        orchestrator.coordinate_workflow()

        logger.info("\n--- Orchestrator.coordinate_workflow() finished ---")
        print(f"Workflow Complete Flag: {mock_wf_state.get_flag('report_generation_complete')}")
        print(f"Final Report MD (from flag): {mock_wf_state.get_flag('final_report_md')}")
        print(f"Total errors: {mock_wf_state.error_count}")

        assert mock_wf_state.get_flag('report_generation_complete') is True
        assert "Mock Final Report" in (mock_wf_state.get_flag('final_report_md') or "")

        print("\nOrchestrator example test successful.")

    except Exception as e:
        print(f"Error during Orchestrator test: {e}")
        import traceback
        traceback.print_exc()

    logger.info("\nOrchestrator Example End")
