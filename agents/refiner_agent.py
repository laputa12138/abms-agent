import logging
import json
from typing import Dict, Optional

from agents.base_agent import BaseAgent
from core.llm_service import LLMService, LLMServiceError
from core.workflow_state import WorkflowState, TASK_TYPE_EVALUATE_CHAPTER, STATUS_EVALUATION_NEEDED # Import constants

logger = logging.getLogger(__name__)

class RefinerAgentError(Exception):
    """Custom exception for RefinerAgent errors."""
    pass

class RefinerAgent(BaseAgent):
    """
    Agent responsible for refining (improving) chapter content based on
    evaluation feedback. Updates WorkflowState and queues re-evaluation.
    """

    DEFAULT_PROMPT_TEMPLATE = """你是一位报告修改专家。请根据以下原始内容和评审反馈，对内容进行修改和完善，输出修改后的中文版本。
你的目标是解决反馈中指出的问题，并提升内容的整体质量，包括相关性、流畅性、完整性和准确性。

章节标题：{chapter_title}

原始内容：
---
{original_content}
---

评审反馈：
---
{evaluation_feedback}
---

请仔细阅读评审反馈，理解需要改进的关键点。
在修改时，请尽量保留原始内容的合理部分，重点针对反馈中提出的不足之处进行优化。
如果反馈中包含具体的修改建议，请优先考虑采纳。

修改后的内容（纯文本）：
"""

    def __init__(self, llm_service: LLMService, prompt_template: Optional[str] = None):
        super().__init__(agent_name="RefinerAgent", llm_service=llm_service)
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE
        if not self.llm_service:
            raise RefinerAgentError("LLMService is required for RefinerAgent.")

    def _format_feedback(self, feedback_data: Dict[str, any]) -> str:
        """Formats the structured feedback from EvaluatorAgent into a string for the prompt."""
        score = feedback_data.get('score', 'N/A')
        feedback_text = feedback_data.get('feedback_cn', '无具体反馈文本。')
        criteria_met = feedback_data.get('evaluation_criteria_met', {})

        criteria_str = "\n具体评估标准反馈：\n"
        if isinstance(criteria_met, dict):
            for k, v in criteria_met.items():
                criteria_str += f"- {k}: {v}\n"
        else: # Handle case where it might not be a dict (e.g. if parsing failed or format changed)
            criteria_str += str(criteria_met) + "\n"

        return f"总体评分: {score}\n\n评审意见:\n{feedback_text}\n{criteria_str if criteria_met else ''}".strip()

    def execute_task(self, workflow_state: WorkflowState, task_payload: Dict) -> None:
        """
        Refines chapter content based on evaluation feedback stored in WorkflowState.
        Updates WorkflowState with refined content and queues for re-evaluation.

        Args:
            workflow_state (WorkflowState): The current state of the workflow.
            task_payload (Dict): Payload, expects 'chapter_key' and 'chapter_title'.
        """
        chapter_key = task_payload.get('chapter_key')
        chapter_title = task_payload.get('chapter_title')

        if not chapter_key or not chapter_title:
            raise RefinerAgentError("Chapter key or title not found in task payload for RefinerAgent.")

        chapter_data = workflow_state.get_chapter_data(chapter_key)
        if not chapter_data or not chapter_data.get('content'):
            workflow_state.log_event(f"No original content to refine for chapter '{chapter_title}'. Skipping refinement.", level="WARNING")
            # Potentially mark as error or re-queue evaluation if this state is unexpected
            workflow_state.update_chapter_status(chapter_key, STATUS_EVALUATION_NEEDED) # Back to eval to see if it can proceed
            return

        original_content = chapter_data['content']

        # Get the latest evaluation feedback
        evaluations = chapter_data.get('evaluations', [])
        if not evaluations:
            workflow_state.log_event(f"No evaluation feedback found for chapter '{chapter_title}' to refine upon. Skipping refinement.", level="WARNING")
            workflow_state.update_chapter_status(chapter_key, STATUS_EVALUATION_NEEDED) # Re-evaluate if no feedback
            return

        latest_evaluation_feedback = evaluations[-1] # Use the most recent one

        self._log_input(chapter_key=chapter_key, original_content_length=len(original_content),
                        evaluation_feedback_score=latest_evaluation_feedback.get('score'))

        formatted_feedback_str = self._format_feedback(latest_evaluation_feedback)

        prompt = self.prompt_template.format(
            chapter_title=chapter_title, # Add chapter_title to prompt for context
            original_content=original_content,
            evaluation_feedback=formatted_feedback_str
        )

        try:
            logger.info(f"Sending request to LLM for content refinement of chapter '{chapter_title}'.")
            refined_text = self.llm_service.chat(
                query=prompt,
                system_prompt="你是一位经验丰富的编辑和内容优化师，擅长根据反馈精确改进文稿。"
            )
            logger.debug(f"Raw LLM response for refinement (first 200 chars): {refined_text[:200]}")

            if not refined_text or not refined_text.strip():
                logger.warning(f"LLM returned empty refined content for '{chapter_title}'. Keeping original.")
                # If LLM returns empty, it implies no changes or failure. We keep original and re-evaluate.
                # The evaluation can then decide if it's "complete" or needs another try/different action.
                refined_text_to_store = original_content
            else:
                refined_text_to_store = refined_text.strip()

            # Update WorkflowState with the refined content (as a new version)
            workflow_state.update_chapter_content(chapter_key, refined_text_to_store, is_new_version=True)
            workflow_state.update_chapter_status(chapter_key, STATUS_EVALUATION_NEEDED) # Always re-evaluate after refinement

            # Add next task: Re-evaluate Chapter
            workflow_state.add_task(
                task_type=TASK_TYPE_EVALUATE_CHAPTER,
                payload={'chapter_key': chapter_key, 'chapter_title': chapter_title},
                priority=task_payload.get('priority', 8) + 1
            )

            self._log_output({"chapter_key": chapter_key, "refined_content_length": len(refined_text_to_store)})
            logger.info(f"Refinement successful for '{chapter_title}'. Next task (Evaluate Chapter) added.")

        except LLMServiceError as e:
            workflow_state.log_event(f"LLM service error during refinement of chapter '{chapter_title}'", {"error": str(e)}, level="ERROR")
            workflow_state.add_chapter_error(chapter_key, f"LLM service error during refinement: {e}")
            raise RefinerAgentError(f"LLM service failed for chapter '{chapter_title}': {e}")
        except Exception as e:
            workflow_state.log_event(f"Unexpected error in RefinerAgent for '{chapter_title}'", {"error": str(e)}, level="CRITICAL")
            workflow_state.add_chapter_error(chapter_key, f"Unexpected error during refinement: {e}")
            raise RefinerAgentError(f"Unexpected error refining chapter '{chapter_title}': {e}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

    class MockLLMService: # Same mock as before
        def chat(self, query: str, system_prompt: str) -> str:
            if "原始内容" in query and "评审反馈" in query:
                original_content_part = query.split("原始内容：\n---\n")[1].split("\n---")[0]
                return original_content_part + "\n\n[LLM Mock Refinement: 针对反馈进行了修改。]"
            return "（模拟的LLM无法处理此refinement请求）"

    from core.workflow_state import WorkflowState, TASK_TYPE_EVALUATE_CHAPTER, STATUS_EVALUATION_NEEDED, STATUS_REFINEMENT_NEEDED

    class MockWorkflowStateRA(WorkflowState): # RA for RefinerAgent
        def __init__(self, user_topic: str, chapter_key: str, chapter_title: str, original_content: str, latest_eval: Dict):
            super().__init__(user_topic)
            self.chapter_data[chapter_key] = {
                'title': chapter_title, 'level': 1, 'status': STATUS_REFINEMENT_NEEDED,
                'content': original_content, 'retrieved_docs': [],
                'evaluations': [latest_eval], # Ensure there's an evaluation to refine upon
                'versions': [], 'errors': []
            }
            self.added_tasks_ra = []

        def update_chapter_content(self, chapter_key: str, content: str,
                                   retrieved_docs: Optional[List[Dict[str, Any]]] = None, # Added List type hint
                                   is_new_version: bool = True):
            super().update_chapter_content(chapter_key, content, retrieved_docs, is_new_version)
            logger.debug(f"MockWorkflowStateRA: Chapter '{chapter_key}' content updated by Refiner.")

        def update_chapter_status(self, chapter_key: str, status: str):
            super().update_chapter_status(chapter_key, status)
            logger.debug(f"MockWorkflowStateRA: Chapter '{chapter_key}' status updated to {status} by Refiner.")

        def add_task(self, task_type: str, payload: Optional[Dict[str, Any]] = None, priority: int = 0):
            self.added_tasks_ra.append({'type': task_type, 'payload': payload, 'priority': priority})
            super().add_task(task_type, payload, priority)


    llm_service_instance = MockLLMService()
    refiner_agent = RefinerAgent(llm_service=llm_service_instance)

    test_chap_key_refine = "chap_to_refine"
    test_chap_title_refine = "Chapter Being Refined"
    original_c = "This is the first draft. It has some issues."
    evaluation_f = {"score": 60, "feedback_cn": "Needs more examples and better flow.",
                    "evaluation_criteria_met": {"completeness": "不足"}}

    mock_state_ra = MockWorkflowStateRA("Refinement Test", test_chap_key_refine, test_chap_title_refine, original_c, evaluation_f)

    task_payload_for_agent_ra = {'chapter_key': test_chap_key_refine, 'chapter_title': test_chap_title_refine}

    print(f"\nExecuting RefinerAgent for chapter: '{test_chap_title_refine}' with MockWorkflowStateRA")
    try:
        refiner_agent.execute_task(mock_state_ra, task_payload_for_agent_ra)

        print("\nWorkflowState after RefinerAgent execution:")
        chapter_info = mock_state_ra.get_chapter_data(test_chap_key_refine)
        if chapter_info:
            print(f"  Chapter '{test_chap_key_refine}' Status: {chapter_info.get('status')}")
            print(f"  Content Preview: {chapter_info.get('content', '')[:100]}...")
            print(f"  Number of versions: {len(chapter_info.get('versions', []))}") # Should be 1 if original was saved

        print(f"  Tasks added by agent: {json.dumps(mock_state_ra.added_tasks_ra, indent=2, ensure_ascii=False)}")

        assert chapter_info is not None
        assert chapter_info.get('status') == STATUS_EVALUATION_NEEDED # Back to evaluation
        assert "[LLM Mock Refinement: 针对反馈进行了修改。]" in chapter_info.get('content', '')
        assert len(mock_state_ra.added_tasks_ra) == 1
        assert mock_state_ra.added_tasks_ra[0]['type'] == TASK_TYPE_EVALUATE_CHAPTER

        print("\nRefinerAgent test successful with MockWorkflowStateRA.")

    except Exception as e:
        print(f"Error during RefinerAgent test: {e}")
        import traceback
        traceback.print_exc()

    print("\nRefinerAgent example finished.")
