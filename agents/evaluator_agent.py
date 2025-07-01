import logging
import json
from typing import Dict, Optional

from agents.base_agent import BaseAgent
from core.llm_service import LLMService, LLMServiceError
from core.workflow_state import WorkflowState, TASK_TYPE_REFINE_CHAPTER, STATUS_REFINEMENT_NEEDED, STATUS_COMPLETED # Import constants
from config import settings # For DEFAULT_MAX_REFINEMENT_ITERATIONS

logger = logging.getLogger(__name__)

class EvaluatorAgentError(Exception):
    """Custom exception for EvaluatorAgent errors."""
    pass

class EvaluatorAgent(BaseAgent):
    """
    Agent responsible for evaluating generated chapter content based on predefined criteria.
    Updates WorkflowState with the evaluation and decides if refinement is needed,
    then queues the next task (refinement or marks chapter as complete).
    """

    DEFAULT_PROMPT_TEMPLATE = """你是一位资深的报告评审员。请根据以下标准评估提供的报告内容：
1.  **相关性**：内容是否紧扣主题和章节要求？信息是否与讨论的核心问题直接相关？
2.  **流畅性**：语句是否通顺自然？段落之间过渡是否平滑？逻辑是否清晰？
3.  **完整性**：信息是否全面？论点是否得到了充分的论证和支持？是否涵盖了应有的关键点？
4.  **准确性**：所陈述的事实、数据和信息是否准确无误？（请基于常识或普遍接受的知识进行判断，除非提供了特定领域的参考标准）

章节标题： {chapter_title}

待评估内容：
---
{content_to_evaluate}
---

请对以上内容进行综合评估，并严格按照以下JSON格式返回你的评分和反馈意见。不要添加任何额外的解释或说明文字。
总评分范围为0-100分。反馈意见应具体指出优点和需要改进的地方。

JSON输出格式：
{
  "score": <总评分，整数>,
  "feedback_cn": "具体的中文反馈意见，包括优点和改进建议。",
  "evaluation_criteria_met": {
    "relevance": "<关于相关性的简短评价，例如：高/中/低，具体说明>",
    "fluency": "<关于流畅性的简短评价，例如：优秀/良好/一般/较差，具体说明>",
    "completeness": "<关于完整性的简短评价，例如：非常全面/基本全面/部分缺失/严重缺失，具体说明>",
    "accuracy": "<关于准确性的简短评价（基于常识），例如：高/待核实/部分存疑/低，具体说明>"
  }
}
"""
    # Arbitrary score threshold for deciding if refinement is needed
    REFINEMENT_SCORE_THRESHOLD = 80


    def __init__(self,
                 llm_service: LLMService,
                 prompt_template: Optional[str] = None,
                 refinement_threshold: int = REFINEMENT_SCORE_THRESHOLD):
        super().__init__(agent_name="EvaluatorAgent", llm_service=llm_service)
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE
        self.refinement_threshold = refinement_threshold
        if not self.llm_service:
            raise EvaluatorAgentError("LLMService is required for EvaluatorAgent.")

    def execute_task(self, workflow_state: WorkflowState, task_payload: Dict) -> None:
        """
        Evaluates chapter content from WorkflowState based on task_payload.
        Updates WorkflowState with evaluation and queues next task (refine or complete).

        Args:
            workflow_state (WorkflowState): The current state of the workflow.
            task_payload (Dict): Payload, expects 'chapter_key' and 'chapter_title'.
        """
        chapter_key = task_payload.get('chapter_key')
        chapter_title = task_payload.get('chapter_title') # Get title for prompt context

        if not chapter_key or not chapter_title:
            raise EvaluatorAgentError("Chapter key or title not found in task payload for EvaluatorAgent.")

        chapter_data = workflow_state.get_chapter_data(chapter_key)
        if not chapter_data or not chapter_data.get('content'):
            # This might happen if writing failed or produced empty content.
            workflow_state.log_event(f"No content to evaluate for chapter '{chapter_title}' (Key: {chapter_key}). Marking as error.", level="ERROR")
            workflow_state.add_chapter_error(chapter_key, "No content available for evaluation.")
            workflow_state.update_chapter_status(chapter_key, STATUS_ERROR) # Mark chapter as error
            # No further task added for this chapter from here.
            return

        content_to_evaluate = chapter_data['content']
        self._log_input(chapter_key=chapter_key, content_length=len(content_to_evaluate))

        if not content_to_evaluate.strip():
            logger.warning(f"EvaluatorAgent received empty content for chapter '{chapter_title}'.")
            evaluation_result = {"score": 0, "feedback_cn": "无法评估空内容。",
                                 "evaluation_criteria_met": {k: "无法评估" for k in ["relevance", "fluency", "completeness", "accuracy"]}}
        else:
            prompt = self.prompt_template.format(content_to_evaluate=content_to_evaluate, chapter_title=chapter_title)
            try:
                logger.info(f"Sending request to LLM for evaluation of chapter '{chapter_title}'.")
                raw_response = self.llm_service.chat(query=prompt, system_prompt="你是一个严格且公正的AI内容评审专家。")
                logger.debug(f"Raw LLM response for evaluation: {raw_response}")

                try:
                    json_start_index = raw_response.find('{')
                    json_end_index = raw_response.rfind('}') + 1
                    if json_start_index != -1 and json_end_index != -1 and json_start_index < json_end_index:
                        json_string = raw_response[json_start_index:json_end_index]
                        evaluation_result = json.loads(json_string)
                    else:
                        raise EvaluatorAgentError(f"LLM eval response no valid JSON: {raw_response}")
                except json.JSONDecodeError as e:
                    raise EvaluatorAgentError(f"LLM eval response not valid JSON: {raw_response}. Error: {e}")

                required_keys = ["score", "feedback_cn", "evaluation_criteria_met"]
                if not all(key in evaluation_result for key in required_keys):
                    raise EvaluatorAgentError(f"LLM eval response missing keys: {evaluation_result}")
                if not isinstance(evaluation_result.get("score"), int) or \
                   not isinstance(evaluation_result.get("feedback_cn"), str) or \
                   not isinstance(evaluation_result.get("evaluation_criteria_met"), dict):
                    raise EvaluatorAgentError(f"LLM eval response malformed types: {evaluation_result}")

            except LLMServiceError as e:
                workflow_state.log_event(f"LLM service error evaluating chapter '{chapter_title}'", {"error": str(e)}, level="ERROR")
                workflow_state.add_chapter_error(chapter_key, f"LLM service error during evaluation: {e}")
                raise EvaluatorAgentError(f"LLM service failed for chapter '{chapter_title}': {e}")
            except Exception as e: # Catch other parsing or validation errors from above
                workflow_state.log_event(f"Error processing LLM evaluation response for '{chapter_title}'", {"error": str(e)}, level="ERROR")
                workflow_state.add_chapter_error(chapter_key, f"Processing LLM evaluation response error: {e}")
                raise EvaluatorAgentError(f"Error processing LLM evaluation response: {e}")

        # Update WorkflowState with the evaluation
        workflow_state.add_chapter_evaluation(chapter_key, evaluation_result)

        # Decide next step based on evaluation score and refinement iterations
        # Max refinement iterations should be a global or pipeline-level config.
        # For now, let's assume it's passed or available via settings.
        max_ref_iters = workflow_state.get_flag('max_refinement_iterations', settings.DEFAULT_MAX_REFINEMENT_ITERATIONS)

        num_evaluations = len(workflow_state.get_chapter_data(chapter_key).get('evaluations', []))

        current_score = evaluation_result.get('score', 0)

        if current_score < self.refinement_threshold and num_evaluations <= max_ref_iters:
            workflow_state.update_chapter_status(chapter_key, STATUS_REFINEMENT_NEEDED)
            workflow_state.add_task(
                task_type=TASK_TYPE_REFINE_CHAPTER,
                payload={'chapter_key': chapter_key, 'chapter_title': chapter_title}, # Pass key and title
                priority=task_payload.get('priority', 7) + 1
            )
            logger.info(f"Evaluation of '{chapter_title}' (Score: {current_score}) requires refinement. "
                        f"Attempt {num_evaluations}/{max_ref_iters}. Next task (Refine Chapter) added.")
        else:
            if current_score >= self.refinement_threshold:
                logger.info(f"Evaluation of '{chapter_title}' (Score: {current_score}) meets threshold. Marking as complete.")
            else: # Score is low, but max refinement iterations reached
                logger.warning(f"Evaluation of '{chapter_title}' (Score: {current_score}) is below threshold, "
                               f"but max refinement iterations ({max_ref_iters}) reached. Marking as complete.")
            workflow_state.update_chapter_status(chapter_key, STATUS_COMPLETED)
            # No further task for this chapter from here. Pipeline will check if all chapters are done.

        self._log_output(evaluation_result)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

    class MockLLMService: # Same mock as before
        def chat(self, query: str, system_prompt: str) -> str:
            if "Chapter Title: Good Chapter" in query:
                return json.dumps({"score": 90, "feedback_cn": "内容优秀，无需修改。", "evaluation_criteria_met": {"relevance": "高", "fluency": "优秀", "completeness": "全面", "accuracy": "高"}})
            elif "Chapter Title: Needs Work Chapter" in query:
                return json.dumps({"score": 65, "feedback_cn": "内容有一些问题，需要改进。", "evaluation_criteria_met": {"relevance": "中", "fluency": "一般", "completeness": "部分", "accuracy": "待核实"}})
            return json.dumps({"score": 50, "feedback_cn": "默认评估。", "evaluation_criteria_met": {}})

    from core.workflow_state import WorkflowState, TASK_TYPE_REFINE_CHAPTER, STATUS_COMPLETED, STATUS_REFINEMENT_NEEDED, STATUS_EVALUATION_NEEDED
    from config import settings # For default max iterations

    class MockWorkflowStateEA(WorkflowState): # EA for EvaluatorAgent
        def __init__(self, user_topic: str, chapter_key: str, chapter_title: str, content: str):
            super().__init__(user_topic)
            self.chapter_data[chapter_key] = {
                'title': chapter_title, 'level': 1, 'status': STATUS_EVALUATION_NEEDED,
                'content': content, 'retrieved_docs': [], 'evaluations': [], 'versions': [], 'errors': []
            }
            self.added_tasks_ea = []
            self.max_ref_iter_config = settings.DEFAULT_MAX_REFINEMENT_ITERATIONS # Store this for test

        def add_task(self, task_type: str, payload: Optional[Dict[str, Any]] = None, priority: int = 0):
            self.added_tasks_ea.append({'type': task_type, 'payload': payload, 'priority': priority})
            super().add_task(task_type, payload, priority)

        def get_flag(self, flag_name: str, default: Optional[Any] = None) -> Any:
            if flag_name == 'max_refinement_iterations': return self.max_ref_iter_config
            return super().get_flag(flag_name, default)


    llm_service_instance = MockLLMService()
    evaluator_agent = EvaluatorAgent(llm_service=llm_service_instance, refinement_threshold=80) # Using 80 as threshold

    # Test case 1: Good content, should complete
    chap_key_good = "chap_good"
    chap_title_good = "Good Chapter"
    content_good = "This is some excellent content for the good chapter."
    mock_state_ea_good = MockWorkflowStateEA("Test Topic", chap_key_good, chap_title_good, content_good)
    task_payload_good = {'chapter_key': chap_key_good, 'chapter_title': chap_title_good}

    print(f"\nExecuting EvaluatorAgent for '{chap_title_good}' (expected to complete)")
    try:
        evaluator_agent.execute_task(mock_state_ea_good, task_payload_good)
        chapter_info = mock_state_ea_good.get_chapter_data(chap_key_good)
        print(f"  Chapter '{chap_key_good}' Status: {chapter_info.get('status')}")
        print(f"  Evaluations: {json.dumps(chapter_info.get('evaluations'), indent=2, ensure_ascii=False)}")
        print(f"  Tasks added: {json.dumps(mock_state_ea_good.added_tasks_ea, indent=2, ensure_ascii=False)}")
        assert chapter_info.get('status') == STATUS_COMPLETED
        assert not mock_state_ea_good.added_tasks_ea # No refinement task for good content
    except Exception as e: print(f"Error: {e}")

    # Test case 2: Needs work, should queue refinement
    chap_key_needs_work = "chap_needs_work"
    chap_title_needs_work = "Needs Work Chapter"
    content_needs_work = "This content needs some improvement."
    mock_state_ea_needs_work = MockWorkflowStateEA("Test Topic", chap_key_needs_work, chap_title_needs_work, content_needs_work)
    task_payload_needs_work = {'chapter_key': chap_key_needs_work, 'chapter_title': chap_title_needs_work}

    print(f"\nExecuting EvaluatorAgent for '{chap_title_needs_work}' (expected to queue refinement)")
    try:
        evaluator_agent.execute_task(mock_state_ea_needs_work, task_payload_needs_work)
        chapter_info = mock_state_ea_needs_work.get_chapter_data(chap_key_needs_work)
        print(f"  Chapter '{chap_key_needs_work}' Status: {chapter_info.get('status')}")
        print(f"  Evaluations: {json.dumps(chapter_info.get('evaluations'), indent=2, ensure_ascii=False)}")
        print(f"  Tasks added: {json.dumps(mock_state_ea_needs_work.added_tasks_ea, indent=2, ensure_ascii=False)}")
        assert chapter_info.get('status') == STATUS_REFINEMENT_NEEDED
        assert len(mock_state_ea_needs_work.added_tasks_ea) == 1
        assert mock_state_ea_needs_work.added_tasks_ea[0]['type'] == TASK_TYPE_REFINE_CHAPTER
    except Exception as e: print(f"Error: {e}")

    # Test case 3: Needs work, but max refinements reached (simulate one prior eval)
    mock_state_ea_max_ref = MockWorkflowStateEA("Test Topic", chap_key_needs_work, chap_title_needs_work, content_needs_work)
    mock_state_ea_max_ref.max_ref_iter_config = 1 # Set max iterations to 1 for this test state
    # Simulate a previous evaluation already happened
    mock_state_ea_max_ref.chapter_data[chap_key_needs_work]['evaluations'].append({"score": 60, "feedback_cn": "First attempt was not good."})

    print(f"\nExecuting EvaluatorAgent for '{chap_title_needs_work}' (max refinements reached)")
    try:
        evaluator_agent.execute_task(mock_state_ea_max_ref, task_payload_needs_work)
        chapter_info = mock_state_ea_max_ref.get_chapter_data(chap_key_needs_work)
        print(f"  Chapter '{chap_key_needs_work}' Status: {chapter_info.get('status')}")
        print(f"  Evaluations (count: {len(chapter_info.get('evaluations'))}): {json.dumps(chapter_info.get('evaluations')[-1], indent=2, ensure_ascii=False)}") # Show last eval
        print(f"  Tasks added: {json.dumps(mock_state_ea_max_ref.added_tasks_ea, indent=2, ensure_ascii=False)}")
        assert chapter_info.get('status') == STATUS_COMPLETED # Should be completed despite low score due to max_iterations
        assert not mock_state_ea_max_ref.added_tasks_ea # No new refinement task
    except Exception as e: print(f"Error: {e}")


    print("\nEvaluatorAgent example finished.")
