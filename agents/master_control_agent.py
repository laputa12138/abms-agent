import logging
import json
from typing import Dict, List, Any, Optional

from agents.base_agent import BaseAgent
from core.llm_service import LLMService, LLMServiceError
from core.workflow_state import WorkflowState, TASK_TYPE_ANALYZE_TOPIC, \
    TASK_TYPE_GENERATE_OUTLINE, TASK_TYPE_PROCESS_CHAPTER, TASK_TYPE_RETRIEVE_FOR_CHAPTER, \
    TASK_TYPE_WRITE_CHAPTER, TASK_TYPE_EVALUATE_CHAPTER, TASK_TYPE_REFINE_CHAPTER, \
    TASK_TYPE_COMPILE_REPORT, STATUS_PENDING, STATUS_COMPLETED, STATUS_ERROR, \
    STATUS_WRITING_NEEDED, STATUS_EVALUATION_NEEDED, STATUS_REFINEMENT_NEEDED

logger = logging.getLogger(__name__)

class MasterControlAgentError(Exception):
    """Custom exception for MasterControlAgent errors."""
    pass

class MasterControlAgent(BaseAgent):
    """
    An LLM-driven agent responsible for high-level decision making and task planning
    for the report generation workflow. It analyzes the current WorkflowState and
    decides the next best actions (tasks) to take.
    """

    DEFAULT_MASTER_PROMPT_TEMPLATE = """你是一位高级项目经理和经验丰富的报告总编。你的目标是指导一个由多个AI助手组成的团队，根据用户提供的主题，高效地生成一份结构合理、内容详实、逻辑清晰、质量上乘的中文报告。

当前用户主题："{user_topic}"
报告暂定标题："{report_title}"

**当前工作流状态摘要：**
{workflow_summary}

**你的任务是：**
根据上述当前状态，并着眼于最终生成高质量报告的目标，决定下一步应该执行的一个或多个任务。
请仔细分析现有信息，特别是各章节的状态、评估反馈（如果有）以及任何已发生的错误。

**可用行动/任务类型及预期负载（payload）结构：**
1.  `{task_analyze_topic}`: 分析初始用户主题。
    - `payload`: {{\"user_topic\": \"{user_topic}\"}} (通常仅在开始时或需要重新分析主题时使用)
2.  `{task_generate_outline}`: 根据主题分析结果生成报告大纲。
    - `payload`: {{\"topic_details\": {{...}} }} (需要 'topic_analysis_results' 字典)
3.  `{task_process_chapter}`: 启动对一个新章节的处理流程（通常包括检索、写作等）。
    - `payload`: {{\"chapter_key\": \"unique_chapter_id\", \"chapter_title\": \"章节标题\", \"level\": int}}
4.  `{task_retrieve_for_chapter}`: (通常由PROCESS_CHAPTER触发) 为指定章节检索信息。
    - `payload`: {{\"chapter_key\": \"unique_chapter_id\", \"chapter_title\": \"章节标题\"}}
5.  `{task_write_chapter}`: (通常由RETRIEVE_FOR_CHAPTER后触发) 撰写指定章节内容。
    - `payload`: {{\"chapter_key\": \"unique_chapter_id\", \"chapter_title\": \"章节标题\"}}
6.  `{task_evaluate_chapter}`: 评估已撰写的章节内容。
    - `payload`: {{\"chapter_key\": \"unique_chapter_id\", \"chapter_title\": \"章节标题\"}}
7.  `{task_refine_chapter}`: 根据评估反馈精炼章节内容。
    - `payload`: {{\"chapter_key\": \"unique_chapter_id\", \"chapter_title\": \"章节标题\"}}
8.  `{task_compile_report}`: 当所有章节完成且大纲最终确定后，编译完整报告。
    - `payload`: {{}} (无特定负载)
(未来可能增加的任务类型: SUGGEST_OUTLINE_REFINEMENT, REQUEST_USER_FEEDBACK)

**决策指南：**
-   如果某个章节评估分数过低且未达到最大精炼次数，优先考虑精炼 (`{task_refine_chapter}`)。
-   如果章节信息不足导致评估不佳，可以考虑重新为该章节检索信息 (`{task_retrieve_for_chapter}`)，或者（更高级）提议修改大纲以更好地覆盖主题。
-   如果所有章节都已标记为“完成” (`{status_completed}`) 并且大纲已最终确定 (`outline_finalized: true` in flags)，则应生成 `{task_compile_report}` 任务。
-   优先处理优先级较高的现有待处理任务（如果任务队列非空且有意义）。但你的主要职责是生成新的、战略性的任务。
-   避免产生重复或冲突的任务。

**输出格式要求：**
请以一个JSON列表的形式返回你决策生成的任务。每个任务是一个字典，必须包含 'type' 和 'payload' 键。'priority' 键可选，默认为0。
例如:
```json
[
  {{
    "type": "{task_process_chapter}",
    "payload": {{\"chapter_key\": "chapter_id_123", "chapter_title": "相关背景介绍", "level": 1}},
    "priority": 10
  }},
  {{
    "type": "{task_evaluate_chapter}",
    "payload": {{\"chapter_key\": "chapter_id_abc", "chapter_title": "核心技术分析"}},
    "priority": 5
  }}
]
```
如果当前无需添加新任务（例如，等待现有高优先级任务完成，或所有工作已完成等待编译），请返回一个空列表 `[]`。
如果遇到无法自行解决的严重问题或流程卡住，可以考虑返回一个包含特定错误或求助信息的任务（暂未定义此类任务，目前请尽量推进流程）。

你的决策：
"""

    def __init__(self, llm_service: LLMService, prompt_template: Optional[str] = None):
        super().__init__(agent_name="MasterControlAgent", llm_service=llm_service)
        self.prompt_template = prompt_template or self.DEFAULT_MASTER_PROMPT_TEMPLATE
        if not self.llm_service:
            raise MasterControlAgentError("LLMService is required for MasterControlAgent.")

    def _summarize_workflow_state(self, workflow_state: WorkflowState) -> str:
        """Generates a text summary of the current workflow state for the LLM."""
        summary_parts = []

        summary_parts.append(f"- **全局标志**: {json.dumps(workflow_state.global_flags)}")
        summary_parts.append(f"- **错误计数**: {workflow_state.error_count}")

        if workflow_state.topic_analysis_results:
            summary_parts.append(f"- **主题分析**: 已完成。")
            summary_parts.append(f"  - 中文泛化主题: {workflow_state.topic_analysis_results.get('generalized_topic_cn')}")
            summary_parts.append(f"  - 中文关键词: {', '.join(workflow_state.topic_analysis_results.get('keywords_cn', []))}")
        else:
            summary_parts.append("- **主题分析**: 尚未进行或未完成。")

        if workflow_state.current_outline_md:
            summary_parts.append(f"- **报告大纲**: 已生成。共 {len(workflow_state.parsed_outline)} 个条目。")
            # Optionally list chapter titles and their status
            for i, item in enumerate(workflow_state.parsed_outline):
                chapter_key = item['id']
                ch_data = workflow_state.chapter_data.get(chapter_key)
                status = ch_data.get('status', STATUS_PENDING) if ch_data else STATUS_PENDING
                eval_score = "N/A"
                if ch_data and ch_data.get('evaluations'):
                    eval_score = ch_data['evaluations'][-1].get('score', "N/A")
                summary_parts.append(f"  - {i+1}. {item['title']} (ID: {chapter_key}, Level: {item['level']}, Status: {status}, Last Score: {eval_score})")
        else:
            summary_parts.append("- **报告大纲**: 尚未生成。")

        # Summarize pending tasks (e.g., count by type, or list a few high-priority ones)
        if workflow_state.pending_tasks:
            summary_parts.append(f"- **待处理任务队列**: {len(workflow_state.pending_tasks)} 个任务。")
            # for task in workflow_state.pending_tasks[:3]: # Show first 3
            #     summary_parts.append(f"  - Type: {task['type']}, Prio: {task['priority']}, Payload: {str(task['payload'])[:50]}...")
        else:
            summary_parts.append("- **待处理任务队列**: 当前为空。")

        # Summarize last few completed tasks
        if workflow_state.completed_tasks:
            summary_parts.append("- **最近完成的任务** (最多3条):")
            for task_info in workflow_state.completed_tasks[-3:]:
                 summary_parts.append(f"  - ID: {task_info['id']}, Status: {task_info['status']}, Summary: {task_info.get('result_summary', 'N/A')}")


        return "\n".join(summary_parts)

    def decide_next_actions(self, workflow_state: WorkflowState) -> List[Dict[str, Any]]:
        """
        Analyzes the workflow state and uses LLM to decide the next set of tasks.
        """
        self._log_input(workflow_state_summary_attempted=True) # Avoid logging full state object

        workflow_summary = self._summarize_workflow_state(workflow_state)

        prompt = self.prompt_template.format(
            user_topic=workflow_state.user_topic,
            report_title=workflow_state.report_title or f"关于“{workflow_state.user_topic}”的分析报告",
            workflow_summary=workflow_summary,
            # Task type constants for the prompt
            task_analyze_topic=TASK_TYPE_ANALYZE_TOPIC,
            task_generate_outline=TASK_TYPE_GENERATE_OUTLINE,
            task_process_chapter=TASK_TYPE_PROCESS_CHAPTER,
            task_retrieve_for_chapter=TASK_TYPE_RETRIEVE_FOR_CHAPTER,
            task_write_chapter=TASK_TYPE_WRITE_CHAPTER,
            task_evaluate_chapter=TASK_TYPE_EVALUATE_CHAPTER,
            task_refine_chapter=TASK_TYPE_REFINE_CHAPTER,
            task_compile_report=TASK_TYPE_COMPILE_REPORT,
            status_completed=STATUS_COMPLETED # For LLM to know what "completed" means
        )

        logger.debug(f"MasterControlAgent Prompt to LLM:\n{prompt}")

        try:
            raw_response = self.llm_service.chat(
                query=prompt,
                system_prompt="你是一个高度智能和有条理的AI项目协调员和决策者。"
            )
            logger.debug(f"MasterControlAgent raw LLM response: {raw_response}")

            # Robust JSON parsing (find JSON block in potentially messy LLM output)
            json_response = None
            try:
                # Try to find the start of a JSON list or object
                json_start_idx = -1
                list_start = raw_response.find('[')
                obj_start = raw_response.find('{')

                if list_start != -1 and (obj_start == -1 or list_start < obj_start) :
                    json_start_idx = list_start
                    json_end_idx = raw_response.rfind(']') + 1
                elif obj_start != -1 : # Could be a single task object, or a list wrapped in something else
                    # This case is trickier if it's not a list of tasks.
                    # For now, assume LLM is asked for a list.
                    # If it returns a single object, we might wrap it in a list.
                    # Let's be strict for now and expect a list.
                    json_start_idx = list_start # Prefer list if both found and list is first.
                    json_end_idx = raw_response.rfind(']') + 1
                    if json_start_idx == -1 : # No list found, maybe it gave one object?
                        # This part needs careful handling if LLM doesn't always return a list.
                        # For now, if no list, assume error or empty.
                        logger.warning("LLM did not return a JSON list as expected for MasterControlAgent decisions.")


                if json_start_idx != -1 and json_end_idx > json_start_idx:
                    json_string = raw_response[json_start_idx:json_end_idx]
                    json_response = json.loads(json_string)
                else: # No JSON array found, maybe it's empty or LLM failed
                    if raw_response.strip() == "[]":
                        json_response = []
                    else:
                        raise json.JSONDecodeError("No JSON array found in LLM response", raw_response, 0)

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response from MasterControlAgent LLM: {raw_response}. Error: {e}")
                workflow_state.log_event("MasterControlAgent LLM response parsing error", {"error": str(e), "raw_response": raw_response}, level="ERROR")
                return [] # Return empty list of tasks on parsing failure

            # Validate tasks
            validated_tasks: List[Dict[str, Any]] = []
            if isinstance(json_response, list):
                for task_dict in json_response:
                    if isinstance(task_dict, dict) and 'type' in task_dict:
                        # Basic validation, more can be added (e.g., payload structure per type)
                        # TODO: Validate task types against a list of known/allowed types.
                        validated_tasks.append({
                            'type': task_dict['type'],
                            'payload': task_dict.get('payload', {}),
                            'priority': task_dict.get('priority', 10) # Default priority for LLM tasks
                        })
                    else:
                        logger.warning(f"Invalid task format from MasterControlAgent LLM: {task_dict}")
                        workflow_state.log_event("MasterControlAgent LLM returned invalid task format", {"task_data": task_dict}, level="WARNING")
            else:
                logger.error(f"MasterControlAgent LLM response was not a list: {json_response}")
                workflow_state.log_event("MasterControlAgent LLM response not a list", {"response": json_response}, level="ERROR")
                return []


            self._log_output(validated_tasks)
            logger.info(f"MasterControlAgent decided on {len(validated_tasks)} new actions.")
            return validated_tasks

        except LLMServiceError as e:
            logger.error(f"LLMServiceError in MasterControlAgent: {e}")
            workflow_state.log_event("MasterControlAgent LLM service error", {"error": str(e)}, level="CRITICAL")
            return [] # Return empty on LLM failure
        except Exception as e:
            logger.error(f"Unexpected error in MasterControlAgent.decide_next_actions: {e}", exc_info=True)
            workflow_state.log_event("MasterControlAgent unexpected error", {"error": str(e)}, level="CRITICAL")
            return []


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

    class MockLLMServiceMCA: # MCA for MasterControlAgent
        def __init__(self, canned_response: str):
            self.canned_response = canned_response
        def chat(self, query: str, system_prompt: str, **kwargs) -> str:
            logger.debug(f"MockLLMServiceMCA received query for MasterControlAgent:\n{query[:500]}...")
            return self.canned_response

    # Mock WorkflowState
    from core.workflow_state import WorkflowState, TASK_TYPE_GENERATE_OUTLINE, TASK_TYPE_PROCESS_CHAPTER

    mock_wf_state_mca = WorkflowState(user_topic="Test LLM-driven MCA", report_title="MCA Test Report")
    # Simulate some initial state
    mock_wf_state_mca.update_topic_analysis({
        "generalized_topic_cn": "LLM驱动的MCA测试",
        "generalized_topic_en": "LLM-driven MCA Test",
        "keywords_cn": ["测试", "MCA", "LLM"], "keywords_en": ["test", "MCA", "LLM"]
    })
    mock_wf_state_mca.set_flag('topic_analyzed', True)
    # No outline yet, so MCA should suggest generating one.

    # --- Test Case 1: LLM suggests generating outline ---
    llm_response_for_outline = json.dumps([
        {"type": TASK_TYPE_GENERATE_OUTLINE, "payload": {"topic_details": mock_wf_state_mca.topic_analysis_results}, "priority": 1}
    ])
    mca_llm_service1 = MockLLMServiceMCA(canned_response=llm_response_for_outline)
    master_agent1 = MasterControlAgent(llm_service=mca_llm_service1)

    print("\n--- Testing MasterControlAgent: Initial state, expecting outline generation task ---")
    next_actions1 = master_agent1.decide_next_actions(mock_wf_state_mca)
    print(f"MCA decided actions: {json.dumps(next_actions1, indent=2, ensure_ascii=False)}")
    assert len(next_actions1) == 1
    assert next_actions1[0]['type'] == TASK_TYPE_GENERATE_OUTLINE

    # --- Test Case 2: LLM suggests processing chapters after outline is 'done' ---
    mock_wf_state_mca.update_outline( # Simulate outline generated
        "- Chapter Foo\n- Chapter Bar",
        [{'id': 'chap_foo', 'title': 'Chapter Foo', 'level': 1}, {'id': 'chap_bar', 'title': 'Chapter Bar', 'level': 1}]
    )
    mock_wf_state_mca.set_flag('outline_generated', True)
    # Clear pending tasks to simulate it's MCA's turn again
    mock_wf_state_mca.pending_tasks = []


    llm_response_for_chapters = json.dumps([
        {"type": TASK_TYPE_PROCESS_CHAPTER, "payload": {"chapter_key": "chap_foo", "chapter_title": "Chapter Foo", "level": 1}},
        {"type": TASK_TYPE_PROCESS_CHAPTER, "payload": {"chapter_key": "chap_bar", "chapter_title": "Chapter Bar", "level": 1}}
    ])
    mca_llm_service2 = MockLLMServiceMCA(canned_response=llm_response_for_chapters)
    master_agent2 = MasterControlAgent(llm_service=mca_llm_service2)

    print("\n--- Testing MasterControlAgent: Outline 'generated', expecting chapter processing tasks ---")
    next_actions2 = master_agent2.decide_next_actions(mock_wf_state_mca)
    print(f"MCA decided actions: {json.dumps(next_actions2, indent=2, ensure_ascii=False)}")
    assert len(next_actions2) == 2
    assert next_actions2[0]['type'] == TASK_TYPE_PROCESS_CHAPTER
    assert next_actions2[1]['type'] == TASK_TYPE_PROCESS_CHAPTER

    # --- Test Case 3: LLM returns malformed JSON ---
    mca_llm_service3 = MockLLMServiceMCA(canned_response="This is not JSON, [ type: blah ")
    master_agent3 = MasterControlAgent(llm_service=mca_llm_service3)
    print("\n--- Testing MasterControlAgent: Malformed LLM JSON response ---")
    next_actions3 = master_agent3.decide_next_actions(mock_wf_state_mca)
    print(f"MCA decided actions (malformed): {json.dumps(next_actions3, indent=2, ensure_ascii=False)}")
    assert len(next_actions3) == 0 # Should return empty list on error

    # --- Test Case 4: LLM returns empty list ---
    mca_llm_service4 = MockLLMServiceMCA(canned_response="[]")
    master_agent4 = MasterControlAgent(llm_service=mca_llm_service4)
    print("\n--- Testing MasterControlAgent: LLM returns empty list ---")
    next_actions4 = master_agent4.decide_next_actions(mock_wf_state_mca)
    print(f"MCA decided actions (empty list): {json.dumps(next_actions4, indent=2, ensure_ascii=False)}")
    assert len(next_actions4) == 0

    logger.info("\nMasterControlAgent Example End")
