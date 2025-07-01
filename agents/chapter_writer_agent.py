import logging
from typing import List, Dict, Optional

from agents.base_agent import BaseAgent
from core.llm_service import LLMService, LLMServiceError
from core.workflow_state import WorkflowState, TASK_TYPE_EVALUATE_CHAPTER, STATUS_EVALUATION_NEEDED # Import constants

logger = logging.getLogger(__name__)

class ChapterWriterAgentError(Exception):
    """Custom exception for ChapterWriterAgent errors."""
    pass

class ChapterWriterAgent(BaseAgent):
    """
    Agent responsible for writing a chapter of a report based on a given
    chapter title and relevant retrieved content (parent chunks).
    Updates WorkflowState with the written content and queues evaluation task.
    """

    DEFAULT_PROMPT_TEMPLATE = """你是一位专业的报告撰写员。请根据以下报告章节标题和相关的参考资料（这些是与章节最相关的父文本块），撰写详细、流畅、专业、连贯的中文章节内容。

章节标题：
{chapter_title}

参考资料（请基于这些资料进行撰写，不要杜撰）：
{retrieved_content_formatted}

请确保内容与章节标题紧密相关，并充分、合理地利用提供的参考资料。避免直接复制粘贴参考资料，而是要理解、整合信息，并用自己的话语有条理地表达出来。
输出的章节内容应具有良好的可读性和专业性。

撰写的章节内容（纯文本，不需要包含章节标题本身）：
"""

    def __init__(self, llm_service: LLMService, prompt_template: Optional[str] = None):
        super().__init__(agent_name="ChapterWriterAgent", llm_service=llm_service)
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE
        if not self.llm_service:
            raise ChapterWriterAgentError("LLMService is required for ChapterWriterAgent.")

    def _format_retrieved_content(self, retrieved_content: List[Dict[str, any]]) -> str:
        """
        Formats the list of retrieved content (parent chunks) for the prompt.
        Input `retrieved_content` is expected to be a list of dicts, where each dict
        has a 'document' key (parent_text) and optionally 'score', 'source'.
        """
        if not retrieved_content:
            return "无参考资料提供。"

        formatted_str = ""
        for i, item in enumerate(retrieved_content):
            # 'document' key now holds the parent_text from RetrievalService output
            doc_text = item.get('document', '无效的参考资料片段')
            score = item.get('score', 'N/A')
            source = item.get('source', 'unknown')
            child_preview = item.get('child_text_preview', '') # Optional preview from retriever

            formatted_str += f"参考资料 {i+1} (来源: {source}, 得分: {score:.4f})\n"
            if child_preview:
                formatted_str += f"[匹配子块预览: {child_preview}]\n"
            formatted_str += f"\"\"\"\n{doc_text}\n\"\"\"\n\n"
        return formatted_str.strip()

    def execute_task(self, workflow_state: WorkflowState, task_payload: Dict) -> None:
        """
        Writes a chapter based on data from WorkflowState and task_payload.
        Updates WorkflowState with the content and adds an evaluation task.

        Args:
            workflow_state (WorkflowState): The current state of the workflow.
            task_payload (Dict): Payload for this task, expects:
                                 'chapter_key': Unique key/ID of the chapter.
                                 'chapter_title': Title of the chapter.
        """
        chapter_key = task_payload.get('chapter_key')
        chapter_title = task_payload.get('chapter_title')

        if not chapter_key or not chapter_title:
            raise ChapterWriterAgentError("Chapter key or title not found in task payload for ChapterWriterAgent.")

        chapter_data = workflow_state.get_chapter_data(chapter_key)
        if not chapter_data:
            # This should ideally not happen if tasks are chained correctly
            workflow_state.log_event(f"Chapter data for key '{chapter_key}' not found.", level="ERROR")
            raise ChapterWriterAgentError(f"Chapter data for key '{chapter_key}' not found in WorkflowState.")

        retrieved_docs = chapter_data.get('retrieved_docs', []) # These are dicts with 'document' as parent_text

        self._log_input(chapter_title=chapter_title, retrieved_docs_count=len(retrieved_docs))

        formatted_content_str = self._format_retrieved_content(retrieved_docs)

        prompt = self.prompt_template.format(
            chapter_title=chapter_title,
            retrieved_content_formatted=formatted_content_str
        )

        try:
            logger.info(f"Sending request to LLM for chapter writing. Chapter: '{chapter_title}' (Key: {chapter_key})")
            chapter_text = self.llm_service.chat(
                query=prompt,
                system_prompt="你是一位精通特定领域知识的专业中文报告撰写者，擅长整合信息并清晰表达。"
            )
            logger.debug(f"Raw LLM response for chapter '{chapter_title}' (first 200 chars): {chapter_text[:200]}")

            if not chapter_text or not chapter_text.strip():
                logger.warning(f"LLM returned empty content for chapter: '{chapter_title}'.")
                # Handle empty content, perhaps by setting an error state or using a placeholder
                # For now, we'll store it as is, and evaluation can pick it up.
                chapter_text = "" # Ensure it's an empty string not None

            # Update WorkflowState
            workflow_state.update_chapter_content(chapter_key, chapter_text.strip(), retrieved_docs=retrieved_docs, is_new_version=False) # First version
            workflow_state.update_chapter_status(chapter_key, STATUS_EVALUATION_NEEDED)

            # Add next task: Evaluate Chapter
            workflow_state.add_task(
                task_type=TASK_TYPE_EVALUATE_CHAPTER,
                payload={'chapter_key': chapter_key, 'chapter_title': chapter_title}, # Pass key and title
                priority=task_payload.get('priority', 6) + 1 # Slightly lower priority
            )

            self._log_output({"chapter_key": chapter_key, "content_length": len(chapter_text)})
            logger.info(f"Chapter writing successful for '{chapter_title}'. Next task (Evaluate Chapter) added.")

        except LLMServiceError as e:
            workflow_state.log_event(f"LLM service error writing chapter '{chapter_title}'", {"error": str(e)}, level="ERROR")
            workflow_state.add_chapter_error(chapter_key, f"LLM service error: {e}")
            raise ChapterWriterAgentError(f"LLM service failed for chapter '{chapter_title}': {e}")
        except Exception as e:
            workflow_state.log_event(f"Unexpected error in ChapterWriterAgent for '{chapter_title}'", {"error": str(e)}, level="CRITICAL")
            workflow_state.add_chapter_error(chapter_key, f"Unexpected error: {e}")
            raise ChapterWriterAgentError(f"Unexpected error writing chapter '{chapter_title}': {e}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

    class MockLLMService: # Same mock as before
        def chat(self, query: str, system_prompt: str) -> str:
            if "ABMS系统概述" in query: return "这是ABMS系统概述的初稿，基于提供的父块上下文。"
            return "模拟生成的章节内容。"

    from core.workflow_state import WorkflowState, TASK_TYPE_EVALUATE_CHAPTER, STATUS_EVALUATION_NEEDED, STATUS_WRITING_NEEDED

    class MockWorkflowStateCWA(WorkflowState): # CWA for ChapterWriterAgent
        def __init__(self, user_topic: str, chapter_key: str, chapter_title: str, retrieved_docs: List):
            super().__init__(user_topic)
            # Pre-populate chapter data as if retrieval was done
            self.chapter_data[chapter_key] = {
                'title': chapter_title, 'level': 1, 'status': STATUS_WRITING_NEEDED,
                'content': None, 'retrieved_docs': retrieved_docs,
                'evaluations': [], 'versions': [], 'errors': []
            }
            self.added_tasks_cwa = []

        def update_chapter_content(self, chapter_key: str, content: str,
                                   retrieved_docs: Optional[List[Dict[str, Any]]] = None,
                                   is_new_version: bool = True):
            super().update_chapter_content(chapter_key, content, retrieved_docs, is_new_version)
            logger.debug(f"MockWorkflowStateCWA: Chapter '{chapter_key}' content updated.")

        def update_chapter_status(self, chapter_key: str, status: str):
            super().update_chapter_status(chapter_key, status)
            logger.debug(f"MockWorkflowStateCWA: Chapter '{chapter_key}' status updated to {status}")

        def add_task(self, task_type: str, payload: Optional[Dict[str, Any]] = None, priority: int = 0):
            self.added_tasks_cwa.append({'type': task_type, 'payload': payload, 'priority': priority})
            super().add_task(task_type, payload, priority)


    llm_service_instance = MockLLMService()
    writer_agent = ChapterWriterAgent(llm_service=llm_service_instance)

    test_chapter_key = "chap_abms_overview"
    test_chapter_title = "ABMS系统概述"
    mock_retrieved_data = [
        {"document": "父块1：ABMS是一个复杂的系统...", "score": 0.9, "source": "hybrid", "child_text_preview": "子块1预览..."},
        {"document": "父块2：JADC2与ABMS的关系...", "score": 0.85, "source": "hybrid", "child_text_preview": "子块2预览..."}
    ]

    mock_state_cwa = MockWorkflowStateCWA(user_topic="ABMS",
                                          chapter_key=test_chapter_key,
                                          chapter_title=test_chapter_title,
                                          retrieved_docs=mock_retrieved_data)

    task_payload_for_agent_cwa = {'chapter_key': test_chapter_key, 'chapter_title': test_chapter_title}

    print(f"\nExecuting ChapterWriterAgent for chapter: '{test_chapter_title}' with MockWorkflowStateCWA")
    try:
        writer_agent.execute_task(mock_state_cwa, task_payload_for_agent_cwa)

        print("\nWorkflowState after ChapterWriterAgent execution:")
        chapter_info = mock_state_cwa.get_chapter_data(test_chapter_key)
        if chapter_info:
            print(f"  Chapter '{test_chapter_key}' Status: {chapter_info.get('status')}")
            print(f"  Content Preview: {chapter_info.get('content', '')[:100]}...")
            print(f"  Retrieved Docs were used: {bool(chapter_info.get('retrieved_docs'))}")

        print(f"  Tasks added by agent: {json.dumps(mock_state_cwa.added_tasks_cwa, indent=2, ensure_ascii=False)}")

        assert chapter_info is not None
        assert chapter_info.get('status') == STATUS_EVALUATION_NEEDED
        assert chapter_info.get('content') == "这是ABMS系统概述的初稿，基于提供的父块上下文。" # From mock LLM
        assert len(mock_state_cwa.added_tasks_cwa) == 1
        assert mock_state_cwa.added_tasks_cwa[0]['type'] == TASK_TYPE_EVALUATE_CHAPTER
        assert mock_state_cwa.added_tasks_cwa[0]['payload']['chapter_key'] == test_chapter_key

        print("\nChapterWriterAgent test successful with MockWorkflowStateCWA.")

    except Exception as e:
        print(f"Error during ChapterWriterAgent test: {e}")
        import traceback
        traceback.print_exc()

    print("\nChapterWriterAgent example finished.")
