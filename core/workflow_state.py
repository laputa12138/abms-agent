import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

# Define task type constants for clarity and to avoid typos
TASK_TYPE_ANALYZE_TOPIC = "analyze_topic"
TASK_TYPE_GENERATE_OUTLINE = "generate_outline"
TASK_TYPE_PROCESS_CHAPTER = "process_chapter" # This might be a meta-task
TASK_TYPE_RETRIEVE_FOR_CHAPTER = "retrieve_for_chapter"
TASK_TYPE_WRITE_CHAPTER = "write_chapter"
TASK_TYPE_EVALUATE_CHAPTER = "evaluate_chapter"
TASK_TYPE_REFINE_CHAPTER = "refine_chapter"
TASK_TYPE_COMPILE_REPORT = "compile_report"
TASK_TYPE_SUGGEST_OUTLINE_REFINEMENT = "suggest_outline_refinement" # Agent can suggest
TASK_TYPE_APPLY_OUTLINE_REFINEMENT = "apply_outline_refinement" # Pipeline/Orchestrator handles this

# Define chapter status constants
STATUS_PENDING = "pending"
STATUS_RETRIEVAL_NEEDED = "retrieval_needed"
STATUS_WRITING_NEEDED = "writing_needed"
STATUS_EVALUATION_NEEDED = "evaluation_needed"
STATUS_REFINEMENT_NEEDED = "refinement_needed"
STATUS_COMPLETED = "completed"
STATUS_ERROR = "error"


class WorkflowState:
    """
    Manages the dynamic state of the report generation workflow.
    Acts as a "working memory" or "central nervous system" for the agents and pipeline.
    """
    def __init__(self, user_topic: str, report_title: Optional[str] = None):
        self.workflow_id: str = str(uuid.uuid4())
        self.start_time: datetime = datetime.now()

        self.user_topic: str = user_topic
        self.report_title: Optional[str] = report_title or f"关于“{user_topic}”的分析报告"

        self.topic_analysis_results: Optional[Dict[str, Any]] = None
        self.current_outline_md: Optional[str] = None
        # Parsed outline: List of {'title': str, 'level': int, 'id': str (unique key for chapter_data)}
        self.parsed_outline: List[Dict[str, Any]] = []

        # Chapter data: key is a unique chapter_id (e.g., from parsed_outline)
        self.chapter_data: Dict[str, Dict[str, Any]] = {}

        # Optional: A pool for caching retrieved information to avoid redundant searches
        # Key could be a normalized query string or a content hash.
        self.retrieved_information_pool: Dict[str, List[Dict[str, Any]]] = {}

        # Task queue: List of {'id': str, 'type': str, 'priority': int, 'payload': Dict, 'status': 'pending'/'processing'}
        self.pending_tasks: List[Dict[str, Any]] = []
        self.completed_tasks: List[Dict[str, Any]] = [] # For logging/auditing

        self.workflow_log: List[Tuple[datetime, str, Dict[str, Any]]] = []
        self.global_flags: Dict[str, Any] = {
            'data_loaded': False, # Becomes true after _process_and_load_data
            'topic_analyzed': False,
            'outline_generated': False,
            'outline_finalized': False, # Can be set to True to prevent further outline changes
            'report_compilation_requested': False, # Flag to trigger compilation when all else is done
            'report_generation_complete': False,
            'max_iterations_reached_for_all_chapters': False # Placeholder
        }
        self.error_count: int = 0
        self.current_processing_task_id: Optional[str] = None

        self.log_event("WorkflowState initialized.", {"user_topic": user_topic, "report_title": self.report_title})

    def log_event(self, message: str, details: Optional[Dict[str, Any]] = None):
        timestamp = datetime.now()
        log_entry = (timestamp, message, details or {})
        self.workflow_log.append(log_entry)
        logger.debug(f"[WorkflowState Log - {timestamp.isoformat()}] {message} {details or ''}")

    def add_task(self, task_type: str, payload: Optional[Dict[str, Any]] = None, priority: int = 0) -> str:
        task_id = str(uuid.uuid4())
        task = {
            'id': task_id,
            'type': task_type,
            'priority': priority, # Lower number = higher priority
            'payload': payload or {},
            'status': 'pending', # 'pending', 'in_progress', 'completed', 'failed'
            'added_at': datetime.now()
        }
        self.pending_tasks.append(task)
        # Sort by priority (lower number first), then by time added (earlier first)
        self.pending_tasks.sort(key=lambda t: (t['priority'], t['added_at']))
        self.log_event(f"Task added: {task_type}", {"task_id": task_id, "priority": priority, "payload": payload})
        return task_id

    def get_next_task(self) -> Optional[Dict[str, Any]]:
        if not self.pending_tasks:
            return None
        # Simple FIFO for tasks of the same priority (already sorted by priority then time)
        task = self.pending_tasks.pop(0)
        task['status'] = 'in_progress'
        self.current_processing_task_id = task['id']
        self.log_event(f"Task started: {task['type']}", {"task_id": task['id'], "payload": task['payload']})
        return task

    def complete_task(self, task_id: str, result_summary: Optional[str] = None, status: str = 'success'):
        if self.current_processing_task_id == task_id:
            self.current_processing_task_id = None

        # Find the task (it should have been moved from pending or handled if it was already processed)
        # For simplicity, we assume it was the one popped by get_next_task or handled if this is called directly
        # A more robust system might search for it in an 'in_progress_tasks' list.
        # For now, we just log its completion and move it to completed_tasks.

        # Find the original task dict to move it (this part is a bit simplified)
        original_task_ref = None
        # This is inefficient, a better way would be to hold the task object.
        # For now, let's assume the calling context has the task dict.
        # We'll just record its completion.

        completed_task_info = {
            'id': task_id,
            'completed_at': datetime.now(),
            'status': status,
            'result_summary': result_summary or "N/A"
        }
        self.completed_tasks.append(completed_task_info)
        self.log_event(f"Task completed: {task_id}", {"status": status, "result": result_summary})
        if status == 'failed':
            self.increment_error_count()

    def update_topic_analysis(self, results: Dict[str, Any]):
        self.topic_analysis_results = results
        self.set_flag('topic_analyzed', True)
        self.log_event("Topic analysis results updated.", results)

    def update_outline(self, outline_md: str, parsed_outline: List[Dict[str, Any]]):
        self.current_outline_md = outline_md
        self.parsed_outline = [] # Reset before populating

        # Initialize chapter_data based on the new parsed outline
        # Ensure each outline item has a unique ID for chapter_data key
        temp_chapter_data = {}
        for i, item in enumerate(parsed_outline):
            # item should be like {'title': str, 'level': int} from ReportCompilerAgent._parse_markdown_outline
            # We need a persistent key. Using title might be problematic if titles change.
            # Let's generate a simple unique key/ID for each outline item.
            chapter_key = item.get('id', f"chapter_{i}_{str(uuid.uuid4())[:4]}") # Use existing ID or generate
            item['id'] = chapter_key # Ensure 'id' exists in parsed_outline item

            self.parsed_outline.append(item) # Add item with ID to state's parsed_outline

            # If chapter_key already exists, try to preserve some data, otherwise initialize
            existing_data = self.chapter_data.get(chapter_key, {})
            temp_chapter_data[chapter_key] = {
                'title': item['title'], # Keep title in chapter_data for convenience
                'level': item['level'],
                'status': existing_data.get('status', STATUS_PENDING), # Preserve status if exists
                'content': existing_data.get('content'),
                'retrieved_docs': existing_data.get('retrieved_docs'),
                'evaluations': existing_data.get('evaluations', []),
                'versions': existing_data.get('versions', []),
                'errors': existing_data.get('errors', [])
            }
        self.chapter_data = temp_chapter_data # Replace with new structure
        self.set_flag('outline_generated', True)
        self.set_flag('outline_finalized', False) # New outline means it's not finalized yet
        self.log_event("Outline updated.", {"outline_md_preview": outline_md[:100]+"...", "num_chapters": len(self.parsed_outline)})

    def _get_chapter_entry(self, chapter_key: str, create_if_missing: bool = False) -> Optional[Dict[str, Any]]:
        """Helper to get or optionally create a chapter_data entry."""
        if chapter_key not in self.chapter_data:
            if create_if_missing:
                # Try to find title/level from parsed_outline if key matches an ID
                outline_item = next((item for item in self.parsed_outline if item.get('id') == chapter_key), None)
                title = outline_item.get('title', chapter_key) if outline_item else chapter_key
                level = outline_item.get('level', 0) if outline_item else 0

                self.chapter_data[chapter_key] = {
                    'title': title, 'level': level, 'status': STATUS_PENDING,
                    'content': None, 'retrieved_docs': None,
                    'evaluations': [], 'versions': [], 'errors': []
                }
                self.log_event(f"Chapter entry created on demand: {chapter_key}")
            else:
                logger.warning(f"Accessing non-existent chapter_key '{chapter_key}' without create_if_missing.")
                return None
        return self.chapter_data[chapter_key]

    def update_chapter_status(self, chapter_key: str, status: str):
        entry = self._get_chapter_entry(chapter_key, create_if_missing=True)
        if entry:
            entry['status'] = status
            self.log_event(f"Chapter '{chapter_key}' status updated to: {status}")

    def update_chapter_content(self, chapter_key: str, content: str,
                               retrieved_docs: Optional[List[Dict[str, Any]]] = None,
                               is_new_version: bool = True):
        entry = self._get_chapter_entry(chapter_key, create_if_missing=True)
        if entry:
            if is_new_version and entry.get('content'): # Save previous version
                entry['versions'].append(entry['content'])
            entry['content'] = content
            if retrieved_docs is not None: # Update if new docs are provided
                entry['retrieved_docs'] = retrieved_docs
            self.log_event(f"Chapter '{chapter_key}' content updated.", {"content_length": len(content)})

    def add_chapter_evaluation(self, chapter_key: str, evaluation: Dict[str, Any]):
        entry = self._get_chapter_entry(chapter_key, create_if_missing=True)
        if entry:
            entry['evaluations'].append(evaluation)
            self.log_event(f"Evaluation added for chapter '{chapter_key}'.", {"score": evaluation.get('score')})

    def add_chapter_error(self, chapter_key: str, error_message: str):
        entry = self._get_chapter_entry(chapter_key, create_if_missing=True)
        if entry:
            entry['errors'].append(f"[{datetime.now().isoformat()}] {error_message}")
            entry['status'] = STATUS_ERROR
            self.log_event(f"Error recorded for chapter '{chapter_key}'.", {"error": error_message})


    def set_flag(self, flag_name: str, value: Any):
        self.global_flags[flag_name] = value
        self.log_event(f"Global flag '{flag_name}' set to: {value}")

    def get_flag(self, flag_name: str, default: Optional[Any] = None) -> Any:
        return self.global_flags.get(flag_name, default)

    def get_chapter_data(self, chapter_key: str) -> Optional[Dict[str, Any]]:
        return self.chapter_data.get(chapter_key)

    def get_all_chapter_keys_by_status(self, status: str) -> List[str]:
        return [key for key, data in self.chapter_data.items() if data.get('status') == status]

    def are_all_chapters_completed(self) -> bool:
        if not self.parsed_outline: return False # No outline means nothing to complete
        for item in self.parsed_outline:
            chapter_key = item['id']
            chapter_info = self.chapter_data.get(chapter_key)
            if not chapter_info or chapter_info.get('status') != STATUS_COMPLETED:
                return False
        return True

    def increment_error_count(self):
        self.error_count += 1
        self.log_event("Global error count incremented.", {"current_error_count": self.error_count})

    def get_full_report_context_for_compilation(self) -> Dict[str, Any]:
        """Prepares data needed by ReportCompilerAgent."""
        # Filter chapter_data to only include chapters present in the current parsed_outline
        # and format it as expected by ReportCompilerAgent (title -> content string)
        valid_chapter_contents = {}
        for item in self.parsed_outline:
            chapter_key = item['id']
            data = self.chapter_data.get(chapter_key)
            if data and data.get('status') == STATUS_COMPLETED and data.get('content'):
                valid_chapter_contents[data['title']] = data['content'] # Use title as key for compiler
            else:
                 logger.warning(f"Chapter '{data.get('title', chapter_key) if data else chapter_key}' not completed or has no content for compilation.")


        return {
            "report_title": self.report_title,
            "markdown_outline": self.current_outline_md,
            "chapter_contents": valid_chapter_contents, # Dict[chapter_title, chapter_text_content]
            "report_topic_details": self.topic_analysis_results
        }

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logger.info("WorkflowState Example Start")

    state = WorkflowState(user_topic="AI in Education", report_title="The Impact of AI on Education")

    # Add initial task
    state.add_task(TASK_TYPE_ANALYZE_TOPIC, payload={"user_topic": state.user_topic}, priority=1)

    next_task = state.get_next_task()
    print(f"\nNext task to process: {json.dumps(next_task, indent=2, default=str)}")

    # Simulate completing the task
    if next_task:
        state.update_topic_analysis({"generalized_topic_cn": "人工智能教育", "keywords_cn": ["AI", "教育"]})
        state.complete_task(next_task['id'], result_summary="Topic analyzed successfully.")
        state.add_task(TASK_TYPE_GENERATE_OUTLINE, payload={"topic_details": state.topic_analysis_results}, priority=2)

    next_task = state.get_next_task()
    print(f"\nNext task to process: {json.dumps(next_task, indent=2, default=str)}")

    # Simulate outline generation
    if next_task and next_task['type'] == TASK_TYPE_GENERATE_OUTLINE:
        mock_outline_md = "- Introduction\n  - Background\n- Main Body\n- Conclusion"
        mock_parsed_outline = [
            {'title': 'Introduction', 'level': 1, 'id': 'chap_intro'},
            {'title': 'Background', 'level': 2, 'id': 'chap_intro_bg'},
            {'title': 'Main Body', 'level': 1, 'id': 'chap_main'},
            {'title': 'Conclusion', 'level': 1, 'id': 'chap_conc'}
        ]
        state.update_outline(mock_outline_md, mock_parsed_outline)
        state.complete_task(next_task['id'], result_summary="Outline generated.")

        # Add tasks for each chapter based on new outline structure
        for item in state.parsed_outline:
            state.add_task(TASK_TYPE_PROCESS_CHAPTER, payload={'chapter_key': item['id'], 'chapter_title': item['title']}, priority=3)

    print(f"\nPending tasks after outline generation: {len(state.pending_tasks)}")
    for task in state.pending_tasks:
        print(task)

    # Simulate processing one chapter
    chapter_task = state.get_next_task() # Should be process_chapter for 'chap_intro'
    if chapter_task and chapter_task['type'] == TASK_TYPE_PROCESS_CHAPTER:
        chap_key = chapter_task['payload']['chapter_key']
        state.update_chapter_status(chap_key, STATUS_RETRIEVAL_NEEDED)
        # Simulate retrieval
        state.chapter_data[chap_key]['retrieved_docs'] = [{"document": "Some retrieved parent context for intro.", "score": 0.9}]
        state.update_chapter_status(chap_key, STATUS_WRITING_NEEDED)
        # Simulate writing
        state.update_chapter_content(chap_key, "This is the written introduction.", retrieved_docs=state.chapter_data[chap_key]['retrieved_docs'])
        state.update_chapter_status(chap_key, STATUS_EVALUATION_NEEDED)
        # Simulate evaluation
        state.add_chapter_evaluation(chap_key, {"score": 70, "feedback_cn": "Needs more detail."})
        state.update_chapter_status(chap_key, STATUS_REFINEMENT_NEEDED)
        # Simulate refinement
        state.update_chapter_content(chap_key, "This is the refined and more detailed introduction.", is_new_version=True)
        state.update_chapter_status(chap_key, STATUS_COMPLETED) # Assume refinement was good enough
        state.complete_task(chapter_task['id'], result_summary=f"Chapter {chap_key} processed.")

    print(f"\nChapter data for '{chapter_task['payload']['chapter_key'] if chapter_task else ''}':")
    if chapter_task : print(json.dumps(state.get_chapter_data(chapter_task['payload']['chapter_key']), indent=2, default=str))

    print(f"\nAre all chapters completed? {state.are_all_chapters_completed()}") # Will be false

    # Simulate completing all other chapters
    while True:
        task = state.get_next_task()
        if not task: break
        if task['type'] == TASK_TYPE_PROCESS_CHAPTER:
            key = task['payload']['chapter_key']
            state.update_chapter_content(key, f"Content for {state.chapter_data[key]['title']}.")
            state.update_chapter_status(key, STATUS_COMPLETED)
            state.complete_task(task['id'])

    print(f"\nAre all chapters completed after mock processing? {state.are_all_chapters_completed()}") # Should be true

    if state.are_all_chapters_completed():
        state.set_flag('outline_finalized', True) # Assuming outline is now final
        state.add_task(TASK_TYPE_COMPILE_REPORT, priority=100) # Low priority, run at the end

    compile_task = state.get_next_task()
    if compile_task and compile_task['type'] == TASK_TYPE_COMPILE_REPORT:
        report_context = state.get_full_report_context_for_compilation()
        print("\nContext for report compilation:")
        print(json.dumps(report_context, indent=2, default=str))
        state.complete_task(compile_task['id'])
        state.set_flag('report_generation_complete', True)

    print(f"\nWorkflow log entries: {len(state.workflow_log)}")
    # print("Last log entry:", state.workflow_log[-1] if state.workflow_log else "None")

    logger.info("WorkflowState Example End")
