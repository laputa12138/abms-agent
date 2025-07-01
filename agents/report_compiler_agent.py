import logging
import re # For more robust anchor generation
from typing import List, Dict, Optional

from agents.base_agent import BaseAgent
from core.workflow_state import WorkflowState # For type hinting if execute_task uses it

logger = logging.getLogger(__name__)

class ReportCompilerAgentError(Exception):
    """Custom exception for ReportCompilerAgent errors."""
    pass

class ReportCompilerAgent(BaseAgent):
    """
    Agent responsible for compiling chapters into a single report string.
    It now primarily uses its `compile_report_from_context` method, which
    can be called by the pipeline when a COMPILE_REPORT task is processed.
    The `_parse_markdown_outline` is a utility that might also be used by
    OutlineGeneratorAgent or WorkflowState to structure the outline.
    """

    def __init__(self, add_table_of_contents: bool = True):
        super().__init__(agent_name="ReportCompilerAgent")
        self.add_table_of_contents = add_table_of_contents
        logger.info(f"ReportCompilerAgent initialized. Add Table of Contents: {self.add_table_of_contents}")

    def _generate_anchor(self, title: str) -> str:
        """Generates a GitHub-style anchor link from a title."""
        anchor = title.lower()
        anchor = re.sub(r'[^\w\s-]', '', anchor) # Remove non-alphanumeric (except spaces and hyphens)
        anchor = re.sub(r'\s+', '-', anchor)    # Replace spaces with hyphens
        return anchor

    def _parse_markdown_outline(self, markdown_outline: str) -> List[Dict[str, any]]:
        """
        Parses a Markdown list/header outline into a structured list.
        Each item: {'title': str, 'level': int, 'id': str (unique ID for chapter_data key)}.
        This method is crucial for structuring content and is also used by WorkflowState via OutlineGenerator.
        The 'id' field is expected by WorkflowState.update_outline.
        """
        parsed_items = []
        lines = markdown_outline.strip().split('\n')

        # For list items, keep track of indentation to determine level
        # This is a simplified heuristic. A proper Markdown AST parser would be more robust.
        # Assuming 2 spaces per indent level for list items.
        # Headers (#, ##, etc.) define their own levels.

        for line_idx, line_content in enumerate(lines):
            stripped_line = line_content.strip()
            if not stripped_line:
                continue

            level = 0
            title = ""
            item_id = f"outline_item_{line_idx}_{self._generate_anchor(stripped_line[:20])}" # Basic unique ID

            if stripped_line.startswith("#"):
                level = stripped_line.count("#", 0, 6) # Max header level 6
                title = stripped_line.lstrip("# ").strip()
            elif stripped_line.startswith(("- ", "* ", "+ ")):
                # Calculate level based on leading spaces of the original line
                leading_spaces = len(line_content) - len(line_content.lstrip())
                level = 1 + (leading_spaces // 2) # Base level 1 for list item, +1 for each 2 spaces
                title = stripped_line.lstrip("-*+ ").strip()
            else:
                # Could be a continuation line or non-standard format, skip for now
                # Or, if we want to capture all non-empty lines as potential sections:
                # title = stripped_line
                # level = 1 # Default level
                logger.debug(f"Skipping non-standard outline line: '{line_content}'")
                continue

            if title:
                parsed_items.append({
                    "id": item_id, # WorkflowState will use this as chapter_key
                    "title": title,
                    "level": level
                })

        logger.debug(f"Parsed outline into {len(parsed_items)} items: {parsed_items}")
        return parsed_items


    def _generate_table_of_contents(self, structured_outline: List[Dict[str, any]]) -> str:
        """Generates a Markdown table of contents from the structured outline."""
        if not self.add_table_of_contents or not structured_outline:
            return ""

        toc = "## 目录\n\n"
        for item in structured_outline: # structured_outline now comes from WorkflowState.parsed_outline
            title = item['title']
            level = item.get('level', 1) # Default to level 1 if not specified
            # Use the 'id' from parsed_outline for anchor, as it's the unique key
            anchor = item.get('id', self._generate_anchor(title))

            indent = "  " * (level - 1) if level > 0 else ""
            toc += f"{indent}- [{title}](#{anchor})\n"
        toc += "\n---\n"
        return toc

    def compile_report_from_context(self, report_context: Dict[str, Any]) -> str:
        """
        Compiles the report using data prepared by WorkflowState.get_full_report_context_for_compilation().
        This is the main method called by the pipeline's task handler.
        """
        report_title = report_context.get('report_title', "未命名报告")
        # markdown_outline_str = report_context.get('markdown_outline', "") # Raw MD outline
        # Use the parsed_outline from workflow_state for structure and IDs
        structured_outline_from_state = report_context.get('parsed_outline', [])

        # chapter_contents is Dict[chapter_title (from original MD), chapter_text_content]
        # We need to map this to chapter_ids if we use ids for anchors.
        # For simplicity, if ReportCompilerAgent's _parse_markdown_outline was used to create
        # the structure in workflow_state, then chapter_contents keys should match titles.
        chapter_contents_by_title = report_context.get('chapter_contents', {})
        report_topic_details = report_context.get('report_topic_details')

        self._log_input(report_title=report_title,
                        num_outline_items=len(structured_outline_from_state),
                        num_chapter_contents=len(chapter_contents_by_title))

        if not report_title: raise ReportCompilerAgentError("Report title cannot be empty.")
        # if not markdown_outline_str and not structured_outline_from_state:
        #     raise ReportCompilerAgentError("Markdown outline or structured outline must be provided.")
        if not structured_outline_from_state:
             logger.warning("Compiling report with no structured outline provided. TOC and chapter structure might be affected.")
             # Fallback: if no structured_outline, but we have MD and contents, try to parse MD here.
             # This shouldn't happen if workflow is correct.
             markdown_outline_str = report_context.get('markdown_outline', "")
             if markdown_outline_str:
                 structured_outline_from_state = self._parse_markdown_outline(markdown_outline_str)
             else:
                 raise ReportCompilerAgentError("No outline information (MD or structured) provided.")


        final_report_parts = [f"# {report_title}\n"]

        if report_topic_details:
            topic_cn = report_topic_details.get("generalized_topic_cn", "未提供")
            keywords_cn_list = report_topic_details.get("keywords_cn", [])
            keywords_cn_str = ", ".join(keywords_cn_list) if keywords_cn_list else "无"
            intro = f"## 引言\n\n本报告围绕主题“**{topic_cn}**”展开"
            if keywords_cn_list: intro += f"，重点探讨与关键词“{keywords_cn_str}”相关的议题。\n"
            else: intro += "。\n"
            intro += "报告旨在提供对此主题的深入分析和全面概述。\n\n---\n"
            final_report_parts.append(intro)

        if self.add_table_of_contents:
            toc_md = self._generate_table_of_contents(structured_outline_from_state) # Use structured outline
            if toc_md: final_report_parts.append(toc_md)

        # Iterate through the structured_outline_from_state to maintain order and hierarchy
        for item in structured_outline_from_state:
            # The key for chapter_contents is the title as parsed by _parse_markdown_outline
            # which should match the titles in structured_outline_from_state.
            chapter_title_key = item['title']
            content = chapter_contents_by_title.get(chapter_title_key)
            level = item.get('level', 1)
            item_id_anchor = item.get('id', self._generate_anchor(chapter_title_key)) # Anchor based on unique ID

            if content is None:
                logger.warning(f"No content found for chapter/section: '{chapter_title_key}'. Omitting.")
                # Optionally add placeholder: final_report_parts.append(f"\n{'#' * level} {chapter_title_key}\n\n*内容待定*\n")
                continue

            final_report_parts.append(f"\n<a id=\"{item_id_anchor}\"></a>\n{'#' * level} {chapter_title_key}\n\n{content}\n")

        compiled_report = "".join(final_report_parts).strip()
        self._log_output(compiled_report[:500] + "...") # Log preview
        return compiled_report

    def execute_task(self, workflow_state: WorkflowState, task_payload: Dict) -> None:
        """
        Called by the pipeline to compile the report.
        It fetches all necessary context from WorkflowState.
        """
        self._log_input(workflow_state_id=workflow_state.workflow_id, task_payload=task_payload)

        if not workflow_state.are_all_chapters_completed():
            msg = "Not all chapters are marked as completed. Report compilation aborted."
            workflow_state.log_event(msg, level="WARNING")
            # This task might be re-queued by the pipeline if called prematurely.
            # Or, we can raise an error to signal the pipeline.
            # For now, let's just log and not produce a report if this happens.
            # The pipeline's main loop should ideally only trigger this when ready.
            workflow_state.complete_task(task_payload['id'], msg, status='deferred') # Use task_payload for ID
            return

        report_context = workflow_state.get_full_report_context_for_compilation()
        # Ensure parsed_outline is included, as it's now the source of truth for structure
        report_context['parsed_outline'] = workflow_state.parsed_outline


        try:
            final_report_md = self.compile_report_from_context(report_context)
            workflow_state.set_flag('final_report_md', final_report_md) # Store in state
            workflow_state.set_flag('report_generation_complete', True) # Signal completion
            logger.info("Report compilation successful and stored in WorkflowState.")
        except ReportCompilerAgentError as e:
            workflow_state.log_event(f"Report compilation failed: {e}", level="ERROR")
            workflow_state.add_chapter_error("REPORT_COMPILATION", f"Compilation error: {e}") # Generic error for compilation
            raise # Re-raise to be caught by pipeline's task handler

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

    # Mock WorkflowState and its get_full_report_context_for_compilation method
    from core.workflow_state import WorkflowState, STATUS_COMPLETED # Import for mock
    from datetime import datetime # For workflow_state log

    class MockWorkflowStateRCA(WorkflowState):
        def __init__(self, user_topic: str):
            super().__init__(user_topic)
            self.mock_report_context = {}

        def get_full_report_context_for_compilation(self) -> Dict[str, Any]:
            # Ensure parsed_outline is part of this context if used by compiler
            self.mock_report_context['parsed_outline'] = self.parsed_outline
            return self.mock_report_context

        def are_all_chapters_completed(self) -> bool: return True # Assume true for test
        def set_flag(self, flag_name: str, value: Any): # Override to see what's set
            super().set_flag(flag_name, value)
            logger.debug(f"MockWorkflowStateRCA: Flag '{flag_name}' set to '{value}'")


    compiler_agent = ReportCompilerAgent(add_table_of_contents=True)

    # --- Test _parse_markdown_outline ---
    print("\n--- Testing _parse_markdown_outline ---")
    md_outline_test = """
    # Chapter 1: Intro
    ## Section 1.1: Background
    - Point 1.1.1
    - Point 1.1.2
      - Sub-point 1.1.2.1
    ## Section 1.2: Significance
    # Chapter 2: Methods
    - Method A
    - Method B
    """
    parsed_test_outline = compiler_agent._parse_markdown_outline(md_outline_test)
    print("Parsed test outline:")
    for item in parsed_test_outline: print(f"  {item}")
    assert len(parsed_test_outline) == 8 # Check number of items
    assert parsed_test_outline[0]['level'] == 1 and parsed_test_outline[0]['title'] == "Chapter 1: Intro"
    assert parsed_test_outline[2]['level'] == 3 and parsed_test_outline[2]['title'] == "Point 1.1.1" # Based on simplified list indent
    assert parsed_test_outline[4]['level'] == 4 and parsed_test_outline[4]['title'] == "Sub-point 1.1.2.1"


    # --- Test execute_task ---
    print("\n--- Testing execute_task with MockWorkflowStateRCA ---")
    mock_state_rca = MockWorkflowStateRCA("Compiler Test Topic")

    # Populate mock_state_rca with data that get_full_report_context_for_compilation would use
    # This data would normally be set by previous agents (OutlineGenerator, ChapterWriter)
    mock_state_rca.report_title = "Final Compiled Report Title"
    mock_state_rca.current_outline_md = "- Chapter Alpha\n  - Section Alpha.1\n- Chapter Beta"
    # WorkflowState.update_outline would have called the parser. We simulate that here.
    # The parser used by OutlineGeneratorAgent should be consistent with this one.
    # For testing compile_report_from_context, we need `parsed_outline` in the context.
    mock_state_rca.parsed_outline = compiler_agent._parse_markdown_outline(mock_state_rca.current_outline_md)

    mock_state_rca.topic_analysis_results = {"generalized_topic_cn": "编译测试", "keywords_cn": ["测试", "报告"]}

    # Chapter contents keys must match titles from the parsed outline
    mock_chapter_contents_for_compiler = {}
    for item in mock_state_rca.parsed_outline:
        # Simulate all chapters are completed and have content
        mock_state_rca.chapter_data[item['id']] = { # Use item['id'] as key
            'title': item['title'], 'level': item['level'],
            'status': STATUS_COMPLETED,
            'content': f"This is the final content for {item['title']}.",
            'evaluations': [], 'versions': [], 'errors': []
        }
        mock_chapter_contents_for_compiler[item['title']] = f"This is the final content for {item['title']}."

    # This is what get_full_report_context_for_compilation in WorkflowState should prepare
    mock_state_rca.mock_report_context = {
        "report_title": mock_state_rca.report_title,
        "markdown_outline": mock_state_rca.current_outline_md, # Raw MD for TOC generation if needed
        "chapter_contents": mock_chapter_contents_for_compiler, # title -> content map
        "report_topic_details": mock_state_rca.topic_analysis_results,
        "parsed_outline": mock_state_rca.parsed_outline # Crucial for structured compilation
    }

    # Task payload for the agent's execute_task method
    task_payload_for_agent_rca = {'id': 'compile_task_id'} # ID is important for complete_task

    try:
        compiler_agent.execute_task(mock_state_rca, task_payload_for_agent_rca)

        final_report_output = mock_state_rca.get_flag('final_report_md')
        assert final_report_output is not None
        assert mock_state_rca.get_flag('report_generation_complete') is True

        print("\nGenerated Report (first 500 chars from WorkflowState flag):")
        print(final_report_output[:500] + "...")
        assert "Chapter Alpha" in final_report_output
        assert "Section Alpha.1" in final_report_output
        assert "Chapter Beta" in final_report_output
        assert "目录" in final_report_output # TOC should be there

        print("\nReportCompilerAgent execute_task test successful.")

    except Exception as e:
        print(f"Error during ReportCompilerAgent execute_task test: {e}")
        import traceback
        traceback.print_exc()

    print("\nReportCompilerAgent example finished.")
