import logging
import json
import os
from typing import List, Dict, Optional, Any
from rank_bm25 import BM25Okapi

from config import settings
from core.llm_service import LLMService
from core.embedding_service import EmbeddingService
from core.reranker_service import RerankerService
from core.document_processor import DocumentProcessor, DocumentProcessorError
from core.vector_store import VectorStore, VectorStoreError
from core.retrieval_service import RetrievalService, RetrievalServiceError
from core.workflow_state import WorkflowState, TASK_TYPE_ANALYZE_TOPIC, \
    TASK_TYPE_GENERATE_OUTLINE, TASK_TYPE_PROCESS_CHAPTER, TASK_TYPE_RETRIEVE_FOR_CHAPTER, \
    TASK_TYPE_WRITE_CHAPTER, TASK_TYPE_EVALUATE_CHAPTER, TASK_TYPE_REFINE_CHAPTER, \
    TASK_TYPE_COMPILE_REPORT # STATUS constants are used by agents & orchestrator
from core.orchestrator import Orchestrator # Import Orchestrator

from agents.topic_analyzer_agent import TopicAnalyzerAgent
from agents.outline_generator_agent import OutlineGeneratorAgent
from agents.content_retriever_agent import ContentRetrieverAgent
from agents.chapter_writer_agent import ChapterWriterAgent
from agents.evaluator_agent import EvaluatorAgent
from agents.refiner_agent import RefinerAgent
from agents.report_compiler_agent import ReportCompilerAgent

logger = logging.getLogger(__name__)

class ReportGenerationPipelineError(Exception):
    pass

class ReportGenerationPipeline:
    def __init__(self,
                 llm_service: LLMService,
                 embedding_service: EmbeddingService,
                 reranker_service: Optional[RerankerService] = None,
                 parent_chunk_size: int = settings.DEFAULT_PARENT_CHUNK_SIZE,
                 parent_chunk_overlap: int = settings.DEFAULT_PARENT_CHUNK_OVERLAP,
                 child_chunk_size: int = settings.DEFAULT_CHILD_CHUNK_SIZE,
                 child_chunk_overlap: int = settings.DEFAULT_CHILD_CHUNK_OVERLAP,
                 vector_top_k: int = settings.DEFAULT_VECTOR_STORE_TOP_K,
                 keyword_top_k: int = settings.DEFAULT_KEYWORD_SEARCH_TOP_K,
                 hybrid_alpha: float = settings.DEFAULT_HYBRID_SEARCH_ALPHA,
                 final_top_n_retrieval: Optional[int] = None,
                 max_refinement_iterations: int = settings.DEFAULT_MAX_REFINEMENT_ITERATIONS,
                 max_workflow_iterations: int = 50,
                 vector_store_path: str = settings.DEFAULT_VECTOR_STORE_PATH,
                 index_name: Optional[str] = None,
                 force_reindex: bool = False
                ):

        self.llm_service = llm_service
        self.embedding_service = embedding_service
        self.reranker_service = reranker_service
        self.max_refinement_iterations = max_refinement_iterations # Used by EvaluatorAgent logic via WorkflowState
        self.max_workflow_iterations = max_workflow_iterations # Passed to Orchestrator

        self.vector_store_path = vector_store_path
        self.index_name = index_name
        self.force_reindex = force_reindex

        self.document_processor = DocumentProcessor(
            parent_chunk_size=parent_chunk_size, parent_chunk_overlap=parent_chunk_overlap,
            child_chunk_size=child_chunk_size, child_chunk_overlap=child_chunk_overlap,
            supported_extensions=settings.SUPPORTED_DOC_EXTENSIONS
        )
        self.vector_store = VectorStore(embedding_service=self.embedding_service)

        self.bm25_index: Optional[BM25Okapi] = None
        self.all_child_chunks_for_bm25_mapping: List[Dict[str, Any]] = []

        self.retrieval_service: Optional[RetrievalService] = None
        self.content_retriever_agent: Optional[ContentRetrieverAgent] = None

        self.retrieval_params = {
            "vector_top_k": vector_top_k, "keyword_top_k": keyword_top_k,
            "hybrid_alpha": hybrid_alpha, "final_top_n": final_top_n_retrieval or vector_top_k
        }

        # Initialize all agents here, they will be passed to the Orchestrator
        self.topic_analyzer = TopicAnalyzerAgent(llm_service=self.llm_service)
        self.outline_generator = OutlineGeneratorAgent(llm_service=self.llm_service)
        # ContentRetrieverAgent is initialized in _initialize_retrieval_and_orchestration_components
        self.chapter_writer = ChapterWriterAgent(llm_service=self.llm_service)
        self.evaluator = EvaluatorAgent(llm_service=self.llm_service, refinement_threshold=80) # Example threshold
        self.refiner = RefinerAgent(llm_service=self.llm_service)
        self.report_compiler = ReportCompilerAgent(add_table_of_contents=True)

        self.workflow_state: Optional[WorkflowState] = None
        self.orchestrator: Optional[Orchestrator] = None

        logger.info("ReportGenerationPipeline initialized.")

    def _initialize_retrieval_and_orchestration_components(self):
        """Initializes RetrievalService, ContentRetrieverAgent, and Orchestrator."""
        if not self.workflow_state:
            raise ReportGenerationPipelineError("WorkflowState must be initialized before retrieval/orchestration components.")

        if not self.retrieval_service:
            self.retrieval_service = RetrievalService(
                vector_store=self.vector_store,
                bm25_index=self.bm25_index,
                all_child_chunks_for_bm25_mapping=self.all_child_chunks_for_bm25_mapping,
                reranker_service=self.reranker_service
            )
            self.workflow_state.log_event("RetrievalService initialized.")

        if not self.content_retriever_agent:
            self.content_retriever_agent = ContentRetrieverAgent(
                retrieval_service=self.retrieval_service,
                default_vector_top_k=self.retrieval_params["vector_top_k"],
                default_keyword_top_k=self.retrieval_params["keyword_top_k"],
                default_hybrid_alpha=self.retrieval_params["hybrid_alpha"],
                default_final_top_n=self.retrieval_params["final_top_n"]
            )
            self.workflow_state.log_event("ContentRetrieverAgent initialized using RetrievalService.")

        if not self.orchestrator:
            self.orchestrator = Orchestrator(
                workflow_state=self.workflow_state,
                topic_analyzer=self.topic_analyzer,
                outline_generator=self.outline_generator,
                content_retriever=self.content_retriever_agent,
                chapter_writer=self.chapter_writer,
                evaluator=self.evaluator,
                refiner=self.refiner,
                report_compiler=self.report_compiler,
                max_workflow_iterations=self.max_workflow_iterations
            )
            self.workflow_state.log_event("Orchestrator initialized.")


    def _process_and_load_data(self, data_path: str):
        self.workflow_state.log_event(f"Data processing: data_path='{data_path}', vs_path='{self.vector_store_path}', "
                                     f"index_name='{self.index_name}', force_reindex={self.force_reindex}")
        loaded_from_file = False
        if not self.force_reindex and self.index_name:
            vs_dir = os.path.abspath(self.vector_store_path)
            faiss_index_path = os.path.join(vs_dir, f"{self.index_name}.faiss")
            metadata_path = os.path.join(vs_dir, f"{self.index_name}.meta.json")

            if os.path.exists(faiss_index_path) and os.path.exists(metadata_path):
                try:
                    self.workflow_state.log_event(f"Attempting to load existing VectorStore: index='{faiss_index_path}', meta='{metadata_path}'")
                    self.vector_store.load_store(faiss_index_path, metadata_path)
                    if self.vector_store.count_child_chunks > 0:
                        loaded_from_file = True
                        self.workflow_state.log_event(f"Successfully loaded VectorStore. {self.vector_store.count_child_chunks} child chunks.")
                    else:
                        self.workflow_state.log_event("Loaded VectorStore files but store is empty. Will re-process.", level="WARNING")
                except Exception as e:
                    self.workflow_state.log_event(f"Failed to load existing VectorStore from {self.index_name}: {e}. Will re-process.", level="WARNING")
            else:
                self.workflow_state.log_event(f"No existing index found for '{self.index_name}' at '{vs_dir}'. Will process documents from data_path.")

        if not loaded_from_file:
            self.workflow_state.log_event(f"Processing documents from directory: {data_path}")
            if not os.path.isdir(data_path):
                raise ReportGenerationPipelineError(f"Invalid data_path for processing: {data_path} is not a directory.")

            all_parent_child_data: List[Dict[str, Any]] = []
            processed_file_count = 0
            for filename in os.listdir(data_path):
                file_path = os.path.join(data_path, filename)
                if not os.path.isfile(file_path): continue
                _, extension = os.path.splitext(filename.lower())
                if extension not in settings.SUPPORTED_DOC_EXTENSIONS: continue
                try:
                    raw_text = self.document_processor.extract_text_from_file(file_path)
                    if not raw_text.strip(): continue
                    doc_id_base = os.path.splitext(filename)[0]
                    parent_child_chunks = self.document_processor.split_text_into_parent_child_chunks(raw_text, doc_id_base)
                    all_parent_child_data.extend(parent_child_chunks)
                    processed_file_count +=1
                except Exception as e:
                    self.workflow_state.log_event(f"Error processing file {file_path}", {"error": str(e)}, level="ERROR")

            if not all_parent_child_data:
                raise ReportGenerationPipelineError("No usable content extracted/chunked from data_path to build a new index.")

            self.vector_store = VectorStore(embedding_service=self.embedding_service)
            self.vector_store.add_documents(all_parent_child_data)
            self.workflow_state.log_event(f"Data from {processed_file_count} files processed and added to new VectorStore.",
                                         {"child_chunks_count": self.vector_store.count_child_chunks})

            effective_index_name_to_save = self.index_name or os.path.basename(os.path.normpath(data_path)) or "default_rag_index"
            vs_dir_to_save = os.path.abspath(self.vector_store_path)
            if not os.path.exists(vs_dir_to_save): os.makedirs(vs_dir_to_save, exist_ok=True)

            save_faiss_path = os.path.join(vs_dir_to_save, f"{effective_index_name_to_save}.faiss")
            save_meta_path = os.path.join(vs_dir_to_save, f"{effective_index_name_to_save}.meta.json")
            try:
                self.vector_store.save_store(save_faiss_path, save_meta_path)
                self.workflow_state.log_event(f"New VectorStore saved: index='{save_faiss_path}', meta='{save_meta_path}'")
            except Exception as e:
                 self.workflow_state.log_event(f"Failed to save new VectorStore: {e}. Processing will continue with in-memory store.", level="ERROR")

        self.workflow_state.set_flag('data_loaded', True)

        self.all_child_chunks_for_bm25_mapping = [{"child_id": i['child_id'], "child_text": i['child_text']}
                                                  for i in self.vector_store.document_store]
        if self.all_child_chunks_for_bm25_mapping:
            # TODO: Use a better tokenizer for Chinese for BM25.
            tokenized_corpus = [item['child_text'].lower().split() for item in self.all_child_chunks_for_bm25_mapping]
            self.bm25_index = BM25Okapi(tokenized_corpus)
            self.workflow_state.log_event(f"BM25 index built with {len(tokenized_corpus)} child chunks.")
        else:
            self.bm25_index = None
            self.workflow_state.log_event("No child chunks available to build BM25 index.", level="WARNING")

        self._initialize_retrieval_and_orchestration_components()


    def run(self, user_topic: str, data_path: str, report_title: Optional[str] = None) -> str:
        self.workflow_state = WorkflowState(user_topic, report_title)
        # Pass max_refinement_iterations to workflow_state so agents (Evaluator) can access it
        self.workflow_state.set_flag('max_refinement_iterations', self.max_refinement_iterations)
        self.workflow_state.log_event("Pipeline run initiated.")

        try:
            self._process_and_load_data(data_path)
        except Exception as e:
            self.workflow_state.log_event(f"Critical error during data processing: {e}", {"level": "CRITICAL"},)
            self.workflow_state.set_flag('report_generation_complete', True)
            self.workflow_state.increment_error_count()
            logger.error(f"Pipeline run failed during data processing: {e}", exc_info=True)
            return f"Error: Data processing failed: {e}. Check logs at {self.workflow_state.get_flag('log_file_path', 'log file (path not set)') if self.workflow_state else 'log file'}."


        if not self.orchestrator:
            msg = "Orchestrator not initialized. This is a critical error in pipeline setup."
            self.workflow_state.log_event(msg, level="CRITICAL")
            raise ReportGenerationPipelineError(msg)

        self.workflow_state.add_task(TASK_TYPE_ANALYZE_TOPIC, payload={'user_topic': user_topic}, priority=0)

        try:
            self.orchestrator.coordinate_workflow()
        except Exception as e:
            self.workflow_state.log_event(f"Critical error during workflow coordination: {e}", {"level": "CRITICAL"})
            self.workflow_state.set_flag('report_generation_complete', True) # Ensure loop terminates
            self.workflow_state.increment_error_count()
            logger.error(f"Orchestrator failed: {e}", exc_info=True)
            # Fall through to return error summary

        final_report_md = self.workflow_state.get_flag('final_report_md')
        if final_report_md and self.workflow_state.get_flag('report_generation_complete'):
            self.workflow_state.log_event("Report generation process concluded successfully.")
            return final_report_md
        else:
            self.workflow_state.log_event("Report generation failed or did not produce a complete report.", level="ERROR")
            error_summary = "Workflow finished without generating a report. Check logs. "
            if self.workflow_state.error_count > 0: error_summary += f"Total errors: {self.workflow_state.error_count}. "
            log_path_info = self.workflow_state.get_flag('log_file_path', 'log file (path not set)') if self.workflow_state else 'log file'
            return error_summary + f"See logs (e.g., {log_path_info}) for details."


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
    logger.info("ReportGenerationPipeline (Orchestrator & Indexing Logic) Example Start")

    class MockLLMServiceForPipeline(LLMService):
        def __init__(self): super().__init__(api_url="mock://llm", model_name="mock-llm")
        def chat(self, query, system_prompt, **kwargs):
            # Simulate different responses based on task type indicators in query
            if "主题分析专家" in query: return json.dumps({"generalized_topic_cn": "测试主题", "keywords_cn": ["测试", "流程"]})
            if "报告大纲撰写助手" in query: return "- 章节 A\n- 章节 B"
            if "专业的报告撰写员" in query: return f"这是关于 {kwargs.get('chapter_title', '未知章节')} 的模拟内容。"
            if "资深的报告评审员" in query: return json.dumps({"score": 88, "feedback_cn": "内容良好，符合预期。"})
            if "报告修改专家" in query: return "这是精炼后的模拟内容。"
            return "Default Mock LLM Response"
        def get_model(self, model_name): return self

    class MockEmbeddingServiceForPipeline(EmbeddingService):
        def __init__(self): super().__init__(api_url="mock://emb", model_name="mock-emb")
        def create_embeddings(self, texts): return [[0.5]*5 for _ in texts] # Ensure consistent dummy embeddings
        def get_model(self, model_name): return self

    # Create dummy dirs for test
    dummy_data_dir_pipe = os.path.abspath("temp_pipeline_main_test_data")
    dummy_vs_dir_pipe = os.path.abspath("temp_pipeline_main_test_vs")
    for d_path in [dummy_data_dir_pipe, dummy_vs_dir_pipe]:
        if not os.path.exists(d_path): os.makedirs(d_path, exist_ok=True)

    # Create a dummy document
    with open(os.path.join(dummy_data_dir_pipe, "sample_doc_for_pipeline.txt"), "w", encoding="utf-8") as f:
        f.write("This is sentence one. This is sentence two about testing.\n\nThis is a new paragraph with sentence three for the test.")

    try:
        pipeline = ReportGenerationPipeline(
            llm_service=MockLLMServiceForPipeline(),
            embedding_service=MockEmbeddingServiceForPipeline(),
            reranker_service=None, # Test without reranker first
            vector_store_path=dummy_vs_dir_pipe,
            index_name="pipeline_test_index",
            force_reindex=True, # Force reindex for consistent test runs
            max_refinement_iterations=0, # No refinement for this test
            max_workflow_iterations=30 # Generous limit for test
        )

        # Set log file path in workflow_state for the error message, if needed by main.py setup
        # In a real run, main.py's setup_logging would handle this.
        # For this test, we can simulate it.
        if pipeline.workflow_state: # Should be created by pipeline.run()
             pipeline.workflow_state.set_flag('log_file_path', './logs/test_pipeline_run.log')


        final_report = pipeline.run(
            user_topic="Comprehensive Test of Pipeline with Orchestrator",
            data_path=dummy_data_dir_pipe
        )
        print("\n" + "="*30 + " FINAL REPORT (Mocked - Pipeline with Orchestrator) " + "="*30)
        print(final_report)
        print("="*80)

        # Verify index files were created
        expected_faiss_path = os.path.join(dummy_vs_dir_pipe, "pipeline_test_index.faiss")
        expected_meta_path = os.path.join(dummy_vs_dir_pipe, "pipeline_test_index.meta.json")
        print(f"Checking for Faiss index: {expected_faiss_path} - Exists: {os.path.exists(expected_faiss_path)}")
        print(f"Checking for Meta json: {expected_meta_path} - Exists: {os.path.exists(expected_meta_path)}")
        assert os.path.exists(expected_faiss_path)
        assert os.path.exists(expected_meta_path)


    except Exception as e:
        logger.error(f"Pipeline example with Orchestrator failed: {e}", exc_info=True)
    finally:
        import shutil
        if os.path.exists(dummy_data_dir_pipe): shutil.rmtree(dummy_data_dir_pipe)
        if os.path.exists(dummy_vs_dir_pipe): shutil.rmtree(dummy_vs_dir_pipe)

    logger.info("ReportGenerationPipeline (Orchestrator & Indexing Logic) Example End")
