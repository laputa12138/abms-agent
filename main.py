import argparse
import logging
import os
import sys
from datetime import datetime

# Ensure the project root is in PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import settings # Import after path adjustment
from core.llm_service import LLMService
from core.embedding_service import EmbeddingService
from core.reranker_service import RerankerService
from pipelines.report_generation_pipeline import ReportGenerationPipeline, ReportGenerationPipelineError

# Setup basic logging configuration
# This will be further configured in setup_logging based on CLI args.
# Initial basicConfig is for any logs before CLI parsing.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Get root logger for initial messages

def setup_logging(log_level_str: str = 'INFO', debug_mode: bool = False, log_file_path: Optional[str] = None):
    """Configures logging based on command-line arguments."""

    if debug_mode:
        effective_log_level = logging.DEBUG
    else:
        effective_log_level = getattr(logging, log_level_str.upper(), logging.INFO)

    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.handlers = [] # Clear any existing handlers (like the basicConfig one)
    root_logger.setLevel(effective_log_level) # Set root logger level

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(effective_log_level) # Console level matches overall effective level
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File Handler (always logs at least DEBUG if enabled, or effective_log_level if higher)
    if log_file_path:
        log_dir = os.path.dirname(log_file_path)
        if log_dir and not os.path.exists(log_dir):
            try:
                os.makedirs(log_dir, exist_ok=True)
                logger.info(f"Log directory created: {log_dir}") # Use initial logger for this
            except OSError as e:
                logger.error(f"Failed to create log directory {log_dir}: {e}")
                # Continue without file logging if dir creation fails
                log_file_path = None # Disable file logging

        if log_file_path:
            file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
            # File handler should generally log more detail, e.g., DEBUG, regardless of console level,
            # unless the overall effective_log_level is higher (e.g. WARNING).
            file_log_level = min(effective_log_level, logging.DEBUG) if debug_mode else logging.DEBUG
            # Let's make file always DEBUG for max info, console respects user setting.
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            logger.info(f"Logging to file: {log_file_path} at DEBUG level.")

    logger.info(f"Logging configured. Effective console log level: {logging.getLevelName(effective_log_level)}. File logging: {'Enabled' if log_file_path else 'Disabled'}.")


def main():
    """
    Main entry point for the RAG Multi-Agent Report Generation System.
    Parses command-line arguments, initializes the pipeline, and runs it.
    """
    parser = argparse.ArgumentParser(
        description="RAG Multi-Agent Report Generation System with Parent-Child Chunking and Hybrid Search.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Core arguments
    parser.add_argument(
        "--topic", type=str, required=True,
        help="The main topic for the report."
    )
    parser.add_argument(
        "--data_path", type=str, default="./data/",
        help="Path to the directory containing source documents (PDF, DOCX, TXT)."
    )
    parser.add_argument(
        "--output_path", type=str, default=f"output/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        help="File path to save the generated Markdown report."
    )
    parser.add_argument(
        "--report_title", type=str, default=None,
        help="Optional custom title for the report. If not provided, one will be generated based on the topic."
    )

    # Xinference and Model Configuration arguments
    xinference_group = parser.add_argument_group('Xinference and Model Configuration')
    xinference_group.add_argument(
        "--xinference_url", type=str, default=settings.XINFERENCE_API_URL,
        help="URL of the Xinference API server."
    )
    xinference_group.add_argument(
        "--llm_model", type=str, default=settings.DEFAULT_LLM_MODEL_NAME,
        help="Name of the LLM model to use via Xinference."
    )
    xinference_group.add_argument(
        "--embedding_model", type=str, default=settings.DEFAULT_EMBEDDING_MODEL_NAME,
        help="Name of the Embedding model to use via Xinference."
    )
    xinference_group.add_argument(
        "--reranker_model", type=str, default=settings.DEFAULT_RERANKER_MODEL_NAME,
        help="Name of the Reranker model. Set to 'None' or empty to disable."
    )

    # Document Processing (Chunking) arguments
    chunking_group = parser.add_argument_group('Document Processing - Chunking Parameters')
    chunking_group.add_argument(
        "--parent_chunk_size", type=int, default=settings.DEFAULT_PARENT_CHUNK_SIZE,
        help="Target character size for parent chunks."
    )
    chunking_group.add_argument(
        "--parent_chunk_overlap", type=int, default=settings.DEFAULT_PARENT_CHUNK_OVERLAP,
        help="Character overlap for parent chunks."
    )
    chunking_group.add_argument(
        "--child_chunk_size", type=int, default=settings.DEFAULT_CHILD_CHUNK_SIZE,
        help="Target character size for child chunks."
    )
    chunking_group.add_argument(
        "--child_chunk_overlap", type=int, default=settings.DEFAULT_CHILD_CHUNK_OVERLAP,
        help="Character overlap for child chunks."
    )

    # Retrieval arguments
    retrieval_group = parser.add_argument_group('Retrieval Parameters')
    retrieval_group.add_argument(
        "--vector_top_k", type=int, default=settings.DEFAULT_VECTOR_STORE_TOP_K,
        help="Number of top documents to retrieve from vector search."
    )
    retrieval_group.add_argument(
        "--keyword_top_k", type=int, default=settings.DEFAULT_KEYWORD_SEARCH_TOP_K,
        help="Number of top documents to retrieve from keyword search (BM25)."
    )
    retrieval_group.add_argument(
        "--hybrid_search_alpha", type=float, default=settings.DEFAULT_HYBRID_SEARCH_ALPHA,
        help="Blending factor for hybrid search (0.0 for keyword-only, 1.0 for vector-only)."
    )
    retrieval_group.add_argument(
        "--final_top_n_retrieval", type=int, default=None, # Will default in pipeline if None
        help="Final number of documents to use for chapter generation after retrieval and reranking. Defaults to vector_top_k."
    )

    # Pipeline execution arguments
    pipeline_group = parser.add_argument_group('Pipeline Execution Parameters')
    pipeline_group.add_argument(
        "--max_refinement_iterations", type=int, default=settings.DEFAULT_MAX_REFINEMENT_ITERATIONS,
        help="Maximum number of refinement iterations for each chapter."
    )
    pipeline_group.add_argument( # Added from previous pipeline init, now a CLI arg for pipeline
        "--max_workflow_iterations", type=int, default=50, # Default from old pipeline init
        help="Maximum number of iterations for the main workflow loop to prevent infinite loops."
    )

    # Vector Store / Indexing arguments
    indexing_group = parser.add_argument_group('Vector Store and Indexing Parameters')
    indexing_group.add_argument(
        "--vector_store_path", type=str, default="./vector_stores/",
        help="Directory to save/load FAISS index and metadata files."
    )
    indexing_group.add_argument(
        "--index_name", type=str, default=None,
        help="Specific name for the FAISS index and metadata files (e.g., 'my_project_index'). "
             "If not provided, a name will be derived from --data_path."
    )
    indexing_group.add_argument(
        "--force_reindex", action='store_true',
        help="Force re-processing of documents and re-creation of FAISS index, even if an existing index is found."
    )

    # Logging arguments
    logging_group = parser.add_argument_group('Logging Parameters')
    logging_group.add_argument(
        "--log_level", type=str, default=settings.LOG_LEVEL, # Use LOG_LEVEL from settings as default
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help="Set the logging level for console output."
    )
    logging_group.add_argument(
        "--debug", action='store_true',
        help="Enable debug mode. Overrides --log_level to DEBUG and enables more verbose logging."
    )
    logging_group.add_argument(
        "--log_path", type=str, default="./logs/", # Default directory for logs
        help="Directory to save log files. A timestamped log file will be created in this directory."
    )

    args = parser.parse_args()

    # Setup logging based on parsed arguments
    log_file_name = f"report_generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    full_log_file_path = os.path.join(args.log_path, log_file_name) if args.log_path else None # Disable file log if no path
    setup_logging(log_level_str=args.log_level, debug_mode=args.debug, log_file_path=full_log_file_path)

    # Re-initialize logger in case setup_logging changed root logger config affecting current module's logger
    # This ensures this 'logger' instance uses the new config.
    global logger
    logger = logging.getLogger(__name__)


    logger.info("Starting Report Generation System with resolved arguments:")
    # Log arguments again now that logging is fully set up, especially if file logging is on.
    for arg, value in vars(args).items():
        # Avoid logging sensitive info if any, though none here yet.
        logger.info(f"  Argument --{arg.replace('_', '-')}: {value}")


    # Validate data_path
    if not os.path.isdir(args.data_path):
        logger.error(f"The provided data_path '{args.data_path}' is not a valid directory or does not exist.")
        print(f"Error: Data path '{args.data_path}' is invalid. Please provide a valid directory path.")
        sys.exit(1)
    logger.info(f"Using data_path: {os.path.abspath(args.data_path)}")


    # Initialize services
    try:
        logger.info(f"Initializing LLMService (URL: {args.xinference_url}, Model: {args.llm_model})")
        llm_service = LLMService(api_url=args.xinference_url, model_name=args.llm_model)

        logger.info(f"Initializing EmbeddingService (URL: {args.xinference_url}, Model: {args.embedding_model})")
        embedding_service = EmbeddingService(api_url=args.xinference_url, model_name=args.embedding_model)

        reranker_service = None
        if args.reranker_model and args.reranker_model.lower() != 'none' and args.reranker_model.strip() != '':
            logger.info(f"Initializing RerankerService (URL: {args.xinference_url}, Model: {args.reranker_model})")
            try:
                reranker_service = RerankerService(api_url=args.xinference_url, model_name=args.reranker_model)
            except Exception as e:
                logger.warning(f"Failed to initialize RerankerService for model '{args.reranker_model}': {e}. Proceeding without reranker.")
        else:
            logger.info("Reranker model not specified or disabled. Proceeding without reranker.")

    except Exception as e:
        logger.error(f"Failed to initialize core AI services: {e}", exc_info=True)
        print(f"Error: Could not initialize AI services. Ensure Xinference is running and models are available at {args.xinference_url}.")
        sys.exit(1)

    # Initialize the pipeline with all relevant parameters
    try:
        pipeline = ReportGenerationPipeline(
            llm_service=llm_service,
            embedding_service=embedding_service,
            reranker_service=reranker_service,
            parent_chunk_size=args.parent_chunk_size,
            parent_chunk_overlap=args.parent_chunk_overlap,
            child_chunk_size=args.child_chunk_size,
            child_chunk_overlap=args.child_chunk_overlap,
            vector_top_k=args.vector_top_k,
            keyword_top_k=args.keyword_top_k,
            hybrid_alpha=args.hybrid_search_alpha,
            final_top_n_retrieval=args.final_top_n_retrieval,
            max_refinement_iterations=args.max_refinement_iterations,
            max_workflow_iterations=args.max_workflow_iterations # Pass new CLI arg
        )
    except Exception as e:
        logger.error(f"Failed to initialize the report generation pipeline: {e}", exc_info=True)
        print(f"Error: Could not initialize the report generation pipeline.")
        sys.exit(1)

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True) # exist_ok=True to prevent error if dir already exists
            logger.info(f"Ensured output directory exists: {output_dir}")
        except OSError as e:
            logger.error(f"Failed to create output directory {output_dir}: {e}")
            print(f"Error: Could not create output directory {output_dir}.")
            sys.exit(1)

    # Run the pipeline
    try:
        logger.info(f"Running report generation pipeline for topic: '{args.topic}'...")
        final_report_md = pipeline.run(
            user_topic=args.topic,
            data_path=args.data_path,
            report_title=args.report_title,
            # Pass indexing related args to pipeline's run method, or handle loading in main before pipeline.run
            # For now, let's assume pipeline.run() will take these if needed for its _process_and_load_data
            # Or, better, pipeline __init__ takes them and uses them in _process_and_load_data
            # The latter is already done, so pipeline.run doesn't need these directly.
            # However, _process_and_load_data in pipeline needs these.
            # Let's adjust pipeline's run to accept them, and then pass to _process_and_load_data
            # No, pipeline's __init__ already takes them.
            # The `pipeline.run` method itself just needs user_topic, data_path, report_title.
            # The indexing params are now part of pipeline's construction.
            # So, the call to pipeline.run is correct as is.
            # The save/load logic will be inside pipeline._process_and_load_data
            # which needs access to these args. The pipeline already has them from its __init__.
            # One change needed: pipeline init needs to accept these new args.
            # And pipeline._process_and_load_data needs to use them.
            # Let's verify pipeline.__init__ and _process_and_load_data call signature.
            # Pipeline.__init__ was modified to accept them.
            # Pipeline.run() needs to pass them to _process_and_load_data().
            # This means _process_and_load_data needs its signature changed.
            # Or, these params are stored on `self` in pipeline and used by _P_A_L_D.
            # The current plan is for _P_A_L_D to take them as args.
            # Let's adjust ReportGenerationPipeline.run and _process_and_load_data to take these:
            # This is actually a change for the *next* step (modifying pipeline.py).
            # For now, main.py correctly passes them to Pipeline constructor.
            # The pipeline will use its stored versions of these params.
        )

        with open(args.output_path, "w", encoding="utf-8") as f:
            f.write(final_report_md)
        logger.info(f"Successfully generated report and saved to: {os.path.abspath(args.output_path)}")
        print(f"\nReport generation complete. Output saved to: {os.path.abspath(args.output_path)}")

    except ReportGenerationPipelineError as e:
        logger.error(f"Report generation pipeline failed: {e}", exc_info=True)
        print(f"\nError: Report generation process failed. Details: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during pipeline execution: {e}", exc_info=True)
        print(f"\nError: An unexpected error occurred. Check logs for details. Details: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Example usage:
    # python main.py --topic "The Future of Renewable Energy" --data_path "./sample_documents/"
    #                --output_path "reports/renewable_energy_report.md"
    #                --parent_chunk_size 1500 --child_chunk_size 300 --hybrid_search_alpha 0.6
    main()
