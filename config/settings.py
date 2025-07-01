import os

# Xinference API Configuration
XINFERENCE_API_URL = os.getenv("XINFERENCE_API_URL", "http://124.128.251.61:1874") # 使用您提供的默认URL

# LLM Configuration
DEFAULT_LLM_MODEL_NAME = os.getenv("DEFAULT_LLM_MODEL_NAME", "qwen3")
DEFAULT_LLM_MAX_TOKENS = int(os.getenv("DEFAULT_LLM_MAX_TOKENS", "14000"))
DEFAULT_LLM_TEMPERATURE = float(os.getenv("DEFAULT_LLM_TEMPERATURE", "0.6"))
DEFAULT_LLM_TOP_P = float(os.getenv("DEFAULT_LLM_TOP_P", "0.95"))
DEFAULT_LLM_ENABLE_THINKING = os.getenv("DEFAULT_LLM_ENABLE_THINKING", "True").lower() == "true"
DEFAULT_LLM_TOP_K = int(os.getenv("DEFAULT_LLM_TOP_K", "20"))
DEFAULT_LLM_MIN_P = float(os.getenv("DEFAULT_LLM_MIN_P", "0"))


# Embedding Model Configuration
DEFAULT_EMBEDDING_MODEL_NAME = os.getenv("DEFAULT_EMBEDDING_MODEL_NAME", "Qwen3-Embedding-0.6B")

# Reranker Model Configuration
DEFAULT_RERANKER_MODEL_NAME = os.getenv("DEFAULT_RERANKER_MODEL_NAME", "Qwen3-Reranker-0.6B")

# Document Processing Configuration
# General settings (can be overridden by parent/child specific if they are used)
DEFAULT_CHUNK_SIZE = int(os.getenv("DEFAULT_CHUNK_SIZE", "1000")) # General fallback if not using parent-child
DEFAULT_CHUNK_OVERLAP = int(os.getenv("DEFAULT_CHUNK_OVERLAP", "100")) # General fallback

# Parent-Child Chunking Configuration
# For parent chunks, aim for larger, context-rich segments (e.g., paragraphs)
DEFAULT_PARENT_CHUNK_SIZE = int(os.getenv("DEFAULT_PARENT_CHUNK_SIZE", "2000")) # Target characters per parent chunk
DEFAULT_PARENT_CHUNK_OVERLAP = int(os.getenv("DEFAULT_PARENT_CHUNK_OVERLAP", "200")) # Overlap for parent chunks
# For child chunks, aim for smaller, more focused segments (e.g., sentences or few sentences)
DEFAULT_CHILD_CHUNK_SIZE = int(os.getenv("DEFAULT_CHILD_CHUNK_SIZE", "400"))  # Target characters per child chunk
DEFAULT_CHILD_CHUNK_OVERLAP = int(os.getenv("DEFAULT_CHILD_CHUNK_OVERLAP", "40"))   # Overlap for child chunks
# Separators can be used for more semantic chunking. E.g., "\n\n" for paragraphs.
# If using NLTK for sentence tokenization, this might not be directly used for child chunks,
# but could be for parent chunks or as a fallback.
# For simplicity, we might primarily rely on size for now, but a regex could be an option.
# Example: DEFAULT_CHUNK_SEPARATOR_REGEX = os.getenv("DEFAULT_CHUNK_SEPARATOR_REGEX", "\\n\\n")
# For sentence-based child chunking, NLTK will be used, so regex might be less critical here.

# Supported document types for processing
SUPPORTED_DOC_EXTENSIONS = [".pdf", ".docx", ".txt"]

# Vector Store Configuration
DEFAULT_VECTOR_STORE_TOP_K = int(os.getenv("DEFAULT_VECTOR_STORE_TOP_K", "5")) # Number of documents to retrieve from vector search

# Hybrid Search Configuration
# Alpha parameter for blending vector search and keyword search scores.
# Alpha = 1.0 means pure vector search, Alpha = 0.0 means pure keyword search.
DEFAULT_HYBRID_SEARCH_ALPHA = float(os.getenv("DEFAULT_HYBRID_SEARCH_ALPHA", "0.5"))
# Top K for keyword search (BM25) before fusion, can be different from vector search K
DEFAULT_KEYWORD_SEARCH_TOP_K = int(os.getenv("DEFAULT_KEYWORD_SEARCH_TOP_K", "10"))


# Pipeline Configuration
DEFAULT_MAX_REFINEMENT_ITERATIONS = int(os.getenv("DEFAULT_MAX_REFINEMENT_ITERATIONS", "1")) # Number of refinement loops

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Example of how to use these settings in other modules:
# from config.settings import XINFERENCE_API_URL, DEFAULT_LLM_MODEL_NAME
#
# client = Client(XINFERENCE_API_URL)
# model = client.get_model(DEFAULT_LLM_MODEL_NAME)

if __name__ == '__main__':
    # Print out all settings for verification
    print(f"XINFERENCE_API_URL: {XINFERENCE_API_URL}")
    print(f"DEFAULT_LLM_MODEL_NAME: {DEFAULT_LLM_MODEL_NAME}")
    print(f"DEFAULT_LLM_MAX_TOKENS: {DEFAULT_LLM_MAX_TOKENS}")
    print(f"DEFAULT_LLM_TEMPERATURE: {DEFAULT_LLM_TEMPERATURE}")
    print(f"DEFAULT_LLM_TOP_P: {DEFAULT_LLM_TOP_P}")
    print(f"DEFAULT_LLM_ENABLE_THINKING: {DEFAULT_LLM_ENABLE_THINKING}")
    print(f"DEFAULT_LLM_TOP_K: {DEFAULT_LLM_TOP_K}")
    print(f"DEFAULT_LLM_MIN_P: {DEFAULT_LLM_MIN_P}")
    print(f"DEFAULT_EMBEDDING_MODEL_NAME: {DEFAULT_EMBEDDING_MODEL_NAME}")
    print(f"DEFAULT_RERANKER_MODEL_NAME: {DEFAULT_RERANKER_MODEL_NAME}")
    print(f"DEFAULT_CHUNK_SIZE (general): {DEFAULT_CHUNK_SIZE}")
    print(f"DEFAULT_CHUNK_OVERLAP (general): {DEFAULT_CHUNK_OVERLAP}")
    print(f"DEFAULT_PARENT_CHUNK_SIZE: {DEFAULT_PARENT_CHUNK_SIZE}")
    print(f"DEFAULT_PARENT_CHUNK_OVERLAP: {DEFAULT_PARENT_CHUNK_OVERLAP}")
    print(f"DEFAULT_CHILD_CHUNK_SIZE: {DEFAULT_CHILD_CHUNK_SIZE}")
    print(f"DEFAULT_CHILD_CHUNK_OVERLAP: {DEFAULT_CHILD_CHUNK_OVERLAP}")
    print(f"SUPPORTED_DOC_EXTENSIONS: {SUPPORTED_DOC_EXTENSIONS}")
    print(f"DEFAULT_VECTOR_STORE_TOP_K: {DEFAULT_VECTOR_STORE_TOP_K}")
    print(f"DEFAULT_HYBRID_SEARCH_ALPHA: {DEFAULT_HYBRID_SEARCH_ALPHA}")
    print(f"DEFAULT_KEYWORD_SEARCH_TOP_K: {DEFAULT_KEYWORD_SEARCH_TOP_K}")
    print(f"DEFAULT_MAX_REFINEMENT_ITERATIONS: {DEFAULT_MAX_REFINEMENT_ITERATIONS}")
    print(f"LOG_LEVEL: {LOG_LEVEL}")
