import logging
from xinference.client import Client as XinferenceClient
from config.settings import XINFERENCE_API_URL, DEFAULT_RERANKER_MODEL_NAME

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RerankerServiceError(Exception):
    """Custom exception for RerankerService errors."""
    pass

class RerankerService:
    """
    A service class to interact with a Reranker model
    deployed via Xinference.
    """
    def __init__(self, api_url: str = None, model_name: str = None):
        """
        Initializes the RerankerService.

        Args:
            api_url (str, optional): The URL of the Xinference API.
                                     Defaults to XINFERENCE_API_URL from settings.
            model_name (str, optional): The name of the Reranker model to use.
                                        Defaults to DEFAULT_RERANKER_MODEL_NAME from settings.
        """
        self.api_url = api_url or XINFERENCE_API_URL
        self.model_name = model_name or DEFAULT_RERANKER_MODEL_NAME

        try:
            self.client = XinferenceClient(self.api_url)
            self.model = self.client.get_model(self.model_name)
            logger.info(f"Successfully connected to Xinference API at {self.api_url} and loaded reranker model {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Xinference client or load reranker model {self.model_name} from {self.api_url}: {e}")
            raise RerankerServiceError(f"Xinference client/reranker model initialization failed: {e}")

    def rerank(self, query: str, documents: list[str], top_n: int = None) -> list[dict]:
        """
        Reranks a list of documents based on a query.

        Args:
            query (str): The query string.
            documents (list[str]): A list of documents (strings) to be reranked.
            top_n (int, optional): The number of top documents to return.
                                   If None, returns all reranked documents.

        Returns:
            list[dict]: A list of reranked results, typically containing 'index',
                        'relevance_score', and potentially 'document'. The 'document'
                        field in the response from Xinference is often None, so we
                        will augment it with the original document content.

        Raises:
            RerankerServiceError: If the rerank request fails or the response is malformed.
        """
        if not query or not documents:
            logger.warning("rerank called with empty query or documents.")
            return []

        logger.info(f"Requesting rerank for query '{query}' with {len(documents)} documents using model {self.model_name}.")

        try:
            # The Xinference rerank method expects 'corpus' for documents.
            response = self.model.rerank(
                documents=documents, # In Xinference SDK `rerank` method, this parameter is `documents`
                query=query,
                top_n=top_n
            )

            if response and "results" in response and isinstance(response["results"], list):
                reranked_results = []
                for result in response["results"]:
                    if isinstance(result, dict) and "index" in result and "relevance_score" in result:
                        original_document_index = result["index"]
                        # Augment the result with the original document text
                        # as Xinference response might have 'document': None
                        reranked_results.append({
                            "document": documents[original_document_index],
                            "relevance_score": result["relevance_score"],
                            "original_index": original_document_index # Keep original index if needed elsewhere
                        })
                    else:
                        logger.warning(f"Skipping malformed result item in reranker response: {result}")

                # Sort by relevance_score descending if not already sorted by the model
                reranked_results.sort(key=lambda x: x["relevance_score"], reverse=True)

                logger.info(f"Successfully reranked {len(documents)} documents. Returned {len(reranked_results)} results.")
                return reranked_results
            else:
                logger.error(f"No valid 'results' in reranker model response. Response: {response}")
                raise RerankerServiceError("No valid 'results' in reranker model response.")
        except Exception as e:
            logger.error(f"Error during rerank operation: {e}")
            # Check if the parameters are mismatched based on SDK version
            if "unexpected keyword argument 'corpus'" in str(e) or "unexpected keyword argument 'documents'" in str(e):
                 logger.warning("The error might indicate a mismatch in parameter names for the rerank method (e.g., 'corpus' vs 'documents'). Check Xinference SDK version.")
            raise RerankerServiceError(f"Rerank operation failed: {e}")

if __name__ == '__main__':
    # This is an example of how to use the RerankerService.
    # It requires a running Xinference server with the 'Qwen3-Reranker-0.6B' model.
    # As per instructions, this will not be run during the automated process.
    print("RerankerService Example (requires running Xinference server)")
    print("This part will not be executed by the agent but is for local testing.")

    try:
        reranker_service = RerankerService() # Uses defaults from settings.py

        # Example: Rerank documents
        # query_example = "A man is eating pasta."
        # corpus_example = [
        #     "A man is eating food.",
        #     "A man is eating a piece of bread.",
        #     "The girl is carrying a baby.",
        #     "A man is riding a horse.",
        #     "A woman is playing violin."
        # ]
        # print(f"\nReranking documents for query: '{query_example}'")
        # print(f"Original corpus: {corpus_example}")

        # reranked_docs = reranker_service.rerank(query_example, corpus_example, top_n=3)

        # if reranked_docs:
        #     print("\nReranked documents (top 3):")
        #     for i, doc_info in enumerate(reranked_docs):
        #         print(f"{i+1}. Document: '{doc_info['document']}' (Score: {doc_info['relevance_score']:.4f}, Original Index: {doc_info['original_index']})")
        # else:
        #     print("No documents were reranked or an error occurred.")

        print("\nRerankerService example finished. If no output, ensure Xinference server is running and configured.")
        print("Note: Actual API calls are commented out to prevent errors if server is not available.")

    except RerankerServiceError as e:
        print(f"Error initializing or using RerankerService: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
