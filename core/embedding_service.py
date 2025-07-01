import logging
from xinference.client import Client as XinferenceClient  # Renamed to avoid conflict if RESTfulClient is also from xinference.client
from config.settings import XINFERENCE_API_URL, DEFAULT_EMBEDDING_MODEL_NAME

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmbeddingServiceError(Exception):
    """Custom exception for EmbeddingService errors."""
    pass

class EmbeddingService:
    """
    A service class to interact with an Embedding model
    deployed via Xinference.
    """
    def __init__(self, api_url: str = None, model_name: str = None):
        """
        Initializes the EmbeddingService.

        Args:
            api_url (str, optional): The URL of the Xinference API.
                                     Defaults to XINFERENCE_API_URL from settings.
            model_name (str, optional): The name of the Embedding model to use.
                                        Defaults to DEFAULT_EMBEDDING_MODEL_NAME from settings.
        """
        self.api_url = api_url or XINFERENCE_API_URL
        self.model_name = model_name or DEFAULT_EMBEDDING_MODEL_NAME

        try:
            # Note: The user's example uses `RESTfulClient` for embeddings,
            # but `xinference.client.Client` is generally the unified client.
            # We will use `xinference.client.Client` for consistency with LLMService
            # and assume it supports embedding models as per typical Xinference usage.
            # If `RESTfulClient` is specifically needed and different, this might need adjustment.
            self.client = XinferenceClient(self.api_url)
            self.model = self.client.get_model(self.model_name)
            logger.info(f"Successfully connected to Xinference API at {self.api_url} and loaded embedding model {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Xinference client or load embedding model {self.model_name} from {self.api_url}: {e}")
            raise EmbeddingServiceError(f"Xinference client/embedding model initialization failed: {e}")

    def create_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Creates embeddings for a list of texts.

        Args:
            texts (list[str]): A list of strings to embed.

        Returns:
            list[list[float]]: A list of embedding vectors, where each vector is a list of floats.

        Raises:
            EmbeddingServiceError: If the embedding request fails or the response is malformed.
        """
        if not texts:
            logger.warning("create_embeddings called with an empty list of texts.")
            return []

        logger.info(f"Requesting embeddings for {len(texts)} texts from model {self.model_name}. First text: '{texts[0][:100]}...'")

        try:
            response = self.model.create_embedding(texts)

            if response and "data" in response and isinstance(response["data"], list):
                embeddings = [item.get("embedding") for item in response["data"]]
                if all(isinstance(emb, list) for emb in embeddings):
                    logger.info(f"Successfully created embeddings for {len(texts)} texts. Embedding dimension: {len(embeddings[0]) if embeddings else 'N/A'}")
                    return embeddings
                else:
                    logger.error(f"Malformed response from embedding model: 'embedding' field missing or not a list in one of the data items. Response: {response}")
                    raise EmbeddingServiceError("Malformed response from embedding model: 'embedding' field missing or not a list.")
            else:
                logger.error(f"No valid 'data' in embedding model response. Response: {response}")
                raise EmbeddingServiceError("No valid 'data' in embedding model response.")
        except Exception as e:
            logger.error(f"Error during embedding creation: {e}")
            # Check if the error message suggests using RESTfulClient
            if "RESTfulClient" in str(e) or "model.create_embedding" in str(e).lower() and "not available" in str(e).lower() :
                 logger.warning("The error might indicate that a RESTfulClient is needed for this embedding model with the current Xinference version.")
            raise EmbeddingServiceError(f"Embedding creation failed: {e}")

if __name__ == '__main__':
    # This is an example of how to use the EmbeddingService.
    # It requires a running Xinference server with the 'Qwen3-Embedding-0.6B' model.
    # As per instructions, this will not be run during the automated process.
    print("EmbeddingService Example (requires running Xinference server)")
    print("This part will not be executed by the agent but is for local testing.")

    try:
        embedding_service = EmbeddingService() # Uses defaults from settings.py

        # Example: Create embeddings
        # sample_texts = ["What is the capital of China?", "Xinference is a great tool."]
        # print(f"\nCreating embeddings for texts: {sample_texts}")
        # embeddings = embedding_service.create_embeddings(sample_texts)

        # if embeddings:
        #     print(f"Number of embeddings: {len(embeddings)}")
        #     print(f"Dimension of first embedding: {len(embeddings[0])}")
        #     print(f"First 10 elements of first embedding: {embeddings[0][:10]}")
        # else:
        #     print("No embeddings were returned.")

        print("\nEmbeddingService example finished. If no output, ensure Xinference server is running and configured.")
        print("Note: Actual API calls are commented out to prevent errors if server is not available.")

    except EmbeddingServiceError as e:
        print(f"Error initializing or using EmbeddingService: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
