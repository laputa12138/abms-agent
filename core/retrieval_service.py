import logging
from typing import List, Dict, Optional, Any
import numpy as np
from rank_bm25 import BM25Okapi

from core.vector_store import VectorStore, VectorStoreError
from core.reranker_service import RerankerService, RerankerServiceError
# Assuming settings will be accessed for defaults if not passed directly,
# or that defaults are handled by the caller (e.g., ContentRetrieverAgent or Pipeline)
# For now, let's make parameters explicit in the retrieve method.
# from config.settings import (...)

logger = logging.getLogger(__name__)

class RetrievalServiceError(Exception):
    """Custom exception for RetrievalService errors."""
    pass

class RetrievalService:
    """
    Service responsible for performing hybrid retrieval (vector + keyword)
    from a knowledge base, optionally followed by reranking.
    It works with parent-child chunked documents stored in a VectorStore.
    """

    def __init__(self,
                 vector_store: VectorStore,
                 bm25_index: Optional[BM25Okapi],
                 all_child_chunks_for_bm25_mapping: List[Dict[str, Any]], # For mapping BM25 results
                 reranker_service: Optional[RerankerService] = None):
        """
        Initializes the RetrievalService.

        Args:
            vector_store (VectorStore): Instance of VectorStore.
            bm25_index (Optional[BM25Okapi]): Pre-computed BM25 index over child chunks.
            all_child_chunks_for_bm25_mapping (List[Dict[str, Any]]): List of dictionaries,
                where each dict contains at least 'child_id' and 'child_text' for every
                child chunk that was used to build the bm25_index. This is crucial for
                mapping BM25's results (which are often indices or raw texts) back to
                the structured child chunk data (including parent context).
            reranker_service (Optional[RerankerService]): Instance of RerankerService.
        """
        if not vector_store:
            raise RetrievalServiceError("VectorStore is required for RetrievalService.")

        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.all_child_chunks_for_bm25_mapping = all_child_chunks_for_bm25_mapping
        self.reranker_service = reranker_service

        # Build a quick lookup map from child_id to its full context (parent, etc.)
        # This map is essential for re-associating BM25 results if they only return child_id or index.
        self.child_id_to_full_context_map: Dict[str, Dict[str, Any]] = {}
        if hasattr(self.vector_store, 'document_store'):
            for item_meta in self.vector_store.document_store:
                self.child_id_to_full_context_map[item_meta['child_id']] = item_meta
        else:
            logger.warning("VectorStore does not have 'document_store' attribute. "
                           "BM25 result mapping might be incomplete if it relies on child_ids not found elsewhere.")


        log_msg = (f"RetrievalService initialized. "
                   f"VectorStore has {self.vector_store.count_child_chunks} child chunks. "
                   f"BM25 index is {'PRESENT' if self.bm25_index else 'ABSENT'}. "
                   f"Reranker service is {'ENABLED' if self.reranker_service else 'DISABLED'}.")
        logger.info(log_msg)

    def _tokenize_query(self, query: str) -> List[str]:
        """Simple whitespace tokenizer. For Chinese, a proper tokenizer is recommended."""
        # TODO: Integrate a proper Chinese tokenizer like jieba if docs are primarily Chinese.
        # e.g., import jieba; return list(jieba.cut_for_search(query))
        return query.lower().split()

    def _normalize_scores(self, scores: List[float], reverse: bool = False) -> List[float]:
        """Min-max normalize scores to [0, 1]. Reverse if higher score is worse (e.g., distance)."""
        if not scores: return []
        min_s, max_s = min(scores), max(scores)
        if max_s == min_s: # Avoid division by zero if all scores are same
            return [0.5] * len(scores) # Default to mid-point

        normalized = [(s - min_s) / (max_s - min_s) for s in scores]
        return [1.0 - s for s in normalized] if reverse else normalized

    def retrieve(self,
                 query_text: str,
                 vector_top_k: int,
                 keyword_top_k: int,
                 # hybrid_alpha: float, # 0 for keyword only, 1 for vector only # Removed
                 final_top_n: Optional[int] = None
                ) -> List[Dict[str, Any]]:
        """
        Performs hybrid retrieval (vector + keyword) and mandatory reranking if available.
        If RerankerService is not available, results from vector and keyword search are combined
        and returned, potentially without ideal ordering.

        Args:
            query_text (str): The user's query.
            vector_top_k (int): Number of candidates to retrieve from vector search.
            keyword_top_k (int): Number of candidates to retrieve from keyword search.
            final_top_n (Optional[int]): Number of final results to return after reranking (or combination).
                                         If None, returns all processed results up to a reasonable limit post-reranking.

        Returns:
            List[Dict[str, Any]]: A list of result dictionaries, structured for consumption
                                  by agents like ChapterWriterAgent. Each dict should contain:
                                  'document' (parent_text), 'score' (final_score from reranker, or default),
                                  'child_text_preview', 'child_id', 'parent_id', 'source' (retrieval method).
        """
        logger.info(f"RetrievalService called with query: '{query_text[:100]}...' "
                    f"v_k={vector_top_k}, k_k={keyword_top_k}, final_n={final_top_n}")

        # --- 1. Vector Search ---
        vector_results_map: Dict[str, Dict[str, Any]] = {} # child_id -> result_dict (metadata + original score)
        if vector_top_k > 0:
            try:
                # VectorStore.search returns List[Dict] with child_id, child_text, parent_id, parent_text, source_document_id, score (distance)
                raw_vector_hits = self.vector_store.search(query_text=query_text, k=vector_top_k)
                # No normalization needed here, primarily collecting candidates.
                # Store original score for potential future use or debugging, but it's not used for combined ranking.
                for hit in raw_vector_hits:
                    child_id = hit['child_id']
                    # Ensure 'score' from vector store (distance) is preserved if needed later,
                    # or transform it if a similarity view is more intuitive (e.g. 1 / (1 + distance))
                    # For now, just pass the hit, which includes its original score.
                    vector_results_map[child_id] = {**hit, 'retrieval_method': 'vector'}
                logger.debug(f"Vector search found {len(vector_results_map)} distinct child chunks.")
            except VectorStoreError as e:
                logger.error(f"VectorStore search failed during retrieval: {e}")
            except Exception as e:
                logger.error(f"Unexpected error during vector search part of retrieval: {e}", exc_info=True)

        # --- 2. Keyword Search (BM25) ---
        keyword_results_map: Dict[str, Dict[str, Any]] = {}
        # Keyword search is always attempted if keyword_top_k > 0 and BM25 index is available
        if keyword_top_k > 0 and self.bm25_index and self.all_child_chunks_for_bm25_mapping:
            try:
                tokenized_query = self._tokenize_query(query_text)
                bm25_doc_scores = self.bm25_index.get_scores(tokenized_query)

                # Get top N indices from BM25 scores.
                num_bm25_candidates = min(keyword_top_k, len(self.all_child_chunks_for_bm25_mapping))
                top_bm25_indices = np.argsort(bm25_doc_scores)[::-1][:num_bm25_candidates]

                # No normalization of BM25 scores needed here. Collect candidates.
                # Store original BM25 score for potential future use or debugging.
                for doc_idx in top_bm25_indices:
                    bm25_score = bm25_doc_scores[doc_idx]
                    if bm25_score <= 1e-6: continue # Skip if score is effectively zero

                    child_meta_from_bm25_corpus = self.all_child_chunks_for_bm25_mapping[doc_idx]
                    child_id = child_meta_from_bm25_corpus['child_id']

                    full_context = self.child_id_to_full_context_map.get(child_id)
                    if not full_context:
                        logger.warning(f"BM25 found child_id '{child_id}' but it's not in child_id_to_full_context_map. Skipping.")
                        continue

                    # Add 'score' (original bm25 score) and 'retrieval_method'
                    keyword_results_map[child_id] = {**full_context, 'score': bm25_score, 'retrieval_method': 'keyword'}
                logger.debug(f"Keyword search (BM25) found {len(keyword_results_map)} distinct child chunks with positive scores.")
            except Exception as e:
                logger.error(f"Keyword search (BM25) failed during retrieval: {e}", exc_info=True)

        # --- 3. Combine candidate results (deduplication) ---
        candidate_pool: Dict[str, Dict[str, Any]] = {} # child_id -> full_data_dict

        # Add vector results first
        for child_id, data in vector_results_map.items():
            candidate_pool[child_id] = data # data already contains 'retrieval_method': 'vector' and original 'score'

        # Add keyword results, potentially updating retrieval_method if also found by vector
        for child_id, data in keyword_results_map.items():
            if child_id in candidate_pool:
                # Already present from vector search, update method to 'hybrid'
                # The 'score' from vector search is kept (or decide on a rule if scores are very different and both needed)
                # For now, vector's metadata (including its score) takes precedence if collision.
                candidate_pool[child_id]['retrieval_method'] = 'hybrid'
                # Optionally, store keyword_score if needed: candidate_pool[child_id]['keyword_score'] = data['score']
            else:
                # New entry from keyword search
                candidate_pool[child_id] = data # data already contains 'retrieval_method': 'keyword' and original 'score'

        logger.debug(f"Combined pool has {len(candidate_pool)} unique child chunks before reranking.")

        # Convert candidate_pool to a list for reranking or final processing
        # The 'score' attribute in candidate_pool items is the original score from their respective methods,
        # it will be replaced by the reranker's score.
        candidates_for_reranking_or_final = list(candidate_pool.values())


        # --- 4. Reranking (operates on parent_text) ---
        results_to_format = [] # This list will hold the items to be formatted for output

        if self.reranker_service and candidates_for_reranking_or_final:
            try:
                parents_for_reranking = [res['parent_text'] for res in candidates_for_reranking_or_final]

                # Reranker service is expected to handle top_n internally if final_top_n is provided to it.
                # The rerank method should accept query, documents, and an optional top_n.
                reranked_outputs = self.reranker_service.rerank(
                    query=query_text,
                    documents=parents_for_reranking,
                    top_n=final_top_n # Pass final_top_n to reranker
                )

                # Re-associate reranked scores and order with original full data
                for reranked_item_from_service in reranked_outputs: # reranked_outputs is already top_n
                    original_idx = reranked_item_from_service['original_index']
                    original_full_data = candidates_for_reranking_or_final[original_idx]

                    # Update score and retrieval_source
                    results_to_format.append({
                        **original_full_data,
                        'final_score': reranked_item_from_service['relevance_score'],
                        'retrieval_source': original_full_data.get('retrieval_method', 'unknown') + "_reranked"
                    })
                logger.debug(f"Reranking complete. Produced {len(results_to_format)} results.")

            except RerankerServiceError as e:
                logger.error(f"Reranker service error: {e}. Using pre-reranked results with simple truncation.")
                # Fallback: use combined candidates, sort by some default if desired, or just truncate
                # For now, no specific sort, just truncate. Reranker was supposed to sort.
                # Scores will be original retrieval scores which are not comparable across methods.
                # Assign a default score or handle this explicitly.
                for item in candidates_for_reranking_or_final:
                    item['final_score'] = item.get('score', 0.0) # Use original score, or 0.0
                    item['retrieval_source'] = item.get('retrieval_method', 'unknown') + "_fallback"
                results_to_format = candidates_for_reranking_or_final
                if final_top_n is not None:
                    results_to_format = results_to_format[:final_top_n]
            except Exception as e:
                logger.error(f"Unexpected error during reranking: {e}. Using pre-reranked results.", exc_info=True)
                for item in candidates_for_reranking_or_final:
                    item['final_score'] = item.get('score', 0.0)
                    item['retrieval_source'] = item.get('retrieval_method', 'unknown') + "_fallback_exception"
                results_to_format = candidates_for_reranking_or_final
                if final_top_n is not None:
                    results_to_format = results_to_format[:final_top_n]
        else: # No reranker or no candidates
            logger.info("No reranker service available or no candidates to rerank. Using combined results directly.")
            # Scores will be original retrieval scores.
            # For consistency, add 'final_score' and 'retrieval_source'
            for item in candidates_for_reranking_or_final:
                item['final_score'] = item.get('score', 0.0) # Use original score (vector distance or bm25 score)
                item['retrieval_source'] = item.get('retrieval_method', 'unknown')
            results_to_format = candidates_for_reranking_or_final
            if final_top_n is not None: # Apply final_top_n if set
                # If no reranker, there's no inherent sorting order across vector/keyword results with raw scores.
                # A simple sort might be needed if a consistent top_n is desired without reranking.
                # However, the problem asks to *use* reranker. If no reranker, it's a fallback.
                # Let's assume for now, if no reranker, the order is somewhat arbitrary from pool conversion.
                results_to_format = results_to_format[:final_top_n]


        # --- 5. Format for Output ---
        # Ensure the output format is suitable for ChapterWriterAgent
        output_for_chapter_writer: List[Dict[str, Any]] = []
        for res_item in results_to_format: # Iterate over what's decided in step 4
            output_for_chapter_writer.append({
                "document": res_item["parent_text"], # Key for ChapterWriterAgent
                "score": res_item["final_score"], # This is reranked_score or fallback score
                "child_text_preview": res_item["child_text"][:150] + "...", # For context/logging
                "child_id": res_item["child_id"],
                "parent_id": res_item["parent_id"],
                "source_document_id": res_item["source_document_id"],
                "retrieval_source": res_item["retrieval_source"] # For tracing
            })


        # --- 5. Format for Output ---
        # Ensure the output format is suitable for ChapterWriterAgent
        output_for_chapter_writer: List[Dict[str, Any]] = []
        for res_item in results_after_rerank_or_hybrid:
            output_for_chapter_writer.append({
                "document": res_item["parent_text"], # Key for ChapterWriterAgent
                "score": res_item["final_score"],
                "child_text_preview": res_item["child_text"][:150] + "...", # For context/logging
                "child_id": res_item["child_id"],
                "parent_id": res_item["parent_id"],
                "source_document_id": res_item["source_document_id"],
                "retrieval_source": res_item["retrieval_source"] # For tracing
            })

        logger.info(f"Retrieval process finished. Returning {len(output_for_chapter_writer)} items.")
        return output_for_chapter_writer

if __name__ == '__main__':
    # This example requires mocks for VectorStore, BM25Okapi, RerankerService
    # It focuses on the internal logic of RetrievalService.
    logging.basicConfig(level=logging.DEBUG)
    logger.info("RetrievalService Example Start")

    # --- Mock Dependencies ---
    class MockEmbeddingServiceForRS:
        def create_embeddings(self, texts: List[str]) -> List[List[float]]:
            return [[np.random.rand() for _ in range(5)] for _ in texts]

    class MockVectorStoreForRS:
        def __init__(self, embedding_service):
            self.embedding_service = embedding_service
            self.document_store = [] # Populated by add_documents_mock
            self.count_child_chunks = 0
            logger.debug("MockVectorStoreForRS initialized.")

        def add_documents_mock(self, data: List[Dict[str, Any]]): # Simulate adding parent_child_data
            # Simplified: just populate document_store for the mock search to use
            for p_data in data:
                for c_data in p_data['children']:
                    self.document_store.append({
                        'child_id': c_data['child_id'], 'child_text': c_data['child_text'],
                        'parent_id': p_data['parent_id'], 'parent_text': p_data['parent_text'],
                        'source_document_id': p_data['source_document_id']
                    })
            self.count_child_chunks = len(self.document_store)
            logger.debug(f"MockVectorStoreForRS populated with {self.count_child_chunks} items via mock add.")


        def search(self, query_text: str, k: int) -> List[Dict[str, Any]]:
            logger.debug(f"MockVectorStoreForRS.search called for '{query_text}', k={k}")
            results = []
            # Return first k/2 items from document_store as mock results
            for i, item in enumerate(self.document_store):
                if i < k // 2 + 1:
                    results.append({**item, 'score': np.random.uniform(0.1, 0.5)}) # distance score
                else: break
            logger.debug(f"MockVectorStore.search returns {len(results)} items.")
            return results

    class MockBM25Okapi:
        def __init__(self, corpus):
            self.corpus = corpus
            logger.debug(f"MockBM25Okapi initialized with {len(corpus)} documents.")
        def get_scores(self, query_tokens: List[str]) -> np.ndarray:
            logger.debug(f"MockBM25Okapi.get_scores called for query: {query_tokens}")
            # Return random scores for all docs in corpus
            return np.random.rand(len(self.corpus)) * 10

    class MockRerankerServiceForRS:
        def rerank(self, query: str, documents: List[str], top_n: Optional[int]) -> List[Dict[str, Any]]:
            logger.debug(f"MockRerankerService.rerank called for '{query}', {len(documents)} docs, top_n={top_n}")
            reranked = []
            for i, doc_text in enumerate(reversed(documents)):
                reranked.append({"document": doc_text, "relevance_score": 0.9 - (i * 0.05), "original_index": documents.index(doc_text)})
            return reranked[:top_n] if top_n else reranked

    # --- Setup Data for Mocks ---
    sample_p_c_data = [
        {"parent_id": "P1", "parent_text": "Parent One: Apples are red. Oranges are orange.", "source_document_id": "DocA",
         "children": [{"child_id": "P1C1", "child_text": "Apples are red."}, {"child_id": "P1C2", "child_text": "Oranges are orange."}]},
        {"parent_id": "P2", "parent_text": "Parent Two: Bananas are yellow. Grapes are purple.", "source_document_id": "DocA",
         "children": [{"child_id": "P2C1", "child_text": "Bananas are yellow."}, {"child_id": "P2C2", "child_text": "Grapes are purple."}]},
        {"parent_id": "P3", "parent_text": "Parent Three: Cars are fast. Bikes are good for exercise.", "source_document_id": "DocB",
         "children": [{"child_id": "P3C1", "child_text": "Cars are fast and come in red or blue."}, {"child_id": "P3C2", "child_text": "Bikes are good for exercise and fun."}]}
    ]

    mock_vs = MockVectorStoreForRS(MockEmbeddingServiceForRS())
    mock_vs.add_documents_mock(sample_p_c_data) # Manually populate its store for this test

    all_child_chunks_map_data = [] # This is List[Dict{'child_id': str, 'child_text': str}]
    bm25_corpus_tokens = []
    for p_item in sample_p_c_data:
        for c_item in p_item['children']:
            all_child_chunks_map_data.append({'child_id': c_item['child_id'], 'child_text': c_item['child_text']})
            bm25_corpus_tokens.append(c_item['child_text'].lower().split())

    mock_bm25 = MockBM25Okapi(bm25_corpus_tokens) if bm25_corpus_tokens else None
    mock_reranker = MockRerankerServiceForRS()

    # --- Initialize RetrievalService with Mocks ---
    retrieval_svc = RetrievalService(
        vector_store=mock_vs,
        bm25_index=mock_bm25,
        all_child_chunks_for_bm25_mapping=all_child_chunks_map_data,
        reranker_service=mock_reranker
    )

    # --- Test Cases ---
    test_queries = ["red apples and fast cars", "yellow bananas", "bikes"]
    for t_query in test_queries:
        logger.info(f"\n--- Testing RetrievalService with query: '{t_query}' ---")
        try:
            results = retrieval_svc.retrieve(
                query_text=t_query,
                vector_top_k=2,  # How many candidates from vector search
                keyword_top_k=2, # How many candidates from keyword search
                # hybrid_alpha is removed
                final_top_n=2    # How many results after reranking (or combination if no reranker)
            )
            if results:
                logger.info(f"Found {len(results)} items for query '{t_query}':")
                for i, item in enumerate(results):
                    print(f"  Result {i+1}:")
                    print(f"    Parent ID: {item['parent_id']}, Child ID: {item['child_id']}")
                    print(f"    Child Preview: {item['child_text_preview']}")
                    print(f"    Parent Text: '{item['document'][:60]}...'")
                    print(f"    Score: {item['score']:.4f}, Source: {item['retrieval_source']}")
            else:
                logger.info(f"No results found for query '{t_query}'.")
        except RetrievalServiceError as e:
            logger.error(f"RetrievalServiceError for query '{t_query}': {e}")
        except Exception as e:
            logger.error(f"Unexpected error for query '{t_query}': {e}", exc_info=True)

    logger.info("\nRetrievalService Example End")
