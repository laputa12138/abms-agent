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
                 hybrid_alpha: float, # 0 for keyword only, 1 for vector only
                 final_top_n: Optional[int] = None
                ) -> List[Dict[str, Any]]:
        """
        Performs hybrid retrieval and optional reranking.

        Args:
            query_text (str): The user's query.
            vector_top_k (int): Number of results from vector search.
            keyword_top_k (int): Number of results from keyword search.
            hybrid_alpha (float): Weight for blending vector and keyword scores.
            final_top_n (Optional[int]): Number of final results to return after all steps.
                                         If None, returns all processed results.

        Returns:
            List[Dict[str, Any]]: A list of result dictionaries, structured for consumption
                                  by agents like ChapterWriterAgent. Each dict should contain:
                                  'document' (parent_text), 'score' (final_score),
                                  'child_text_preview', 'child_id', 'parent_id', 'source' (retrieval method).
        """
        logger.info(f"RetrievalService called with query: '{query_text[:100]}...' "
                    f"v_k={vector_top_k}, k_k={keyword_top_k}, alpha={hybrid_alpha}, final_n={final_top_n}")

        # --- 1. Vector Search ---
        vector_results_map: Dict[str, Dict[str, Any]] = {} # child_id -> result_dict_with_scores
        if hybrid_alpha > 0:
            try:
                # VectorStore.search returns List[Dict] with child_id, child_text, parent_id, parent_text, source_document_id, score (distance)
                raw_vector_hits = self.vector_store.search(query_text=query_text, k=vector_top_k)
                distances = [hit['score'] for hit in raw_vector_hits]
                norm_similarity_scores = self._normalize_scores(distances, reverse=True)

                for i, hit in enumerate(raw_vector_hits):
                    child_id = hit['child_id']
                    vector_results_map[child_id] = {**hit, 'vector_score': norm_similarity_scores[i], 'keyword_score': 0.0}
                logger.debug(f"Vector search found {len(vector_results_map)} distinct child chunks.")
            except VectorStoreError as e:
                logger.error(f"VectorStore search failed during retrieval: {e}")
            except Exception as e:
                logger.error(f"Unexpected error during vector search part of retrieval: {e}", exc_info=True)

        # --- 2. Keyword Search (BM25) ---
        keyword_results_map: Dict[str, Dict[str, Any]] = {}
        if hybrid_alpha < 1.0 and self.bm25_index and self.all_child_chunks_for_bm25_mapping:
            try:
                tokenized_query = self._tokenize_query(query_text)
                bm25_doc_scores = self.bm25_index.get_scores(tokenized_query)

                # Get top N indices from BM25 scores. These map to `all_child_chunks_for_bm25_mapping`.
                # Ensure we don't request more than available.
                num_bm25_candidates = min(keyword_top_k, len(self.all_child_chunks_for_bm25_mapping))
                top_bm25_indices = np.argsort(bm25_doc_scores)[::-1][:num_bm25_candidates]

                top_bm25_scores_only = [bm25_doc_scores[i] for i in top_bm25_indices]
                norm_bm25_scores = self._normalize_scores(top_bm25_scores_only, reverse=False)

                for i, doc_idx in enumerate(top_bm25_indices):
                    if norm_bm25_scores[i] <= 1e-6: continue # Skip if score is effectively zero

                    # child_meta_from_bm25_corpus is like {'child_id': ..., 'child_text': ...}
                    child_meta_from_bm25_corpus = self.all_child_chunks_for_bm25_mapping[doc_idx]
                    child_id = child_meta_from_bm25_corpus['child_id']

                    # Fetch full context (parent_text, etc.) using child_id from the main map
                    full_context = self.child_id_to_full_context_map.get(child_id)
                    if not full_context:
                        logger.warning(f"BM25 found child_id '{child_id}' but it's not in child_id_to_full_context_map. Skipping.")
                        continue

                    keyword_results_map[child_id] = {**full_context, 'vector_score': 0.0, 'keyword_score': norm_bm25_scores[i]}
                logger.debug(f"Keyword search (BM25) found {len(keyword_results_map)} distinct child chunks with positive scores.")
            except Exception as e:
                logger.error(f"Keyword search (BM25) failed during retrieval: {e}", exc_info=True)

        # --- 3. Combine and Rank Results ---
        combined_results: Dict[str, Dict[str, Any]] = {}
        for child_id, data in vector_results_map.items(): combined_results[child_id] = data
        for child_id, data in keyword_results_map.items():
            if child_id in combined_results: combined_results[child_id]['keyword_score'] = data['keyword_score']
            else: combined_results[child_id] = data

        scored_results = []
        for child_id, data in combined_results.items():
            final_score = (hybrid_alpha * data['vector_score']) + ((1.0 - hybrid_alpha) * data['keyword_score'])
            source = "hybrid"
            if hybrid_alpha == 1.0 and data['vector_score'] > 0: source = "vector"
            elif hybrid_alpha == 0.0 and data['keyword_score'] > 0: source = "keyword"
            elif data['vector_score'] == 0 and data['keyword_score'] == 0 : source = "none" # Should not happen if filtered before

            # Only include if there's some score
            if final_score > 1e-6 : # Tolerance for float precision
                 scored_results.append({**data, 'final_score': final_score, 'retrieval_source': source})

        scored_results.sort(key=lambda x: x['final_score'], reverse=True)

        # --- 4. Optional Reranking (operates on parent_text) ---
        results_after_rerank_or_hybrid = scored_results # Default if no reranking
        if self.reranker_service and scored_results:
            parents_for_reranking = [res['parent_text'] for res in scored_results]
            # Keep track of original full data to re-associate
            original_items_before_rerank = list(scored_results) # shallow copy
            try:
                reranked_outputs = self.reranker_service.rerank(query=query_text, documents=parents_for_reranking, top_n=final_top_n) # Reranker handles top_n

                temp_reranked_list = []
                for reranked_item_from_service in reranked_outputs:
                    original_idx = reranked_item_from_service['original_index']
                    original_full_data = original_items_before_rerank[original_idx]
                    temp_reranked_list.append({
                        **original_full_data,
                        'parent_text': reranked_item_from_service['document'], # This is the parent_text
                        'final_score': reranked_item_from_service['relevance_score'], # Update score
                        'retrieval_source': original_full_data['retrieval_source'] + "_reranked"
                    })
                results_after_rerank_or_hybrid = temp_reranked_list
                logger.debug(f"Reranking complete. Produced {len(results_after_rerank_or_hybrid)} results.")
            except RerankerServiceError as e:
                logger.error(f"Reranker service error: {e}. Using pre-reranked results.")
                # Fallback to results before reranking, respecting final_top_n if reranker was supposed to do it
                results_after_rerank_or_hybrid = scored_results[:final_top_n] if final_top_n else scored_results
            except Exception as e:
                 logger.error(f"Unexpected error during reranking: {e}. Using pre-reranked results.", exc_info=True)
                 results_after_rerank_or_hybrid = scored_results[:final_top_n] if final_top_n else scored_results
        elif final_top_n is not None: # No reranker, but final_top_n is set
            results_after_rerank_or_hybrid = scored_results[:final_top_n]


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
                vector_top_k=2,
                keyword_top_k=2,
                hybrid_alpha=0.5,
                final_top_n=2
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
