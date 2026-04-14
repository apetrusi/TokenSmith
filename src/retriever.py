"""
retriever.py

Stores core retrieval logic using FAISS and BM25 scoring.
It also contains helpers for loading artifacts and filtering chunks.
"""

from __future__ import annotations

import math
import pathlib
import os
import pickle
from abc import ABC, abstractmethod
from typing import Any, Collection, Dict, List, Optional, Tuple
import nltk
from nltk.stem import WordNetLemmatizer

import faiss
import numpy as np
from src.embedder import CachedEmbedder

from src.config import RAGConfig
from src.index_builder import preprocess_for_bm25


# -------------------------- Embedder cache ------------------------------

_EMBED_CACHE: Dict[str, CachedEmbedder] = {}

def _get_embedder(model_name: str) -> CachedEmbedder:
    if model_name not in _EMBED_CACHE:
        # Use the cached embedding model to avoid reloading it on every call
        _EMBED_CACHE[model_name] = CachedEmbedder(model_name)
    return _EMBED_CACHE[model_name]


# -------------------------- Read artifacts -------------------------------

def load_artifacts(artifacts_dir: os.PathLike, index_prefix: str) -> Tuple[faiss.Index, List[str], List[str], Any]:
    """
    Loads:
      - FAISS index: {index_prefix}.faiss
      - chunks:      {index_prefix}_chunks.pkl
      - sources:     {index_prefix}_sources.pkl
    """
    artifacts_dir = pathlib.Path(artifacts_dir)
    faiss_index = faiss.read_index(str(artifacts_dir / f"{index_prefix}.faiss"))
    bm25_index  = pickle.load(open(artifacts_dir / f"{index_prefix}_bm25.pkl", "rb"))
    chunks      = pickle.load(open(artifacts_dir / f"{index_prefix}_chunks.pkl", "rb"))
    sources     = pickle.load(open(artifacts_dir / f"{index_prefix}_sources.pkl", "rb"))
    metadata = pickle.load(open(artifacts_dir / f"{index_prefix}_meta.pkl", "rb"))

    return faiss_index, bm25_index, chunks, sources, metadata


# -------------------------- Helper to get page nums for chunks -------------------------------

def get_page_numbers(chunk_indices: list[int], metadata: list[dict]) -> dict[int, List[int]]:
    if not metadata or not chunk_indices:
        return {}

    page_map: dict[int, List[int]] = {}

    for chunk_idx in chunk_indices:
        chunk_idx = int(chunk_idx)
        if 0 <= chunk_idx < len(metadata):
            chunk_pages = metadata[chunk_idx].get("page_numbers")
            if chunk_pages is None:
                continue  # don't store None; callers can default to [1]
            page_map[chunk_idx] = chunk_pages

    return page_map

# -------------------------- Filtering logic -----------------------------

def filter_retrieved_chunks(cfg: RAGConfig, chunks, ordered):
    topk_idxs = ordered[:cfg.top_k]
    return topk_idxs

# -------------------------- Retrieval core ------------------------------

class Retriever(ABC):
    @abstractmethod
    def get_scores(self, query: str, pool_size: int, chunks: List[str]):
        """Retrieves the top 'pool_size' chunks cores for a given query."""
        pass


class FAISSRetriever(Retriever):
    name = "faiss"

    def __init__(self, index, embed_model: str):
        self.index = index
        self.embedder = _get_embedder(embed_model)

    def get_scores(self,
                query: str,
                pool_size: int,
                chunks: List[str]) -> Dict[int, float]:
        """
        Returns FAISS scores for top 'pool_size' keyed by global chunk index.
        """
        # FAISS expects a 2D array
        q_vec = self.embedder.encode([query]).astype("float32")
        
        # Safety check on vector dimensions
        if q_vec.shape[1] !=  self.index.d:
            raise ValueError(
                f"Embedding dim mismatch: index={ self.index.d} vs query={q_vec.shape[1]}"
            )

        # Perform the search
        distances, indices =  self.index.search(q_vec, pool_size)

        # Remove invalid indices and ensure they are within bounds
        cand_idxs = [i for i in indices[0] if 0 <= i < len(chunks)]

        # Create the distance dictionary, ensuring we only include valid candidates
        dists = {idx: float(dist) for idx, dist in zip(cand_idxs, distances[0][:len(cand_idxs)])}

        # Invert distance to score: 1 / (1 + distance). Adding 1 avoids division by zero.
        return {
            idx: 1.0 / (1.0 + dist)
            for idx, dist in dists.items()
        }


class BM25Retriever(Retriever):
    name = "bm25"

    def __init__(self, index):
        self.index = index

    def get_scores(self,
                 query: str,
                 pool_size: int,
                 chunks: List[str]) -> Dict[int, float]:
        """
        Returns BM25 scores for top 'pool_size' keyed by global chunk index.
        """
        # Tokenize the query in the same way the index was built
        tokenized_query = preprocess_for_bm25(query)

        # Get scores for all documents in the corpus
        all_scores = self.index.get_scores(tokenized_query)

        # Find the indices of the top 'pool_size' scores
        num_candidates = min(pool_size, len(all_scores))
        top_k_indices = np.argpartition(-all_scores, kth=num_candidates-1)[:num_candidates]

        # Remove invalid indices and ensure they are within bounds
        top_k_indices = [i for i in top_k_indices if 0 <= i < len(chunks)]
        
        # Get the corresponding scores for the top indices
        top_scores = all_scores[top_k_indices]

        # Format the output as a dictionary of scores
        scores = {int(idx): float(score) for idx, score in zip(top_k_indices, top_scores)}

        return scores


class IndexKeywordRetriever(Retriever):
    name = "index_keywords"
    
    def __init__(self, extracted_index_path: os.PathLike, page_to_chunk_map_path: os.PathLike):
        """
        Retriever that uses textbook index keywords to boost chunks on relevant pages.
        
        Args:
            extracted_index_path: Path to extracted_index.json (keyword -> page numbers)
            page_to_chunk_map_path: Path to page_to_chunk_map.json (page -> chunk IDs)
        """
        import json
        nltk.download('wordnet', quiet=True)
        self.page_to_chunk_map = {}
        
        # Load and normalize index: lemmatize phrases as units
        # Build token->phrase mapping for fast lookup
        if os.path.exists(extracted_index_path):
            lemmatizer = WordNetLemmatizer()
            
            with open(extracted_index_path, 'r') as f:
                raw_index = json.load(f)
                self.phrase_to_pages = {}  # phrase -> pages
                self.token_to_phrases = {}  # token -> [phrases]
                
                for key, pages in raw_index.items():
                    # Lemmatize each word in the phrase but keep phrase together
                    key_lower = key.lower()
                    words = key_lower.split()
                    lemmatized_words = []
                    
                    for word in words:
                        cleaned = word.strip('.,!?()[]:"\'')
                        if not cleaned:
                            continue
                        lemmatized_words.append(self._lemmatize_word(cleaned, lemmatizer))
                    
                    lemmatized_phrase = ' '.join(lemmatized_words)
                    self.phrase_to_pages[lemmatized_phrase] = pages
                    
                    # Build reverse index: each token points to phrases containing it
                    for token in lemmatized_words:
                        if token not in self.token_to_phrases:
                            self.token_to_phrases[token] = []
                        self.token_to_phrases[token].append(lemmatized_phrase)
        else:
            self.phrase_to_pages = {}
            self.token_to_phrases = {}
        
        if os.path.exists(page_to_chunk_map_path):
            with open(page_to_chunk_map_path, 'r') as f:
                self.page_to_chunk_map = json.load(f)
    
    def get_scores(self, query: str, pool_size: int, chunks: List[str]) -> Dict[int, float]:
        """
        Returns scores for chunks that match index keywords.
        Score is proportional to the number of keyword hits.
        """
        keywords = self._extract_keywords(query)
        # chunk_id -> hit count
        chunk_hit_counts: Dict[int, int] = {} 
        
        # Match query keywords against index phrases (token overlap)
        for keyword in keywords:
            if keyword not in self.token_to_phrases:
                continue
            
            # Get all phrases containing this keyword token
            matching_phrases = self.token_to_phrases[keyword]
            
            for phrase in matching_phrases:
                page_numbers = self.phrase_to_pages[phrase]
                
                # Map pages to chunks
                for page_no in page_numbers:
                    chunk_ids = self.page_to_chunk_map.get(str(page_no), [])
                    for chunk_id in chunk_ids:
                        if chunk_id >= 0 and chunk_id < len(chunks):
                            chunk_hit_counts[chunk_id] = chunk_hit_counts.get(chunk_id, 0) + 1
        
        if not chunk_hit_counts:
            return {}
        
        # Normalize scores: more keyword hits = higher score
        max_hits = max(chunk_hit_counts.values())
        scores = {
            chunk_id: float(hit_count) / max_hits
            for chunk_id, hit_count in chunk_hit_counts.items()
        }
        
        return scores
    
    @staticmethod
    def _lemmatize_word(word: str, lemmatizer) -> str:
        """Lemmatize a word, trying noun then verb."""
        lemma = lemmatizer.lemmatize(word, pos='n')
        if lemma == word:
            lemma = lemmatizer.lemmatize(word, pos='v')
        return lemma
    
    @staticmethod
    def _extract_keywords(query: str) -> List[str]:
        """Extract keywords from query by removing stopwords and lemmatizing."""
        
        stopwords = {
            "the", "is", "at", "which", "on", "for", "a", "an", "and", "or", "in",
            "to", "of", "by", "with", "that", "this", "it", "as", "are", "was", 
            "what", "how", "why", "when", "where", "who", "does", "do", "be"
        }
        
        lemmatizer = WordNetLemmatizer()
        words = query.lower().split()
        keywords = []
        for word in words:
            cleaned = word.strip('.,!?()[]:"\'')
            if not cleaned or cleaned in stopwords:
                continue
            keywords.append(IndexKeywordRetriever._lemmatize_word(cleaned, lemmatizer))
        return keywords

# Grover sampling, used by GroverRetriever
def _grover_num_qubits_for_states(n_states: int) -> int:
    if n_states <= 1:
        return 1
    return max(1, int(math.ceil(math.log2(n_states))))

def _grover_default_iterations(n_states_pad: int, n_marked: int) -> int:
    if n_states_pad <= 1 or n_marked <= 0:
        return 0
    ratio = n_states_pad / float(n_marked)
    it = int(round((math.pi / 4.0) * math.sqrt(ratio)))
    return max(1, min(it, 50))

def _grover_measurement_distribution(
    n_states: int,
    marked_indices: Collection[int],
    num_iterations: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import Diagonal as DiagonalGate
    from qiskit.circuit.library import GroverOperator
    from qiskit.quantum_info import Statevector

    if n_states <= 0:
        return np.array([], dtype=np.float64), 0

    n_qubits = _grover_num_qubits_for_states(n_states)
    dim = 2**n_qubits
    marked = {int(i) for i in marked_indices if 0 <= int(i) < n_states}
    if not marked:
        marked = {0}

    diag = np.ones(dim, dtype=np.complex128)
    for idx in marked:
        if idx < dim:
            diag[idx] = -1.0
    oracle = QuantumCircuit(n_qubits)
    oracle.append(DiagonalGate(diag.tolist()), range(n_qubits))
    grover_op = GroverOperator(oracle)

    iters = num_iterations
    if iters is None:
        iters = _grover_default_iterations(dim, len(marked))

    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))
    if iters > 0:
        qc.compose(grover_op.power(iters), qubits=range(n_qubits), inplace=True)

    sv = Statevector(qc)
    probs = np.abs(sv.data) ** 2
    return probs.real.astype(np.float64), n_qubits

def _sample_grover_indices(
    n_states: int,
    marked_indices: Collection[int],
    shots: int,
    num_iterations: Optional[int] = None,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if n_states <= 0 or shots <= 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    probs, _ = _grover_measurement_distribution(
        n_states, marked_indices, num_iterations=num_iterations
    )
    dim = probs.size
    rng = np.random.default_rng(seed)
    draws = rng.choice(dim, size=shots, p=probs)

    mask = draws < n_states
    filtered = draws[mask]
    if filtered.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    unique, counts = np.unique(filtered, return_counts=True)
    return unique.astype(np.int64), counts.astype(np.int64)

class GroverRetriever(Retriever):
    """
    Grover search over a capped chunk pool: cosine similarity marks states, Qiskit's
    GroverOperator biases sampling; returned scores mix shot frequencies with a small
    classical tie-break.
    """

    name = "grover"

    def __init__(self, cfg: RAGConfig):
        self.cfg = cfg
        self._embedder = _get_embedder(cfg.embed_model)

    def get_scores(
        self,
        query: str,
        pool_size: int,
        chunks: List[str],
    ) -> Dict[int, float]:
        if not chunks or pool_size <= 0:
            return {}

        cap = min(
            len(chunks),
            pool_size,
            max(1, int(self.cfg.grover_max_pool)),
        )
        n = cap
        pool = chunks[:n]

        q = self._embedder.encode([query], normalize=True).astype(np.float32)
        emb = self._embedder.encode(
            pool,
            normalize=True,
            batch_size=max(1, int(self.cfg.grover_embed_batch_size)),
        )
        emb = np.asarray(emb, dtype=np.float32)
        sims = (q @ emb.T).ravel()

        m_mark = max(1, min(int(self.cfg.grover_mark_top_m), n))
        marked = set(np.argpartition(-sims, kth=m_mark - 1)[:m_mark].tolist())

        shots = max(1, int(self.cfg.grover_shots))
        unique, counts = _sample_grover_indices(
            n,
            marked,
            shots=shots,
            num_iterations=self.cfg.grover_iterations,
            seed=self.cfg.grover_seed,
        )

        pq = np.zeros(n, dtype=np.float64)
        if unique.size > 0:
            pq[unique] = counts.astype(np.float64) / float(shots)

        sim_min, sim_max = float(sims.min()), float(sims.max())
        if sim_max <= sim_min:
            classical_norm = np.ones(n, dtype=np.float64)
        else:
            classical_norm = ((sims.astype(np.float64) - sim_min) / (sim_max - sim_min)).ravel()

        tie = 1e-6
        comb = pq + tie * classical_norm

        k = min(pool_size, n)
        top_local = np.argpartition(-comb, kth=k - 1)[:k]
        global_indices = top_local.astype(int)
        return {int(i): float(comb[i]) for i in global_indices}
