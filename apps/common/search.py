"""
Vector and keyword search utilities for RAG retrieval.

Provides three search modes:
- Embedding search: Cosine similarity-based vector search
- Keyword search: Text-based search using term matching
- Blend search: Combines embedding and keyword scores
"""
import math
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class SearchMode(Enum):
    """Search mode options."""
    EMBEDDING = 'embedding'
    KEYWORDS = 'keywords'
    BLEND = 'blend'


@dataclass
class SearchResult:
    """Result of a search operation."""
    paragraph_id: str
    document_id: str
    knowledge_id: str
    content: str
    title: str
    similarity: float
    source_type: str = 'p'

    def to_dict(self) -> dict:
        return {
            'paragraph_id': self.paragraph_id,
            'document_id': self.document_id,
            'knowledge_id': self.knowledge_id,
            'content': self.content,
            'title': self.title,
            'similarity': self.similarity,
            'source_type': self.source_type,
        }


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        vec1: First vector as list of floats
        vec2: Second vector as list of floats

    Returns:
        Cosine similarity score between 0 and 1
    """
    if not vec1 or not vec2:
        return 0.0

    if len(vec1) != len(vec2):
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)


def euclidean_distance(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate euclidean distance between two vectors.

    Args:
        vec1: First vector as list of floats
        vec2: Second vector as list of floats

    Returns:
        Euclidean distance (lower is more similar)
    """
    if not vec1 or not vec2:
        return float('inf')

    if len(vec1) != len(vec2):
        return float('inf')

    return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2)))


def normalize_text(text: str) -> str:
    """
    Normalize text for search by lowercasing and removing extra whitespace.

    Args:
        text: Raw text string

    Returns:
        Normalized text string
    """
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def tokenize(text: str) -> List[str]:
    """
    Tokenize text into words for keyword search.

    Handles both English and Chinese text.

    Args:
        text: Text to tokenize

    Returns:
        List of tokens
    """
    if not text:
        return []

    text = normalize_text(text)
    chinese_pattern = re.compile(r'[一-鿿]')
    tokens = []
    current_word = ""

    for char in text:
        if chinese_pattern.match(char):
            if current_word:
                tokens.append(current_word)
                current_word = ""
            tokens.append(char)
        elif char.isalnum():
            current_word += char
        else:
            if current_word:
                tokens.append(current_word)
                current_word = ""

    if current_word:
        tokens.append(current_word)

    return [t for t in tokens if t.strip()]


def keyword_match_score(query: str, text: str) -> float:
    """
    Calculate keyword match score between query and text.

    Uses term frequency normalization for scoring.

    Args:
        query: Search query string
        text: Text to match against

    Returns:
        Score between 0 and 1
    """
    if not query or not text:
        return 0.0

    query_tokens = set(tokenize(query))
    text_tokens = tokenize(text)

    if not query_tokens or not text_tokens:
        return 0.0

    text_token_set = set(text_tokens)
    matches = query_tokens & text_token_set

    if not matches:
        return 0.0

    query_coverage = len(matches) / len(query_tokens)
    text_freq = sum(1 for t in text_tokens if t in matches) / len(text_tokens)

    return (query_coverage + text_freq) / 2


def embedding_search(
    query_vector: List[float],
    embeddings: List[Dict],
    top_n: int = 10,
    similarity_threshold: float = 0.0
) -> List[Dict]:
    """
    Perform vector similarity search.

    Args:
        query_vector: Query embedding vector
        embeddings: List of embedding records with 'embedding' field
        top_n: Maximum number of results to return
        similarity_threshold: Minimum similarity score

    Returns:
        List of matching embeddings with similarity scores
    """
    if not query_vector or not embeddings:
        return []

    results = []
    seen_paragraphs = set()

    for emb in embeddings:
        if not emb.get('is_active', True):
            continue

        vector = emb.get('embedding', [])
        if not vector:
            continue

        similarity = cosine_similarity(query_vector, vector)

        if similarity >= similarity_threshold:
            para_id = str(emb.get('paragraph_id', ''))

            if para_id in seen_paragraphs:
                continue
            seen_paragraphs.add(para_id)

            results.append({
                **emb,
                'similarity': similarity,
            })

    results.sort(key=lambda x: x['similarity'], reverse=True)
    return results[:top_n]


def keyword_search(
    query: str,
    paragraphs: List[Dict],
    top_n: int = 10,
    similarity_threshold: float = 0.0
) -> List[Dict]:
    """
    Perform keyword-based text search.

    Args:
        query: Search query string
        paragraphs: List of paragraph records with 'content' field
        top_n: Maximum number of results to return
        similarity_threshold: Minimum match score (exclusive, 0 returns only matches)

    Returns:
        List of matching paragraphs with similarity scores
    """
    if not query or not paragraphs:
        return []

    results = []

    for para in paragraphs:
        if not para.get('is_active', True):
            continue

        content = para.get('content', '')
        title = para.get('title', '')
        combined = f"{title} {content}"

        score = keyword_match_score(query, combined)

        if score > similarity_threshold:
            results.append({
                **para,
                'similarity': score,
            })

    results.sort(key=lambda x: x['similarity'], reverse=True)
    return results[:top_n]


def blend_search(
    query: str,
    query_vector: List[float],
    embeddings: List[Dict],
    paragraphs: List[Dict],
    top_n: int = 10,
    similarity_threshold: float = 0.0,
    embedding_weight: float = 0.7,
    keyword_weight: float = 0.3
) -> List[Dict]:
    """
    Perform combined embedding and keyword search.

    Blends vector similarity with keyword matching scores.

    Args:
        query: Search query string
        query_vector: Query embedding vector
        embeddings: List of embedding records
        paragraphs: List of paragraph records
        top_n: Maximum number of results
        similarity_threshold: Minimum combined score
        embedding_weight: Weight for embedding score (0-1)
        keyword_weight: Weight for keyword score (0-1)

    Returns:
        List of matching results with combined scores
    """
    if not query_vector or not embeddings:
        return keyword_search(query, paragraphs, top_n, similarity_threshold)

    if not query or not paragraphs:
        return embedding_search(query_vector, embeddings, top_n, similarity_threshold)

    para_map = {str(p.get('id', '')): p for p in paragraphs}

    embedding_results = embedding_search(
        query_vector, embeddings,
        top_n=len(embeddings),
        similarity_threshold=0.0
    )

    emb_scores = {}
    for emb in embedding_results:
        para_id = str(emb.get('paragraph_id', ''))
        if para_id not in emb_scores:
            emb_scores[para_id] = emb['similarity']

    kw_scores = {}
    for para in paragraphs:
        if not para.get('is_active', True):
            continue
        para_id = str(para.get('id', ''))
        content = para.get('content', '')
        title = para.get('title', '')
        combined = f"{title} {content}"
        kw_scores[para_id] = keyword_match_score(query, combined)

    all_para_ids = set(emb_scores.keys()) | set(kw_scores.keys())
    results = []

    for para_id in all_para_ids:
        emb_score = emb_scores.get(para_id, 0.0)
        kw_score = kw_scores.get(para_id, 0.0)

        combined_score = (
            embedding_weight * emb_score +
            keyword_weight * kw_score
        )

        if combined_score >= similarity_threshold:
            para = para_map.get(para_id, {})
            results.append({
                **para,
                'id': para_id,
                'paragraph_id': para_id,
                'similarity': combined_score,
                'embedding_score': emb_score,
                'keyword_score': kw_score,
            })

    results.sort(key=lambda x: x['similarity'], reverse=True)
    return results[:top_n]


def search(
    query: str,
    query_vector: Optional[List[float]],
    embeddings: List[Dict],
    paragraphs: List[Dict],
    mode: SearchMode = SearchMode.EMBEDDING,
    top_n: int = 10,
    similarity_threshold: float = 0.0,
    knowledge_ids: Optional[List[str]] = None,
    document_ids: Optional[List[str]] = None,
    exclude_paragraph_ids: Optional[List[str]] = None
) -> List[SearchResult]:
    """
    Unified search interface supporting multiple search modes.

    Args:
        query: Search query string
        query_vector: Query embedding vector (required for embedding/blend modes)
        embeddings: List of embedding records
        paragraphs: List of paragraph records
        mode: Search mode (embedding, keywords, or blend)
        top_n: Maximum number of results
        similarity_threshold: Minimum similarity score
        knowledge_ids: Filter by knowledge base IDs
        document_ids: Filter by document IDs
        exclude_paragraph_ids: Paragraph IDs to exclude

    Returns:
        List of SearchResult objects
    """
    filtered_embeddings = embeddings
    filtered_paragraphs = paragraphs

    if knowledge_ids:
        knowledge_id_set = set(str(k) for k in knowledge_ids)
        filtered_embeddings = [
            e for e in filtered_embeddings
            if str(e.get('knowledge_id', '')) in knowledge_id_set
        ]
        filtered_paragraphs = [
            p for p in filtered_paragraphs
            if str(p.get('knowledge_id', '')) in knowledge_id_set
        ]

    if document_ids:
        document_id_set = set(str(d) for d in document_ids)
        filtered_embeddings = [
            e for e in filtered_embeddings
            if str(e.get('document_id', '')) in document_id_set
        ]
        filtered_paragraphs = [
            p for p in filtered_paragraphs
            if str(p.get('document_id', '')) in document_id_set
        ]

    if exclude_paragraph_ids:
        exclude_set = set(str(p) for p in exclude_paragraph_ids)
        filtered_embeddings = [
            e for e in filtered_embeddings
            if str(e.get('paragraph_id', '')) not in exclude_set
        ]
        filtered_paragraphs = [
            p for p in filtered_paragraphs
            if str(p.get('id', '')) not in exclude_set
        ]

    if mode == SearchMode.EMBEDDING:
        if not query_vector:
            return []
        raw_results = embedding_search(
            query_vector, filtered_embeddings, top_n, similarity_threshold
        )
    elif mode == SearchMode.KEYWORDS:
        raw_results = keyword_search(
            query, filtered_paragraphs, top_n, similarity_threshold
        )
    else:
        raw_results = blend_search(
            query, query_vector, filtered_embeddings, filtered_paragraphs,
            top_n, similarity_threshold
        )

    para_map = {str(p.get('id', '')): p for p in paragraphs}

    results = []
    for item in raw_results:
        para_id = str(item.get('paragraph_id', item.get('id', '')))
        para = para_map.get(para_id, {})

        results.append(SearchResult(
            paragraph_id=para_id,
            document_id=str(item.get('document_id', para.get('document_id', ''))),
            knowledge_id=str(item.get('knowledge_id', para.get('knowledge_id', ''))),
            content=para.get('content', item.get('content', '')),
            title=para.get('title', item.get('title', '')),
            similarity=item.get('similarity', 0.0),
            source_type=item.get('source_type', 'p'),
        ))

    return results
