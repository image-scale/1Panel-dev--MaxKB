"""
Document processing pipeline for the KnowledgeBot platform.

Handles chunking documents and creating embeddings for vector search.
"""
from typing import List, Optional
from dataclasses import dataclass

from knowledgebot.knowledge.documents import (
    Document,
    DocumentStatus,
    get_document,
    update_document,
    get_doc_store,
)
from knowledgebot.core.chunking import chunk_text, ChunkingConfig
from knowledgebot.core.embeddings import get_embedding_provider, EmbeddingProvider
from knowledgebot.core.vectorstore import get_vector_store, VectorStore


@dataclass
class ProcessingResult:
    """Result of processing a document."""
    document_id: str
    success: bool
    chunks_created: int
    error_message: Optional[str] = None


def process_document(
    document_id: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    embedding_provider: Optional[EmbeddingProvider] = None,
    vector_store: Optional[VectorStore] = None,
) -> ProcessingResult:
    """
    Process a document: chunk it and create embeddings.

    Args:
        document_id: ID of the document to process
        chunk_size: Size of chunks in characters
        chunk_overlap: Overlap between chunks
        embedding_provider: Provider for embeddings (uses global if None)
        vector_store: Store for vectors (uses global if None)

    Returns:
        ProcessingResult with status and chunk count
    """
    # Get the document
    document = get_document(document_id)
    if document is None:
        return ProcessingResult(
            document_id=document_id,
            success=False,
            chunks_created=0,
            error_message=f"Document {document_id} not found"
        )

    # Update status to processing
    update_document(document_id, status=DocumentStatus.PROCESSING)

    try:
        # Use defaults if not provided
        provider = embedding_provider or get_embedding_provider()
        store = vector_store or get_vector_store()

        # Delete any existing vectors for this document
        store.delete_by_metadata(document.knowledge_base_id, "document_id", document_id)

        # Chunk the document
        chunks = chunk_text(document.content, chunk_size, chunk_overlap)

        if not chunks:
            # No content to process
            update_document(document_id, status=DocumentStatus.COMPLETED)
            return ProcessingResult(
                document_id=document_id,
                success=True,
                chunks_created=0
            )

        # Create embeddings and store vectors
        embeddings = provider.embed_texts(chunks)

        for i, (chunk_text_content, embedding) in enumerate(zip(chunks, embeddings)):
            store.add_vector(
                vector=embedding,
                text=chunk_text_content,
                knowledge_base_id=document.knowledge_base_id,
                metadata={
                    "document_id": document_id,
                    "document_name": document.name,
                    "chunk_index": i,
                }
            )

        # Update status to completed
        update_document(document_id, status=DocumentStatus.COMPLETED)

        return ProcessingResult(
            document_id=document_id,
            success=True,
            chunks_created=len(chunks)
        )

    except Exception as e:
        # Update status to error
        update_document(document_id, status=DocumentStatus.ERROR)
        return ProcessingResult(
            document_id=document_id,
            success=False,
            chunks_created=0,
            error_message=str(e)
        )


def process_documents(
    document_ids: List[str],
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> List[ProcessingResult]:
    """
    Process multiple documents.

    Args:
        document_ids: List of document IDs to process
        chunk_size: Size of chunks
        chunk_overlap: Overlap between chunks

    Returns:
        List of ProcessingResult for each document
    """
    return [
        process_document(doc_id, chunk_size, chunk_overlap)
        for doc_id in document_ids
    ]


def reprocess_knowledge_base(
    knowledge_base_id: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> List[ProcessingResult]:
    """
    Reprocess all documents in a knowledge base.

    Args:
        knowledge_base_id: ID of the knowledge base
        chunk_size: Size of chunks
        chunk_overlap: Overlap between chunks

    Returns:
        List of ProcessingResult for each document
    """
    doc_store = get_doc_store()
    documents = doc_store.list_by_knowledge_base(knowledge_base_id)
    return process_documents([doc.id for doc in documents], chunk_size, chunk_overlap)


def get_document_chunks(document_id: str) -> List[str]:
    """
    Get the chunks stored for a document.

    Args:
        document_id: ID of the document

    Returns:
        List of chunk texts
    """
    document = get_document(document_id)
    if document is None:
        return []

    store = get_vector_store()
    entries = store.list_by_knowledge_base(document.knowledge_base_id)

    # Filter to this document and sort by chunk_index
    doc_entries = [
        e for e in entries
        if e.metadata.get("document_id") == document_id
    ]
    doc_entries.sort(key=lambda e: e.metadata.get("chunk_index", 0))

    return [e.text for e in doc_entries]


def search_knowledge_base(
    query: str,
    knowledge_base_ids: List[str],
    top_k: int = 5,
    min_score: float = 0.0,
    embedding_provider: Optional[EmbeddingProvider] = None,
    vector_store: Optional[VectorStore] = None,
) -> List[dict]:
    """
    Search for relevant content in knowledge bases.

    Args:
        query: The search query text
        knowledge_base_ids: Knowledge bases to search
        top_k: Maximum number of results
        min_score: Minimum similarity score
        embedding_provider: Provider for embeddings
        vector_store: Store for vectors

    Returns:
        List of search results with text, score, and metadata
    """
    provider = embedding_provider or get_embedding_provider()
    store = vector_store or get_vector_store()

    # Embed the query
    query_embedding = provider.embed_text(query)

    # Search
    results = store.search(
        query_vector=query_embedding,
        knowledge_base_ids=knowledge_base_ids,
        top_k=top_k,
        min_score=min_score,
    )

    return [result.to_dict() for result in results]
