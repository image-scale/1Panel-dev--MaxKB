# Goal

## Project
MaxKB — a Python/Django project for building enterprise-grade AI agents with RAG capabilities.

## Description
MaxKB (Max Knowledge Brain) is an open-source platform for building enterprise-grade AI agents. It integrates Retrieval-Augmented Generation (RAG) pipelines, supports workflow orchestration, and provides LLM model integration. The platform enables intelligent Q&A capabilities through knowledge base management, document processing, text vectorization, and semantic search. It is used for intelligent customer service, corporate knowledge bases, academic research, and education.

## Core Features
1. **User Management** - User authentication with secure password handling
2. **Knowledge Base Management** - Create, update, delete knowledge bases with documents and paragraphs
3. **Document Processing** - Upload documents, split into paragraphs, manage document lifecycle
4. **Text Chunking** - Split text content into manageable chunks for RAG
5. **Vector Store** - Store and query embeddings for semantic search (pgvector-based)
6. **Search Capabilities** - Embedding search, keyword search, and hybrid blend search
7. **Application/Agent Management** - Create AI applications that use knowledge bases
8. **Chat System** - Pipeline for processing chat messages through RAG
9. **Model Provider Integration** - Support multiple LLM providers (OpenAI-compatible)

## Scope
- Focus on core backend functionality with Django REST API
- Implement essential data models, serializers, and views
- Include text processing utilities for RAG pipeline
- Provide vector storage and search capabilities
- Tests covering all implemented functionality
