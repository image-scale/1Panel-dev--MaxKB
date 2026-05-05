# Todo

## Plan
Build the RAG platform in layers: first establish core data models and utilities, then implement knowledge base with document processing and embeddings, followed by application/agent configuration, and finally the chat interface with RAG pipeline. Each task delivers a complete, testable feature.

## Tasks
- [x] Task 1: Implement user authentication system with registration, login, JWT token generation, and password hashing (core/auth.py, core/models.py + tests)
- [x] Task 2: Implement knowledge base management allowing users to create, list, update, and delete knowledge bases with metadata (knowledge/models.py, knowledge/service.py + tests)
- [x] Task 3: Implement document upload and storage with file handling and document metadata management (knowledge/documents.py + tests)
- [x] Task 4: Implement text chunking to split documents into smaller pieces based on configurable chunk size and overlap (core/chunking.py + tests)
- [x] Task 5: Implement embedding service interface and a simple embedding provider for creating vector representations of text (core/embeddings.py + tests)
- [x] Task 6: Implement vector store for storing and querying embeddings with similarity search capabilities (core/vectorstore.py + tests)
- [x] Task 7: Implement document processing pipeline that chunks documents and creates embeddings (knowledge/processing.py + tests)
- [x] Task 8: Implement LLM provider interface with OpenAI-compatible implementation for chat completions (providers/llm.py + tests)
- [x] Task 9: Implement application/agent configuration with prompt templates, RAG settings, and model selection (applications/models.py, applications/service.py + tests)
- [x] Task 10: Implement chat service with RAG pipeline that retrieves relevant context and generates responses (chat/service.py + tests)
- [x] Task 11: Implement REST API endpoints for user operations (api/users.py + tests)
- [ ] Task 12: Implement REST API endpoints for knowledge base and document operations (api/knowledge.py + tests)
- [ ] Task 13: Implement REST API endpoints for application management (api/applications.py + tests)
- [ ] Task 14: Implement REST API endpoints for chat with streaming support (api/chat.py + tests)
