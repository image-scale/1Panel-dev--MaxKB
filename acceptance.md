# Acceptance Criteria

(Updated before each feature implementation. Define what "done" means for each task.)

## Task 1: User Model with Secure Password Hashing and CRUD Operations

### Acceptance Criteria
- [x] User model has fields: id (UUID), email, phone, nick_name, username, password, role, source, is_active, language, create_time, update_time
- [x] Password is securely hashed using SHA-256 before storage (never stored in plain text)
- [x] set_password("mysecret") stores a hash, not "mysecret"
- [x] Users can be created with required fields (username, password, role)
- [x] Users can be retrieved by ID or username
- [x] Users can be updated (change nick_name, email, etc.)
- [x] Users can be deleted (soft delete by setting is_active=False or hard delete)
- [x] Duplicate usernames are rejected
- [x] Django REST API endpoints for user CRUD operations work correctly

## Task 2: Knowledge Base Model with Folder Organization and CRUD API

### Acceptance Criteria
- [x] Knowledge base model has fields: id (UUID), name, description, workspace_id, user (foreign key), type, scope, file_size_limit, file_count_limit, meta, create_time, update_time
- [x] Knowledge base types include: BASE (general), WEB (web crawl), WORKFLOW
- [x] Knowledge base scopes include: SHARED (shared across workspaces), WORKSPACE (workspace-specific)
- [x] Folder model for organizing knowledge bases with hierarchical structure (parent-child)
- [x] Knowledge bases can be created with name, description, and workspace_id
- [x] Knowledge bases can be retrieved by ID or listed with filtering/pagination
- [x] Knowledge bases can be updated (name, description, settings)
- [x] Knowledge bases can be deleted
- [x] REST API endpoints for knowledge base CRUD operations work correctly
- [x] Knowledge bases can be associated with folders for organization

## Task 3: Document Model with Status Tracking and Document Management API

### Acceptance Criteria
- [x] Document model has fields: id (UUID), knowledge (FK), name, char_length, status, status_meta, is_active, type, hit_handling_method, directly_return_similarity, meta, create_time, update_time
- [x] Document status supports states: PENDING, STARTED, SUCCESS, FAILURE, REVOKE, REVOKED, IGNORED
- [x] Hit handling methods include: optimization (model optimization), directly_return (direct answer)
- [x] Documents can be created within a knowledge base
- [x] Documents can be retrieved by ID or listed with filtering/pagination
- [x] Documents can be updated (name, status, settings)
- [x] Documents can be deleted or deactivated (is_active=False)
- [x] REST API endpoints for document CRUD operations work correctly
- [x] Documents track character length and can store arbitrary metadata

## Task 4: Paragraph Model for Storing Document Segments

### Acceptance Criteria
- [x] Paragraph model has fields: id (UUID), document (FK), knowledge (FK), content, title, status, status_meta, hit_num, is_active, position, create_time, update_time
- [x] Paragraphs belong to documents and knowledge bases
- [x] Paragraphs track hit count for analytics
- [x] Paragraphs can be ordered by position
- [x] Paragraphs can be created with content and title
- [x] Paragraphs can be retrieved by ID or listed with filtering/pagination
- [x] Paragraphs can be updated (content, title, status, position)
- [x] Paragraphs can be deleted or deactivated
- [x] REST API endpoints for paragraph CRUD operations work correctly

## Task 5: Text Chunking Utilities

### Acceptance Criteria
- [x] Text chunking function that splits text into segments of configurable size
- [x] Chunking respects sentence boundaries (splits at punctuation like . ! ? ; newlines)
- [x] Chunks do not exceed specified maximum length
- [x] Empty chunks are filtered out
- [x] Whitespace is properly trimmed from chunks
- [x] Chinese and English text both work correctly
- [x] Large texts exceeding chunk size are handled gracefully

## Task 6: Embedding Storage Model

### Acceptance Criteria
- [x] Embedding model stores vector representations with paragraph associations
- [x] Embeddings link to source (paragraph, problem, or title)
- [x] Embeddings can be queried by knowledge base, document, or paragraph
- [x] Embedding vectors can be stored as JSON list of floats
- [x] Source types include: paragraph, problem, title
- [x] Embeddings can be activated/deactivated
- [x] Batch save functionality for efficiency
- [x] Embeddings can be deleted by source ID or knowledge base

## Task 7: Vector Search Capabilities

### Acceptance Criteria
- [x] Embedding search function performs vector similarity calculations
- [x] Keyword search function performs text-based search on paragraphs
- [x] Blend search combines vector and keyword search results
- [x] Search results include relevance scores
- [x] Search supports filtering by knowledge base and document
- [x] Search respects active status of paragraphs and embeddings
- [x] Search results are ranked by similarity/relevance
- [x] Search returns paragraph content with metadata

## Task 8: Problem/Question Model

### Acceptance Criteria
- [x] Problem model links questions to paragraphs for Q&A
- [x] Problems have content with hit tracking
- [x] Problems can be associated with multiple paragraphs
- [x] Problems can be queried by paragraph or knowledge base
- [x] Problems support CRUD operations through API
- [x] Problem-paragraph mappings support CRUD

## Task 9: Application/Agent Model

### Acceptance Criteria
- [x] Application model stores agent configuration
- [x] Applications link to knowledge bases for RAG
- [x] Applications have configurable settings (prompts, limits)
- [x] Applications track usage statistics
- [x] Applications support CRUD operations through API
- [x] Applications can be activated/deactivated

## Task 10: Chat Pipeline

### Acceptance Criteria
- [x] Chat model stores conversation history
- [x] Chat messages have role (user, assistant, system)
- [x] Chat sessions belong to applications
- [x] Chat supports context retrieval from linked knowledge bases
- [x] Chat API supports sending messages and getting responses
- [x] Chat history can be cleared or retrieved

## Task 11: Model Provider Abstraction

### Acceptance Criteria
- [x] Base model provider interface with abstract methods
- [x] Model types enum (LLM, EMBEDDING, STT, TTS, IMAGE, RERANKER)
- [x] Model configuration storage (name, provider, credentials, type)
- [x] Provider credential validation interface
- [x] Model instance creation from provider and credentials
- [x] Provider registry for managing available providers
- [x] Model CRUD operations through API
- [x] Models can be filtered by type and provider

## Task 12: Tag System

### Acceptance Criteria
- [x] Tag model with key-value pairs scoped to knowledge bases
- [x] DocumentTag mapping for associating tags with documents
- [x] Tags can be created, retrieved, updated, and deleted
- [x] Documents can have multiple tags assigned
- [x] Tags are unique per key-value within a knowledge base
- [x] API endpoints for tag CRUD and document-tag associations
- [x] Documents can be filtered by tags
- [x] Tag operations cascade properly (delete tag removes mappings)
