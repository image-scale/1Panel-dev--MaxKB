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
- [ ] Text chunking function that splits text into segments of configurable size
- [ ] Chunking respects sentence boundaries (splits at punctuation like . ! ? ; newlines)
- [ ] Chunks do not exceed specified maximum length
- [ ] Empty chunks are filtered out
- [ ] Whitespace is properly trimmed from chunks
- [ ] Chinese and English text both work correctly
- [ ] Large texts exceeding chunk size are handled gracefully
