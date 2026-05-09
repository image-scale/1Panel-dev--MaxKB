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
- [ ] Document model has fields: id (UUID), knowledge (FK), name, char_length, status, status_meta, is_active, type, hit_handling_method, directly_return_similarity, meta, create_time, update_time
- [ ] Document status supports states: PENDING, STARTED, SUCCESS, FAILURE, REVOKE, REVOKED, IGNORED
- [ ] Hit handling methods include: optimization (model optimization), directly_return (direct answer)
- [ ] Documents can be created within a knowledge base
- [ ] Documents can be retrieved by ID or listed with filtering/pagination
- [ ] Documents can be updated (name, status, settings)
- [ ] Documents can be deleted or deactivated (is_active=False)
- [ ] REST API endpoints for document CRUD operations work correctly
- [ ] Documents track character length and can store arbitrary metadata
