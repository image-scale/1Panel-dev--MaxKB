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
- [ ] Knowledge base model has fields: id (UUID), name, description, workspace_id, user (foreign key), type, scope, file_size_limit, file_count_limit, meta, create_time, update_time
- [ ] Knowledge base types include: BASE (general), WEB (web crawl), WORKFLOW
- [ ] Knowledge base scopes include: SHARED (shared across workspaces), WORKSPACE (workspace-specific)
- [ ] Folder model for organizing knowledge bases with hierarchical structure (parent-child)
- [ ] Knowledge bases can be created with name, description, and workspace_id
- [ ] Knowledge bases can be retrieved by ID or listed with filtering/pagination
- [ ] Knowledge bases can be updated (name, description, settings)
- [ ] Knowledge bases can be deleted
- [ ] REST API endpoints for knowledge base CRUD operations work correctly
- [ ] Knowledge bases can be associated with folders for organization
