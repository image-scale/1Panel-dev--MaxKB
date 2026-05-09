# Acceptance Criteria

(Updated before each feature implementation. Define what "done" means for each task.)

## Task 1: User Model with Secure Password Hashing and CRUD Operations

### Acceptance Criteria
- [ ] User model has fields: id (UUID), email, phone, nick_name, username, password, role, source, is_active, language, create_time, update_time
- [ ] Password is securely hashed using SHA-256 before storage (never stored in plain text)
- [ ] set_password("mysecret") stores a hash, not "mysecret"
- [ ] Users can be created with required fields (username, password, role)
- [ ] Users can be retrieved by ID or username
- [ ] Users can be updated (change nick_name, email, etc.)
- [ ] Users can be deleted (soft delete by setting is_active=False or hard delete)
- [ ] Duplicate usernames are rejected
- [ ] Django REST API endpoints for user CRUD operations work correctly
