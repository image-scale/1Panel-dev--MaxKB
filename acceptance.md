# Acceptance Criteria

## Task 1: User Authentication System (COMPLETED)

### Acceptance Criteria
- [x] User model stores id, username, email, password_hash, role, is_active, created_at, updated_at
- [x] password_encrypt("password123") returns a consistent MD5 hash different from the original password
- [x] User.set_password("password123") stores hashed password, not plaintext
- [x] User.check_password("password123") returns True for correct password, False for wrong password
- [x] create_user(username="testuser", email="test@test.com", password="pass123") creates and returns a new user
- [x] create_user with existing username raises ValueError
- [x] authenticate_user("testuser", "pass123") returns user if credentials are valid
- [x] authenticate_user("testuser", "wrongpass") returns None
- [x] generate_token(user) returns a JWT token string containing user_id
- [x] verify_token(token) returns the user_id from a valid token
- [x] verify_token with expired/invalid token returns None
- [x] UserRole enum has ADMIN and USER values

## Task 2: Knowledge Base Management

### Acceptance Criteria
- [ ] KnowledgeBase model stores id, name, description, user_id, created_at, updated_at, settings
- [ ] create_knowledge_base(name="My KB", description="Test", user_id="123") creates and returns a new knowledge base
- [ ] create_knowledge_base with duplicate name for same user raises ValueError
- [ ] get_knowledge_base(kb_id) returns the knowledge base if it exists, None otherwise
- [ ] list_knowledge_bases(user_id="123") returns all knowledge bases for that user
- [ ] update_knowledge_base(kb_id, name="New Name") updates the knowledge base
- [ ] delete_knowledge_base(kb_id) removes the knowledge base and returns True
- [ ] delete_knowledge_base with non-existent id returns False
- [ ] KnowledgeBase.to_dict() returns a dictionary representation without internal state
- [ ] Knowledge base settings include default chunk_size and similarity_threshold
