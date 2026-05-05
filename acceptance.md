# Acceptance Criteria

## Task 1: User Authentication System

### Acceptance Criteria
- [ ] User model stores id, username, email, password_hash, role, is_active, created_at, updated_at
- [ ] password_encrypt("password123") returns a consistent MD5 hash different from the original password
- [ ] User.set_password("password123") stores hashed password, not plaintext
- [ ] User.check_password("password123") returns True for correct password, False for wrong password
- [ ] create_user(username="testuser", email="test@test.com", password="pass123") creates and returns a new user
- [ ] create_user with existing username raises ValueError
- [ ] authenticate_user("testuser", "pass123") returns user if credentials are valid
- [ ] authenticate_user("testuser", "wrongpass") returns None
- [ ] generate_token(user) returns a JWT token string containing user_id
- [ ] verify_token(token) returns the user_id from a valid token
- [ ] verify_token with expired/invalid token returns None
- [ ] UserRole enum has ADMIN and USER values
