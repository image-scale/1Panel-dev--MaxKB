"""
Tests for User model and API endpoints.
"""
import hashlib
from django.test import TestCase
from rest_framework.test import APITestCase
from rest_framework import status
from apps.users.models import User, generate_password_hash, UserRole, UserSource


class PasswordHashingTest(TestCase):
    """Test password hashing functionality."""

    def test_generate_password_hash_returns_sha256(self):
        """Password hash should be SHA-256."""
        password = "mysecretpassword"
        expected = hashlib.sha256(password.encode('utf-8')).hexdigest()
        result = generate_password_hash(password)
        self.assertEqual(result, expected)
        self.assertEqual(len(result), 64)

    def test_generate_password_hash_deterministic(self):
        """Same password should generate same hash."""
        password = "test123"
        hash1 = generate_password_hash(password)
        hash2 = generate_password_hash(password)
        self.assertEqual(hash1, hash2)

    def test_generate_password_hash_different_passwords(self):
        """Different passwords should generate different hashes."""
        hash1 = generate_password_hash("password1")
        hash2 = generate_password_hash("password2")
        self.assertNotEqual(hash1, hash2)


class UserModelTest(TestCase):
    """Test User model functionality."""

    def test_create_user_stores_hashed_password(self):
        """User creation should store hashed password, not plain text."""
        user = User.create_user(
            username="testuser",
            password="mysecret",
            nick_name="Test User"
        )
        self.assertNotEqual(user.password, "mysecret")
        expected_hash = generate_password_hash("mysecret")
        self.assertEqual(user.password, expected_hash)

    def test_set_password_hashes_correctly(self):
        """set_password should store a hash, not the raw password."""
        user = User(username="testuser", nick_name="Test")
        user.set_password("mysecret")
        self.assertNotEqual(user.password, "mysecret")
        self.assertEqual(len(user.password), 64)

    def test_check_password_correct(self):
        """check_password returns True for correct password."""
        user = User(username="testuser", nick_name="Test")
        user.set_password("correctpassword")
        self.assertTrue(user.check_password("correctpassword"))

    def test_check_password_incorrect(self):
        """check_password returns False for incorrect password."""
        user = User(username="testuser", nick_name="Test")
        user.set_password("correctpassword")
        self.assertFalse(user.check_password("wrongpassword"))

    def test_user_has_uuid_id(self):
        """User ID should be a UUID."""
        user = User.create_user(
            username="uuiduser",
            password="pass123",
            nick_name="UUID User"
        )
        self.assertIsNotNone(user.id)
        self.assertEqual(len(str(user.id)), 36)

    def test_user_default_role_is_user(self):
        """Default role should be USER."""
        user = User.create_user(
            username="roleuser",
            password="pass123",
            nick_name="Role User"
        )
        self.assertEqual(user.role, UserRole.USER)

    def test_user_default_source_is_local(self):
        """Default source should be LOCAL."""
        user = User.create_user(
            username="sourceuser",
            password="pass123",
            nick_name="Source User"
        )
        self.assertEqual(user.source, UserSource.LOCAL)

    def test_user_is_active_by_default(self):
        """User should be active by default."""
        user = User.create_user(
            username="activeuser",
            password="pass123",
            nick_name="Active User"
        )
        self.assertTrue(user.is_active)

    def test_duplicate_username_raises_error(self):
        """Creating user with duplicate username should fail."""
        User.create_user(
            username="dupuser",
            password="pass123",
            nick_name="First User"
        )
        with self.assertRaises(Exception):
            User.create_user(
                username="dupuser",
                password="pass456",
                nick_name="Second User"
            )

    def test_user_timestamps_auto_set(self):
        """create_time and update_time should be auto-set."""
        user = User.create_user(
            username="timeuser",
            password="pass123",
            nick_name="Time User"
        )
        self.assertIsNotNone(user.create_time)
        self.assertIsNotNone(user.update_time)


class UserAPITest(APITestCase):
    """Test User API endpoints."""

    def setUp(self):
        """Create test users."""
        self.user1 = User.create_user(
            username="apiuser1",
            password="testpass1",
            nick_name="API User 1",
            email="user1@example.com"
        )
        self.user2 = User.create_user(
            username="apiuser2",
            password="testpass2",
            nick_name="API User 2",
            email="user2@example.com"
        )

    def test_list_users(self):
        """GET /api/users/ should return list of users."""
        response = self.client.get('/api/users/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['code'], 200)
        self.assertIsInstance(response.data['data'], list)
        self.assertEqual(len(response.data['data']), 2)

    def test_create_user(self):
        """POST /api/users/ should create a new user."""
        data = {
            'username': 'newuser',
            'password': 'newpass123',
            'nick_name': 'New User',
            'email': 'new@example.com',
            'role': 'ADMIN'
        }
        response = self.client.post('/api/users/', data, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response.data['code'], 201)
        self.assertEqual(response.data['data']['username'], 'newuser')
        self.assertNotIn('password', response.data['data'])

    def test_create_user_password_not_exposed(self):
        """Created user response should not contain password."""
        data = {
            'username': 'secureuser',
            'password': 'secretpass',
            'nick_name': 'Secure User'
        }
        response = self.client.post('/api/users/', data, format='json')
        self.assertNotIn('password', response.data['data'])

    def test_create_user_duplicate_username_fails(self):
        """POST /api/users/ with duplicate username should fail."""
        data = {
            'username': 'apiuser1',
            'password': 'somepass',
            'nick_name': 'Duplicate User'
        }
        response = self.client.post('/api/users/', data, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_get_user_by_id(self):
        """GET /api/users/{id} should return user details."""
        response = self.client.get(f'/api/users/{self.user1.id}')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['data']['username'], 'apiuser1')

    def test_get_user_by_username(self):
        """GET /api/users/username/{username} should return user."""
        response = self.client.get('/api/users/username/apiuser1')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['data']['username'], 'apiuser1')

    def test_update_user(self):
        """PUT /api/users/{id} should update user."""
        data = {
            'nick_name': 'Updated Name',
            'phone': '1234567890'
        }
        response = self.client.put(
            f'/api/users/{self.user1.id}',
            data,
            format='json'
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['data']['nick_name'], 'Updated Name')
        self.assertEqual(response.data['data']['phone'], '1234567890')

    def test_update_user_password(self):
        """PUT /api/users/{id} with password should update password hash."""
        data = {'password': 'newpassword123'}
        response = self.client.put(
            f'/api/users/{self.user1.id}',
            data,
            format='json'
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.user1.refresh_from_db()
        self.assertTrue(self.user1.check_password('newpassword123'))

    def test_delete_user(self):
        """DELETE /api/users/{id} should soft delete user."""
        response = self.client.delete(f'/api/users/{self.user1.id}')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.user1.refresh_from_db()
        self.assertFalse(self.user1.is_active)

    def test_get_nonexistent_user_returns_404(self):
        """GET /api/users/{nonexistent_id} should return 404."""
        fake_id = '00000000-0000-0000-0000-000000000000'
        response = self.client.get(f'/api/users/{fake_id}')
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

    def test_user_fields_in_response(self):
        """Response should contain expected fields."""
        response = self.client.get(f'/api/users/{self.user1.id}')
        data = response.data['data']
        expected_fields = [
            'id', 'email', 'phone', 'nick_name', 'username',
            'role', 'source', 'is_active', 'language',
            'create_time', 'update_time'
        ]
        for field in expected_fields:
            self.assertIn(field, data)
