"""
Tests for Application model and API endpoints.
"""
import uuid
from django.test import TestCase
from rest_framework.test import APITestCase
from rest_framework import status
from apps.application.models import (
    Application, ApplicationKnowledgeMapping, ApplicationType
)
from apps.knowledge.models import Knowledge
from apps.users.models import User


class ApplicationModelTest(TestCase):
    """Test Application model functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.user = User.objects.create(
            username="testuser",
            password="hashed",
            role="user"
        )

    def test_create_application(self):
        """Application can be created."""
        app = Application.create_application(
            name="Test App",
            workspace_id="test-ws",
            description="Test description",
            user=self.user
        )
        self.assertIsNotNone(app.id)
        self.assertEqual(app.name, "Test App")
        self.assertEqual(app.description, "Test description")
        self.assertEqual(app.user, self.user)

    def test_application_has_uuid_id(self):
        """Application has UUID primary key."""
        app = Application.create_application(
            name="Test App",
            workspace_id="test-ws"
        )
        self.assertIsInstance(app.id, uuid.UUID)

    def test_application_default_type_is_simple(self):
        """Application default type is SIMPLE."""
        app = Application.create_application(
            name="Test App",
            workspace_id="test-ws"
        )
        self.assertEqual(app.type, ApplicationType.SIMPLE)

    def test_application_is_active_by_default(self):
        """Application is active by default."""
        app = Application.create_application(
            name="Test App",
            workspace_id="test-ws"
        )
        self.assertTrue(app.is_active)

    def test_application_not_published_by_default(self):
        """Application is not published by default."""
        app = Application.create_application(
            name="Test App",
            workspace_id="test-ws"
        )
        self.assertFalse(app.is_published)

    def test_application_default_dialogue_count_is_zero(self):
        """Application default dialogue_count is 0."""
        app = Application.create_application(
            name="Test App",
            workspace_id="test-ws"
        )
        self.assertEqual(app.dialogue_count, 0)

    def test_application_timestamps_auto_set(self):
        """Application has auto-set timestamps."""
        app = Application.create_application(
            name="Test App",
            workspace_id="test-ws"
        )
        self.assertIsNotNone(app.create_time)
        self.assertIsNotNone(app.update_time)

    def test_application_publish(self):
        """Application can be published."""
        app = Application.create_application(
            name="Test App",
            workspace_id="test-ws"
        )
        self.assertFalse(app.is_published)
        app.publish()
        app.refresh_from_db()
        self.assertTrue(app.is_published)

    def test_application_unpublish(self):
        """Application can be unpublished."""
        app = Application.objects.create(
            name="Test App",
            workspace_id="test-ws",
            is_published=True
        )
        self.assertTrue(app.is_published)
        app.unpublish()
        app.refresh_from_db()
        self.assertFalse(app.is_published)

    def test_application_activate(self):
        """Application can be activated."""
        app = Application.objects.create(
            name="Test App",
            workspace_id="test-ws",
            is_active=False
        )
        self.assertFalse(app.is_active)
        app.activate()
        app.refresh_from_db()
        self.assertTrue(app.is_active)

    def test_application_deactivate(self):
        """Application can be deactivated."""
        app = Application.create_application(
            name="Test App",
            workspace_id="test-ws"
        )
        self.assertTrue(app.is_active)
        app.deactivate()
        app.refresh_from_db()
        self.assertFalse(app.is_active)

    def test_application_increment_dialogue_count(self):
        """Application dialogue count increments."""
        app = Application.create_application(
            name="Test App",
            workspace_id="test-ws"
        )
        self.assertEqual(app.dialogue_count, 0)
        app.increment_dialogue_count()
        app.refresh_from_db()
        self.assertEqual(app.dialogue_count, 1)
        app.increment_dialogue_count()
        app.refresh_from_db()
        self.assertEqual(app.dialogue_count, 2)

    def test_application_str_representation(self):
        """Application has string representation."""
        app = Application.create_application(
            name="Test App",
            workspace_id="test-ws"
        )
        self.assertEqual(str(app), "Test App (Simple Chat)")

    def test_application_default_knowledge_setting(self):
        """Application has default knowledge settings."""
        settings = Application.get_default_knowledge_setting()
        self.assertIn('top_n', settings)
        self.assertIn('similarity', settings)
        self.assertIn('search_mode', settings)

    def test_application_default_system_prompt(self):
        """Application has default system prompt."""
        prompt = Application.get_default_system_prompt()
        self.assertIn('{context}', prompt)
        self.assertIn('{question}', prompt)


class ApplicationKnowledgeMappingModelTest(TestCase):
    """Test ApplicationKnowledgeMapping model functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.user = User.objects.create(
            username="testuser",
            password="hashed",
            role="user"
        )
        self.application = Application.create_application(
            name="Test App",
            workspace_id="test-ws",
            user=self.user
        )
        self.knowledge1 = Knowledge.objects.create(
            name="KB 1",
            workspace_id="test-ws",
            user=self.user
        )
        self.knowledge2 = Knowledge.objects.create(
            name="KB 2",
            workspace_id="test-ws",
            user=self.user
        )

    def test_create_mapping(self):
        """Mapping can be created."""
        mapping = ApplicationKnowledgeMapping.create_mapping(
            application=self.application,
            knowledge=self.knowledge1
        )
        self.assertIsNotNone(mapping.id)
        self.assertEqual(mapping.application, self.application)
        self.assertEqual(mapping.knowledge, self.knowledge1)

    def test_mapping_has_uuid_id(self):
        """Mapping has UUID primary key."""
        mapping = ApplicationKnowledgeMapping.create_mapping(
            application=self.application,
            knowledge=self.knowledge1
        )
        self.assertIsInstance(mapping.id, uuid.UUID)

    def test_get_knowledge_bases(self):
        """Get knowledge bases for an application."""
        ApplicationKnowledgeMapping.create_mapping(
            self.application, self.knowledge1
        )
        ApplicationKnowledgeMapping.create_mapping(
            self.application, self.knowledge2
        )
        kbs = ApplicationKnowledgeMapping.get_knowledge_bases(
            str(self.application.id)
        )
        self.assertEqual(len(kbs), 2)
        self.assertIn(self.knowledge1, kbs)
        self.assertIn(self.knowledge2, kbs)

    def test_get_applications(self):
        """Get applications using a knowledge base."""
        app2 = Application.create_application(
            name="App 2",
            workspace_id="test-ws"
        )
        ApplicationKnowledgeMapping.create_mapping(
            self.application, self.knowledge1
        )
        ApplicationKnowledgeMapping.create_mapping(
            app2, self.knowledge1
        )
        apps = ApplicationKnowledgeMapping.get_applications(
            str(self.knowledge1.id)
        )
        self.assertEqual(len(apps), 2)
        self.assertIn(self.application, apps)
        self.assertIn(app2, apps)

    def test_delete_by_application(self):
        """Mappings can be deleted by application."""
        ApplicationKnowledgeMapping.create_mapping(
            self.application, self.knowledge1
        )
        ApplicationKnowledgeMapping.create_mapping(
            self.application, self.knowledge2
        )
        self.assertEqual(ApplicationKnowledgeMapping.objects.count(), 2)
        count = ApplicationKnowledgeMapping.delete_by_application(
            str(self.application.id)
        )
        self.assertEqual(count, 2)
        self.assertEqual(ApplicationKnowledgeMapping.objects.count(), 0)

    def test_delete_by_knowledge(self):
        """Mappings can be deleted by knowledge base."""
        app2 = Application.create_application(
            name="App 2",
            workspace_id="test-ws"
        )
        ApplicationKnowledgeMapping.create_mapping(
            self.application, self.knowledge1
        )
        ApplicationKnowledgeMapping.create_mapping(
            app2, self.knowledge1
        )
        self.assertEqual(ApplicationKnowledgeMapping.objects.count(), 2)
        count = ApplicationKnowledgeMapping.delete_by_knowledge(
            str(self.knowledge1.id)
        )
        self.assertEqual(count, 2)
        self.assertEqual(ApplicationKnowledgeMapping.objects.count(), 0)

    def test_unique_constraint(self):
        """Application-knowledge mapping is unique."""
        ApplicationKnowledgeMapping.create_mapping(
            self.application, self.knowledge1
        )
        with self.assertRaises(Exception):
            ApplicationKnowledgeMapping.create_mapping(
                self.application, self.knowledge1
            )

    def test_cascade_delete_with_application(self):
        """Mappings deleted when application is deleted."""
        ApplicationKnowledgeMapping.create_mapping(
            self.application, self.knowledge1
        )
        self.assertEqual(ApplicationKnowledgeMapping.objects.count(), 1)
        self.application.delete()
        self.assertEqual(ApplicationKnowledgeMapping.objects.count(), 0)

    def test_cascade_delete_with_knowledge(self):
        """Mappings deleted when knowledge base is deleted."""
        ApplicationKnowledgeMapping.create_mapping(
            self.application, self.knowledge1
        )
        self.assertEqual(ApplicationKnowledgeMapping.objects.count(), 1)
        self.knowledge1.delete()
        self.assertEqual(ApplicationKnowledgeMapping.objects.count(), 0)


class ApplicationAPITest(APITestCase):
    """Test Application API endpoints."""

    def setUp(self):
        """Set up test fixtures."""
        self.user = User.objects.create(
            username="testuser",
            password="hashed",
            role="user"
        )
        self.base_url = '/api/workspace/test-ws/application'

    def test_create_application(self):
        """Create an application via API."""
        data = {
            'name': 'My App',
            'description': 'Test application'
        }
        response = self.client.post(self.base_url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response.data['data']['name'], 'My App')

    def test_create_application_empty_name_fails(self):
        """Create application with empty name fails."""
        data = {'name': ''}
        response = self.client.post(self.base_url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_list_applications(self):
        """List applications via API."""
        Application.create_application("App 1", "test-ws")
        Application.create_application("App 2", "test-ws")
        response = self.client.get(self.base_url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data['data']['items']), 2)

    def test_list_applications_pagination(self):
        """List applications with pagination."""
        for i in range(5):
            Application.create_application(f"App {i}", "test-ws")
        response = self.client.get(f'{self.base_url}?page=1&page_size=2')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data['data']['items']), 2)
        self.assertEqual(response.data['data']['total'], 5)

    def test_list_applications_with_type_filter(self):
        """List applications filtered by type."""
        Application.create_application("Simple", "test-ws", app_type="SIMPLE")
        Application.create_application("Workflow", "test-ws", app_type="WORKFLOW")
        response = self.client.get(f'{self.base_url}?type=SIMPLE')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data['data']['items']), 1)

    def test_list_applications_with_published_filter(self):
        """List applications filtered by published status."""
        app1 = Application.create_application("Published", "test-ws")
        app1.publish()
        Application.create_application("Draft", "test-ws")
        response = self.client.get(f'{self.base_url}?is_published=true')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data['data']['items']), 1)

    def test_list_applications_with_name_filter(self):
        """List applications filtered by name."""
        Application.create_application("Test App", "test-ws")
        Application.create_application("Demo App", "test-ws")
        response = self.client.get(f'{self.base_url}?name=Test')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data['data']['items']), 1)

    def test_get_application_by_id(self):
        """Get application by ID."""
        app = Application.create_application("Test App", "test-ws")
        response = self.client.get(f'{self.base_url}/{app.id}')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['data']['name'], 'Test App')

    def test_get_nonexistent_application(self):
        """Get nonexistent application returns 404."""
        fake_id = uuid.uuid4()
        response = self.client.get(f'{self.base_url}/{fake_id}')
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

    def test_update_application(self):
        """Update application via API."""
        app = Application.create_application("Original", "test-ws")
        data = {'name': 'Updated', 'description': 'New description'}
        response = self.client.put(
            f'{self.base_url}/{app.id}', data, format='json'
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['data']['name'], 'Updated')
        self.assertEqual(response.data['data']['description'], 'New description')

    def test_update_application_publish(self):
        """Publish application via API."""
        app = Application.create_application("App", "test-ws")
        data = {'is_published': True}
        response = self.client.put(
            f'{self.base_url}/{app.id}', data, format='json'
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertTrue(response.data['data']['is_published'])

    def test_update_application_deactivate(self):
        """Deactivate application via API."""
        app = Application.create_application("App", "test-ws")
        data = {'is_active': False}
        response = self.client.put(
            f'{self.base_url}/{app.id}', data, format='json'
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertFalse(response.data['data']['is_active'])

    def test_delete_application(self):
        """Delete application via API."""
        app = Application.create_application("To delete", "test-ws")
        response = self.client.delete(f'{self.base_url}/{app.id}')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(Application.objects.count(), 0)

    def test_application_fields_in_response(self):
        """Response includes expected fields."""
        app = Application.create_application("Test", "test-ws")
        response = self.client.get(f'{self.base_url}/{app.id}')
        data = response.data['data']
        self.assertIn('id', data)
        self.assertIn('name', data)
        self.assertIn('description', data)
        self.assertIn('type', data)
        self.assertIn('type_display', data)
        self.assertIn('prologue', data)
        self.assertIn('system_prompt', data)
        self.assertIn('is_published', data)
        self.assertIn('is_active', data)
        self.assertIn('dialogue_count', data)
        self.assertIn('knowledge_setting', data)
        self.assertIn('model_setting', data)
        self.assertIn('knowledge_count', data)
        self.assertIn('create_time', data)
        self.assertIn('update_time', data)


class ApplicationKnowledgeMappingAPITest(APITestCase):
    """Test Application-Knowledge mapping API endpoints."""

    def setUp(self):
        """Set up test fixtures."""
        self.user = User.objects.create(
            username="testuser",
            password="hashed",
            role="user"
        )
        self.application = Application.create_application(
            "Test App", "test-ws", user=self.user
        )
        self.knowledge1 = Knowledge.objects.create(
            name="KB 1",
            workspace_id="test-ws",
            user=self.user
        )
        self.knowledge2 = Knowledge.objects.create(
            name="KB 2",
            workspace_id="test-ws",
            user=self.user
        )
        self.base_url = (
            f'/api/workspace/test-ws/application/{self.application.id}/knowledge'
        )

    def test_list_mappings(self):
        """List mappings for an application."""
        ApplicationKnowledgeMapping.create_mapping(
            self.application, self.knowledge1
        )
        ApplicationKnowledgeMapping.create_mapping(
            self.application, self.knowledge2
        )
        response = self.client.get(self.base_url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data['data']), 2)

    def test_create_mapping(self):
        """Create a mapping via API."""
        data = {'knowledge_id': str(self.knowledge1.id)}
        response = self.client.post(self.base_url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(ApplicationKnowledgeMapping.objects.count(), 1)

    def test_create_duplicate_mapping_fails(self):
        """Creating duplicate mapping fails."""
        ApplicationKnowledgeMapping.create_mapping(
            self.application, self.knowledge1
        )
        data = {'knowledge_id': str(self.knowledge1.id)}
        response = self.client.post(self.base_url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_create_mapping_without_knowledge_id_fails(self):
        """Creating mapping without knowledge_id fails."""
        data = {}
        response = self.client.post(self.base_url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_delete_mapping(self):
        """Delete a mapping via API."""
        mapping = ApplicationKnowledgeMapping.create_mapping(
            self.application, self.knowledge1
        )
        response = self.client.delete(f'{self.base_url}/{mapping.id}')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(ApplicationKnowledgeMapping.objects.count(), 0)

    def test_mapping_fields_in_response(self):
        """Response includes expected fields."""
        ApplicationKnowledgeMapping.create_mapping(
            self.application, self.knowledge1
        )
        response = self.client.get(self.base_url)
        data = response.data['data'][0]
        self.assertIn('id', data)
        self.assertIn('application', data)
        self.assertIn('application_name', data)
        self.assertIn('knowledge', data)
        self.assertIn('knowledge_name', data)
        self.assertIn('create_time', data)
