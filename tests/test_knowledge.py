"""
Tests for Knowledge Base model and API endpoints.
"""
from django.test import TestCase
from rest_framework.test import APITestCase
from rest_framework import status
from apps.knowledge.models import (
    Knowledge, KnowledgeFolder, KnowledgeType, KnowledgeScope
)
from apps.users.models import User


class KnowledgeModelTest(TestCase):
    """Test Knowledge model functionality."""

    def setUp(self):
        """Create test user."""
        self.user = User.create_user(
            username="testuser",
            password="testpass",
            nick_name="Test User"
        )

    def test_create_knowledge_base(self):
        """Knowledge base can be created with basic fields."""
        kb = Knowledge.create_knowledge(
            name="My Knowledge Base",
            workspace_id="ws1",
            description="Test description",
            user=self.user
        )
        self.assertIsNotNone(kb.id)
        self.assertEqual(kb.name, "My Knowledge Base")
        self.assertEqual(kb.workspace_id, "ws1")
        self.assertEqual(kb.description, "Test description")

    def test_knowledge_has_uuid_id(self):
        """Knowledge base ID should be a UUID."""
        kb = Knowledge.create_knowledge(
            name="UUID Test",
            workspace_id="ws1"
        )
        self.assertEqual(len(str(kb.id)), 36)

    def test_knowledge_default_type_is_base(self):
        """Default type should be BASE."""
        kb = Knowledge.create_knowledge(
            name="Type Test",
            workspace_id="ws1"
        )
        self.assertEqual(kb.type, KnowledgeType.BASE)

    def test_knowledge_default_scope_is_workspace(self):
        """Default scope should be WORKSPACE."""
        kb = Knowledge.create_knowledge(
            name="Scope Test",
            workspace_id="ws1"
        )
        self.assertEqual(kb.scope, KnowledgeScope.WORKSPACE)

    def test_knowledge_types_exist(self):
        """Knowledge types should include BASE, WEB, WORKFLOW."""
        self.assertEqual(KnowledgeType.BASE, 0)
        self.assertEqual(KnowledgeType.WEB, 1)
        self.assertEqual(KnowledgeType.WORKFLOW, 2)

    def test_knowledge_scopes_exist(self):
        """Knowledge scopes should include SHARED, WORKSPACE."""
        self.assertEqual(KnowledgeScope.SHARED, 'SHARED')
        self.assertEqual(KnowledgeScope.WORKSPACE, 'WORKSPACE')

    def test_knowledge_default_limits(self):
        """Default file limits should be set."""
        kb = Knowledge.create_knowledge(
            name="Limits Test",
            workspace_id="ws1"
        )
        self.assertEqual(kb.file_size_limit, 100)
        self.assertEqual(kb.file_count_limit, 50)

    def test_knowledge_timestamps_auto_set(self):
        """create_time and update_time should be auto-set."""
        kb = Knowledge.create_knowledge(
            name="Time Test",
            workspace_id="ws1"
        )
        self.assertIsNotNone(kb.create_time)
        self.assertIsNotNone(kb.update_time)

    def test_knowledge_with_custom_type(self):
        """Knowledge base can be created with custom type."""
        kb = Knowledge.create_knowledge(
            name="Web KB",
            workspace_id="ws1",
            kb_type=KnowledgeType.WEB
        )
        self.assertEqual(kb.type, KnowledgeType.WEB)

    def test_knowledge_with_shared_scope(self):
        """Knowledge base can be created with SHARED scope."""
        kb = Knowledge.create_knowledge(
            name="Shared KB",
            workspace_id="ws1",
            scope=KnowledgeScope.SHARED
        )
        self.assertEqual(kb.scope, KnowledgeScope.SHARED)

    def test_knowledge_metadata(self):
        """Knowledge base can store metadata."""
        kb = Knowledge.create_knowledge(
            name="Meta KB",
            workspace_id="ws1",
            meta={"key": "value", "number": 42}
        )
        self.assertEqual(kb.meta, {"key": "value", "number": 42})


class KnowledgeFolderModelTest(TestCase):
    """Test KnowledgeFolder model functionality."""

    def setUp(self):
        """Create test user."""
        self.user = User.create_user(
            username="folderuser",
            password="testpass",
            nick_name="Folder User"
        )

    def test_create_folder(self):
        """Folder can be created with basic fields."""
        folder = KnowledgeFolder(
            name="Test Folder",
            workspace_id="ws1",
            user=self.user
        )
        folder.save()
        self.assertIsNotNone(folder.id)
        self.assertEqual(folder.name, "Test Folder")

    def test_folder_hierarchy(self):
        """Folders can have parent-child relationships."""
        parent = KnowledgeFolder(
            name="Parent Folder",
            workspace_id="ws1"
        )
        parent.save()

        child = KnowledgeFolder(
            name="Child Folder",
            workspace_id="ws1",
            parent=parent
        )
        child.save()

        self.assertEqual(child.parent, parent)
        self.assertIn(child, parent.children.all())

    def test_folder_get_ancestors(self):
        """get_ancestors returns all parent folders."""
        grandparent = KnowledgeFolder(name="Grandparent", workspace_id="ws1")
        grandparent.save()

        parent = KnowledgeFolder(
            name="Parent",
            workspace_id="ws1",
            parent=grandparent
        )
        parent.save()

        child = KnowledgeFolder(
            name="Child",
            workspace_id="ws1",
            parent=parent
        )
        child.save()

        ancestors = child.get_ancestors()
        self.assertEqual(len(ancestors), 2)
        self.assertIn(parent, ancestors)
        self.assertIn(grandparent, ancestors)

    def test_folder_get_descendants(self):
        """get_descendants returns all child folders."""
        root = KnowledgeFolder(name="Root", workspace_id="ws1")
        root.save()

        child1 = KnowledgeFolder(name="Child1", workspace_id="ws1", parent=root)
        child1.save()

        child2 = KnowledgeFolder(name="Child2", workspace_id="ws1", parent=root)
        child2.save()

        grandchild = KnowledgeFolder(
            name="Grandchild",
            workspace_id="ws1",
            parent=child1
        )
        grandchild.save()

        descendants = root.get_descendants()
        self.assertEqual(len(descendants), 3)
        self.assertIn(child1, descendants)
        self.assertIn(child2, descendants)
        self.assertIn(grandchild, descendants)

    def test_knowledge_in_folder(self):
        """Knowledge base can be associated with a folder."""
        folder = KnowledgeFolder(name="KB Folder", workspace_id="ws1")
        folder.save()

        kb = Knowledge.create_knowledge(
            name="Foldered KB",
            workspace_id="ws1",
            folder=folder
        )

        self.assertEqual(kb.folder, folder)


class KnowledgeAPITest(APITestCase):
    """Test Knowledge Base API endpoints."""

    def setUp(self):
        """Create test data."""
        self.user = User.create_user(
            username="apiuser",
            password="testpass",
            nick_name="API User"
        )
        self.workspace_id = "test-workspace"
        self.kb1 = Knowledge.create_knowledge(
            name="KB 1",
            workspace_id=self.workspace_id,
            description="First knowledge base",
            user=self.user
        )
        self.kb2 = Knowledge.create_knowledge(
            name="KB 2",
            workspace_id=self.workspace_id,
            description="Second knowledge base",
            kb_type=KnowledgeType.WEB
        )

    def test_list_knowledge_bases(self):
        """GET /api/workspace/{workspace_id}/knowledge returns list."""
        response = self.client.get(
            f'/api/workspace/{self.workspace_id}/knowledge'
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['code'], 200)
        self.assertEqual(len(response.data['data']['items']), 2)

    def test_list_knowledge_bases_with_type_filter(self):
        """GET with type filter returns filtered results."""
        response = self.client.get(
            f'/api/workspace/{self.workspace_id}/knowledge',
            {'type': KnowledgeType.WEB}
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data['data']['items']), 1)
        self.assertEqual(
            response.data['data']['items'][0]['type'], KnowledgeType.WEB
        )

    def test_list_knowledge_bases_pagination(self):
        """GET with pagination returns correct page."""
        response = self.client.get(
            f'/api/workspace/{self.workspace_id}/knowledge',
            {'page': 1, 'page_size': 1}
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data['data']['items']), 1)
        self.assertEqual(response.data['data']['total'], 2)

    def test_create_knowledge_base(self):
        """POST /api/workspace/{workspace_id}/knowledge creates new KB."""
        data = {
            'name': 'New KB',
            'description': 'A new knowledge base',
            'type': KnowledgeType.BASE,
            'scope': KnowledgeScope.WORKSPACE
        }
        response = self.client.post(
            f'/api/workspace/{self.workspace_id}/knowledge',
            data,
            format='json'
        )
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response.data['code'], 201)
        self.assertEqual(response.data['data']['name'], 'New KB')
        self.assertEqual(
            response.data['data']['workspace_id'], self.workspace_id
        )

    def test_get_knowledge_base_by_id(self):
        """GET /api/workspace/{workspace_id}/knowledge/{id} returns KB."""
        response = self.client.get(
            f'/api/workspace/{self.workspace_id}/knowledge/{self.kb1.id}'
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['data']['name'], 'KB 1')

    def test_update_knowledge_base(self):
        """PUT /api/workspace/{workspace_id}/knowledge/{id} updates KB."""
        data = {
            'name': 'Updated KB',
            'description': 'Updated description'
        }
        response = self.client.put(
            f'/api/workspace/{self.workspace_id}/knowledge/{self.kb1.id}',
            data,
            format='json'
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['data']['name'], 'Updated KB')
        self.assertEqual(
            response.data['data']['description'], 'Updated description'
        )

    def test_delete_knowledge_base(self):
        """DELETE /api/workspace/{workspace_id}/knowledge/{id} deletes KB."""
        response = self.client.delete(
            f'/api/workspace/{self.workspace_id}/knowledge/{self.kb1.id}'
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertFalse(Knowledge.objects.filter(id=self.kb1.id).exists())

    def test_get_nonexistent_knowledge_base(self):
        """GET nonexistent KB returns 404."""
        fake_id = '00000000-0000-0000-0000-000000000000'
        response = self.client.get(
            f'/api/workspace/{self.workspace_id}/knowledge/{fake_id}'
        )
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

    def test_knowledge_base_fields_in_response(self):
        """Response contains expected fields."""
        response = self.client.get(
            f'/api/workspace/{self.workspace_id}/knowledge/{self.kb1.id}'
        )
        data = response.data['data']
        expected_fields = [
            'id', 'name', 'workspace_id', 'description', 'type',
            'type_display', 'scope', 'scope_display', 'file_size_limit',
            'file_count_limit', 'meta', 'create_time', 'update_time'
        ]
        for field in expected_fields:
            self.assertIn(field, data)


class KnowledgeFolderAPITest(APITestCase):
    """Test Knowledge Folder API endpoints."""

    def setUp(self):
        """Create test data."""
        self.workspace_id = "test-workspace"
        self.folder1 = KnowledgeFolder(
            name="Folder 1",
            workspace_id=self.workspace_id
        )
        self.folder1.save()

    def test_list_folders(self):
        """GET /api/workspace/{workspace_id}/folders returns list."""
        response = self.client.get(
            f'/api/workspace/{self.workspace_id}/folders'
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['code'], 200)
        self.assertEqual(len(response.data['data']), 1)

    def test_create_folder(self):
        """POST /api/workspace/{workspace_id}/folders creates folder."""
        data = {
            'name': 'New Folder',
            'description': 'A new folder'
        }
        response = self.client.post(
            f'/api/workspace/{self.workspace_id}/folders',
            data,
            format='json'
        )
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response.data['data']['name'], 'New Folder')

    def test_create_nested_folder(self):
        """POST with parent creates nested folder."""
        data = {
            'name': 'Child Folder',
            'parent': self.folder1.id
        }
        response = self.client.post(
            f'/api/workspace/{self.workspace_id}/folders',
            data,
            format='json'
        )
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response.data['data']['parent'], self.folder1.id)

    def test_get_folder_by_id(self):
        """GET /api/workspace/{workspace_id}/folders/{id} returns folder."""
        response = self.client.get(
            f'/api/workspace/{self.workspace_id}/folders/{self.folder1.id}'
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['data']['name'], 'Folder 1')

    def test_update_folder(self):
        """PUT /api/workspace/{workspace_id}/folders/{id} updates folder."""
        data = {'name': 'Updated Folder'}
        response = self.client.put(
            f'/api/workspace/{self.workspace_id}/folders/{self.folder1.id}',
            data,
            format='json'
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['data']['name'], 'Updated Folder')

    def test_delete_folder(self):
        """DELETE /api/workspace/{workspace_id}/folders/{id} deletes folder."""
        response = self.client.delete(
            f'/api/workspace/{self.workspace_id}/folders/{self.folder1.id}'
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertFalse(
            KnowledgeFolder.objects.filter(id=self.folder1.id).exists()
        )

    def test_filter_folders_by_parent(self):
        """GET with parent_id filters to children."""
        child = KnowledgeFolder(
            name="Child",
            workspace_id=self.workspace_id,
            parent=self.folder1
        )
        child.save()

        response = self.client.get(
            f'/api/workspace/{self.workspace_id}/folders',
            {'parent_id': self.folder1.id}
        )
        self.assertEqual(len(response.data['data']), 1)
        self.assertEqual(response.data['data'][0]['name'], 'Child')
