"""
Tests for Document model and API endpoints.
"""
from django.test import TestCase
from rest_framework.test import APITestCase
from rest_framework import status
from apps.knowledge.models import (
    Knowledge, Document, DocumentStatus, HitHandlingMethod, KnowledgeType
)
from apps.users.models import User


class DocumentModelTest(TestCase):
    """Test Document model functionality."""

    def setUp(self):
        """Create test user and knowledge base."""
        self.user = User.create_user(
            username="docuser",
            password="testpass",
            nick_name="Doc User"
        )
        self.knowledge = Knowledge.create_knowledge(
            name="Test KB",
            workspace_id="ws1",
            user=self.user
        )

    def test_create_document(self):
        """Document can be created within a knowledge base."""
        doc = Document.create_document(
            knowledge=self.knowledge,
            name="Test Document",
            content="This is test content."
        )
        self.assertIsNotNone(doc.id)
        self.assertEqual(doc.name, "Test Document")
        self.assertEqual(doc.knowledge, self.knowledge)

    def test_document_has_uuid_id(self):
        """Document ID should be a UUID."""
        doc = Document.create_document(
            knowledge=self.knowledge,
            name="UUID Test"
        )
        self.assertEqual(len(str(doc.id)), 36)

    def test_document_tracks_char_length(self):
        """Document should track content character length."""
        content = "Hello World"
        doc = Document.create_document(
            knowledge=self.knowledge,
            name="Length Test",
            content=content
        )
        self.assertEqual(doc.char_length, len(content))

    def test_document_default_status_is_pending(self):
        """Default status should be PENDING."""
        doc = Document.create_document(
            knowledge=self.knowledge,
            name="Status Test"
        )
        self.assertEqual(doc.status, DocumentStatus.PENDING)

    def test_document_status_values_exist(self):
        """Document status values should exist."""
        self.assertEqual(DocumentStatus.PENDING, '0')
        self.assertEqual(DocumentStatus.STARTED, '1')
        self.assertEqual(DocumentStatus.SUCCESS, '2')
        self.assertEqual(DocumentStatus.FAILURE, '3')
        self.assertEqual(DocumentStatus.REVOKE, '4')
        self.assertEqual(DocumentStatus.REVOKED, '5')
        self.assertEqual(DocumentStatus.IGNORED, 'n')

    def test_hit_handling_methods_exist(self):
        """Hit handling methods should exist."""
        self.assertEqual(HitHandlingMethod.OPTIMIZATION, 'optimization')
        self.assertEqual(HitHandlingMethod.DIRECTLY_RETURN, 'directly_return')

    def test_document_default_hit_handling(self):
        """Default hit handling method should be optimization."""
        doc = Document.create_document(
            knowledge=self.knowledge,
            name="Hit Test"
        )
        self.assertEqual(doc.hit_handling_method, HitHandlingMethod.OPTIMIZATION)

    def test_document_is_active_by_default(self):
        """Document should be active by default."""
        doc = Document.create_document(
            knowledge=self.knowledge,
            name="Active Test"
        )
        self.assertTrue(doc.is_active)

    def test_document_update_status(self):
        """update_status should change status and add metadata."""
        doc = Document.create_document(
            knowledge=self.knowledge,
            name="Update Test"
        )
        doc.update_status(DocumentStatus.SUCCESS, {'completed_at': '2026-05-09'})
        self.assertEqual(doc.status, DocumentStatus.SUCCESS)
        self.assertIn('completed_at', doc.status_meta)

    def test_document_activate(self):
        """activate should set is_active to True."""
        doc = Document.create_document(
            knowledge=self.knowledge,
            name="Activate Test"
        )
        doc.is_active = False
        doc.save()
        doc.activate()
        self.assertTrue(doc.is_active)

    def test_document_deactivate(self):
        """deactivate should set is_active to False."""
        doc = Document.create_document(
            knowledge=self.knowledge,
            name="Deactivate Test"
        )
        doc.deactivate()
        self.assertFalse(doc.is_active)

    def test_document_timestamps_auto_set(self):
        """create_time and update_time should be auto-set."""
        doc = Document.create_document(
            knowledge=self.knowledge,
            name="Time Test"
        )
        self.assertIsNotNone(doc.create_time)
        self.assertIsNotNone(doc.update_time)

    def test_document_with_custom_similarity(self):
        """Document can have custom similarity threshold."""
        doc = Document.create_document(
            knowledge=self.knowledge,
            name="Similarity Test",
            directly_return_similarity=0.85
        )
        self.assertEqual(doc.directly_return_similarity, 0.85)

    def test_document_metadata(self):
        """Document can store metadata."""
        doc = Document.create_document(
            knowledge=self.knowledge,
            name="Meta Test",
            meta={"source": "upload", "pages": 10}
        )
        self.assertEqual(doc.meta, {"source": "upload", "pages": 10})


class DocumentAPITest(APITestCase):
    """Test Document API endpoints."""

    def setUp(self):
        """Create test data."""
        self.user = User.create_user(
            username="apidocuser",
            password="testpass",
            nick_name="API Doc User"
        )
        self.workspace_id = "test-workspace"
        self.knowledge = Knowledge.create_knowledge(
            name="Test KB",
            workspace_id=self.workspace_id,
            user=self.user
        )
        self.doc1 = Document.create_document(
            knowledge=self.knowledge,
            name="Document 1",
            content="Content for document 1"
        )
        self.doc2 = Document.create_document(
            knowledge=self.knowledge,
            name="Document 2",
            content="Content for document 2"
        )
        self.doc2.update_status(DocumentStatus.SUCCESS)

    def test_list_documents(self):
        """GET returns list of documents."""
        url = f'/api/workspace/{self.workspace_id}/knowledge/{self.knowledge.id}/document'
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['code'], 200)
        self.assertEqual(len(response.data['data']['items']), 2)

    def test_list_documents_with_status_filter(self):
        """GET with status filter returns filtered results."""
        url = f'/api/workspace/{self.workspace_id}/knowledge/{self.knowledge.id}/document'
        response = self.client.get(url, {'status': DocumentStatus.SUCCESS})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data['data']['items']), 1)
        self.assertEqual(
            response.data['data']['items'][0]['status'],
            DocumentStatus.SUCCESS
        )

    def test_list_documents_with_active_filter(self):
        """GET with is_active filter returns filtered results."""
        self.doc1.deactivate()
        url = f'/api/workspace/{self.workspace_id}/knowledge/{self.knowledge.id}/document'
        response = self.client.get(url, {'is_active': 'true'})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data['data']['items']), 1)

    def test_list_documents_pagination(self):
        """GET with pagination returns correct page."""
        url = f'/api/workspace/{self.workspace_id}/knowledge/{self.knowledge.id}/document'
        response = self.client.get(url, {'page': 1, 'page_size': 1})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data['data']['items']), 1)
        self.assertEqual(response.data['data']['total'], 2)

    def test_create_document(self):
        """POST creates new document."""
        url = f'/api/workspace/{self.workspace_id}/knowledge/{self.knowledge.id}/document'
        data = {
            'name': 'New Document',
            'content': 'This is new content.',
            'hit_handling_method': HitHandlingMethod.DIRECTLY_RETURN
        }
        response = self.client.post(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response.data['code'], 201)
        self.assertEqual(response.data['data']['name'], 'New Document')
        self.assertEqual(response.data['data']['char_length'], len('This is new content.'))

    def test_get_document_by_id(self):
        """GET returns document by ID."""
        url = f'/api/workspace/{self.workspace_id}/knowledge/{self.knowledge.id}/document/{self.doc1.id}'
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['data']['name'], 'Document 1')

    def test_update_document(self):
        """PUT updates document."""
        url = f'/api/workspace/{self.workspace_id}/knowledge/{self.knowledge.id}/document/{self.doc1.id}'
        data = {
            'name': 'Updated Document',
            'status': DocumentStatus.STARTED
        }
        response = self.client.put(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['data']['name'], 'Updated Document')
        self.assertEqual(response.data['data']['status'], DocumentStatus.STARTED)

    def test_update_document_deactivate(self):
        """PUT can deactivate document."""
        url = f'/api/workspace/{self.workspace_id}/knowledge/{self.knowledge.id}/document/{self.doc1.id}'
        data = {'is_active': False}
        response = self.client.put(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertFalse(response.data['data']['is_active'])

    def test_delete_document(self):
        """DELETE removes document."""
        url = f'/api/workspace/{self.workspace_id}/knowledge/{self.knowledge.id}/document/{self.doc1.id}'
        response = self.client.delete(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertFalse(Document.objects.filter(id=self.doc1.id).exists())

    def test_get_nonexistent_document(self):
        """GET nonexistent document returns 404."""
        fake_id = '00000000-0000-0000-0000-000000000000'
        url = f'/api/workspace/{self.workspace_id}/knowledge/{self.knowledge.id}/document/{fake_id}'
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

    def test_document_fields_in_response(self):
        """Response contains expected fields."""
        url = f'/api/workspace/{self.workspace_id}/knowledge/{self.knowledge.id}/document/{self.doc1.id}'
        response = self.client.get(url)
        data = response.data['data']
        expected_fields = [
            'id', 'knowledge', 'knowledge_name', 'name', 'char_length',
            'status', 'status_display', 'status_meta', 'is_active',
            'type', 'type_display', 'hit_handling_method', 'hit_handling_display',
            'directly_return_similarity', 'meta', 'create_time', 'update_time'
        ]
        for field in expected_fields:
            self.assertIn(field, data)

    def test_create_document_invalid_hit_handling(self):
        """POST with invalid hit_handling_method returns error."""
        url = f'/api/workspace/{self.workspace_id}/knowledge/{self.knowledge.id}/document'
        data = {
            'name': 'Invalid Doc',
            'hit_handling_method': 'invalid_method'
        }
        response = self.client.post(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
