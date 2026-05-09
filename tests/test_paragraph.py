"""
Tests for Paragraph model and API endpoints.
"""
from django.test import TestCase
from rest_framework.test import APITestCase
from rest_framework import status
from apps.knowledge.models import (
    Knowledge, Document, Paragraph, DocumentStatus
)
from apps.users.models import User


class ParagraphModelTest(TestCase):
    """Test Paragraph model functionality."""

    def setUp(self):
        """Create test user, knowledge base, and document."""
        self.user = User.create_user(
            username="parauser",
            password="testpass",
            nick_name="Para User"
        )
        self.knowledge = Knowledge.create_knowledge(
            name="Test KB",
            workspace_id="ws1",
            user=self.user
        )
        self.document = Document.create_document(
            knowledge=self.knowledge,
            name="Test Document",
            content="Test content"
        )

    def test_create_paragraph(self):
        """Paragraph can be created within a document."""
        para = Paragraph.create_paragraph(
            document=self.document,
            content="This is paragraph content.",
            title="Paragraph Title"
        )
        self.assertIsNotNone(para.id)
        self.assertEqual(para.content, "This is paragraph content.")
        self.assertEqual(para.title, "Paragraph Title")
        self.assertEqual(para.document, self.document)
        self.assertEqual(para.knowledge, self.knowledge)

    def test_paragraph_has_uuid_id(self):
        """Paragraph ID should be a UUID."""
        para = Paragraph.create_paragraph(
            document=self.document,
            content="UUID test content"
        )
        self.assertEqual(len(str(para.id)), 36)

    def test_paragraph_default_status_is_pending(self):
        """Default status should be PENDING."""
        para = Paragraph.create_paragraph(
            document=self.document,
            content="Status test"
        )
        self.assertEqual(para.status, DocumentStatus.PENDING)

    def test_paragraph_default_hit_num_is_zero(self):
        """Default hit count should be 0."""
        para = Paragraph.create_paragraph(
            document=self.document,
            content="Hit test"
        )
        self.assertEqual(para.hit_num, 0)

    def test_paragraph_record_hit(self):
        """record_hit should increment hit count."""
        para = Paragraph.create_paragraph(
            document=self.document,
            content="Record hit test"
        )
        para.record_hit()
        self.assertEqual(para.hit_num, 1)
        para.record_hit()
        self.assertEqual(para.hit_num, 2)

    def test_paragraph_is_active_by_default(self):
        """Paragraph should be active by default."""
        para = Paragraph.create_paragraph(
            document=self.document,
            content="Active test"
        )
        self.assertTrue(para.is_active)

    def test_paragraph_position(self):
        """Paragraph can have a position."""
        para1 = Paragraph.create_paragraph(
            document=self.document,
            content="First paragraph",
            position=0
        )
        para2 = Paragraph.create_paragraph(
            document=self.document,
            content="Second paragraph",
            position=1
        )
        self.assertEqual(para1.position, 0)
        self.assertEqual(para2.position, 1)

    def test_paragraph_ordering(self):
        """Paragraphs should be ordered by position."""
        para2 = Paragraph.create_paragraph(
            document=self.document,
            content="Second",
            position=1
        )
        para1 = Paragraph.create_paragraph(
            document=self.document,
            content="First",
            position=0
        )
        paragraphs = list(Paragraph.objects.filter(document=self.document))
        self.assertEqual(paragraphs[0].content, "First")
        self.assertEqual(paragraphs[1].content, "Second")

    def test_paragraph_update_status(self):
        """update_status should change status and add metadata."""
        para = Paragraph.create_paragraph(
            document=self.document,
            content="Update status test"
        )
        para.update_status(DocumentStatus.SUCCESS, {'processed_at': '2026-05-09'})
        self.assertEqual(para.status, DocumentStatus.SUCCESS)
        self.assertIn('processed_at', para.status_meta)

    def test_paragraph_activate(self):
        """activate should set is_active to True."""
        para = Paragraph.create_paragraph(
            document=self.document,
            content="Activate test"
        )
        para.is_active = False
        para.save()
        para.activate()
        self.assertTrue(para.is_active)

    def test_paragraph_deactivate(self):
        """deactivate should set is_active to False."""
        para = Paragraph.create_paragraph(
            document=self.document,
            content="Deactivate test"
        )
        para.deactivate()
        self.assertFalse(para.is_active)

    def test_paragraph_timestamps_auto_set(self):
        """create_time and update_time should be auto-set."""
        para = Paragraph.create_paragraph(
            document=self.document,
            content="Time test"
        )
        self.assertIsNotNone(para.create_time)
        self.assertIsNotNone(para.update_time)


class ParagraphAPITest(APITestCase):
    """Test Paragraph API endpoints."""

    def setUp(self):
        """Create test data."""
        self.user = User.create_user(
            username="apiparauser",
            password="testpass",
            nick_name="API Para User"
        )
        self.workspace_id = "test-workspace"
        self.knowledge = Knowledge.create_knowledge(
            name="Test KB",
            workspace_id=self.workspace_id,
            user=self.user
        )
        self.document = Document.create_document(
            knowledge=self.knowledge,
            name="Test Document"
        )
        self.para1 = Paragraph.create_paragraph(
            document=self.document,
            content="First paragraph content",
            title="Para 1",
            position=0
        )
        self.para2 = Paragraph.create_paragraph(
            document=self.document,
            content="Second paragraph content",
            title="Para 2",
            position=1
        )
        self.para2.update_status(DocumentStatus.SUCCESS)

    def test_list_paragraphs(self):
        """GET returns list of paragraphs."""
        url = f'/api/workspace/{self.workspace_id}/knowledge/{self.knowledge.id}/document/{self.document.id}/paragraph'
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['code'], 200)
        self.assertEqual(len(response.data['data']['items']), 2)

    def test_list_paragraphs_with_status_filter(self):
        """GET with status filter returns filtered results."""
        url = f'/api/workspace/{self.workspace_id}/knowledge/{self.knowledge.id}/document/{self.document.id}/paragraph'
        response = self.client.get(url, {'status': DocumentStatus.SUCCESS})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data['data']['items']), 1)
        self.assertEqual(
            response.data['data']['items'][0]['status'],
            DocumentStatus.SUCCESS
        )

    def test_list_paragraphs_with_active_filter(self):
        """GET with is_active filter returns filtered results."""
        self.para1.deactivate()
        url = f'/api/workspace/{self.workspace_id}/knowledge/{self.knowledge.id}/document/{self.document.id}/paragraph'
        response = self.client.get(url, {'is_active': 'true'})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data['data']['items']), 1)

    def test_list_paragraphs_pagination(self):
        """GET with pagination returns correct page."""
        url = f'/api/workspace/{self.workspace_id}/knowledge/{self.knowledge.id}/document/{self.document.id}/paragraph'
        response = self.client.get(url, {'page': 1, 'page_size': 1})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data['data']['items']), 1)
        self.assertEqual(response.data['data']['total'], 2)

    def test_create_paragraph(self):
        """POST creates new paragraph."""
        url = f'/api/workspace/{self.workspace_id}/knowledge/{self.knowledge.id}/document/{self.document.id}/paragraph'
        data = {
            'content': 'New paragraph content.',
            'title': 'New Paragraph',
            'position': 2
        }
        response = self.client.post(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response.data['code'], 201)
        self.assertEqual(response.data['data']['content'], 'New paragraph content.')
        self.assertEqual(response.data['data']['position'], 2)

    def test_create_paragraph_empty_content_fails(self):
        """POST with empty content returns error."""
        url = f'/api/workspace/{self.workspace_id}/knowledge/{self.knowledge.id}/document/{self.document.id}/paragraph'
        data = {
            'content': '',
            'title': 'Empty Para'
        }
        response = self.client.post(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_get_paragraph_by_id(self):
        """GET returns paragraph by ID."""
        url = f'/api/workspace/{self.workspace_id}/knowledge/{self.knowledge.id}/document/{self.document.id}/paragraph/{self.para1.id}'
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['data']['title'], 'Para 1')

    def test_update_paragraph(self):
        """PUT updates paragraph."""
        url = f'/api/workspace/{self.workspace_id}/knowledge/{self.knowledge.id}/document/{self.document.id}/paragraph/{self.para1.id}'
        data = {
            'content': 'Updated content',
            'title': 'Updated Title',
            'position': 5
        }
        response = self.client.put(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['data']['content'], 'Updated content')
        self.assertEqual(response.data['data']['title'], 'Updated Title')
        self.assertEqual(response.data['data']['position'], 5)

    def test_update_paragraph_status(self):
        """PUT can update paragraph status."""
        url = f'/api/workspace/{self.workspace_id}/knowledge/{self.knowledge.id}/document/{self.document.id}/paragraph/{self.para1.id}'
        data = {'status': DocumentStatus.STARTED}
        response = self.client.put(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['data']['status'], DocumentStatus.STARTED)

    def test_update_paragraph_deactivate(self):
        """PUT can deactivate paragraph."""
        url = f'/api/workspace/{self.workspace_id}/knowledge/{self.knowledge.id}/document/{self.document.id}/paragraph/{self.para1.id}'
        data = {'is_active': False}
        response = self.client.put(url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertFalse(response.data['data']['is_active'])

    def test_delete_paragraph(self):
        """DELETE removes paragraph."""
        url = f'/api/workspace/{self.workspace_id}/knowledge/{self.knowledge.id}/document/{self.document.id}/paragraph/{self.para1.id}'
        response = self.client.delete(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertFalse(Paragraph.objects.filter(id=self.para1.id).exists())

    def test_get_nonexistent_paragraph(self):
        """GET nonexistent paragraph returns 404."""
        fake_id = '00000000-0000-0000-0000-000000000000'
        url = f'/api/workspace/{self.workspace_id}/knowledge/{self.knowledge.id}/document/{self.document.id}/paragraph/{fake_id}'
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

    def test_paragraph_fields_in_response(self):
        """Response contains expected fields."""
        url = f'/api/workspace/{self.workspace_id}/knowledge/{self.knowledge.id}/document/{self.document.id}/paragraph/{self.para1.id}'
        response = self.client.get(url)
        data = response.data['data']
        expected_fields = [
            'id', 'document', 'document_name', 'knowledge', 'knowledge_name',
            'content', 'title', 'status', 'status_display', 'status_meta',
            'hit_num', 'is_active', 'position', 'create_time', 'update_time'
        ]
        for field in expected_fields:
            self.assertIn(field, data)
