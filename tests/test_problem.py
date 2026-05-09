"""
Tests for Problem model and API endpoints.
"""
import uuid
from django.test import TestCase
from rest_framework.test import APITestCase
from rest_framework import status
from apps.knowledge.models import (
    Knowledge, Document, Paragraph, Problem, ProblemParagraphMapping,
    DocumentStatus
)
from apps.users.models import User


class ProblemModelTest(TestCase):
    """Test Problem model functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.user = User.objects.create(
            username="testuser",
            password="hashed",
            role="user"
        )
        self.knowledge = Knowledge.objects.create(
            name="Test KB",
            workspace_id="test-ws",
            user=self.user
        )
        self.document = Document.objects.create(
            knowledge=self.knowledge,
            name="Test Doc",
            status=DocumentStatus.SUCCESS
        )
        self.paragraph = Paragraph.objects.create(
            document=self.document,
            knowledge=self.knowledge,
            content="Test paragraph content",
            title="Test Title"
        )

    def test_create_problem(self):
        """Problem can be created."""
        problem = Problem.create_problem(
            knowledge=self.knowledge,
            content="What is Python?"
        )
        self.assertIsNotNone(problem.id)
        self.assertEqual(problem.content, "What is Python?")
        self.assertEqual(problem.knowledge, self.knowledge)

    def test_problem_has_uuid_id(self):
        """Problem has UUID primary key."""
        problem = Problem.create_problem(
            knowledge=self.knowledge,
            content="Test question"
        )
        self.assertIsInstance(problem.id, uuid.UUID)

    def test_problem_default_hit_num_is_zero(self):
        """Problem default hit_num is 0."""
        problem = Problem.create_problem(
            knowledge=self.knowledge,
            content="Test question"
        )
        self.assertEqual(problem.hit_num, 0)

    def test_problem_is_active_by_default(self):
        """Problem is active by default."""
        problem = Problem.create_problem(
            knowledge=self.knowledge,
            content="Test question"
        )
        self.assertTrue(problem.is_active)

    def test_problem_timestamps_auto_set(self):
        """Problem has auto-set timestamps."""
        problem = Problem.create_problem(
            knowledge=self.knowledge,
            content="Test question"
        )
        self.assertIsNotNone(problem.create_time)
        self.assertIsNotNone(problem.update_time)

    def test_problem_record_hit(self):
        """Problem hit count increments."""
        problem = Problem.create_problem(
            knowledge=self.knowledge,
            content="Test question"
        )
        self.assertEqual(problem.hit_num, 0)
        problem.record_hit()
        problem.refresh_from_db()
        self.assertEqual(problem.hit_num, 1)
        problem.record_hit()
        problem.refresh_from_db()
        self.assertEqual(problem.hit_num, 2)

    def test_problem_activate(self):
        """Problem can be activated."""
        problem = Problem.objects.create(
            knowledge=self.knowledge,
            content="Test",
            is_active=False
        )
        self.assertFalse(problem.is_active)
        problem.activate()
        problem.refresh_from_db()
        self.assertTrue(problem.is_active)

    def test_problem_deactivate(self):
        """Problem can be deactivated."""
        problem = Problem.create_problem(
            knowledge=self.knowledge,
            content="Test"
        )
        self.assertTrue(problem.is_active)
        problem.deactivate()
        problem.refresh_from_db()
        self.assertFalse(problem.is_active)

    def test_problem_str_representation(self):
        """Problem has string representation."""
        problem = Problem.create_problem(
            knowledge=self.knowledge,
            content="What is Python programming?"
        )
        self.assertEqual(str(problem), "What is Python programming?")

    def test_problem_ordering(self):
        """Problems are ordered by create_time descending."""
        p1 = Problem.create_problem(self.knowledge, "First")
        p2 = Problem.create_problem(self.knowledge, "Second")
        problems = list(Problem.objects.filter(knowledge=self.knowledge))
        self.assertEqual(problems[0], p2)
        self.assertEqual(problems[1], p1)


class ProblemParagraphMappingModelTest(TestCase):
    """Test ProblemParagraphMapping model functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.user = User.objects.create(
            username="testuser",
            password="hashed",
            role="user"
        )
        self.knowledge = Knowledge.objects.create(
            name="Test KB",
            workspace_id="test-ws",
            user=self.user
        )
        self.document = Document.objects.create(
            knowledge=self.knowledge,
            name="Test Doc"
        )
        self.paragraph1 = Paragraph.objects.create(
            document=self.document,
            knowledge=self.knowledge,
            content="Paragraph 1",
            title="P1"
        )
        self.paragraph2 = Paragraph.objects.create(
            document=self.document,
            knowledge=self.knowledge,
            content="Paragraph 2",
            title="P2"
        )
        self.problem = Problem.create_problem(
            knowledge=self.knowledge,
            content="Test question"
        )

    def test_create_mapping(self):
        """Mapping can be created."""
        mapping = ProblemParagraphMapping.create_mapping(
            problem=self.problem,
            paragraph=self.paragraph1
        )
        self.assertIsNotNone(mapping.id)
        self.assertEqual(mapping.problem, self.problem)
        self.assertEqual(mapping.paragraph, self.paragraph1)
        self.assertEqual(mapping.document, self.document)
        self.assertEqual(mapping.knowledge, self.knowledge)

    def test_mapping_has_uuid_id(self):
        """Mapping has UUID primary key."""
        mapping = ProblemParagraphMapping.create_mapping(
            problem=self.problem,
            paragraph=self.paragraph1
        )
        self.assertIsInstance(mapping.id, uuid.UUID)

    def test_get_paragraphs_for_problem(self):
        """Get paragraphs linked to a problem."""
        ProblemParagraphMapping.create_mapping(self.problem, self.paragraph1)
        ProblemParagraphMapping.create_mapping(self.problem, self.paragraph2)
        paragraphs = ProblemParagraphMapping.get_paragraphs_for_problem(
            str(self.problem.id)
        )
        self.assertEqual(len(paragraphs), 2)
        self.assertIn(self.paragraph1, paragraphs)
        self.assertIn(self.paragraph2, paragraphs)

    def test_get_problems_for_paragraph(self):
        """Get problems linked to a paragraph."""
        problem2 = Problem.create_problem(self.knowledge, "Another question")
        ProblemParagraphMapping.create_mapping(self.problem, self.paragraph1)
        ProblemParagraphMapping.create_mapping(problem2, self.paragraph1)
        problems = ProblemParagraphMapping.get_problems_for_paragraph(
            str(self.paragraph1.id)
        )
        self.assertEqual(len(problems), 2)
        self.assertIn(self.problem, problems)
        self.assertIn(problem2, problems)

    def test_delete_by_problem(self):
        """Mappings can be deleted by problem."""
        ProblemParagraphMapping.create_mapping(self.problem, self.paragraph1)
        ProblemParagraphMapping.create_mapping(self.problem, self.paragraph2)
        self.assertEqual(ProblemParagraphMapping.objects.count(), 2)
        count = ProblemParagraphMapping.delete_by_problem(str(self.problem.id))
        self.assertEqual(count, 2)
        self.assertEqual(ProblemParagraphMapping.objects.count(), 0)

    def test_delete_by_paragraph(self):
        """Mappings can be deleted by paragraph."""
        problem2 = Problem.create_problem(self.knowledge, "Another question")
        ProblemParagraphMapping.create_mapping(self.problem, self.paragraph1)
        ProblemParagraphMapping.create_mapping(problem2, self.paragraph1)
        self.assertEqual(ProblemParagraphMapping.objects.count(), 2)
        count = ProblemParagraphMapping.delete_by_paragraph(str(self.paragraph1.id))
        self.assertEqual(count, 2)
        self.assertEqual(ProblemParagraphMapping.objects.count(), 0)

    def test_unique_constraint(self):
        """Problem-paragraph mapping is unique."""
        ProblemParagraphMapping.create_mapping(self.problem, self.paragraph1)
        with self.assertRaises(Exception):
            ProblemParagraphMapping.create_mapping(self.problem, self.paragraph1)

    def test_cascade_delete_with_problem(self):
        """Mappings deleted when problem is deleted."""
        ProblemParagraphMapping.create_mapping(self.problem, self.paragraph1)
        self.assertEqual(ProblemParagraphMapping.objects.count(), 1)
        self.problem.delete()
        self.assertEqual(ProblemParagraphMapping.objects.count(), 0)

    def test_cascade_delete_with_paragraph(self):
        """Mappings deleted when paragraph is deleted."""
        ProblemParagraphMapping.create_mapping(self.problem, self.paragraph1)
        self.assertEqual(ProblemParagraphMapping.objects.count(), 1)
        self.paragraph1.delete()
        self.assertEqual(ProblemParagraphMapping.objects.count(), 0)


class ProblemAPITest(APITestCase):
    """Test Problem API endpoints."""

    def setUp(self):
        """Set up test fixtures."""
        self.user = User.objects.create(
            username="testuser",
            password="hashed",
            role="user"
        )
        self.knowledge = Knowledge.objects.create(
            name="Test KB",
            workspace_id="test-ws",
            user=self.user
        )
        self.document = Document.objects.create(
            knowledge=self.knowledge,
            name="Test Doc"
        )
        self.paragraph = Paragraph.objects.create(
            document=self.document,
            knowledge=self.knowledge,
            content="Test paragraph"
        )
        self.base_url = f'/api/workspace/test-ws/knowledge/{self.knowledge.id}/problem'

    def test_create_problem(self):
        """Create a problem via API."""
        data = {'content': 'What is Python?'}
        response = self.client.post(self.base_url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response.data['data']['content'], 'What is Python?')

    def test_create_problem_empty_content_fails(self):
        """Create problem with empty content fails."""
        data = {'content': ''}
        response = self.client.post(self.base_url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_list_problems(self):
        """List problems via API."""
        Problem.create_problem(self.knowledge, "Question 1")
        Problem.create_problem(self.knowledge, "Question 2")
        response = self.client.get(self.base_url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data['data']['items']), 2)

    def test_list_problems_pagination(self):
        """List problems with pagination."""
        for i in range(5):
            Problem.create_problem(self.knowledge, f"Question {i}")
        response = self.client.get(f'{self.base_url}?page=1&page_size=2')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data['data']['items']), 2)
        self.assertEqual(response.data['data']['total'], 5)

    def test_list_problems_with_content_filter(self):
        """List problems filtered by content."""
        Problem.create_problem(self.knowledge, "What is Python?")
        Problem.create_problem(self.knowledge, "What is Java?")
        Problem.create_problem(self.knowledge, "How to learn?")
        response = self.client.get(f'{self.base_url}?content=Python')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data['data']['items']), 1)

    def test_list_problems_with_active_filter(self):
        """List problems filtered by active status."""
        p1 = Problem.create_problem(self.knowledge, "Active")
        p2 = Problem.create_problem(self.knowledge, "Inactive")
        p2.deactivate()
        response = self.client.get(f'{self.base_url}?is_active=true')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data['data']['items']), 1)

    def test_get_problem_by_id(self):
        """Get problem by ID."""
        problem = Problem.create_problem(self.knowledge, "Test question")
        response = self.client.get(f'{self.base_url}/{problem.id}')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['data']['content'], 'Test question')

    def test_get_nonexistent_problem(self):
        """Get nonexistent problem returns 404."""
        fake_id = uuid.uuid4()
        response = self.client.get(f'{self.base_url}/{fake_id}')
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

    def test_update_problem(self):
        """Update problem via API."""
        problem = Problem.create_problem(self.knowledge, "Original")
        data = {'content': 'Updated question'}
        response = self.client.put(
            f'{self.base_url}/{problem.id}', data, format='json'
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['data']['content'], 'Updated question')

    def test_update_problem_deactivate(self):
        """Deactivate problem via API."""
        problem = Problem.create_problem(self.knowledge, "Active")
        data = {'is_active': False}
        response = self.client.put(
            f'{self.base_url}/{problem.id}', data, format='json'
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertFalse(response.data['data']['is_active'])

    def test_delete_problem(self):
        """Delete problem via API."""
        problem = Problem.create_problem(self.knowledge, "To delete")
        response = self.client.delete(f'{self.base_url}/{problem.id}')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(Problem.objects.count(), 0)

    def test_problem_fields_in_response(self):
        """Response includes expected fields."""
        problem = Problem.create_problem(self.knowledge, "Test")
        response = self.client.get(f'{self.base_url}/{problem.id}')
        data = response.data['data']
        self.assertIn('id', data)
        self.assertIn('content', data)
        self.assertIn('hit_num', data)
        self.assertIn('is_active', data)
        self.assertIn('knowledge', data)
        self.assertIn('knowledge_name', data)
        self.assertIn('paragraph_count', data)
        self.assertIn('create_time', data)
        self.assertIn('update_time', data)


class ProblemMappingAPITest(APITestCase):
    """Test Problem-Paragraph mapping API endpoints."""

    def setUp(self):
        """Set up test fixtures."""
        self.user = User.objects.create(
            username="testuser",
            password="hashed",
            role="user"
        )
        self.knowledge = Knowledge.objects.create(
            name="Test KB",
            workspace_id="test-ws",
            user=self.user
        )
        self.document = Document.objects.create(
            knowledge=self.knowledge,
            name="Test Doc"
        )
        self.paragraph1 = Paragraph.objects.create(
            document=self.document,
            knowledge=self.knowledge,
            content="Paragraph 1"
        )
        self.paragraph2 = Paragraph.objects.create(
            document=self.document,
            knowledge=self.knowledge,
            content="Paragraph 2"
        )
        self.problem = Problem.create_problem(
            self.knowledge, "Test question"
        )
        self.base_url = (
            f'/api/workspace/test-ws/knowledge/{self.knowledge.id}'
            f'/problem/{self.problem.id}/mapping'
        )

    def test_list_mappings(self):
        """List mappings for a problem."""
        ProblemParagraphMapping.create_mapping(self.problem, self.paragraph1)
        ProblemParagraphMapping.create_mapping(self.problem, self.paragraph2)
        response = self.client.get(self.base_url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data['data']), 2)

    def test_create_mapping(self):
        """Create a mapping via API."""
        data = {'paragraph_id': str(self.paragraph1.id)}
        response = self.client.post(self.base_url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(ProblemParagraphMapping.objects.count(), 1)

    def test_create_duplicate_mapping_fails(self):
        """Creating duplicate mapping fails."""
        ProblemParagraphMapping.create_mapping(self.problem, self.paragraph1)
        data = {'paragraph_id': str(self.paragraph1.id)}
        response = self.client.post(self.base_url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_create_mapping_without_paragraph_id_fails(self):
        """Creating mapping without paragraph_id fails."""
        data = {}
        response = self.client.post(self.base_url, data, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_delete_mapping(self):
        """Delete a mapping via API."""
        mapping = ProblemParagraphMapping.create_mapping(
            self.problem, self.paragraph1
        )
        response = self.client.delete(f'{self.base_url}/{mapping.id}')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(ProblemParagraphMapping.objects.count(), 0)

    def test_mapping_fields_in_response(self):
        """Response includes expected fields."""
        ProblemParagraphMapping.create_mapping(self.problem, self.paragraph1)
        response = self.client.get(self.base_url)
        data = response.data['data'][0]
        self.assertIn('id', data)
        self.assertIn('problem', data)
        self.assertIn('problem_content', data)
        self.assertIn('paragraph', data)
        self.assertIn('paragraph_title', data)
        self.assertIn('paragraph_content', data)
        self.assertIn('document', data)
        self.assertIn('knowledge', data)
