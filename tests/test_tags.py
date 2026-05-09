"""
Tests for Tag models and API endpoints.
"""
import pytest
from django.test import TestCase
from rest_framework.test import APIClient

from apps.users.models import User
from apps.knowledge.models import (
    Knowledge, Document, Tag, DocumentTag
)


class TestTagModel(TestCase):
    """Tests for Tag model."""

    def setUp(self):
        self.user = User.objects.create(
            username='taguser',
            password='hashedpassword',
            role='USER'
        )
        self.knowledge = Knowledge.create_knowledge(
            name='Tag Test KB',
            workspace_id='ws-tag',
            user=self.user
        )

    def test_create_tag(self):
        """Test creating a tag."""
        tag = Tag.create_tag(
            knowledge=self.knowledge,
            key='category',
            value='technical'
        )
        assert tag.id is not None
        assert tag.knowledge == self.knowledge
        assert tag.key == 'category'
        assert tag.value == 'technical'

    def test_tag_str(self):
        """Test tag string representation."""
        tag = Tag.create_tag(
            knowledge=self.knowledge,
            key='type',
            value='document'
        )
        assert str(tag) == 'type=document'

    def test_get_or_create_tag_creates_new(self):
        """Test get_or_create creates a new tag."""
        tag, created = Tag.get_or_create_tag(
            knowledge=self.knowledge,
            key='status',
            value='active'
        )
        assert created is True
        assert tag.key == 'status'

    def test_get_or_create_tag_returns_existing(self):
        """Test get_or_create returns existing tag."""
        Tag.create_tag(
            knowledge=self.knowledge,
            key='status',
            value='active'
        )
        tag, created = Tag.get_or_create_tag(
            knowledge=self.knowledge,
            key='status',
            value='active'
        )
        assert created is False
        assert tag.key == 'status'

    def test_unique_key_value_per_knowledge(self):
        """Test unique constraint on key-value per knowledge base."""
        Tag.create_tag(
            knowledge=self.knowledge,
            key='category',
            value='tech'
        )
        with pytest.raises(Exception):
            Tag.create_tag(
                knowledge=self.knowledge,
                key='category',
                value='tech'
            )

    def test_same_key_value_different_knowledge(self):
        """Test same key-value allowed in different knowledge bases."""
        kb2 = Knowledge.create_knowledge(
            name='KB2',
            workspace_id='ws-tag',
            user=self.user
        )
        tag1 = Tag.create_tag(
            knowledge=self.knowledge,
            key='category',
            value='tech'
        )
        tag2 = Tag.create_tag(
            knowledge=kb2,
            key='category',
            value='tech'
        )
        assert tag1.id != tag2.id

    def test_get_tags_for_knowledge(self):
        """Test getting all tags for a knowledge base."""
        Tag.create_tag(self.knowledge, 'key1', 'value1')
        Tag.create_tag(self.knowledge, 'key2', 'value2')
        tags = Tag.get_tags_for_knowledge(str(self.knowledge.id))
        assert len(tags) == 2

    def test_get_unique_keys(self):
        """Test getting unique tag keys."""
        Tag.create_tag(self.knowledge, 'category', 'tech')
        Tag.create_tag(self.knowledge, 'category', 'business')
        Tag.create_tag(self.knowledge, 'status', 'active')
        keys = Tag.get_unique_keys(str(self.knowledge.id))
        assert len(keys) == 2
        assert 'category' in keys
        assert 'status' in keys

    def test_get_values_for_key(self):
        """Test getting values for a specific key."""
        Tag.create_tag(self.knowledge, 'category', 'tech')
        Tag.create_tag(self.knowledge, 'category', 'business')
        Tag.create_tag(self.knowledge, 'status', 'active')
        values = Tag.get_values_for_key(str(self.knowledge.id), 'category')
        assert len(values) == 2
        assert 'tech' in values
        assert 'business' in values


class TestDocumentTagModel(TestCase):
    """Tests for DocumentTag model."""

    def setUp(self):
        self.user = User.objects.create(
            username='doctaguser',
            password='hashedpassword',
            role='USER'
        )
        self.knowledge = Knowledge.create_knowledge(
            name='DocTag Test KB',
            workspace_id='ws-doctag',
            user=self.user
        )
        self.document = Document.create_document(
            knowledge=self.knowledge,
            name='Test Document'
        )
        self.tag1 = Tag.create_tag(self.knowledge, 'category', 'tech')
        self.tag2 = Tag.create_tag(self.knowledge, 'status', 'active')

    def test_create_mapping(self):
        """Test creating a document-tag mapping."""
        mapping = DocumentTag.create_mapping(self.document, self.tag1)
        assert mapping.id is not None
        assert mapping.document == self.document
        assert mapping.tag == self.tag1

    def test_add_tags_to_document(self):
        """Test adding multiple tags to a document."""
        mappings = DocumentTag.add_tags_to_document(
            self.document, [self.tag1, self.tag2]
        )
        assert len(mappings) == 2

    def test_add_duplicate_tags_ignored(self):
        """Test that adding duplicate tags is idempotent."""
        DocumentTag.add_tags_to_document(self.document, [self.tag1])
        mappings = DocumentTag.add_tags_to_document(self.document, [self.tag1])
        assert len(mappings) == 0
        total = DocumentTag.objects.filter(document=self.document).count()
        assert total == 1

    def test_remove_tags_from_document(self):
        """Test removing tags from a document."""
        DocumentTag.add_tags_to_document(
            self.document, [self.tag1, self.tag2]
        )
        count = DocumentTag.remove_tags_from_document(
            self.document, [self.tag1]
        )
        assert count == 1
        tags = DocumentTag.get_tags_for_document(str(self.document.id))
        assert len(tags) == 1
        assert tags[0] == self.tag2

    def test_get_tags_for_document(self):
        """Test getting tags for a document."""
        DocumentTag.add_tags_to_document(
            self.document, [self.tag1, self.tag2]
        )
        tags = DocumentTag.get_tags_for_document(str(self.document.id))
        assert len(tags) == 2

    def test_get_documents_for_tag(self):
        """Test getting documents with a specific tag."""
        doc2 = Document.create_document(
            knowledge=self.knowledge,
            name='Document 2'
        )
        DocumentTag.add_tags_to_document(self.document, [self.tag1])
        DocumentTag.add_tags_to_document(doc2, [self.tag1])
        documents = DocumentTag.get_documents_for_tag(str(self.tag1.id))
        assert len(documents) == 2

    def test_get_documents_by_tags_single_filter(self):
        """Test filtering documents by a single tag."""
        doc2 = Document.create_document(
            knowledge=self.knowledge,
            name='Document 2'
        )
        DocumentTag.add_tags_to_document(self.document, [self.tag1])
        DocumentTag.add_tags_to_document(doc2, [self.tag2])

        documents = DocumentTag.get_documents_by_tags(
            str(self.knowledge.id),
            [{'key': 'category', 'value': 'tech'}]
        )
        assert len(documents) == 1
        assert documents[0] == self.document

    def test_get_documents_by_tags_multiple_filters(self):
        """Test filtering documents by multiple tags (AND logic)."""
        doc2 = Document.create_document(
            knowledge=self.knowledge,
            name='Document 2'
        )
        DocumentTag.add_tags_to_document(self.document, [self.tag1, self.tag2])
        DocumentTag.add_tags_to_document(doc2, [self.tag1])

        documents = DocumentTag.get_documents_by_tags(
            str(self.knowledge.id),
            [
                {'key': 'category', 'value': 'tech'},
                {'key': 'status', 'value': 'active'}
            ]
        )
        assert len(documents) == 1
        assert documents[0] == self.document

    def test_cascade_delete_tag_removes_mappings(self):
        """Test that deleting a tag removes its mappings."""
        DocumentTag.add_tags_to_document(self.document, [self.tag1])
        assert DocumentTag.objects.filter(tag=self.tag1).count() == 1
        self.tag1.delete()
        assert DocumentTag.objects.filter(tag_id=self.tag1.id).count() == 0

    def test_cascade_delete_document_removes_mappings(self):
        """Test that deleting a document removes its tag mappings."""
        DocumentTag.add_tags_to_document(self.document, [self.tag1, self.tag2])
        doc_id = self.document.id
        self.document.delete()
        assert DocumentTag.objects.filter(document_id=doc_id).count() == 0


class TestTagAPI(TestCase):
    """Tests for Tag API endpoints."""

    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create(
            username='tagapiuser',
            password='hashedpassword',
            role='USER'
        )
        self.knowledge = Knowledge.create_knowledge(
            name='API Tag KB',
            workspace_id='ws-tagapi',
            user=self.user
        )
        self.base_url = f'/api/workspace/ws-tagapi/knowledge/{self.knowledge.id}/tag'

    def test_create_tag(self):
        """Test creating a tag via API."""
        response = self.client.post(
            self.base_url,
            {'key': 'category', 'value': 'technical'},
            format='json'
        )
        assert response.status_code == 201
        assert response.data['data']['key'] == 'category'
        assert response.data['data']['value'] == 'technical'

    def test_create_duplicate_tag_returns_existing(self):
        """Test creating duplicate tag returns existing."""
        self.client.post(
            self.base_url,
            {'key': 'category', 'value': 'tech'},
            format='json'
        )
        response = self.client.post(
            self.base_url,
            {'key': 'category', 'value': 'tech'},
            format='json'
        )
        assert response.status_code == 200
        assert 'already exists' in response.data['message']

    def test_list_tags(self):
        """Test listing tags."""
        Tag.create_tag(self.knowledge, 'key1', 'val1')
        Tag.create_tag(self.knowledge, 'key2', 'val2')
        response = self.client.get(self.base_url)
        assert response.status_code == 200
        assert response.data['data']['total'] == 2

    def test_filter_tags_by_key(self):
        """Test filtering tags by key."""
        Tag.create_tag(self.knowledge, 'category', 'tech')
        Tag.create_tag(self.knowledge, 'category', 'business')
        Tag.create_tag(self.knowledge, 'status', 'active')
        response = self.client.get(f'{self.base_url}?key=category')
        assert response.data['data']['total'] == 2

    def test_get_tag(self):
        """Test getting a specific tag."""
        tag = Tag.create_tag(self.knowledge, 'test', 'value')
        response = self.client.get(f'{self.base_url}/{tag.id}')
        assert response.status_code == 200
        assert response.data['data']['key'] == 'test'

    def test_update_tag(self):
        """Test updating a tag."""
        tag = Tag.create_tag(self.knowledge, 'old', 'value')
        response = self.client.put(
            f'{self.base_url}/{tag.id}',
            {'key': 'new'},
            format='json'
        )
        assert response.status_code == 200
        assert response.data['data']['key'] == 'new'

    def test_delete_tag(self):
        """Test deleting a tag."""
        tag = Tag.create_tag(self.knowledge, 'delete', 'me')
        tag_id = tag.id
        response = self.client.delete(f'{self.base_url}/{tag.id}')
        assert response.status_code == 200
        assert not Tag.objects.filter(id=tag_id).exists()

    def test_get_unique_keys(self):
        """Test getting unique tag keys via API."""
        Tag.create_tag(self.knowledge, 'category', 'tech')
        Tag.create_tag(self.knowledge, 'category', 'business')
        Tag.create_tag(self.knowledge, 'status', 'active')
        response = self.client.get(f'{self.base_url}/keys')
        assert response.status_code == 200
        assert len(response.data['data']) == 2

    def test_get_values_for_key(self):
        """Test getting values for a key via API."""
        Tag.create_tag(self.knowledge, 'category', 'tech')
        Tag.create_tag(self.knowledge, 'category', 'business')
        response = self.client.get(f'{self.base_url}/key/category/values')
        assert response.status_code == 200
        assert len(response.data['data']) == 2


class TestDocumentTagAPI(TestCase):
    """Tests for DocumentTag API endpoints."""

    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create(
            username='doctagapiuser',
            password='hashedpassword',
            role='USER'
        )
        self.knowledge = Knowledge.create_knowledge(
            name='API DocTag KB',
            workspace_id='ws-doctagapi',
            user=self.user
        )
        self.document = Document.create_document(
            knowledge=self.knowledge,
            name='API Test Doc'
        )
        self.tag1 = Tag.create_tag(self.knowledge, 'category', 'tech')
        self.tag2 = Tag.create_tag(self.knowledge, 'status', 'active')
        self.base_url = (
            f'/api/workspace/ws-doctagapi/knowledge/{self.knowledge.id}'
            f'/document/{self.document.id}/tag'
        )

    def test_add_tags_by_id(self):
        """Test adding tags to document by tag IDs."""
        response = self.client.post(
            self.base_url,
            {'tag_ids': [str(self.tag1.id), str(self.tag2.id)]},
            format='json'
        )
        assert response.status_code == 201
        assert '2 tags added' in response.data['message']

    def test_add_tags_by_key_value(self):
        """Test adding tags by key-value pairs (creates if needed)."""
        response = self.client.post(
            self.base_url,
            {'tags': [{'key': 'new', 'value': 'tag'}]},
            format='json'
        )
        assert response.status_code == 201
        assert Tag.objects.filter(key='new', value='tag').exists()

    def test_get_document_tags(self):
        """Test getting tags for a document."""
        DocumentTag.add_tags_to_document(
            self.document, [self.tag1, self.tag2]
        )
        response = self.client.get(self.base_url)
        assert response.status_code == 200
        assert len(response.data['data']) == 2

    def test_remove_specific_tag(self):
        """Test removing a specific tag from document."""
        DocumentTag.add_tags_to_document(
            self.document, [self.tag1, self.tag2]
        )
        response = self.client.delete(f'{self.base_url}/{self.tag1.id}')
        assert response.status_code == 200
        tags = DocumentTag.get_tags_for_document(str(self.document.id))
        assert len(tags) == 1

    def test_remove_all_tags(self):
        """Test removing all tags from document."""
        DocumentTag.add_tags_to_document(
            self.document, [self.tag1, self.tag2]
        )
        response = self.client.delete(self.base_url)
        assert response.status_code == 200
        tags = DocumentTag.get_tags_for_document(str(self.document.id))
        assert len(tags) == 0


class TestDocumentsByTagAPI(TestCase):
    """Tests for filtering documents by tags."""

    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create(
            username='filteruser',
            password='hashedpassword',
            role='USER'
        )
        self.knowledge = Knowledge.create_knowledge(
            name='Filter KB',
            workspace_id='ws-filter',
            user=self.user
        )
        self.doc1 = Document.create_document(
            knowledge=self.knowledge,
            name='Doc 1'
        )
        self.doc2 = Document.create_document(
            knowledge=self.knowledge,
            name='Doc 2'
        )
        self.tag_tech = Tag.create_tag(self.knowledge, 'category', 'tech')
        self.tag_active = Tag.create_tag(self.knowledge, 'status', 'active')
        DocumentTag.add_tags_to_document(
            self.doc1, [self.tag_tech, self.tag_active]
        )
        DocumentTag.add_tags_to_document(self.doc2, [self.tag_tech])
        self.filter_url = (
            f'/api/workspace/ws-filter/knowledge/{self.knowledge.id}'
            f'/documents-by-tag'
        )

    def test_filter_by_single_tag(self):
        """Test filtering documents by single tag."""
        response = self.client.post(
            self.filter_url,
            {'tags': [{'key': 'category', 'value': 'tech'}]},
            format='json'
        )
        assert response.status_code == 200
        assert len(response.data['data']) == 2

    def test_filter_by_multiple_tags(self):
        """Test filtering documents by multiple tags (AND)."""
        response = self.client.post(
            self.filter_url,
            {
                'tags': [
                    {'key': 'category', 'value': 'tech'},
                    {'key': 'status', 'value': 'active'}
                ]
            },
            format='json'
        )
        assert response.status_code == 200
        assert len(response.data['data']) == 1
        assert response.data['data'][0]['name'] == 'Doc 1'

    def test_filter_no_match(self):
        """Test filtering with no matching documents."""
        response = self.client.post(
            self.filter_url,
            {'tags': [{'key': 'category', 'value': 'nonexistent'}]},
            format='json'
        )
        assert response.status_code == 200
        assert len(response.data['data']) == 0

    def test_filter_requires_tags(self):
        """Test that filtering requires tags parameter."""
        response = self.client.post(
            self.filter_url,
            {},
            format='json'
        )
        assert response.status_code == 400
