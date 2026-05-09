"""
Tests for Embedding model and vector storage.
"""
import uuid
from django.test import TestCase
from apps.knowledge.models import (
    Knowledge, Document, Paragraph, Embedding,
    SourceType, KnowledgeType, DocumentStatus
)
from apps.users.models import User


class EmbeddingModelTest(TestCase):
    """Test Embedding model functionality."""

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
        self.test_vector = [0.1, 0.2, 0.3, 0.4, 0.5]

    def test_create_embedding(self):
        """Embedding can be created with vector."""
        embedding = Embedding.create_embedding(
            paragraph=self.paragraph,
            vector=self.test_vector
        )
        self.assertIsNotNone(embedding.id)
        self.assertEqual(embedding.embedding, self.test_vector)
        self.assertEqual(embedding.paragraph, self.paragraph)
        self.assertEqual(embedding.document, self.document)
        self.assertEqual(embedding.knowledge, self.knowledge)

    def test_embedding_source_type_default(self):
        """Embedding defaults to paragraph source type."""
        embedding = Embedding.create_embedding(
            paragraph=self.paragraph,
            vector=self.test_vector
        )
        self.assertEqual(embedding.source_type, SourceType.PARAGRAPH)

    def test_embedding_source_type_problem(self):
        """Embedding can have problem source type."""
        embedding = Embedding.create_embedding(
            paragraph=self.paragraph,
            vector=self.test_vector,
            source_type=SourceType.PROBLEM,
            source_id="problem-123"
        )
        self.assertEqual(embedding.source_type, SourceType.PROBLEM)
        self.assertEqual(embedding.source_id, "problem-123")

    def test_embedding_source_type_title(self):
        """Embedding can have title source type."""
        embedding = Embedding.create_embedding(
            paragraph=self.paragraph,
            vector=self.test_vector,
            source_type=SourceType.TITLE
        )
        self.assertEqual(embedding.source_type, SourceType.TITLE)

    def test_embedding_is_active_default(self):
        """Embedding is active by default."""
        embedding = Embedding.create_embedding(
            paragraph=self.paragraph,
            vector=self.test_vector
        )
        self.assertTrue(embedding.is_active)

    def test_embedding_can_be_inactive(self):
        """Embedding can be created as inactive."""
        embedding = Embedding.create_embedding(
            paragraph=self.paragraph,
            vector=self.test_vector,
            is_active=False
        )
        self.assertFalse(embedding.is_active)

    def test_embedding_stores_metadata(self):
        """Embedding can store metadata."""
        meta = {"model": "text-embedding-ada-002", "dimensions": 1536}
        embedding = Embedding.create_embedding(
            paragraph=self.paragraph,
            vector=self.test_vector,
            meta=meta
        )
        self.assertEqual(embedding.meta, meta)

    def test_embedding_has_timestamps(self):
        """Embedding has create and update timestamps."""
        embedding = Embedding.create_embedding(
            paragraph=self.paragraph,
            vector=self.test_vector
        )
        self.assertIsNotNone(embedding.create_time)
        self.assertIsNotNone(embedding.update_time)

    def test_embedding_str_representation(self):
        """Embedding has useful string representation."""
        embedding = Embedding.create_embedding(
            paragraph=self.paragraph,
            vector=self.test_vector
        )
        self.assertIn("Embedding", str(embedding))

    def test_activate_embedding(self):
        """Embedding can be activated."""
        embedding = Embedding.create_embedding(
            paragraph=self.paragraph,
            vector=self.test_vector,
            is_active=False
        )
        self.assertFalse(embedding.is_active)
        embedding.activate()
        embedding.refresh_from_db()
        self.assertTrue(embedding.is_active)

    def test_deactivate_embedding(self):
        """Embedding can be deactivated."""
        embedding = Embedding.create_embedding(
            paragraph=self.paragraph,
            vector=self.test_vector
        )
        self.assertTrue(embedding.is_active)
        embedding.deactivate()
        embedding.refresh_from_db()
        self.assertFalse(embedding.is_active)


class EmbeddingBatchCreateTest(TestCase):
    """Test batch creation of embeddings."""

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
        self.paragraphs = []
        for i in range(5):
            p = Paragraph.objects.create(
                document=self.document,
                knowledge=self.knowledge,
                content=f"Paragraph {i} content",
                position=i
            )
            self.paragraphs.append(p)

    def test_batch_create_embeddings(self):
        """Multiple embeddings can be created in batch."""
        embeddings_data = [
            {"paragraph": p, "vector": [0.1 * i, 0.2 * i, 0.3 * i]}
            for i, p in enumerate(self.paragraphs)
        ]
        created = Embedding.batch_create(embeddings_data)
        self.assertEqual(len(created), 5)
        self.assertEqual(Embedding.objects.count(), 5)

    def test_batch_create_with_metadata(self):
        """Batch creation supports metadata."""
        embeddings_data = [
            {
                "paragraph": self.paragraphs[0],
                "vector": [0.1, 0.2],
                "meta": {"index": 0}
            },
            {
                "paragraph": self.paragraphs[1],
                "vector": [0.3, 0.4],
                "meta": {"index": 1}
            }
        ]
        created = Embedding.batch_create(embeddings_data)
        self.assertEqual(created[0].meta, {"index": 0})
        self.assertEqual(created[1].meta, {"index": 1})

    def test_batch_create_with_source_types(self):
        """Batch creation supports different source types."""
        embeddings_data = [
            {
                "paragraph": self.paragraphs[0],
                "vector": [0.1],
                "source_type": SourceType.PARAGRAPH
            },
            {
                "paragraph": self.paragraphs[1],
                "vector": [0.2],
                "source_type": SourceType.TITLE
            }
        ]
        created = Embedding.batch_create(embeddings_data)
        self.assertEqual(created[0].source_type, SourceType.PARAGRAPH)
        self.assertEqual(created[1].source_type, SourceType.TITLE)

    def test_batch_create_empty_list(self):
        """Batch creation with empty list returns empty list."""
        created = Embedding.batch_create([])
        self.assertEqual(created, [])


class EmbeddingQueryTest(TestCase):
    """Test querying embeddings."""

    def setUp(self):
        """Set up test fixtures with multiple embeddings."""
        self.user = User.objects.create(
            username="testuser",
            password="hashed",
            role="user"
        )
        self.kb1 = Knowledge.objects.create(
            name="KB 1",
            workspace_id="test-ws",
            user=self.user
        )
        self.kb2 = Knowledge.objects.create(
            name="KB 2",
            workspace_id="test-ws",
            user=self.user
        )
        self.doc1 = Document.objects.create(
            knowledge=self.kb1,
            name="Doc 1"
        )
        self.doc2 = Document.objects.create(
            knowledge=self.kb1,
            name="Doc 2"
        )
        self.para1 = Paragraph.objects.create(
            document=self.doc1,
            knowledge=self.kb1,
            content="Para 1"
        )
        self.para2 = Paragraph.objects.create(
            document=self.doc1,
            knowledge=self.kb1,
            content="Para 2"
        )
        self.para3 = Paragraph.objects.create(
            document=self.doc2,
            knowledge=self.kb1,
            content="Para 3"
        )
        self.para4 = Paragraph.objects.create(
            document=Document.objects.create(
                knowledge=self.kb2,
                name="Doc in KB2"
            ),
            knowledge=self.kb2,
            content="Para in KB2"
        )

        for i, para in enumerate([self.para1, self.para2, self.para3, self.para4]):
            Embedding.create_embedding(
                paragraph=para,
                vector=[0.1 * i]
            )

    def test_query_by_knowledge(self):
        """Embeddings can be queried by knowledge base."""
        embeddings = Embedding.objects.filter(knowledge=self.kb1)
        self.assertEqual(embeddings.count(), 3)

    def test_query_by_document(self):
        """Embeddings can be queried by document."""
        embeddings = Embedding.objects.filter(document=self.doc1)
        self.assertEqual(embeddings.count(), 2)

    def test_query_by_paragraph(self):
        """Embeddings can be queried by paragraph."""
        embeddings = Embedding.objects.filter(paragraph=self.para1)
        self.assertEqual(embeddings.count(), 1)

    def test_query_active_only(self):
        """Active embeddings can be filtered."""
        embedding = Embedding.objects.filter(paragraph=self.para1).first()
        embedding.deactivate()
        active = Embedding.objects.filter(knowledge=self.kb1, is_active=True)
        self.assertEqual(active.count(), 2)

    def test_query_by_source_type(self):
        """Embeddings can be queried by source type."""
        Embedding.create_embedding(
            paragraph=self.para1,
            vector=[0.5],
            source_type=SourceType.TITLE
        )
        title_embeddings = Embedding.objects.filter(source_type=SourceType.TITLE)
        self.assertEqual(title_embeddings.count(), 1)


class EmbeddingDeleteTest(TestCase):
    """Test deletion of embeddings."""

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
            content="Test content"
        )

    def test_delete_by_knowledge(self):
        """Embeddings can be deleted by knowledge base."""
        for i in range(3):
            Embedding.create_embedding(
                paragraph=self.paragraph,
                vector=[0.1 * i]
            )
        self.assertEqual(Embedding.objects.count(), 3)
        count = Embedding.delete_by_knowledge(str(self.knowledge.id))
        self.assertEqual(count, 3)
        self.assertEqual(Embedding.objects.count(), 0)

    def test_delete_by_document(self):
        """Embeddings can be deleted by document."""
        for i in range(3):
            Embedding.create_embedding(
                paragraph=self.paragraph,
                vector=[0.1 * i]
            )
        count = Embedding.delete_by_document(str(self.document.id))
        self.assertEqual(count, 3)
        self.assertEqual(Embedding.objects.count(), 0)

    def test_delete_by_paragraph(self):
        """Embeddings can be deleted by paragraph."""
        for i in range(3):
            Embedding.create_embedding(
                paragraph=self.paragraph,
                vector=[0.1 * i]
            )
        count = Embedding.delete_by_paragraph(str(self.paragraph.id))
        self.assertEqual(count, 3)
        self.assertEqual(Embedding.objects.count(), 0)

    def test_delete_by_source_id(self):
        """Embeddings can be deleted by source ID."""
        source_id = "custom-source-123"
        Embedding.create_embedding(
            paragraph=self.paragraph,
            vector=[0.1],
            source_id=source_id
        )
        Embedding.create_embedding(
            paragraph=self.paragraph,
            vector=[0.2],
            source_id=source_id
        )
        Embedding.create_embedding(
            paragraph=self.paragraph,
            vector=[0.3],
            source_id="other-source"
        )
        count = Embedding.delete_by_source_id(source_id)
        self.assertEqual(count, 2)
        self.assertEqual(Embedding.objects.count(), 1)

    def test_delete_by_source_id_with_type(self):
        """Embeddings can be deleted by source ID and type."""
        source_id = "shared-source"
        Embedding.create_embedding(
            paragraph=self.paragraph,
            vector=[0.1],
            source_id=source_id,
            source_type=SourceType.PARAGRAPH
        )
        Embedding.create_embedding(
            paragraph=self.paragraph,
            vector=[0.2],
            source_id=source_id,
            source_type=SourceType.TITLE
        )
        count = Embedding.delete_by_source_id(source_id, SourceType.PARAGRAPH)
        self.assertEqual(count, 1)
        self.assertEqual(Embedding.objects.count(), 1)
        remaining = Embedding.objects.first()
        self.assertEqual(remaining.source_type, SourceType.TITLE)

    def test_cascade_delete_with_paragraph(self):
        """Embeddings are deleted when paragraph is deleted."""
        Embedding.create_embedding(
            paragraph=self.paragraph,
            vector=[0.1]
        )
        self.assertEqual(Embedding.objects.count(), 1)
        self.paragraph.delete()
        self.assertEqual(Embedding.objects.count(), 0)

    def test_cascade_delete_with_document(self):
        """Embeddings are deleted when document is deleted."""
        Embedding.create_embedding(
            paragraph=self.paragraph,
            vector=[0.1]
        )
        self.assertEqual(Embedding.objects.count(), 1)
        self.document.delete()
        self.assertEqual(Embedding.objects.count(), 0)

    def test_cascade_delete_with_knowledge(self):
        """Embeddings are deleted when knowledge base is deleted."""
        Embedding.create_embedding(
            paragraph=self.paragraph,
            vector=[0.1]
        )
        self.assertEqual(Embedding.objects.count(), 1)
        self.knowledge.delete()
        self.assertEqual(Embedding.objects.count(), 0)


class SourceTypeTest(TestCase):
    """Test SourceType constants."""

    def test_source_type_values(self):
        """Source types have correct values."""
        self.assertEqual(SourceType.PARAGRAPH, 'p')
        self.assertEqual(SourceType.PROBLEM, 'q')
        self.assertEqual(SourceType.TITLE, 't')

    def test_source_type_choices(self):
        """Source type choices are properly defined."""
        choices = dict(SourceType.CHOICES)
        self.assertEqual(choices['p'], 'Paragraph')
        self.assertEqual(choices['q'], 'Problem')
        self.assertEqual(choices['t'], 'Title')
