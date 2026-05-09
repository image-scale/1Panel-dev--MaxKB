"""
Views for Knowledge Base API endpoints.
"""
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from django.shortcuts import get_object_or_404

from .models import Knowledge, KnowledgeFolder, Document
from .serializers import (
    KnowledgeSerializer, KnowledgeCreateSerializer, KnowledgeUpdateSerializer,
    KnowledgeFolderSerializer, KnowledgeFolderCreateSerializer,
    DocumentSerializer, DocumentCreateSerializer, DocumentUpdateSerializer
)


class KnowledgeListView(APIView):
    """List all knowledge bases or create a new one."""

    def get(self, request, workspace_id):
        """Get list of knowledge bases for a workspace."""
        kb_type = request.query_params.get('type')
        scope = request.query_params.get('scope')
        folder_id = request.query_params.get('folder_id')
        page = int(request.query_params.get('page', 1))
        page_size = int(request.query_params.get('page_size', 20))

        queryset = Knowledge.objects.filter(workspace_id=workspace_id)

        if kb_type is not None:
            queryset = queryset.filter(type=kb_type)
        if scope:
            queryset = queryset.filter(scope=scope)
        if folder_id:
            queryset = queryset.filter(folder_id=folder_id)

        total = queryset.count()
        start = (page - 1) * page_size
        end = start + page_size
        knowledge_bases = queryset[start:end]

        serializer = KnowledgeSerializer(knowledge_bases, many=True)
        return Response({
            'code': 200,
            'message': 'success',
            'data': {
                'items': serializer.data,
                'total': total,
                'page': page,
                'page_size': page_size
            }
        })

    def post(self, request, workspace_id):
        """Create a new knowledge base."""
        data = request.data.copy()
        data['workspace_id'] = workspace_id
        serializer = KnowledgeCreateSerializer(data=data)
        if serializer.is_valid():
            knowledge = serializer.save()
            return Response({
                'code': 201,
                'message': 'Knowledge base created successfully',
                'data': KnowledgeSerializer(knowledge).data
            }, status=status.HTTP_201_CREATED)
        return Response({
            'code': 400,
            'message': 'Validation error',
            'errors': serializer.errors
        }, status=status.HTTP_400_BAD_REQUEST)


class KnowledgeDetailView(APIView):
    """Retrieve, update, or delete a knowledge base."""

    def get(self, request, workspace_id, knowledge_id):
        """Get knowledge base by ID."""
        knowledge = get_object_or_404(
            Knowledge, id=knowledge_id, workspace_id=workspace_id
        )
        serializer = KnowledgeSerializer(knowledge)
        return Response({
            'code': 200,
            'message': 'success',
            'data': serializer.data
        })

    def put(self, request, workspace_id, knowledge_id):
        """Update knowledge base by ID."""
        knowledge = get_object_or_404(
            Knowledge, id=knowledge_id, workspace_id=workspace_id
        )
        serializer = KnowledgeUpdateSerializer(
            knowledge, data=request.data, partial=True
        )
        if serializer.is_valid():
            knowledge = serializer.save()
            return Response({
                'code': 200,
                'message': 'Knowledge base updated successfully',
                'data': KnowledgeSerializer(knowledge).data
            })
        return Response({
            'code': 400,
            'message': 'Validation error',
            'errors': serializer.errors
        }, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, workspace_id, knowledge_id):
        """Delete knowledge base by ID."""
        knowledge = get_object_or_404(
            Knowledge, id=knowledge_id, workspace_id=workspace_id
        )
        knowledge.delete()
        return Response({
            'code': 200,
            'message': 'Knowledge base deleted successfully'
        })


class KnowledgeFolderListView(APIView):
    """List all folders or create a new one."""

    def get(self, request, workspace_id):
        """Get list of folders for a workspace."""
        parent_id = request.query_params.get('parent_id')

        queryset = KnowledgeFolder.objects.filter(workspace_id=workspace_id)

        if parent_id:
            queryset = queryset.filter(parent_id=parent_id)
        else:
            queryset = queryset.filter(parent__isnull=True)

        serializer = KnowledgeFolderSerializer(queryset, many=True)
        return Response({
            'code': 200,
            'message': 'success',
            'data': serializer.data
        })

    def post(self, request, workspace_id):
        """Create a new folder."""
        data = request.data.copy()
        data['workspace_id'] = workspace_id
        serializer = KnowledgeFolderCreateSerializer(data=data)
        if serializer.is_valid():
            folder = serializer.save()
            return Response({
                'code': 201,
                'message': 'Folder created successfully',
                'data': KnowledgeFolderSerializer(folder).data
            }, status=status.HTTP_201_CREATED)
        return Response({
            'code': 400,
            'message': 'Validation error',
            'errors': serializer.errors
        }, status=status.HTTP_400_BAD_REQUEST)


class KnowledgeFolderDetailView(APIView):
    """Retrieve, update, or delete a folder."""

    def get(self, request, workspace_id, folder_id):
        """Get folder by ID."""
        folder = get_object_or_404(
            KnowledgeFolder, id=folder_id, workspace_id=workspace_id
        )
        serializer = KnowledgeFolderSerializer(folder)
        return Response({
            'code': 200,
            'message': 'success',
            'data': serializer.data
        })

    def put(self, request, workspace_id, folder_id):
        """Update folder by ID."""
        folder = get_object_or_404(
            KnowledgeFolder, id=folder_id, workspace_id=workspace_id
        )
        serializer = KnowledgeFolderCreateSerializer(
            folder, data=request.data, partial=True
        )
        if serializer.is_valid():
            folder = serializer.save()
            return Response({
                'code': 200,
                'message': 'Folder updated successfully',
                'data': KnowledgeFolderSerializer(folder).data
            })
        return Response({
            'code': 400,
            'message': 'Validation error',
            'errors': serializer.errors
        }, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, workspace_id, folder_id):
        """Delete folder by ID."""
        folder = get_object_or_404(
            KnowledgeFolder, id=folder_id, workspace_id=workspace_id
        )
        folder.delete()
        return Response({
            'code': 200,
            'message': 'Folder deleted successfully'
        })


class DocumentListView(APIView):
    """List all documents or create a new one."""

    def get(self, request, workspace_id, knowledge_id):
        """Get list of documents for a knowledge base."""
        knowledge = get_object_or_404(
            Knowledge, id=knowledge_id, workspace_id=workspace_id
        )
        doc_status = request.query_params.get('status')
        is_active = request.query_params.get('is_active')
        page = int(request.query_params.get('page', 1))
        page_size = int(request.query_params.get('page_size', 20))

        queryset = Document.objects.filter(knowledge=knowledge)

        if doc_status is not None:
            queryset = queryset.filter(status=doc_status)
        if is_active is not None:
            queryset = queryset.filter(is_active=is_active.lower() == 'true')

        total = queryset.count()
        start = (page - 1) * page_size
        end = start + page_size
        documents = queryset[start:end]

        serializer = DocumentSerializer(documents, many=True)
        return Response({
            'code': 200,
            'message': 'success',
            'data': {
                'items': serializer.data,
                'total': total,
                'page': page,
                'page_size': page_size
            }
        })

    def post(self, request, workspace_id, knowledge_id):
        """Create a new document."""
        knowledge = get_object_or_404(
            Knowledge, id=knowledge_id, workspace_id=workspace_id
        )
        data = request.data.copy()
        data['knowledge'] = knowledge.id
        serializer = DocumentCreateSerializer(data=data)
        if serializer.is_valid():
            document = serializer.save()
            return Response({
                'code': 201,
                'message': 'Document created successfully',
                'data': DocumentSerializer(document).data
            }, status=status.HTTP_201_CREATED)
        return Response({
            'code': 400,
            'message': 'Validation error',
            'errors': serializer.errors
        }, status=status.HTTP_400_BAD_REQUEST)


class DocumentDetailView(APIView):
    """Retrieve, update, or delete a document."""

    def get(self, request, workspace_id, knowledge_id, document_id):
        """Get document by ID."""
        knowledge = get_object_or_404(
            Knowledge, id=knowledge_id, workspace_id=workspace_id
        )
        document = get_object_or_404(
            Document, id=document_id, knowledge=knowledge
        )
        serializer = DocumentSerializer(document)
        return Response({
            'code': 200,
            'message': 'success',
            'data': serializer.data
        })

    def put(self, request, workspace_id, knowledge_id, document_id):
        """Update document by ID."""
        knowledge = get_object_or_404(
            Knowledge, id=knowledge_id, workspace_id=workspace_id
        )
        document = get_object_or_404(
            Document, id=document_id, knowledge=knowledge
        )
        serializer = DocumentUpdateSerializer(
            document, data=request.data, partial=True
        )
        if serializer.is_valid():
            document = serializer.save()
            return Response({
                'code': 200,
                'message': 'Document updated successfully',
                'data': DocumentSerializer(document).data
            })
        return Response({
            'code': 400,
            'message': 'Validation error',
            'errors': serializer.errors
        }, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, workspace_id, knowledge_id, document_id):
        """Delete document by ID."""
        knowledge = get_object_or_404(
            Knowledge, id=knowledge_id, workspace_id=workspace_id
        )
        document = get_object_or_404(
            Document, id=document_id, knowledge=knowledge
        )
        document.delete()
        return Response({
            'code': 200,
            'message': 'Document deleted successfully'
        })
