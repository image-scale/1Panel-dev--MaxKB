"""
Views for Application API endpoints.
"""
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from django.shortcuts import get_object_or_404

from .models import Application, ApplicationKnowledgeMapping
from .serializers import (
    ApplicationSerializer, ApplicationCreateSerializer,
    ApplicationUpdateSerializer, ApplicationKnowledgeMappingSerializer,
    ApplicationKnowledgeMappingCreateSerializer
)
from apps.knowledge.models import Knowledge


class ApplicationListView(APIView):
    """List all applications or create a new one."""

    def get(self, request, workspace_id):
        """Get list of applications for a workspace."""
        app_type = request.query_params.get('type')
        is_published = request.query_params.get('is_published')
        is_active = request.query_params.get('is_active')
        name = request.query_params.get('name')
        page = int(request.query_params.get('page', 1))
        page_size = int(request.query_params.get('page_size', 20))

        queryset = Application.objects.filter(workspace_id=workspace_id)

        if app_type is not None:
            queryset = queryset.filter(type=app_type)
        if is_published is not None:
            queryset = queryset.filter(is_published=is_published.lower() == 'true')
        if is_active is not None:
            queryset = queryset.filter(is_active=is_active.lower() == 'true')
        if name is not None:
            queryset = queryset.filter(name__icontains=name)

        total = queryset.count()
        start = (page - 1) * page_size
        end = start + page_size
        applications = queryset[start:end]

        serializer = ApplicationSerializer(applications, many=True)
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
        """Create a new application."""
        data = request.data.copy()
        data['workspace_id'] = workspace_id
        serializer = ApplicationCreateSerializer(data=data)
        if serializer.is_valid():
            application = serializer.save()
            return Response({
                'code': 201,
                'message': 'Application created successfully',
                'data': ApplicationSerializer(application).data
            }, status=status.HTTP_201_CREATED)
        return Response({
            'code': 400,
            'message': 'Validation error',
            'errors': serializer.errors
        }, status=status.HTTP_400_BAD_REQUEST)


class ApplicationDetailView(APIView):
    """Retrieve, update, or delete an application."""

    def get(self, request, workspace_id, application_id):
        """Get application by ID."""
        application = get_object_or_404(
            Application, id=application_id, workspace_id=workspace_id
        )
        serializer = ApplicationSerializer(application)
        return Response({
            'code': 200,
            'message': 'success',
            'data': serializer.data
        })

    def put(self, request, workspace_id, application_id):
        """Update application by ID."""
        application = get_object_or_404(
            Application, id=application_id, workspace_id=workspace_id
        )
        serializer = ApplicationUpdateSerializer(
            application, data=request.data, partial=True
        )
        if serializer.is_valid():
            application = serializer.save()
            return Response({
                'code': 200,
                'message': 'Application updated successfully',
                'data': ApplicationSerializer(application).data
            })
        return Response({
            'code': 400,
            'message': 'Validation error',
            'errors': serializer.errors
        }, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, workspace_id, application_id):
        """Delete application by ID."""
        application = get_object_or_404(
            Application, id=application_id, workspace_id=workspace_id
        )
        application.delete()
        return Response({
            'code': 200,
            'message': 'Application deleted successfully'
        })


class ApplicationKnowledgeMappingListView(APIView):
    """List and manage application-knowledge mappings."""

    def get(self, request, workspace_id, application_id):
        """Get knowledge bases linked to an application."""
        application = get_object_or_404(
            Application, id=application_id, workspace_id=workspace_id
        )
        mappings = ApplicationKnowledgeMapping.objects.filter(
            application=application
        )
        serializer = ApplicationKnowledgeMappingSerializer(mappings, many=True)
        return Response({
            'code': 200,
            'message': 'success',
            'data': serializer.data
        })

    def post(self, request, workspace_id, application_id):
        """Link a knowledge base to an application."""
        application = get_object_or_404(
            Application, id=application_id, workspace_id=workspace_id
        )
        knowledge_id = request.data.get('knowledge_id')
        if not knowledge_id:
            return Response({
                'code': 400,
                'message': 'knowledge_id is required'
            }, status=status.HTTP_400_BAD_REQUEST)

        knowledge = get_object_or_404(
            Knowledge, id=knowledge_id, workspace_id=workspace_id
        )

        if ApplicationKnowledgeMapping.objects.filter(
            application=application, knowledge=knowledge
        ).exists():
            return Response({
                'code': 400,
                'message': 'Mapping already exists'
            }, status=status.HTTP_400_BAD_REQUEST)

        mapping = ApplicationKnowledgeMapping.create_mapping(
            application, knowledge
        )
        return Response({
            'code': 201,
            'message': 'Knowledge base linked successfully',
            'data': ApplicationKnowledgeMappingSerializer(mapping).data
        }, status=status.HTTP_201_CREATED)


class ApplicationKnowledgeMappingDetailView(APIView):
    """Delete an application-knowledge mapping."""

    def delete(self, request, workspace_id, application_id, mapping_id):
        """Unlink a knowledge base from an application."""
        application = get_object_or_404(
            Application, id=application_id, workspace_id=workspace_id
        )
        mapping = get_object_or_404(
            ApplicationKnowledgeMapping, id=mapping_id, application=application
        )
        mapping.delete()
        return Response({
            'code': 200,
            'message': 'Mapping deleted successfully'
        })
