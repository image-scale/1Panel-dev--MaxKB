"""
Views for Application API endpoints.
"""
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from django.shortcuts import get_object_or_404

from .models import Application, ApplicationKnowledgeMapping
from .chat_models import Chat, ChatRecord, ChatMessage, ChatUserType, VoteStatus
from .serializers import (
    ApplicationSerializer, ApplicationCreateSerializer,
    ApplicationUpdateSerializer, ApplicationKnowledgeMappingSerializer,
    ApplicationKnowledgeMappingCreateSerializer,
    ChatSerializer, ChatCreateSerializer,
    ChatRecordSerializer, ChatRecordCreateSerializer, ChatRecordVoteSerializer,
    ChatMessageSerializer, ChatMessageCreateSerializer, SendMessageSerializer
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


class ChatListView(APIView):
    """List chats for an application or create a new chat."""

    def get(self, request, workspace_id, application_id):
        """Get list of chats for an application."""
        application = get_object_or_404(
            Application, id=application_id, workspace_id=workspace_id
        )
        include_deleted = request.query_params.get('include_deleted', 'false')
        chat_user_id = request.query_params.get('chat_user_id')
        page = int(request.query_params.get('page', 1))
        page_size = int(request.query_params.get('page_size', 20))

        queryset = Chat.objects.filter(application=application)
        if include_deleted.lower() != 'true':
            queryset = queryset.filter(is_deleted=False)
        if chat_user_id:
            queryset = queryset.filter(chat_user_id=chat_user_id)

        total = queryset.count()
        start = (page - 1) * page_size
        end = start + page_size
        chats = queryset[start:end]

        serializer = ChatSerializer(chats, many=True)
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

    def post(self, request, workspace_id, application_id):
        """Create a new chat session."""
        application = get_object_or_404(
            Application, id=application_id, workspace_id=workspace_id
        )
        chat = Chat.create_chat(
            application=application,
            user=request.data.get('user'),
            chat_user_id=request.data.get('chat_user_id', ''),
            chat_user_type=request.data.get('chat_user_type', ChatUserType.ANONYMOUS),
            title=request.data.get('title', '')
        )
        return Response({
            'code': 201,
            'message': 'Chat created successfully',
            'data': ChatSerializer(chat).data
        }, status=status.HTTP_201_CREATED)


class ChatDetailView(APIView):
    """Retrieve, update, or delete a chat."""

    def get(self, request, workspace_id, application_id, chat_id):
        """Get chat by ID."""
        chat = get_object_or_404(
            Chat, id=chat_id, application_id=application_id
        )
        serializer = ChatSerializer(chat)
        return Response({
            'code': 200,
            'message': 'success',
            'data': serializer.data
        })

    def put(self, request, workspace_id, application_id, chat_id):
        """Update chat by ID."""
        chat = get_object_or_404(
            Chat, id=chat_id, application_id=application_id
        )
        if 'title' in request.data:
            chat.title = request.data['title']
        if 'meta' in request.data:
            chat.meta = request.data['meta']
        chat.save()
        return Response({
            'code': 200,
            'message': 'Chat updated successfully',
            'data': ChatSerializer(chat).data
        })

    def delete(self, request, workspace_id, application_id, chat_id):
        """Soft delete chat by ID."""
        chat = get_object_or_404(
            Chat, id=chat_id, application_id=application_id
        )
        hard_delete = request.query_params.get('hard', 'false').lower() == 'true'
        if hard_delete:
            chat.delete()
        else:
            chat.soft_delete()
        return Response({
            'code': 200,
            'message': 'Chat deleted successfully'
        })


class ChatRecordListView(APIView):
    """List records for a chat or create a new record."""

    def get(self, request, workspace_id, application_id, chat_id):
        """Get records for a chat."""
        chat = get_object_or_404(
            Chat, id=chat_id, application_id=application_id
        )
        records = ChatRecord.objects.filter(chat=chat)
        serializer = ChatRecordSerializer(records, many=True)
        return Response({
            'code': 200,
            'message': 'success',
            'data': serializer.data
        })

    def post(self, request, workspace_id, application_id, chat_id):
        """Create a new chat record."""
        chat = get_object_or_404(
            Chat, id=chat_id, application_id=application_id
        )
        record = ChatRecord.create_record(
            chat=chat,
            problem_text=request.data.get('problem_text', ''),
            answer_text=request.data.get('answer_text', ''),
            message_tokens=request.data.get('message_tokens', 0),
            answer_tokens=request.data.get('answer_tokens', 0),
            run_time=request.data.get('run_time', 0.0),
            paragraph_ids=request.data.get('paragraph_ids'),
            details=request.data.get('details')
        )
        return Response({
            'code': 201,
            'message': 'Record created successfully',
            'data': ChatRecordSerializer(record).data
        }, status=status.HTTP_201_CREATED)


class ChatRecordDetailView(APIView):
    """Retrieve or update a chat record."""

    def get(self, request, workspace_id, application_id, chat_id, record_id):
        """Get a specific chat record."""
        record = get_object_or_404(ChatRecord, id=record_id, chat_id=chat_id)
        return Response({
            'code': 200,
            'message': 'success',
            'data': ChatRecordSerializer(record).data
        })

    def put(self, request, workspace_id, application_id, chat_id, record_id):
        """Update a chat record (set answer)."""
        record = get_object_or_404(ChatRecord, id=record_id, chat_id=chat_id)
        if 'answer_text' in request.data:
            record.set_answer(
                answer_text=request.data['answer_text'],
                answer_tokens=request.data.get('answer_tokens', 0),
                run_time=request.data.get('run_time', 0.0),
                paragraph_ids=request.data.get('paragraph_ids')
            )
        return Response({
            'code': 200,
            'message': 'Record updated successfully',
            'data': ChatRecordSerializer(record).data
        })


class ChatRecordVoteView(APIView):
    """Vote on a chat record."""

    def post(self, request, workspace_id, application_id, chat_id, record_id):
        """Set vote status on a record."""
        record = get_object_or_404(ChatRecord, id=record_id, chat_id=chat_id)
        vote_status = request.data.get('vote_status', VoteStatus.NONE)

        if vote_status == VoteStatus.UP:
            record.upvote()
        elif vote_status == VoteStatus.DOWN:
            record.downvote()
        else:
            record.clear_vote()

        return Response({
            'code': 200,
            'message': 'Vote recorded',
            'data': ChatRecordSerializer(record).data
        })


class ChatMessageListView(APIView):
    """List messages for a chat or add a message."""

    def get(self, request, workspace_id, application_id, chat_id):
        """Get message history for a chat."""
        chat = get_object_or_404(
            Chat, id=chat_id, application_id=application_id
        )
        limit = request.query_params.get('limit')
        if limit:
            messages = ChatMessage.get_history(str(chat.id), int(limit))
        else:
            messages = ChatMessage.get_history(str(chat.id))
        serializer = ChatMessageSerializer(messages, many=True)
        return Response({
            'code': 200,
            'message': 'success',
            'data': serializer.data
        })

    def post(self, request, workspace_id, application_id, chat_id):
        """Add a message to the chat."""
        chat = get_object_or_404(
            Chat, id=chat_id, application_id=application_id
        )
        message = ChatMessage.create_message(
            chat=chat,
            role=request.data.get('role', 'user'),
            content=request.data.get('content', ''),
            tokens=request.data.get('tokens', 0),
            meta=request.data.get('meta')
        )
        return Response({
            'code': 201,
            'message': 'Message created successfully',
            'data': ChatMessageSerializer(message).data
        }, status=status.HTTP_201_CREATED)

    def delete(self, request, workspace_id, application_id, chat_id):
        """Clear chat message history."""
        chat = get_object_or_404(
            Chat, id=chat_id, application_id=application_id
        )
        ChatMessage.objects.filter(chat=chat).delete()
        return Response({
            'code': 200,
            'message': 'Chat history cleared'
        })


class ChatHistoryView(APIView):
    """Get formatted chat history for an application."""

    def get(self, request, workspace_id, application_id, chat_id):
        """Get formatted history as role/content list."""
        chat = get_object_or_404(
            Chat, id=chat_id, application_id=application_id
        )
        limit = request.query_params.get('limit')
        if limit:
            history = ChatMessage.get_formatted_history(str(chat.id), int(limit))
        else:
            history = ChatMessage.get_formatted_history(str(chat.id))
        return Response({
            'code': 200,
            'message': 'success',
            'data': history
        })
