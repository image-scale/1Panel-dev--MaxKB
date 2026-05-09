"""
URL routing for Application API endpoints.
"""
from django.urls import path
from . import views

app_name = 'application'

urlpatterns = [
    path(
        'workspace/<str:workspace_id>/application',
        views.ApplicationListView.as_view(),
        name='application-list'
    ),
    path(
        'workspace/<str:workspace_id>/application/<uuid:application_id>',
        views.ApplicationDetailView.as_view(),
        name='application-detail'
    ),
    path(
        'workspace/<str:workspace_id>/application/<uuid:application_id>/knowledge',
        views.ApplicationKnowledgeMappingListView.as_view(),
        name='application-knowledge-list'
    ),
    path(
        'workspace/<str:workspace_id>/application/<uuid:application_id>/knowledge/<uuid:mapping_id>',
        views.ApplicationKnowledgeMappingDetailView.as_view(),
        name='application-knowledge-detail'
    ),
    # Chat endpoints
    path(
        'workspace/<str:workspace_id>/application/<uuid:application_id>/chat',
        views.ChatListView.as_view(),
        name='chat-list'
    ),
    path(
        'workspace/<str:workspace_id>/application/<uuid:application_id>/chat/<uuid:chat_id>',
        views.ChatDetailView.as_view(),
        name='chat-detail'
    ),
    path(
        'workspace/<str:workspace_id>/application/<uuid:application_id>/chat/<uuid:chat_id>/record',
        views.ChatRecordListView.as_view(),
        name='chat-record-list'
    ),
    path(
        'workspace/<str:workspace_id>/application/<uuid:application_id>/chat/<uuid:chat_id>/record/<uuid:record_id>',
        views.ChatRecordDetailView.as_view(),
        name='chat-record-detail'
    ),
    path(
        'workspace/<str:workspace_id>/application/<uuid:application_id>/chat/<uuid:chat_id>/record/<uuid:record_id>/vote',
        views.ChatRecordVoteView.as_view(),
        name='chat-record-vote'
    ),
    path(
        'workspace/<str:workspace_id>/application/<uuid:application_id>/chat/<uuid:chat_id>/message',
        views.ChatMessageListView.as_view(),
        name='chat-message-list'
    ),
    path(
        'workspace/<str:workspace_id>/application/<uuid:application_id>/chat/<uuid:chat_id>/history',
        views.ChatHistoryView.as_view(),
        name='chat-history'
    ),
]
