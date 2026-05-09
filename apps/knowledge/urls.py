"""
URL routing for Knowledge Base API endpoints.
"""
from django.urls import path
from . import views

app_name = 'knowledge'

urlpatterns = [
    path(
        'workspace/<str:workspace_id>/knowledge',
        views.KnowledgeListView.as_view(),
        name='knowledge-list'
    ),
    path(
        'workspace/<str:workspace_id>/knowledge/<uuid:knowledge_id>',
        views.KnowledgeDetailView.as_view(),
        name='knowledge-detail'
    ),
    path(
        'workspace/<str:workspace_id>/folders',
        views.KnowledgeFolderListView.as_view(),
        name='folder-list'
    ),
    path(
        'workspace/<str:workspace_id>/folders/<str:folder_id>',
        views.KnowledgeFolderDetailView.as_view(),
        name='folder-detail'
    ),
]
