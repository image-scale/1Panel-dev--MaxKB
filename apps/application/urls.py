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
]
