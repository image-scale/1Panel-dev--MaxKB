"""
URL routing for Model Provider API endpoints.
"""
from django.urls import path
from . import views

app_name = 'models_provider'

urlpatterns = [
    # Provider endpoints
    path(
        'providers',
        views.ProviderListView.as_view(),
        name='provider-list'
    ),
    path(
        'providers/validate',
        views.ValidateCredentialView.as_view(),
        name='validate-credential'
    ),
    path(
        'providers/<str:provider_id>',
        views.ProviderDetailView.as_view(),
        name='provider-detail'
    ),
    path(
        'providers/<str:provider_id>/models',
        views.ProviderModelsView.as_view(),
        name='provider-models'
    ),
    # Model configuration endpoints
    path(
        'workspace/<str:workspace_id>/model',
        views.ModelConfigListView.as_view(),
        name='model-config-list'
    ),
    path(
        'workspace/<str:workspace_id>/model/<uuid:config_id>',
        views.ModelConfigDetailView.as_view(),
        name='model-config-detail'
    ),
    path(
        'workspace/<str:workspace_id>/model/<uuid:config_id>/activate',
        views.ModelConfigActivateView.as_view(),
        name='model-config-activate'
    ),
]
