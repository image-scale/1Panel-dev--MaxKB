"""
URL configuration for MaxKB project.
"""
from django.urls import path, include

urlpatterns = [
    path('api/users/', include('apps.users.urls')),
    path('api/', include('apps.knowledge.urls')),
]
