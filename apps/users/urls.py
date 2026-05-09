"""
URL routing for User API endpoints.
"""
from django.urls import path
from . import views

app_name = 'users'

urlpatterns = [
    path('', views.UserListView.as_view(), name='user-list'),
    path('<uuid:user_id>', views.UserDetailView.as_view(), name='user-detail'),
    path('username/<str:username>', views.UserByUsernameView.as_view(), name='user-by-username'),
]
