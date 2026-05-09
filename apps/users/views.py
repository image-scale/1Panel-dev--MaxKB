"""
Views for User API endpoints.
"""
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from django.shortcuts import get_object_or_404

from .models import User
from .serializers import UserSerializer, UserCreateSerializer, UserUpdateSerializer


class UserListView(APIView):
    """List all users or create a new user."""

    def get(self, request):
        """Get list of all users."""
        users = User.objects.filter(is_active=True)
        serializer = UserSerializer(users, many=True)
        return Response({
            'code': 200,
            'message': 'success',
            'data': serializer.data
        })

    def post(self, request):
        """Create a new user."""
        serializer = UserCreateSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            return Response({
                'code': 201,
                'message': 'User created successfully',
                'data': UserSerializer(user).data
            }, status=status.HTTP_201_CREATED)
        return Response({
            'code': 400,
            'message': 'Validation error',
            'errors': serializer.errors
        }, status=status.HTTP_400_BAD_REQUEST)


class UserDetailView(APIView):
    """Retrieve, update, or delete a user."""

    def get(self, request, user_id):
        """Get user by ID."""
        user = get_object_or_404(User, id=user_id)
        serializer = UserSerializer(user)
        return Response({
            'code': 200,
            'message': 'success',
            'data': serializer.data
        })

    def put(self, request, user_id):
        """Update user by ID."""
        user = get_object_or_404(User, id=user_id)
        serializer = UserUpdateSerializer(user, data=request.data, partial=True)
        if serializer.is_valid():
            user = serializer.save()
            return Response({
                'code': 200,
                'message': 'User updated successfully',
                'data': UserSerializer(user).data
            })
        return Response({
            'code': 400,
            'message': 'Validation error',
            'errors': serializer.errors
        }, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, user_id):
        """Delete user by ID (soft delete)."""
        user = get_object_or_404(User, id=user_id)
        user.is_active = False
        user.save()
        return Response({
            'code': 200,
            'message': 'User deleted successfully'
        })


class UserByUsernameView(APIView):
    """Get user by username."""

    def get(self, request, username):
        """Get user by username."""
        user = get_object_or_404(User, username=username)
        serializer = UserSerializer(user)
        return Response({
            'code': 200,
            'message': 'success',
            'data': serializer.data
        })
