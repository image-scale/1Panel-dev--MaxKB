"""
Serializers for User model.
"""
from rest_framework import serializers
from .models import User, generate_password_hash


class UserSerializer(serializers.ModelSerializer):
    """Serializer for reading user data."""

    class Meta:
        model = User
        fields = [
            'id', 'email', 'phone', 'nick_name', 'username',
            'role', 'source', 'is_active', 'language',
            'create_time', 'update_time'
        ]
        read_only_fields = ['id', 'create_time', 'update_time']


class UserCreateSerializer(serializers.ModelSerializer):
    """Serializer for creating a new user."""
    password = serializers.CharField(write_only=True, min_length=6)

    class Meta:
        model = User
        fields = [
            'username', 'password', 'nick_name', 'email',
            'phone', 'role', 'source', 'language'
        ]

    def create(self, validated_data):
        raw_password = validated_data.pop('password')
        user = User(**validated_data)
        user.set_password(raw_password)
        user.save()
        return user


class UserUpdateSerializer(serializers.ModelSerializer):
    """Serializer for updating user data."""
    password = serializers.CharField(write_only=True, min_length=6, required=False)

    class Meta:
        model = User
        fields = [
            'email', 'phone', 'nick_name', 'password',
            'role', 'is_active', 'language'
        ]

    def update(self, instance, validated_data):
        raw_password = validated_data.pop('password', None)
        for attr, value in validated_data.items():
            setattr(instance, attr, value)
        if raw_password:
            instance.set_password(raw_password)
        instance.save()
        return instance
