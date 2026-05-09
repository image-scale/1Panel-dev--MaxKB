"""
Tests for Chat models and API endpoints.
"""
import pytest
from django.test import TestCase
from rest_framework.test import APIClient

from apps.users.models import User
from apps.application.models import Application
from apps.application.chat_models import (
    Chat, ChatRecord, ChatMessage,
    ChatUserType, MessageRole, VoteStatus
)


class TestChatModel(TestCase):
    """Tests for Chat model."""

    def setUp(self):
        self.user = User.objects.create(
            username='testuser',
            password='hashedpassword',
            role='USER'
        )
        self.application = Application.create_application(
            name='Test App',
            workspace_id='workspace-1',
            user=self.user
        )

    def test_create_chat(self):
        """Test creating a chat session."""
        chat = Chat.create_chat(
            application=self.application,
            user=self.user,
            chat_user_id='user-123',
            chat_user_type=ChatUserType.REGISTERED,
            title='Test Chat'
        )
        assert chat.id is not None
        assert chat.application == self.application
        assert chat.user == self.user
        assert chat.chat_user_id == 'user-123'
        assert chat.chat_user_type == ChatUserType.REGISTERED
        assert chat.title == 'Test Chat'
        assert chat.message_count == 0
        assert chat.is_deleted is False

    def test_create_anonymous_chat(self):
        """Test creating an anonymous chat."""
        chat = Chat.create_chat(
            application=self.application,
            chat_user_type=ChatUserType.ANONYMOUS
        )
        assert chat.user is None
        assert chat.chat_user_type == ChatUserType.ANONYMOUS

    def test_chat_increments_dialogue_count(self):
        """Test that creating a chat increments application dialogue count."""
        initial_count = self.application.dialogue_count
        Chat.create_chat(application=self.application)
        self.application.refresh_from_db()
        assert self.application.dialogue_count == initial_count + 1

    def test_soft_delete_chat(self):
        """Test soft deleting a chat."""
        chat = Chat.create_chat(application=self.application)
        assert chat.is_deleted is False
        chat.soft_delete()
        assert chat.is_deleted is True

    def test_restore_chat(self):
        """Test restoring a soft-deleted chat."""
        chat = Chat.create_chat(application=self.application)
        chat.soft_delete()
        assert chat.is_deleted is True
        chat.restore()
        assert chat.is_deleted is False

    def test_update_abstract(self):
        """Test updating chat abstract and title."""
        chat = Chat.create_chat(application=self.application)
        chat.update_abstract('This is the first message in the chat')
        assert chat.abstract == 'This is the first message in the chat'
        assert chat.title == 'This is the first message in the chat'

    def test_update_abstract_truncates_long_text(self):
        """Test that update_abstract truncates long text."""
        chat = Chat.create_chat(application=self.application)
        long_text = 'x' * 2000
        chat.update_abstract(long_text)
        assert len(chat.abstract) == 1024
        assert len(chat.title) == 256

    def test_increment_message_count(self):
        """Test incrementing message count."""
        chat = Chat.create_chat(application=self.application)
        assert chat.message_count == 0
        chat.increment_message_count()
        assert chat.message_count == 1
        chat.increment_message_count()
        assert chat.message_count == 2


class TestChatRecordModel(TestCase):
    """Tests for ChatRecord model."""

    def setUp(self):
        self.user = User.objects.create(
            username='testuser2',
            password='hashedpassword',
            role='USER'
        )
        self.application = Application.create_application(
            name='Test App 2',
            workspace_id='workspace-2',
            user=self.user
        )
        self.chat = Chat.create_chat(application=self.application)

    def test_create_record(self):
        """Test creating a chat record."""
        record = ChatRecord.create_record(
            chat=self.chat,
            problem_text='What is Python?',
            answer_text='Python is a programming language.',
            message_tokens=10,
            answer_tokens=20,
            run_time=0.5,
            paragraph_ids=['para-1', 'para-2']
        )
        assert record.id is not None
        assert record.chat == self.chat
        assert record.index == 0
        assert record.problem_text == 'What is Python?'
        assert record.answer_text == 'Python is a programming language.'
        assert record.message_tokens == 10
        assert record.answer_tokens == 20
        assert record.run_time == 0.5
        assert record.paragraph_ids == ['para-1', 'para-2']
        assert record.vote_status == VoteStatus.NONE

    def test_create_record_increments_message_count(self):
        """Test that creating a record increments chat message count."""
        assert self.chat.message_count == 0
        ChatRecord.create_record(chat=self.chat, problem_text='Question 1')
        self.chat.refresh_from_db()
        assert self.chat.message_count == 1

    def test_create_record_sets_abstract_for_first_message(self):
        """Test that first record sets chat abstract."""
        ChatRecord.create_record(chat=self.chat, problem_text='First question here')
        self.chat.refresh_from_db()
        assert self.chat.abstract == 'First question here'
        assert self.chat.title == 'First question here'

    def test_record_index_increments(self):
        """Test that record index increments for each new record."""
        r1 = ChatRecord.create_record(chat=self.chat, problem_text='Q1')
        r2 = ChatRecord.create_record(chat=self.chat, problem_text='Q2')
        r3 = ChatRecord.create_record(chat=self.chat, problem_text='Q3')
        assert r1.index == 0
        assert r2.index == 1
        assert r3.index == 2

    def test_set_answer(self):
        """Test setting answer on a record."""
        record = ChatRecord.create_record(
            chat=self.chat,
            problem_text='What is AI?'
        )
        assert record.answer_text == ''
        record.set_answer(
            answer_text='AI is artificial intelligence.',
            answer_tokens=15,
            run_time=0.3,
            paragraph_ids=['para-3']
        )
        record.refresh_from_db()
        assert record.answer_text == 'AI is artificial intelligence.'
        assert record.answer_tokens == 15
        assert record.run_time == 0.3
        assert record.paragraph_ids == ['para-3']

    def test_upvote(self):
        """Test upvoting a record."""
        record = ChatRecord.create_record(chat=self.chat, problem_text='Q')
        assert record.vote_status == VoteStatus.NONE
        record.upvote()
        assert record.vote_status == VoteStatus.UP

    def test_downvote(self):
        """Test downvoting a record."""
        record = ChatRecord.create_record(chat=self.chat, problem_text='Q')
        record.downvote()
        assert record.vote_status == VoteStatus.DOWN

    def test_clear_vote(self):
        """Test clearing vote status."""
        record = ChatRecord.create_record(chat=self.chat, problem_text='Q')
        record.upvote()
        assert record.vote_status == VoteStatus.UP
        record.clear_vote()
        assert record.vote_status == VoteStatus.NONE


class TestChatMessageModel(TestCase):
    """Tests for ChatMessage model."""

    def setUp(self):
        self.user = User.objects.create(
            username='testuser3',
            password='hashedpassword',
            role='USER'
        )
        self.application = Application.create_application(
            name='Test App 3',
            workspace_id='workspace-3',
            user=self.user
        )
        self.chat = Chat.create_chat(application=self.application)

    def test_create_message(self):
        """Test creating a chat message."""
        message = ChatMessage.create_message(
            chat=self.chat,
            role=MessageRole.USER,
            content='Hello, how are you?',
            tokens=5,
            meta={'source': 'web'}
        )
        assert message.id is not None
        assert message.chat == self.chat
        assert message.role == MessageRole.USER
        assert message.content == 'Hello, how are you?'
        assert message.tokens == 5
        assert message.meta == {'source': 'web'}

    def test_create_system_message(self):
        """Test creating a system message."""
        message = ChatMessage.create_message(
            chat=self.chat,
            role=MessageRole.SYSTEM,
            content='You are a helpful assistant.'
        )
        assert message.role == MessageRole.SYSTEM

    def test_create_assistant_message(self):
        """Test creating an assistant message."""
        message = ChatMessage.create_message(
            chat=self.chat,
            role=MessageRole.ASSISTANT,
            content='I am doing well, thank you!'
        )
        assert message.role == MessageRole.ASSISTANT

    def test_get_history(self):
        """Test getting message history."""
        ChatMessage.create_message(
            chat=self.chat, role=MessageRole.USER, content='Msg 1'
        )
        ChatMessage.create_message(
            chat=self.chat, role=MessageRole.ASSISTANT, content='Msg 2'
        )
        ChatMessage.create_message(
            chat=self.chat, role=MessageRole.USER, content='Msg 3'
        )
        history = ChatMessage.get_history(str(self.chat.id))
        assert len(history) == 3
        assert history[0].content == 'Msg 1'
        assert history[1].content == 'Msg 2'
        assert history[2].content == 'Msg 3'

    def test_get_history_with_limit(self):
        """Test getting limited message history."""
        for i in range(5):
            ChatMessage.create_message(
                chat=self.chat, role=MessageRole.USER, content=f'Msg {i}'
            )
        history = ChatMessage.get_history(str(self.chat.id), limit=3)
        assert len(history) == 3

    def test_get_formatted_history(self):
        """Test getting formatted history."""
        ChatMessage.create_message(
            chat=self.chat, role=MessageRole.USER, content='Hello'
        )
        ChatMessage.create_message(
            chat=self.chat, role=MessageRole.ASSISTANT, content='Hi there!'
        )
        formatted = ChatMessage.get_formatted_history(str(self.chat.id))
        assert len(formatted) == 2
        assert formatted[0] == {'role': 'user', 'content': 'Hello'}
        assert formatted[1] == {'role': 'assistant', 'content': 'Hi there!'}


class TestChatAPI(TestCase):
    """Tests for Chat API endpoints."""

    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create(
            username='apiuser',
            password='hashedpassword',
            role='USER'
        )
        self.application = Application.create_application(
            name='API Test App',
            workspace_id='ws-api',
            user=self.user
        )
        self.base_url = f'/api/workspace/ws-api/application/{self.application.id}/chat'

    def test_create_chat(self):
        """Test creating a chat via API."""
        response = self.client.post(
            self.base_url,
            {'title': 'API Chat', 'chat_user_type': ChatUserType.ANONYMOUS},
            format='json'
        )
        assert response.status_code == 201
        assert response.data['code'] == 201
        assert response.data['data']['title'] == 'API Chat'

    def test_list_chats(self):
        """Test listing chats via API."""
        Chat.create_chat(application=self.application, title='Chat 1')
        Chat.create_chat(application=self.application, title='Chat 2')
        response = self.client.get(self.base_url)
        assert response.status_code == 200
        assert response.data['data']['total'] == 2

    def test_list_chats_excludes_deleted(self):
        """Test that deleted chats are excluded by default."""
        Chat.create_chat(application=self.application, title='Active')
        deleted = Chat.create_chat(application=self.application, title='Deleted')
        deleted.soft_delete()
        response = self.client.get(self.base_url)
        assert response.data['data']['total'] == 1

    def test_list_chats_include_deleted(self):
        """Test including deleted chats."""
        Chat.create_chat(application=self.application, title='Active')
        deleted = Chat.create_chat(application=self.application, title='Deleted')
        deleted.soft_delete()
        response = self.client.get(f'{self.base_url}?include_deleted=true')
        assert response.data['data']['total'] == 2

    def test_get_chat(self):
        """Test getting a chat by ID."""
        chat = Chat.create_chat(application=self.application, title='Get Me')
        response = self.client.get(f'{self.base_url}/{chat.id}')
        assert response.status_code == 200
        assert response.data['data']['title'] == 'Get Me'

    def test_update_chat(self):
        """Test updating a chat."""
        chat = Chat.create_chat(application=self.application, title='Original')
        response = self.client.put(
            f'{self.base_url}/{chat.id}',
            {'title': 'Updated'},
            format='json'
        )
        assert response.status_code == 200
        assert response.data['data']['title'] == 'Updated'

    def test_delete_chat_soft(self):
        """Test soft deleting a chat."""
        chat = Chat.create_chat(application=self.application)
        response = self.client.delete(f'{self.base_url}/{chat.id}')
        assert response.status_code == 200
        chat.refresh_from_db()
        assert chat.is_deleted is True

    def test_delete_chat_hard(self):
        """Test hard deleting a chat."""
        chat = Chat.create_chat(application=self.application)
        chat_id = chat.id
        response = self.client.delete(f'{self.base_url}/{chat.id}?hard=true')
        assert response.status_code == 200
        assert not Chat.objects.filter(id=chat_id).exists()


class TestChatRecordAPI(TestCase):
    """Tests for ChatRecord API endpoints."""

    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create(
            username='recorduser',
            password='hashedpassword',
            role='USER'
        )
        self.application = Application.create_application(
            name='Record Test App',
            workspace_id='ws-record',
            user=self.user
        )
        self.chat = Chat.create_chat(application=self.application)
        self.base_url = (
            f'/api/workspace/ws-record/application/{self.application.id}'
            f'/chat/{self.chat.id}/record'
        )

    def test_create_record(self):
        """Test creating a chat record via API."""
        response = self.client.post(
            self.base_url,
            {
                'problem_text': 'What is Django?',
                'answer_text': 'Django is a web framework.'
            },
            format='json'
        )
        assert response.status_code == 201
        assert response.data['data']['problem_text'] == 'What is Django?'

    def test_list_records(self):
        """Test listing chat records."""
        ChatRecord.create_record(chat=self.chat, problem_text='Q1')
        ChatRecord.create_record(chat=self.chat, problem_text='Q2')
        response = self.client.get(self.base_url)
        assert response.status_code == 200
        assert len(response.data['data']) == 2

    def test_get_record(self):
        """Test getting a specific record."""
        record = ChatRecord.create_record(chat=self.chat, problem_text='Q')
        response = self.client.get(f'{self.base_url}/{record.id}')
        assert response.status_code == 200
        assert response.data['data']['problem_text'] == 'Q'

    def test_update_record_answer(self):
        """Test updating record with an answer."""
        record = ChatRecord.create_record(chat=self.chat, problem_text='Q')
        response = self.client.put(
            f'{self.base_url}/{record.id}',
            {'answer_text': 'A', 'answer_tokens': 10},
            format='json'
        )
        assert response.status_code == 200
        assert response.data['data']['answer_text'] == 'A'

    def test_vote_upvote(self):
        """Test upvoting a record."""
        record = ChatRecord.create_record(chat=self.chat, problem_text='Q')
        response = self.client.post(
            f'{self.base_url}/{record.id}/vote',
            {'vote_status': 'up'},
            format='json'
        )
        assert response.status_code == 200
        assert response.data['data']['vote_status'] == 'up'

    def test_vote_downvote(self):
        """Test downvoting a record."""
        record = ChatRecord.create_record(chat=self.chat, problem_text='Q')
        response = self.client.post(
            f'{self.base_url}/{record.id}/vote',
            {'vote_status': 'down'},
            format='json'
        )
        assert response.status_code == 200
        assert response.data['data']['vote_status'] == 'down'

    def test_vote_clear(self):
        """Test clearing vote."""
        record = ChatRecord.create_record(chat=self.chat, problem_text='Q')
        record.upvote()
        response = self.client.post(
            f'{self.base_url}/{record.id}/vote',
            {'vote_status': 'none'},
            format='json'
        )
        assert response.status_code == 200
        assert response.data['data']['vote_status'] == 'none'


class TestChatMessageAPI(TestCase):
    """Tests for ChatMessage API endpoints."""

    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create(
            username='msguser',
            password='hashedpassword',
            role='USER'
        )
        self.application = Application.create_application(
            name='Message Test App',
            workspace_id='ws-msg',
            user=self.user
        )
        self.chat = Chat.create_chat(application=self.application)
        self.base_url = (
            f'/api/workspace/ws-msg/application/{self.application.id}'
            f'/chat/{self.chat.id}/message'
        )

    def test_create_message(self):
        """Test creating a chat message via API."""
        response = self.client.post(
            self.base_url,
            {'role': 'user', 'content': 'Hello!'},
            format='json'
        )
        assert response.status_code == 201
        assert response.data['data']['content'] == 'Hello!'
        assert response.data['data']['role'] == 'user'

    def test_list_messages(self):
        """Test listing chat messages."""
        ChatMessage.create_message(
            chat=self.chat, role=MessageRole.USER, content='Hi'
        )
        ChatMessage.create_message(
            chat=self.chat, role=MessageRole.ASSISTANT, content='Hello'
        )
        response = self.client.get(self.base_url)
        assert response.status_code == 200
        assert len(response.data['data']) == 2

    def test_list_messages_with_limit(self):
        """Test listing messages with limit."""
        for i in range(5):
            ChatMessage.create_message(
                chat=self.chat, role=MessageRole.USER, content=f'Msg {i}'
            )
        response = self.client.get(f'{self.base_url}?limit=3')
        assert len(response.data['data']) == 3

    def test_clear_messages(self):
        """Test clearing chat history."""
        ChatMessage.create_message(
            chat=self.chat, role=MessageRole.USER, content='Hi'
        )
        ChatMessage.create_message(
            chat=self.chat, role=MessageRole.ASSISTANT, content='Hello'
        )
        response = self.client.delete(self.base_url)
        assert response.status_code == 200
        assert ChatMessage.objects.filter(chat=self.chat).count() == 0

    def test_get_formatted_history(self):
        """Test getting formatted history via API."""
        ChatMessage.create_message(
            chat=self.chat, role=MessageRole.USER, content='Hello'
        )
        ChatMessage.create_message(
            chat=self.chat, role=MessageRole.ASSISTANT, content='Hi!'
        )
        history_url = (
            f'/api/workspace/ws-msg/application/{self.application.id}'
            f'/chat/{self.chat.id}/history'
        )
        response = self.client.get(history_url)
        assert response.status_code == 200
        assert len(response.data['data']) == 2
        assert response.data['data'][0] == {'role': 'user', 'content': 'Hello'}


class TestChatUserTypes(TestCase):
    """Tests for different chat user types."""

    def setUp(self):
        self.user = User.objects.create(
            username='typeuser',
            password='hashedpassword',
            role='USER'
        )
        self.application = Application.create_application(
            name='Type Test App',
            workspace_id='ws-type',
            user=self.user
        )

    def test_anonymous_user(self):
        """Test anonymous chat user type."""
        chat = Chat.create_chat(
            application=self.application,
            chat_user_type=ChatUserType.ANONYMOUS
        )
        assert chat.chat_user_type == 'ANONYMOUS'

    def test_registered_user(self):
        """Test registered chat user type."""
        chat = Chat.create_chat(
            application=self.application,
            user=self.user,
            chat_user_type=ChatUserType.REGISTERED
        )
        assert chat.chat_user_type == 'REGISTERED'

    def test_api_key_user(self):
        """Test API key chat user type."""
        chat = Chat.create_chat(
            application=self.application,
            chat_user_id='api-key-abc123',
            chat_user_type=ChatUserType.API_KEY
        )
        assert chat.chat_user_type == 'API_KEY'
        assert chat.chat_user_id == 'api-key-abc123'


class TestMessageRoles(TestCase):
    """Tests for different message roles."""

    def setUp(self):
        self.user = User.objects.create(
            username='roleuser',
            password='hashedpassword',
            role='USER'
        )
        self.application = Application.create_application(
            name='Role Test App',
            workspace_id='ws-role',
            user=self.user
        )
        self.chat = Chat.create_chat(application=self.application)

    def test_system_role(self):
        """Test system message role."""
        msg = ChatMessage.create_message(
            chat=self.chat,
            role=MessageRole.SYSTEM,
            content='You are helpful.'
        )
        assert msg.role == 'system'

    def test_user_role(self):
        """Test user message role."""
        msg = ChatMessage.create_message(
            chat=self.chat,
            role=MessageRole.USER,
            content='Hello'
        )
        assert msg.role == 'user'

    def test_assistant_role(self):
        """Test assistant message role."""
        msg = ChatMessage.create_message(
            chat=self.chat,
            role=MessageRole.ASSISTANT,
            content='Hi!'
        )
        assert msg.role == 'assistant'


class TestChatCascadeDelete(TestCase):
    """Tests for cascade delete behavior."""

    def setUp(self):
        self.user = User.objects.create(
            username='cascadeuser',
            password='hashedpassword',
            role='USER'
        )
        self.application = Application.create_application(
            name='Cascade Test App',
            workspace_id='ws-cascade',
            user=self.user
        )

    def test_chat_deletes_records(self):
        """Test that deleting a chat deletes its records."""
        chat = Chat.create_chat(application=self.application)
        ChatRecord.create_record(chat=chat, problem_text='Q1')
        ChatRecord.create_record(chat=chat, problem_text='Q2')
        assert ChatRecord.objects.filter(chat=chat).count() == 2
        chat.delete()
        assert ChatRecord.objects.filter(chat_id=chat.id).count() == 0

    def test_chat_deletes_messages(self):
        """Test that deleting a chat deletes its messages."""
        chat = Chat.create_chat(application=self.application)
        ChatMessage.create_message(chat=chat, role='user', content='Hi')
        ChatMessage.create_message(chat=chat, role='assistant', content='Hello')
        assert ChatMessage.objects.filter(chat=chat).count() == 2
        chat.delete()
        assert ChatMessage.objects.filter(chat_id=chat.id).count() == 0

    def test_application_deletes_chats(self):
        """Test that deleting an application deletes its chats."""
        chat1 = Chat.create_chat(application=self.application)
        chat2 = Chat.create_chat(application=self.application)
        ChatRecord.create_record(chat=chat1, problem_text='Q1')
        ChatMessage.create_message(chat=chat2, role='user', content='Hi')

        self.application.delete()
        assert Chat.objects.filter(application_id=self.application.id).count() == 0
        assert ChatRecord.objects.filter(chat__application_id=self.application.id).count() == 0
        assert ChatMessage.objects.filter(chat__application_id=self.application.id).count() == 0
