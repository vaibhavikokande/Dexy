"""
Test suite for the memory manager module
"""

import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import json
import numpy as np
from datetime import datetime, timedelta

# Import the memory manager module
try:
    from core.memory_manager import MemoryManager, MemoryError
except ImportError:
    import sys
    sys.path.append('..')
    from core.memory_manager import MemoryManager, MemoryError


class TestMemoryManager(unittest.TestCase):
    """Test cases for MemoryManager class"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.memory_path = os.path.join(self.temp_dir, 'test_memory')
        
        # Sample meeting data
        self.sample_meeting = {
            'id': 'meeting_001',
            'date': '2024-01-15',
            'participants': ['John Doe', 'Jane Smith', 'Bob Johnson'],
            'summary': 'Discussed quarterly sales performance and customer retention strategies.',
            'key_points': [
                'Sales increased by 15% this quarter',
                'Customer retention is a priority',
                'New marketing strategies needed'
            ],
            'action_items': [
                {
                    'task': 'Develop customer retention plan',
                    'assignee': 'Jane Smith',
                    'due_date': '2024-02-01'
                }
            ],
            'topics': ['Sales Performance', 'Customer Retention', 'Marketing'],
            'transcript': 'Full meeting transcript text here...'
        }

    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('faiss.IndexFlatL2')
    @patch('sentence_transformers.SentenceTransformer')
    def test_initialize_memory_manager(self, mock_sentence_transformer, mock_faiss):
        """Test memory manager initialization"""
        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model
        mock_index = Mock()
        mock_faiss.return_value = mock_index
        
        memory_manager = MemoryManager(self.memory_path)
        
        mock_sentence_transformer.assert_called_once()
        self.assertEqual(memory_manager.model, mock_model)
        self.assertEqual(memory_manager.memory_path, self.memory_path)

    @patch('faiss.IndexFlatL2')
    @patch('sentence_transformers.SentenceTransformer')
    def test_store_meeting(self, mock_sentence_transformer, mock_faiss):
        """Test storing meeting data"""
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(384)  # Mock embedding
        mock_sentence_transformer.return_value = mock_model
        
        mock_index = Mock()
        mock_faiss.return_value = mock_index
        
        memory_manager = MemoryManager(self.memory_path)
        meeting_id = memory_manager.store_meeting(self.sample_meeting)
        
        self.assertIsInstance(meeting_id, str)
        self.assertTrue(len(meeting_id) > 0)
        
        # Verify embedding was created
        mock_model.encode.assert_called()
        
        # Verify index was updated
        mock_index.add.assert_called_once()

    @patch('faiss.IndexFlatL2')
    @patch('sentence_transformers.SentenceTransformer')
    def test_retrieve_meeting(self, mock_sentence_transformer, mock_faiss):
        """Test retrieving meeting data"""
        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model
        mock_index = Mock()
        mock_faiss.return_value = mock_index
        
        memory_manager = MemoryManager(self.memory_path)
        
        # Store meeting first
        meeting_id = memory_manager.store_meeting(self.sample_meeting)
        
        # Retrieve meeting
        retrieved = memory_manager.retrieve_meeting(meeting_id)
        
        self.assertIsInstance(retrieved, dict)
        self.assertEqual(retrieved['id'], meeting_id)
        self.assertEqual(retrieved['summary'], self.sample_meeting['summary'])

    @patch('faiss.IndexFlatL2')
    @patch('sentence_transformers.SentenceTransformer')
    def test_search_similar_meetings(self, mock_sentence_transformer, mock_faiss):
        """Test searching for similar meetings"""
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(384)
        mock_sentence_transformer.return_value = mock_model
        
        mock_index = Mock()
        mock_index.search.return_value = (
            np.array([[0.1, 0.3, 0.5]]),  # distances
            np.array([[0, 1, 2]])  # indices
        )
        mock_faiss.return_value = mock_index
        
        memory_manager = MemoryManager(self.memory_path)
        
        # Store some meetings
        memory_manager.store_meeting(self.sample_meeting)
        memory_manager.store_meeting({
            **self.sample_meeting,
            'id': 'meeting_002',
            'summary': 'Different meeting about budget planning'
        })
        
        # Search for similar meetings
        query = "sales performance discussion"
        results = memory_manager.search_similar_meetings(query, k=2)
        
        self.assertIsInstance(results, list)
        self.assertLessEqual(len(results), 2)
        
        # Verify search was called
        mock_index.search.assert_called_once()

    @patch('faiss.IndexFlatL2')
    @patch('sentence_transformers.SentenceTransformer')
    def test_get_meeting_context(self, mock_sentence_transformer, mock_faiss):
        """Test getting meeting context"""
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(384)
        mock_sentence_transformer.return_value = mock_model
        
        mock_index = Mock()
        mock_index.search.return_value = (
            np.array([[0.1, 0.2]]),
            np.array([[0, 1]])
        )
        mock_faiss.return_value = mock_index
        
        memory_manager = MemoryManager(self.memory_path)
        
        # Store meetings
        memory_manager.store_meeting(self.sample_meeting)
        memory_manager.store_meeting({
            **self.sample_meeting,
            'id': 'meeting_002',
            'date': '2024-01-08',
            'summary': 'Previous meeting about project planning'
        })
        
        # Get context
        context = memory_manager.get_meeting_context('sales performance')
        
        self.assertIsInstance(context, dict)
        self.assertIn('relevant_meetings', context)
        self.assertIn('common_topics', context)
        self.assertIn('recurring_participants', context)

    @patch('faiss.IndexFlatL2')
    @patch('sentence_transformers.SentenceTransformer')
    def test_get_action_items_history(self, mock_sentence_transformer, mock_faiss):
        """Test getting action items history"""
        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model
        mock_index = Mock()
        mock_faiss.return_value = mock_index
        
        memory_manager = MemoryManager(self.memory_path)
        
        # Store meeting with action items
        memory_manager.store_meeting(self.sample_meeting)
        
        # Get action items history
        history = memory_manager.get_action_items_history('Jane Smith')
        
        self.assertIsInstance(history, list)
        if history:
            self.assertIn('task', history[0])
            self.assertIn('assignee', history[0])

    @patch('faiss.IndexFlatL2')
    @patch('sentence_transformers.SentenceTransformer')
    def test_get_participant_history(self, mock_sentence_transformer, mock_faiss):
        """Test getting participant history"""
        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model
        mock_index = Mock()
        mock_faiss.return_value = mock_index
        
        memory_manager = MemoryManager(self.memory_path)
        
        # Store meeting
        memory_manager.store_meeting(self.sample_meeting)
        
        # Get participant history
        history = memory_manager.get_participant_history('John Doe')
        
        self.assertIsInstance(history, list)
        if history:
            self.assertIn('participants', history[0])
            self.assertIn('John Doe', history[0]['participants'])

    @patch('faiss.IndexFlatL2')
    @patch('sentence_transformers.SentenceTransformer')
    def test_get_topic_trends(self, mock_sentence_transformer, mock_faiss):
        """Test getting topic trends"""
        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model
        mock_index = Mock()
        mock_faiss.return_value = mock_index
        
        memory_manager = MemoryManager(self.memory_path)
        
        # Store multiple meetings with different topics
        for i in range(3):
            meeting_data = {
                **self.sample_meeting,
                'id': f'meeting_{i:03d}',
                'date': (datetime.now() - timedelta(days=i*7)).strftime('%Y-%m-%d'),
                'topics': ['Sales Performance', 'Customer Retention'] if i < 2 else ['Budget Planning']
            }
            memory_manager.store_meeting(meeting_data)
        
        # Get topic trends
        trends = memory_manager.get_topic_trends()
        
        self.assertIsInstance(trends, dict)
        self.assertIn('Sales Performance', trends)
        self.assertIn('Customer Retention', trends)

    @patch('faiss.IndexFlatL2')
    @patch('sentence_transformers.SentenceTransformer')
    def test_update_meeting(self, mock_sentence_transformer, mock_faiss):
        """Test updating meeting data"""
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(384)
        mock_sentence_transformer.return_value = mock_model
        mock_index = Mock()
        mock_faiss.return_value = mock_index
        
        memory_manager = MemoryManager(self.memory_path)
        
        # Store meeting
        meeting_id = memory_manager.store_meeting(self.sample_meeting)
        
        # Update meeting
        updated_data = {
            **self.sample_meeting,
            'summary': 'Updated summary with new information',
            'key_points': ['Updated point 1', 'Updated point 2']
        }
        
        success = memory_manager.update_meeting(meeting_id, updated_data)
        
        self.assertTrue(success)
        
        # Verify update
        retrieved = memory_manager.retrieve_meeting(meeting_id)
        self.assertEqual(retrieved['summary'], 'Updated summary with new information')

    @patch('faiss.IndexFlatL2')
    @patch('sentence_transformers.SentenceTransformer')
    def test_delete_meeting(self, mock_sentence_transformer, mock_faiss):
        """Test deleting meeting data"""
        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model
        mock_index = Mock()
        mock_faiss.return_value = mock_index
        
        memory_manager = MemoryManager(self.memory_path)
        
        # Store meeting
        meeting_id = memory_manager.store_meeting(self.sample_meeting)
        
        # Delete meeting
        success = memory_manager.delete_meeting(meeting_id)
        
        self.assertTrue(success)
        
        # Verify deletion
        retrieved = memory_manager.retrieve_meeting(meeting_id)
        self.assertIsNone(retrieved)

    @patch('faiss.IndexFlatL2')
    @patch('sentence_transformers.SentenceTransformer')
    def test_save_and_load_memory(self, mock_sentence_transformer, mock_faiss):
        """Test saving and loading memory state"""
        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model
        mock_index = Mock()
        mock_faiss.return_value = mock_index
        
        memory_manager = MemoryManager(self.memory_path)
        
        # Store meeting
        meeting_id = memory_manager.store_meeting(self.sample_meeting)
        
        # Save memory
        memory_manager.save_memory()
        
        # Create new memory manager and load
        new_memory_manager = MemoryManager(self.memory_path)
        new_memory_manager.load_memory()
        
        # Verify data is loaded
        retrieved = new_memory_manager.retrieve_meeting(meeting_id)
        self.assertIsNotNone(retrieved)

    @patch('faiss.IndexFlatL2')
    @patch('sentence_transformers.SentenceTransformer')
    def test_get_memory_statistics(self, mock_sentence_transformer, mock_faiss):
        """Test getting memory statistics"""
        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model
        mock_index = Mock()
        mock_faiss.return_value = mock_index
        
        memory_manager = MemoryManager(self.memory_path)
        
        # Store meetings
        memory_manager.store_meeting(self.sample_meeting)
        memory_manager.store_meeting({
            **self.sample_meeting,
            'id': 'meeting_002',
            'participants': ['Alice', 'Bob']
        })
        
        # Get statistics
        stats = memory_manager.get_memory_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('total_meetings', stats)
        self.assertIn('unique_participants', stats)
        self.assertIn('total_action_items', stats)
        self.assertIn('common_topics', stats)

    @patch('faiss.IndexFlatL2')
    @patch('sentence_transformers.SentenceTransformer')
    def test_clear_memory(self, mock_sentence_transformer, mock_faiss):
        """Test clearing memory"""
        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model
        mock_index = Mock()
        mock_faiss.return_value = mock_index
        
        memory_manager = MemoryManager(self.memory_path)
        
        # Store meeting
        memory_manager.store_meeting(self.sample_meeting)
        
        # Clear memory
        memory_manager.clear_memory()
        
        # Verify memory is cleared
        stats = memory_manager.get_memory_statistics()
        self.assertEqual(stats['total_meetings'], 0)

    @patch('faiss.IndexFlatL2')
    @patch('sentence_transformers.SentenceTransformer')
    def test_memory_error_handling(self, mock_sentence_transformer, mock_faiss):
        """Test memory error handling"""
        mock_model = Mock()
        mock_model.encode.side_effect = Exception("Encoding error")
        mock_sentence_transformer.return_value = mock_model
        mock_index = Mock()
        mock_faiss.return_value = mock_index
        
        memory_manager = MemoryManager(self.memory_path)
        
        with self.assertRaises(MemoryError):
            memory_manager.store_meeting(self.sample_meeting)

    @patch('faiss.IndexFlatL2')
    @patch('sentence_transformers.SentenceTransformer')
    def test_retrieve_nonexistent_meeting(self, mock_sentence_transformer, mock_faiss):
        """Test retrieving non-existent meeting"""
        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model
        mock_index = Mock()
        mock_faiss.return_value = mock_index
        
        memory_manager = MemoryManager(self.memory_path)
        
        # Try to retrieve non-existent meeting
        result = memory_manager.retrieve_meeting('nonexistent_id')
        
        self.assertIsNone(result)

    @patch('faiss.IndexFlatL2')
    @patch('sentence_transformers.SentenceTransformer')
    def test_search_with_filters(self, mock_sentence_transformer, mock_faiss):
        """Test searching with filters"""
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(384)
        mock_sentence_transformer.return_value = mock_model
        
        mock_index = Mock()
        mock_index.search.return_value = (
            np.array([[0.1, 0.2]]),
            np.array([[0, 1]])
        )
        mock_faiss.return_value = mock_index
        
        memory_manager = MemoryManager(self.memory_path)
        
        # Store meetings
        memory_manager.store_meeting(self.sample_meeting)
        memory_manager.store_meeting({
            **self.sample_meeting,
            'id': 'meeting_002',
            'date': '2024-01-20',
            'participants': ['Alice', 'Bob']
        })
        
        # Search with filters
        filters = {
            'date_range': ('2024-01-01', '2024-01-31'),
            'participants': ['John Doe'],
            'topics': ['Sales Performance']
        }
        
        results = memory_manager.search_similar_meetings(
            'sales discussion', 
            k=5, 
            filters=filters
        )
        
        self.assertIsInstance(results, list)

    @patch('faiss.IndexFlatL2')
    @patch('sentence_transformers.SentenceTransformer')
    def test_backup_and_restore(self, mock_sentence_transformer, mock_faiss):
        """Test memory backup and restore"""
        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model
        mock_index = Mock()
        mock_faiss.return_value = mock_index
        
        memory_manager = MemoryManager(self.memory_path)
        
        # Store meeting
        meeting_id = memory_manager.store_meeting(self.sample_meeting)
        
        # Create backup
        backup_path = os.path.join(self.temp_dir, 'backup')
        memory_manager.create_backup(backup_path)
        
        # Clear memory
        memory_manager.clear_memory()
        
        # Restore from backup
        memory_manager.restore_from_backup(backup_path)
        
        # Verify restoration
        retrieved = memory_manager.retrieve_meeting(meeting_id)
        self.assertIsNotNone(retrieved)


class TestMemoryError(unittest.TestCase):
    """Test cases for MemoryError exception"""

    def test_memory_error_creation(self):
        """Test MemoryError exception creation"""
        error_msg = "Test memory error"
        
        with self.assertRaises(MemoryError) as context:
            raise MemoryError(error_msg)
        
        self.assertEqual(str(context.exception), error_msg)

    def test_memory_error_with_cause(self):
        """Test MemoryError with underlying cause"""
        original_error = ValueError("Original error")
        
        with self.assertRaises(MemoryError) as context:
            try:
                raise original_error
            except ValueError as e:
                raise MemoryError("Memory operation failed") from e
        
        self.assertEqual(str(context.exception), "Memory operation failed")
        self.assertEqual(context.exception.__cause__, original_error)


if __name__ == '__main__':
    unittest.main()