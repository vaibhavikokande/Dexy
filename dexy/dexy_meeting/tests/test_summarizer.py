"""
Test suite for the summarizer module
"""

import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import json
from datetime import datetime

# Import the summarizer module
try:
    from core.summarizer import MeetingSummarizer, SummarizerError
except ImportError:
    import sys
    sys.path.append('..')
    from core.summarizer import MeetingSummarizer, SummarizerError


class TestMeetingSummarizer(unittest.TestCase):
    """Test cases for MeetingSummarizer class"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.sample_transcript = {
            'text': 'Hello everyone. Let us discuss the quarterly sales report. The sales have increased by 15% this quarter. We need to focus on customer retention strategies.',
            'segments': [
                {'start': 0.0, 'end': 2.0, 'text': 'Hello everyone.'},
                {'start': 2.0, 'end': 5.0, 'text': 'Let us discuss the quarterly sales report.'},
                {'start': 5.0, 'end': 8.0, 'text': 'The sales have increased by 15% this quarter.'},
                {'start': 8.0, 'end': 12.0, 'text': 'We need to focus on customer retention strategies.'}
            ]
        }
        
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('openai.OpenAI')
    def test_initialize_summarizer(self, mock_openai):
        """Test summarizer initialization"""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        summarizer = MeetingSummarizer()
        
        mock_openai.assert_called_once()
        self.assertEqual(summarizer.client, mock_client)

    @patch('openai.OpenAI')
    def test_summarize_transcript(self, mock_openai):
        """Test basic transcript summarization"""
        # Setup mock
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            'summary': 'Meeting discussed quarterly sales report with 15% increase.',
            'key_points': ['Sales increased by 15%', 'Need customer retention strategies'],
            'action_items': ['Develop customer retention plan'],
            'participants': ['Team members'],
            'topics': ['Quarterly sales', 'Customer retention']
        })
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        summarizer = MeetingSummarizer()
        result = summarizer.summarize_transcript(self.sample_transcript)
        
        # Assertions
        self.assertIsInstance(result, dict)
        self.assertIn('summary', result)
        self.assertIn('key_points', result)
        self.assertIn('action_items', result)
        self.assertIn('participants', result)
        self.assertIn('topics', result)
        
        # Verify OpenAI was called
        mock_client.chat.completions.create.assert_called_once()

    @patch('openai.OpenAI')
    def test_summarize_with_context(self, mock_openai):
        """Test summarization with previous meeting context"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            'summary': 'Follow-up meeting on sales report implementation.',
            'key_points': ['Implementation of previous action items'],
            'action_items': ['Continue monitoring sales'],
            'participants': ['Team members'],
            'topics': ['Sales implementation']
        })
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        previous_context = {
            'summary': 'Previous meeting discussed budget allocation.',
            'action_items': ['Allocate budget for Q4']
        }
        
        summarizer = MeetingSummarizer()
        result = summarizer.summarize_transcript(
            self.sample_transcript, 
            previous_context=previous_context
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('summary', result)
        
        # Verify context was included in the API call
        call_args = mock_client.chat.completions.create.call_args
        self.assertIn('previous_context', str(call_args))

    @patch('openai.OpenAI')
    def test_extract_action_items(self, mock_openai):
        """Test action item extraction"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps([
            {
                'task': 'Develop customer retention plan',
                'assignee': 'Sales Team',
                'due_date': '2024-02-15',
                'priority': 'High'
            },
            {
                'task': 'Prepare quarterly report',
                'assignee': 'Analytics Team',
                'due_date': '2024-02-10',
                'priority': 'Medium'
            }
        ])
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        summarizer = MeetingSummarizer()
        action_items = summarizer.extract_action_items(self.sample_transcript)
        
        self.assertIsInstance(action_items, list)
        self.assertEqual(len(action_items), 2)
        
        # Check structure of action items
        for item in action_items:
            self.assertIn('task', item)
            self.assertIn('assignee', item)
            self.assertIn('due_date', item)
            self.assertIn('priority', item)

    @patch('openai.OpenAI')
    def test_identify_key_topics(self, mock_openai):
        """Test key topic identification"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps([
            {
                'topic': 'Quarterly Sales Performance',
                'importance': 'High',
                'time_spent': '45%',
                'key_discussion_points': ['15% increase', 'Performance metrics']
            },
            {
                'topic': 'Customer Retention',
                'importance': 'High',
                'time_spent': '35%',
                'key_discussion_points': ['Retention strategies', 'Customer satisfaction']
            }
        ])
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        summarizer = MeetingSummarizer()
        topics = summarizer.identify_key_topics(self.sample_transcript)
        
        self.assertIsInstance(topics, list)
        self.assertEqual(len(topics), 2)
        
        # Check structure of topics
        for topic in topics:
            self.assertIn('topic', topic)
            self.assertIn('importance', topic)
            self.assertIn('time_spent', topic)
            self.assertIn('key_discussion_points', topic)

    @patch('openai.OpenAI')
    def test_generate_insights(self, mock_openai):
        """Test insight generation"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            'overall_sentiment': 'Positive',
            'decision_quality': 'High',
            'meeting_effectiveness': 'Good',
            'recommendations': [
                'Continue monitoring sales performance',
                'Implement customer retention strategies quickly'
            ],
            'risks_identified': ['Potential customer churn'],
            'opportunities': ['Expand successful sales strategies']
        })
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        summarizer = MeetingSummarizer()
        insights = summarizer.generate_insights(self.sample_transcript)
        
        self.assertIsInstance(insights, dict)
        self.assertIn('overall_sentiment', insights)
        self.assertIn('decision_quality', insights)
        self.assertIn('meeting_effectiveness', insights)
        self.assertIn('recommendations', insights)
        self.assertIn('risks_identified', insights)
        self.assertIn('opportunities', insights)

    @patch('openai.OpenAI')
    def test_create_comprehensive_summary(self, mock_openai):
        """Test comprehensive summary creation"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            'executive_summary': 'Meeting focused on quarterly sales performance and customer retention.',
            'detailed_summary': 'The team discussed the 15% increase in quarterly sales...',
            'key_decisions': ['Implement customer retention strategies'],
            'next_steps': ['Develop retention plan', 'Monitor sales performance'],
            'meeting_metadata': {
                'duration': '60 minutes',
                'participants_count': 5,
                'topics_covered': 2
            }
        })
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        summarizer = MeetingSummarizer()
        summary = summarizer.create_comprehensive_summary(self.sample_transcript)
        
        self.assertIsInstance(summary, dict)
        self.assertIn('executive_summary', summary)
        self.assertIn('detailed_summary', summary)
        self.assertIn('key_decisions', summary)
        self.assertIn('next_steps', summary)
        self.assertIn('meeting_metadata', summary)

    @patch('openai.OpenAI')
    def test_summarize_with_custom_prompt(self, mock_openai):
        """Test summarization with custom prompt"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            'custom_summary': 'Custom analysis of the meeting transcript.'
        })
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        custom_prompt = "Analyze this meeting transcript for technical decisions only."
        
        summarizer = MeetingSummarizer()
        result = summarizer.summarize_with_custom_prompt(
            self.sample_transcript, 
            custom_prompt
        )
        
        self.assertIsInstance(result, dict)
        
        # Verify custom prompt was used
        call_args = mock_client.chat.completions.create.call_args
        self.assertIn('technical decisions', str(call_args))

    @patch('openai.OpenAI')
    def test_api_error_handling(self, mock_openai):
        """Test API error handling"""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai.return_value = mock_client
        
        summarizer = MeetingSummarizer()
        
        with self.assertRaises(SummarizerError):
            summarizer.summarize_transcript(self.sample_transcript)

    @patch('openai.OpenAI')
    def test_invalid_json_response(self, mock_openai):
        """Test handling of invalid JSON response"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Invalid JSON response"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        summarizer = MeetingSummarizer()
        
        with self.assertRaises(SummarizerError):
            summarizer.summarize_transcript(self.sample_transcript)

    def test_save_summary(self):
        """Test saving summary to file"""
        summary_data = {
            'summary': 'Test meeting summary',
            'key_points': ['Point 1', 'Point 2'],
            'action_items': ['Action 1'],
            'timestamp': datetime.now().isoformat()
        }
        
        output_file = os.path.join(self.temp_dir, 'test_summary.json')
        
        summarizer = MeetingSummarizer()
        summarizer.save_summary(summary_data, output_file)
        
        self.assertTrue(os.path.exists(output_file))
        
        # Verify file content
        with open(output_file, 'r') as f:
            saved_data = json.load(f)
            
        self.assertEqual(saved_data['summary'], 'Test meeting summary')
        self.assertEqual(len(saved_data['key_points']), 2)

    def test_load_summary(self):
        """Test loading summary from file"""
        summary_data = {
            'summary': 'Loaded meeting summary',
            'key_points': ['Loaded point'],
            'timestamp': datetime.now().isoformat()
        }
        
        # Save summary first
        summary_file = os.path.join(self.temp_dir, 'load_test.json')
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f)
        
        summarizer = MeetingSummarizer()
        loaded_data = summarizer.load_summary(summary_file)
        
        self.assertEqual(loaded_data['summary'], 'Loaded meeting summary')
        self.assertEqual(len(loaded_data['key_points']), 1)

    @patch('openai.OpenAI')
    def test_batch_summarization(self, mock_openai):
        """Test batch summarization of multiple transcripts"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            'summary': 'Batch summary',
            'key_points': ['Batch point'],
            'action_items': ['Batch action'],
            'participants': ['Team'],
            'topics': ['Batch topic']
        })
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        transcripts = [self.sample_transcript, self.sample_transcript]
        
        summarizer = MeetingSummarizer()
        results = summarizer.batch_summarize(transcripts)
        
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertIn('summary', result)
            self.assertIn('key_points', result)

    @patch('openai.OpenAI')
    def test_different_summary_styles(self, mock_openai):
        """Test different summary styles"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            'summary': 'Executive style summary',
            'key_points': ['Executive point'],
            'action_items': ['Executive action'],
            'participants': ['Executives'],
            'topics': ['Executive topic']
        })
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        summarizer = MeetingSummarizer()
        
        # Test different styles
        styles = ['executive', 'detailed', 'bullet_points', 'technical']
        
        for style in styles:
            result = summarizer.summarize_transcript(
                self.sample_transcript, 
                style=style
            )
            self.assertIsInstance(result, dict)
            self.assertIn('summary', result)


class TestSummarizerError(unittest.TestCase):
    """Test cases for SummarizerError exception"""

    def test_summarizer_error_creation(self):
        """Test SummarizerError exception creation"""
        error_msg = "Test summarizer error"
        
        with self.assertRaises(SummarizerError) as context:
            raise SummarizerError(error_msg)
        
        self.assertEqual(str(context.exception), error_msg)

    def test_summarizer_error_with_cause(self):
        """Test SummarizerError with underlying cause"""
        original_error = ValueError("Original error")
        
        with self.assertRaises(SummarizerError) as context:
            try:
                raise original_error
            except ValueError as e:
                raise SummarizerError("Summarization failed") from e
        
        self.assertEqual(str(context.exception), "Summarization failed")
        self.assertEqual(context.exception.__cause__, original_error)


if __name__ == '__main__':
    unittest.main()