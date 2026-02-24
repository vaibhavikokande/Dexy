import unittest
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import tempfile
import json

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.emailer import EmailSender

class TestEmailSender(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_config = {
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'email_user': 'test@example.com',
            'email_password': 'test_password'
        }
        self.emailer = EmailSender(self.test_config)
        
        # Sample meeting data
        self.sample_meeting_data = {
            'meeting_id': 'test_meeting_123',
            'date': '2024-01-15',
            'time': '14:30',
            'title': 'Project Review Meeting',
            'participants': ['alice@example.com', 'bob@example.com', 'charlie@example.com'],
            'summary': 'Discussed project progress and next steps.',
            'key_points': [
                'Project is on track for Q1 delivery',
                'Need to address performance issues',
                'Budget review scheduled for next week'
            ],
            'action_items': [
                {'task': 'Fix performance issues', 'assignee': 'alice@example.com', 'due_date': '2024-01-20'},
                {'task': 'Prepare budget report', 'assignee': 'bob@example.com', 'due_date': '2024-01-22'}
            ],
            'next_meeting': '2024-01-22 15:00'
        }
    
    def test_init_with_valid_config(self):
        """Test EmailSender initialization with valid configuration."""
        self.assertEqual(self.emailer.smtp_server, 'smtp.gmail.com')
        self.assertEqual(self.emailer.smtp_port, 587)
        self.assertEqual(self.emailer.email_user, 'test@example.com')
        self.assertEqual(self.emailer.email_password, 'test_password')
    
    def test_init_with_missing_config(self):
        """Test EmailSender initialization with missing configuration."""
        incomplete_config = {
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587
            # Missing email_user and email_password
        }
        
        with self.assertRaises(ValueError):
            EmailSender(incomplete_config)
    
    def test_validate_email_addresses_valid(self):
        """Test email validation with valid addresses."""
        valid_emails = ['test@example.com', 'user.name@domain.co.uk', 'admin@company.org']
        
        for email in valid_emails:
            self.assertTrue(self.emailer.validate_email(email))
    
    def test_validate_email_addresses_invalid(self):
        """Test email validation with invalid addresses."""
        invalid_emails = ['invalid.email', '@domain.com', 'user@', 'user@domain', '']
        
        for email in invalid_emails:
            self.assertFalse(self.emailer.validate_email(email))
    
    def test_format_meeting_summary_html(self):
        """Test HTML formatting of meeting summary."""
        html_content = self.emailer.format_meeting_summary(self.sample_meeting_data, format_type='html')
        
        # Check if HTML content contains expected elements
        self.assertIn('<html>', html_content)
        self.assertIn('<h1>Meeting Summary</h1>', html_content)
        self.assertIn('Project Review Meeting', html_content)
        self.assertIn('alice@example.com', html_content)
        self.assertIn('Fix performance issues', html_content)
        self.assertIn('</html>', html_content)
    
    def test_format_meeting_summary_plain(self):
        """Test plain text formatting of meeting summary."""
        plain_content = self.emailer.format_meeting_summary(self.sample_meeting_data, format_type='plain')
        
        # Check if plain text content contains expected elements
        self.assertIn('MEETING SUMMARY', plain_content)
        self.assertIn('Project Review Meeting', plain_content)
        self.assertIn('alice@example.com', plain_content)
        self.assertIn('Fix performance issues', plain_content)
        self.assertNotIn('<html>', plain_content)
        self.assertNotIn('<h1>', plain_content)
    
    def test_create_email_message(self):
        """Test email message creation."""
        recipients = ['alice@example.com', 'bob@example.com']
        subject = 'Test Meeting Summary'
        body = 'This is a test meeting summary.'
        
        message = self.emailer.create_email_message(
            recipients=recipients,
            subject=subject,
            body=body,
            format_type='plain'
        )
        
        self.assertEqual(message['From'], 'test@example.com')
        self.assertEqual(message['To'], 'alice@example.com, bob@example.com')
        self.assertEqual(message['Subject'], subject)
        self.assertIn(body, message.get_payload())
    
    def test_create_email_message_html(self):
        """Test HTML email message creation."""
        recipients = ['alice@example.com']
        subject = 'Test HTML Meeting Summary'
        body = '<h1>Meeting Summary</h1><p>This is a test.</p>'
        
        message = self.emailer.create_email_message(
            recipients=recipients,
            subject=subject,
            body=body,
            format_type='html'
        )
        
        self.assertEqual(message['From'], 'test@example.com')
        self.assertEqual(message['To'], 'alice@example.com')
        self.assertEqual(message['Subject'], subject)
        self.assertEqual(message.get_content_type(), 'text/html')
    
    @patch('smtplib.SMTP')
    def test_send_email_success(self, mock_smtp):
        """Test successful email sending."""
        # Mock SMTP server
        mock_server = Mock()
        mock_smtp.return_value = mock_server
        mock_server.starttls.return_value = None
        mock_server.login.return_value = None
        mock_server.send_message.return_value = {}
        
        recipients = ['alice@example.com', 'bob@example.com']
        subject = 'Test Meeting Summary'
        body = 'This is a test meeting summary.'
        
        result = self.emailer.send_email(
            recipients=recipients,
            subject=subject,
            body=body
        )
        
        self.assertTrue(result)
        mock_smtp.assert_called_once_with('smtp.gmail.com', 587)
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once_with('test@example.com', 'test_password')
        mock_server.send_message.assert_called_once()
        mock_server.quit.assert_called_once()
    
    @patch('smtplib.SMTP')
    def test_send_email_smtp_error(self, mock_smtp):
        """Test email sending with SMTP error."""
        # Mock SMTP server to raise an exception
        mock_smtp.side_effect = Exception("SMTP connection failed")
        
        recipients = ['alice@example.com']
        subject = 'Test Meeting Summary'
        body = 'This is a test meeting summary.'
        
        result = self.emailer.send_email(
            recipients=recipients,
            subject=subject,
            body=body
        )
        
        self.assertFalse(result)
    
    @patch('smtplib.SMTP')
    def test_send_meeting_summary_success(self, mock_smtp):
        """Test sending complete meeting summary."""
        # Mock SMTP server
        mock_server = Mock()
        mock_smtp.return_value = mock_server
        mock_server.starttls.return_value = None
        mock_server.login.return_value = None
        mock_server.send_message.return_value = {}
        
        result = self.emailer.send_meeting_summary(self.sample_meeting_data)
        
        self.assertTrue(result)
        mock_server.send_message.assert_called_once()
    
    def test_send_meeting_summary_invalid_participants(self):
        """Test sending meeting summary with invalid participants."""
        invalid_meeting_data = self.sample_meeting_data.copy()
        invalid_meeting_data['participants'] = ['invalid.email', 'another.invalid']
        
        result = self.emailer.send_meeting_summary(invalid_meeting_data)
        
        self.assertFalse(result)
    
    def test_send_meeting_summary_no_participants(self):
        """Test sending meeting summary with no participants."""
        no_participants_data = self.sample_meeting_data.copy()
        no_participants_data['participants'] = []
        
        result = self.emailer.send_meeting_summary(no_participants_data)
        
        self.assertFalse(result)
    
    @patch('smtplib.SMTP')
    def test_send_custom_email_success(self, mock_smtp):
        """Test sending custom email with attachments."""
        # Mock SMTP server
        mock_server = Mock()
        mock_smtp.return_value = mock_server
        mock_server.starttls.return_value = None
        mock_server.login.return_value = None
        mock_server.send_message.return_value = {}
        
        # Create a temporary file for attachment
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
            tmp_file.write("Test attachment content")
            tmp_file_path = tmp_file.name
        
        try:
            result = self.emailer.send_custom_email(
                recipients=['alice@example.com'],
                subject='Custom Email with Attachment',
                body='This is a custom email.',
                attachments=[tmp_file_path]
            )
            
            self.assertTrue(result)
            mock_server.send_message.assert_called_once()
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)
    
    def test_generate_subject_line(self):
        """Test automatic subject line generation."""
        subject = self.emailer.generate_subject_line(self.sample_meeting_data)
        
        expected_subject = "Meeting Summary: Project Review Meeting - 2024-01-15"
        self.assertEqual(subject, expected_subject)
    
    def test_save_sent_email_log(self):
        """Test saving email send log."""
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'recipients': ['alice@example.com', 'bob@example.com'],
            'subject': 'Test Meeting Summary',
            'status': 'sent',
            'meeting_id': 'test_meeting_123'
        }
        
        # Create a temporary directory for logs
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_file = os.path.join(tmp_dir, 'email_log.json')
            
            result = self.emailer.save_email_log(log_data, log_file)
            
            self.assertTrue(result)
            self.assertTrue(os.path.exists(log_file))
            
            # Verify log content
            with open(log_file, 'r') as f:
                saved_data = json.load(f)
                self.assertEqual(saved_data['meeting_id'], 'test_meeting_123')
                self.assertEqual(saved_data['status'], 'sent')
    
    def test_batch_send_emails(self):
        """Test batch sending of emails to multiple meetings."""
        meetings = [
            self.sample_meeting_data,
            {
                'meeting_id': 'test_meeting_456',
                'date': '2024-01-16',
                'time': '10:00',
                'title': 'Weekly Standup',
                'participants': ['dave@example.com', 'eve@example.com'],
                'summary': 'Weekly team standup.',
                'key_points': ['Sprint progress review'],
                'action_items': [],
                'next_meeting': '2024-01-23 10:00'
            }
        ]
        
        with patch.object(self.emailer, 'send_meeting_summary') as mock_send:
            mock_send.return_value = True
            
            results = self.emailer.batch_send_summaries(meetings)
            
            self.assertEqual(len(results), 2)
            self.assertTrue(all(results))
            self.assertEqual(mock_send.call_count, 2)
    
    def tearDown(self):
        """Clean up after each test."""
        pass

if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)