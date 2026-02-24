"""
Test suite for Dexy Meeting Agent
"""

import os
import sys
import pytest
from unittest.mock import MagicMock

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Test configuration
TEST_CONFIG = {
    'test_audio_path': os.path.join(project_root, 'data', 'audio', 'test_audio.wav'),
    'test_transcript_path': os.path.join(project_root, 'data', 'transcripts', 'test_transcript.txt'),
    'test_summary_path': os.path.join(project_root, 'data', 'summaries', 'test_summary.json'),
    'test_memory_path': os.path.join(project_root, 'data', 'memory', 'test_memory'),
}

# Mock environment variables for testing
os.environ.setdefault('OPENAI_API_KEY', 'test_key')
os.environ.setdefault('EMAIL_PASSWORD', 'test_password')
os.environ.setdefault('EMAIL_ADDRESS', 'test@example.com')

def create_test_directories():
    """Create test directories if they don't exist"""
    for path in TEST_CONFIG.values():
        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)

def cleanup_test_files():
    """Clean up test files after testing"""
    for path in TEST_CONFIG.values():
        if os.path.exists(path):
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                import shutil
                shutil.rmtree(path)

# Create test directories on import
create_test_directories()