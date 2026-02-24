"""
Test suite for the transcriber module
"""

import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import numpy as np
from io import BytesIO

# Import the transcriber module
try:
    from core.transcriber import AudioTranscriber, TranscriptionError
except ImportError:
    # Fallback for testing
    import sys
    sys.path.append('..')
    from core.transcriber import AudioTranscriber, TranscriptionError


class TestAudioTranscriber(unittest.TestCase):
    """Test cases for AudioTranscriber class"""

    def setUp(self):
        """Set up test fixtures"""
        self.transcriber = AudioTranscriber()
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_mock_audio_file(self, duration=5):
        """Create a mock audio file for testing"""
        # Create a temporary WAV file
        temp_file = os.path.join(self.temp_dir, 'test_audio.wav')
        
        # Generate simple sine wave audio data
        sample_rate = 16000
        samples = int(sample_rate * duration)
        audio_data = np.sin(2 * np.pi * 440 * np.linspace(0, duration, samples))
        
        # Convert to 16-bit PCM
        audio_data = (audio_data * 32767).astype(np.int16)
        
        # Create a simple WAV file
        with open(temp_file, 'wb') as f:
            # WAV header
            f.write(b'RIFF')
            f.write((36 + len(audio_data) * 2).to_bytes(4, 'little'))
            f.write(b'WAVE')
            f.write(b'fmt ')
            f.write((16).to_bytes(4, 'little'))
            f.write((1).to_bytes(2, 'little'))  # PCM
            f.write((1).to_bytes(2, 'little'))  # mono
            f.write(sample_rate.to_bytes(4, 'little'))
            f.write((sample_rate * 2).to_bytes(4, 'little'))
            f.write((2).to_bytes(2, 'little'))
            f.write((16).to_bytes(2, 'little'))
            f.write(b'data')
            f.write((len(audio_data) * 2).to_bytes(4, 'little'))
            f.write(audio_data.tobytes())
            
        return temp_file

    @patch('whisper.load_model')
    def test_initialize_transcriber(self, mock_load_model):
        """Test transcriber initialization"""
        mock_model = Mock()
        mock_load_model.return_value = mock_model
        
        transcriber = AudioTranscriber(model_size='base')
        
        mock_load_model.assert_called_once_with('base')
        self.assertEqual(transcriber.model, mock_model)

    @patch('whisper.load_model')
    def test_transcribe_audio_file(self, mock_load_model):
        """Test audio file transcription"""
        # Setup mock
        mock_model = Mock()
        mock_model.transcribe.return_value = {
            'text': 'Hello, this is a test transcription.',
            'segments': [
                {'start': 0.0, 'end': 2.0, 'text': 'Hello, this is a test'},
                {'start': 2.0, 'end': 4.0, 'text': 'transcription.'}
            ]
        }
        mock_load_model.return_value = mock_model
        
        # Create test audio file
        audio_file = self.create_mock_audio_file()
        
        # Test transcription
        transcriber = AudioTranscriber()
        result = transcriber.transcribe_file(audio_file)
        
        # Assertions
        self.assertIsInstance(result, dict)
        self.assertIn('text', result)
        self.assertIn('segments', result)
        self.assertEqual(result['text'], 'Hello, this is a test transcription.')
        mock_model.transcribe.assert_called_once_with(audio_file)

    @patch('whisper.load_model')
    def test_transcribe_with_timestamps(self, mock_load_model):
        """Test transcription with timestamps"""
        mock_model = Mock()
        mock_model.transcribe.return_value = {
            'text': 'Meeting started at 9 AM.',
            'segments': [
                {'start': 0.0, 'end': 3.0, 'text': 'Meeting started at 9 AM.'}
            ]
        }
        mock_load_model.return_value = mock_model
        
        audio_file = self.create_mock_audio_file()
        transcriber = AudioTranscriber()
        
        result = transcriber.transcribe_file(audio_file, include_timestamps=True)
        
        self.assertIn('segments', result)
        self.assertTrue(len(result['segments']) > 0)
        self.assertIn('start', result['segments'][0])
        self.assertIn('end', result['segments'][0])

    @patch('whisper.load_model')
    def test_transcribe_nonexistent_file(self, mock_load_model):
        """Test transcription of non-existent file"""
        mock_load_model.return_value = Mock()
        
        transcriber = AudioTranscriber()
        
        with self.assertRaises(TranscriptionError):
            transcriber.transcribe_file('/nonexistent/file.wav')

    @patch('whisper.load_model')
    def test_transcribe_empty_audio(self, mock_load_model):
        """Test transcription of empty audio"""
        mock_model = Mock()
        mock_model.transcribe.return_value = {
            'text': '',
            'segments': []
        }
        mock_load_model.return_value = mock_model
        
        audio_file = self.create_mock_audio_file(duration=0.1)
        transcriber = AudioTranscriber()
        
        result = transcriber.transcribe_file(audio_file)
        
        self.assertEqual(result['text'], '')
        self.assertEqual(len(result['segments']), 0)

    @patch('whisper.load_model')
    @patch('pyaudio.PyAudio')
    def test_real_time_transcription(self, mock_pyaudio, mock_load_model):
        """Test real-time transcription"""
        mock_model = Mock()
        mock_model.transcribe.return_value = {
            'text': 'Real-time transcription test',
            'segments': []
        }
        mock_load_model.return_value = mock_model
        
        # Mock PyAudio
        mock_audio = Mock()
        mock_stream = Mock()
        mock_stream.read.return_value = b'\x00' * 1024  # Mock audio data
        mock_audio.open.return_value = mock_stream
        mock_pyaudio.return_value = mock_audio
        
        transcriber = AudioTranscriber()
        
        # Test real-time transcription (simulate for testing)
        result = transcriber.transcribe_realtime(duration=1)
        
        self.assertIsInstance(result, dict)
        self.assertIn('text', result)

    @patch('whisper.load_model')
    def test_language_detection(self, mock_load_model):
        """Test language detection"""
        mock_model = Mock()
        mock_model.transcribe.return_value = {
            'text': 'Bonjour, comment allez-vous?',
            'language': 'fr',
            'segments': []
        }
        mock_load_model.return_value = mock_model
        
        audio_file = self.create_mock_audio_file()
        transcriber = AudioTranscriber()
        
        result = transcriber.transcribe_file(audio_file, language='auto')
        
        self.assertEqual(result['language'], 'fr')

    @patch('whisper.load_model')
    def test_transcribe_with_speaker_diarization(self, mock_load_model):
        """Test transcription with speaker diarization"""
        mock_model = Mock()
        mock_model.transcribe.return_value = {
            'text': 'Speaker 1: Hello. Speaker 2: Hi there.',
            'segments': [
                {'start': 0.0, 'end': 1.0, 'text': 'Hello.', 'speaker': 'Speaker 1'},
                {'start': 1.0, 'end': 2.0, 'text': 'Hi there.', 'speaker': 'Speaker 2'}
            ]
        }
        mock_load_model.return_value = mock_model
        
        audio_file = self.create_mock_audio_file()
        transcriber = AudioTranscriber()
        
        result = transcriber.transcribe_file(audio_file, enable_speaker_diarization=True)
        
        self.assertIn('segments', result)
        # Check if speaker information is included
        if result['segments']:
            self.assertTrue(any('speaker' in segment for segment in result['segments']))

    @patch('whisper.load_model')
    def test_batch_transcription(self, mock_load_model):
        """Test batch transcription of multiple files"""
        mock_model = Mock()
        mock_model.transcribe.side_effect = [
            {'text': 'First file transcription', 'segments': []},
            {'text': 'Second file transcription', 'segments': []}
        ]
        mock_load_model.return_value = mock_model
        
        # Create multiple test files
        files = [
            self.create_mock_audio_file(),
            self.create_mock_audio_file()
        ]
        
        transcriber = AudioTranscriber()
        results = transcriber.transcribe_batch(files)
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]['text'], 'First file transcription')
        self.assertEqual(results[1]['text'], 'Second file transcription')

    def test_save_transcript(self):
        """Test saving transcript to file"""
        transcript_data = {
            'text': 'Test transcript',
            'segments': [
                {'start': 0.0, 'end': 2.0, 'text': 'Test transcript'}
            ],
            'timestamp': '2024-01-01T12:00:00'
        }
        
        output_file = os.path.join(self.temp_dir, 'test_transcript.json')
        
        transcriber = AudioTranscriber()
        transcriber.save_transcript(transcript_data, output_file)
        
        self.assertTrue(os.path.exists(output_file))
        
        # Verify file content
        import json
        with open(output_file, 'r') as f:
            saved_data = json.load(f)
            
        self.assertEqual(saved_data['text'], 'Test transcript')
        self.assertEqual(len(saved_data['segments']), 1)

    def test_load_transcript(self):
        """Test loading transcript from file"""
        transcript_data = {
            'text': 'Loaded transcript',
            'segments': [],
            'timestamp': '2024-01-01T12:00:00'
        }
        
        # Save transcript first
        import json
        transcript_file = os.path.join(self.temp_dir, 'load_test.json')
        with open(transcript_file, 'w') as f:
            json.dump(transcript_data, f)
        
        transcriber = AudioTranscriber()
        loaded_data = transcriber.load_transcript(transcript_file)
        
        self.assertEqual(loaded_data['text'], 'Loaded transcript')
        self.assertEqual(loaded_data['timestamp'], '2024-01-01T12:00:00')


class TestTranscriptionError(unittest.TestCase):
    """Test cases for TranscriptionError exception"""

    def test_transcription_error_creation(self):
        """Test TranscriptionError exception creation"""
        error_msg = "Test transcription error"
        
        with self.assertRaises(TranscriptionError) as context:
            raise TranscriptionError(error_msg)
        
        self.assertEqual(str(context.exception), error_msg)

    def test_transcription_error_with_cause(self):
        """Test TranscriptionError with underlying cause"""
        original_error = ValueError("Original error")
        
        with self.assertRaises(TranscriptionError) as context:
            try:
                raise original_error
            except ValueError as e:
                raise TranscriptionError("Transcription failed") from e
        
        self.assertEqual(str(context.exception), "Transcription failed")
        self.assertEqual(context.exception.__cause__, original_error)


if __name__ == '__main__':
    unittest.main()