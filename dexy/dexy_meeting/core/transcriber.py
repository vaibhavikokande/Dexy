import whisper
import pyaudio
import wave
import threading
import queue
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable
import numpy as np

from config.settings import Settings

logger = logging.getLogger(__name__)

class AudioTranscriber:
    def __init__(self):
        self.settings = Settings()
        self.model = whisper.load_model(self.settings.WHISPER_MODEL)
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.transcript_callback: Optional[Callable] = None
        self.audio_thread = None
        self.transcription_thread = None
        
    def set_transcript_callback(self, callback: Callable[[str], None]):
        """Set callback function to receive transcription updates"""
        self.transcript_callback = callback
        
    def start_recording(self, callback: Optional[Callable] = None):
        """Start real-time audio recording and transcription"""
        if self.is_recording:
            logger.warning("Recording already in progress")
            return
            
        if callback:
            self.transcript_callback = callback
            
        self.is_recording = True
        
        # Start audio recording thread
        self.audio_thread = threading.Thread(target=self._record_audio)
        self.audio_thread.daemon = True
        self.audio_thread.start()
        
        # Start transcription thread
        self.transcription_thread = threading.Thread(target=self._process_audio)
        self.transcription_thread.daemon = True
        self.transcription_thread.start()
        
        logger.info("Started real-time transcription")
        
    def stop_recording(self):
        """Stop recording and transcription"""
        if not self.is_recording:
            return
            
        self.is_recording = False
        
        # Wait for threads to finish
        if self.audio_thread:
            self.audio_thread.join(timeout=5)
        if self.transcription_thread:
            self.transcription_thread.join(timeout=5)
            
        logger.info("Stopped real-time transcription")
        
    def _record_audio(self):
        """Record audio in chunks"""
        try:
            audio = pyaudio.PyAudio()
            
            # Audio stream configuration
            stream = audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.settings.SAMPLE_RATE,
                input=True,
                frames_per_buffer=self.settings.CHUNK_SIZE
            )
            
            logger.info("Audio recording started")
            
            frames = []
            chunk_duration = 30  # Process every 30 seconds
            chunks_per_duration = int(self.settings.SAMPLE_RATE * chunk_duration / self.settings.CHUNK_SIZE)
            
            while self.is_recording:
                try:
                    data = stream.read(self.settings.CHUNK_SIZE, exception_on_overflow=False)
                    frames.append(data)
                    
                    # Process accumulated audio every 30 seconds
                    if len(frames) >= chunks_per_duration:
                        audio_data = b''.join(frames)
                        self.audio_queue.put(audio_data)
                        frames = []
                        
                except Exception as e:
                    logger.error(f"Error reading audio: {e}")
                    break
                    
            # Process remaining audio
            if frames:
                audio_data = b''.join(frames)
                self.audio_queue.put(audio_data)
                
            stream.stop_stream()
            stream.close()
            audio.terminate()
            
        except Exception as e:
            logger.error(f"Audio recording error: {e}")
            
    def _process_audio(self):
        """Process audio chunks for transcription"""
        while self.is_recording or not self.audio_queue.empty():
            try:
                # Get audio data with timeout
                try:
                    audio_data = self.audio_queue.get(timeout=1)
                except queue.Empty:
                    continue
                    
                # Save audio chunk temporarily
                temp_file = self.settings.AUDIO_DIR / f"temp_chunk_{int(time.time())}.wav"
                self._save_audio_chunk(audio_data, temp_file)
                
                # Transcribe audio
                transcript = self._transcribe_audio(temp_file)
                
                # Clean up temp file
                temp_file.unlink(missing_ok=True)
                
                # Send transcript to callback
                if transcript and self.transcript_callback:
                    self.transcript_callback(transcript)
                    
            except Exception as e:
                logger.error(f"Audio processing error: {e}")
                
    def _save_audio_chunk(self, audio_data: bytes, file_path: Path):
        """Save audio chunk to file"""
        try:
            with wave.open(str(file_path), 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.settings.SAMPLE_RATE)
                wav_file.writeframes(audio_data)
        except Exception as e:
            logger.error(f"Error saving audio chunk: {e}")
            
    def _transcribe_audio(self, audio_file: Path) -> str:
        """Transcribe audio file using Whisper"""
        try:
            result = self.model.transcribe(str(audio_file))
            return result["text"].strip()
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""
            
    def transcribe_file(self, audio_file: Path) -> str:
        """Transcribe a complete audio file"""
        try:
            logger.info(f"Transcribing file: {audio_file}")
            result = self.model.transcribe(str(audio_file))
            transcript = result["text"].strip()
            
            # Save transcript
            transcript_file = self.settings.TRANSCRIPT_DIR / f"{audio_file.stem}_transcript.txt"
            with open(transcript_file, 'w', encoding='utf-8') as f:
                f.write(transcript)
                
            logger.info(f"Transcript saved to: {transcript_file}")
            return transcript
            
        except Exception as e:
            logger.error(f"File transcription error: {e}")
            return ""
            
    def save_meeting_audio(self, meeting_id: str) -> Path:
        """Save complete meeting audio"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_file = self.settings.AUDIO_DIR / f"meeting_{meeting_id}_{timestamp}.wav"
        
        # This would be implemented to save the complete meeting audio
        # For now, return the expected path
        return audio_file
        
    def get_audio_level(self) -> float:
        """Get current audio input level (for UI feedback)"""
        try:
            audio = pyaudio.PyAudio()
            stream = audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.settings.SAMPLE_RATE,
                input=True,
                frames_per_buffer=self.settings.CHUNK_SIZE
            )
            
            data = stream.read(self.settings.CHUNK_SIZE)
            stream.close()
            audio.terminate()
            
            # Calculate RMS (Root Mean Square) for volume level
            audio_data = np.frombuffer(data, dtype=np.int16)
            rms = np.sqrt(np.mean(audio_data**2))
            
            # Normalize to 0-1 range
            return min(rms / 32768.0, 1.0)
            
        except Exception as e:
            logger.error(f"Error getting audio level: {e}")
            return 0.0
            
    def test_microphone(self) -> bool:
        """Test if microphone is working"""
        try:
            audio = pyaudio.PyAudio()
            stream = audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.settings.SAMPLE_RATE,
                input=True,
                frames_per_buffer=self.settings.CHUNK_SIZE
            )
            
            # Try to read some audio
            data = stream.read(self.settings.CHUNK_SIZE)
            stream.close()
            audio.terminate()
            
            return len(data) > 0
            
        except Exception as e:
            logger.error(f"Microphone test failed: {e}")
            return False