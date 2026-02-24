import logging
import pyttsx3
from gtts import gTTS
import tempfile
import os
import pygame
from pathlib import Path
from typing import Optional, Dict, List
import threading
import time
from config.settings import AUDIO_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextToSpeech:
    """
    Text-to-Speech functionality using pyttsx3 and gTTS
    """
    
    def __init__(self, engine_type: str = "pyttsx3"):
        """
        Initialize TTS engine
        
        Args:
            engine_type: TTS engine type ('pyttsx3' or 'gtts')
        """
        self.engine_type = engine_type
        self.engine = None
        self.is_speaking = False
        self.speak_thread = None
        
        # Voice settings
        self.voice_rate = 200
        self.voice_volume = 0.8
        self.voice_id = None
        
        self._setup_engine()
    
    def _setup_engine(self):
        """Setup TTS engine"""
        try:
            if self.engine_type == "pyttsx3":
                self._setup_pyttsx3()
            elif self.engine_type == "gtts":
                self._setup_gtts()
            else:
                raise ValueError(f"Unsupported engine type: {self.engine_type}")
                
            logger.info(f"TTS engine initialized: {self.engine_type}")
            
        except Exception as e:
            logger.error(f"Failed to initialize TTS engine: {e}")
            # Fallback to pyttsx3
            if self.engine_type != "pyttsx3":
                self.engine_type = "pyttsx3"
                self._setup_pyttsx3()
    
    def _setup_pyttsx3(self):
        """Setup pyttsx3 engine"""
        try:
            self.engine = pyttsx3.init()
            
            # Set voice properties
            self.engine.setProperty('rate', self.voice_rate)
            self.engine.setProperty('volume', self.voice_volume)
            
            # Get available voices
            voices = self.engine.getProperty('voices')
            if voices:
                # Try to find a female voice
                female_voice = None
                for voice in voices:
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        female_voice = voice
                        break
                
                if female_voice:
                    self.engine.setProperty('voice', female_voice.id)
                    self.voice_id = female_voice.id
                    logger.info(f"Using voice: {female_voice.name}")
                else:
                    # Use first available voice
                    self.engine.setProperty('voice', voices[0].id)
                    self.voice_id = voices[0].id
                    logger.info(f"Using voice: {voices[0].name}")
            
        except Exception as e:
            logger.error(f"Error setting up pyttsx3: {e}")
            raise
    
    def _setup_gtts(self):
        """Setup gTTS engine"""
        try:
            # Initialize pygame mixer for audio playback
            pygame.mixer.init()
            logger.info("gTTS engine ready")
            
        except Exception as e:
            logger.error(f"Error setting up gTTS: {e}")
            raise
    
    def speak(self, text: str, async_mode: bool = False) -> bool:
        """
        Speak text using TTS
        
        Args:
            text: Text to speak
            async_mode: Whether to speak asynchronously
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not text or not text.strip():
                logger.warning("Empty text provided for TTS")
                return False
            
            logger.info(f"Speaking: {text[:50]}...")
            
            if async_mode:
                self.speak_thread = threading.Thread(
                    target=self._speak_sync, 
                    args=(text,)
                )
                self.speak_thread.daemon = True
                self.speak_thread.start()
                return True
            else:
                return self._speak_sync(text)
                
        except Exception as e:
            logger.error(f"Error in TTS speak: {e}")
            return False
    
    def _speak_sync(self, text: str) -> bool:
        """
        Synchronous speak implementation
        
        Args:
            text: Text to speak
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.is_speaking = True
            
            if self.engine_type == "pyttsx3":
                self._speak_pyttsx3(text)
            elif self.engine_type == "gtts":
                self._speak_gtts(text)
            
            self.is_speaking = False
            return True
            
        except Exception as e:
            logger.error(f"Error in synchronous speak: {e}")
            self.is_speaking = False
            return False
    
    def _speak_pyttsx3(self, text: str):
        """Speak using pyttsx3"""
        try:
            self.engine.say(text)
            self.engine.runAndWait()
            
        except Exception as e:
            logger.error(f"Error in pyttsx3 speak: {e}")
            raise
    
    def _speak_gtts(self, text: str):
        """Speak using gTTS"""
        try:
            # Create temporary audio file
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            # Generate speech
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(tmp_path)
            
            # Play audio
            pygame.mixer.music.load(tmp_path)
            pygame.mixer.music.play()
            
            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            
            # Clean up
            os.unlink(tmp_path)
            
        except Exception as e:
            logger.error(f"Error in gTTS speak: {e}")
            raise
    
    def stop_speaking(self):
        """Stop current TTS playback"""
        try:
            if self.engine_type == "pyttsx3" and self.engine:
                self.engine.stop()
            elif self.engine_type == "gtts":
                pygame.mixer.music.stop()
            
            self.is_speaking = False
            logger.info("TTS playback stopped")
            
        except Exception as e:
            logger.error(f"Error stopping TTS: {e}")
    
    def set_voice_properties(self, rate: int = None, volume: float = None):
        """
        Set voice properties
        
        Args:
            rate: Speech rate (words per minute)
            volume: Voice volume (0.0 to 1.0)
        """
        try:
            if self.engine_type == "pyttsx3" and self.engine:
                if rate is not None:
                    self.engine.setProperty('rate', rate)
                    self.voice_rate = rate
                
                if volume is not None:
                    self.engine.setProperty('volume', volume)
                    self.voice_volume = volume
                
                logger.info(f"Voice properties updated: rate={rate}, volume={volume}")
            
        except Exception as e:
            logger.error(f"Error setting voice properties: {e}")
    
    def get_available_voices(self) -> List[Dict]:
        """
        Get available voices
        
        Returns:
            List of available voice information
        """
        try:
            voices_info = []
            
            if self.engine_type == "pyttsx3" and self.engine:
                voices = self.engine.getProperty('voices')
                for voice in voices:
                    voices_info.append({
                        'id': voice.id,
                        'name': voice.name,
                        'gender': 'female' if 'female' in voice.name.lower() else 'male',
                        'age': voice.age if hasattr(voice, 'age') else 'unknown'
                    })
            
            return voices_info
            
        except Exception as e:
            logger.error(f"Error getting available voices: {e}")
            return []
    
    def set_voice(self, voice_id: str) -> bool:
        """
        Set specific voice
        
        Args:
            voice_id: Voice identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.engine_type == "pyttsx3" and self.engine:
                self.engine.setProperty('voice', voice_id)
                self.voice_id = voice_id
                logger.info(f"Voice changed to: {voice_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error setting voice: {e}")
            return False
    
    def save_speech(self, text: str, output_path: str) -> bool:
        """
        Save speech to audio file
        
        Args:
            text: Text to convert to speech
            output_path: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if self.engine_type == "pyttsx3":
                self.engine.save_to_file(text, str(output_path))
                self.engine.runAndWait()
            elif self.engine_type == "gtts":
                tts = gTTS(text=text, lang='en', slow=False)
                tts.save(str(output_path))
            
            logger.info(f"Speech saved to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving speech: {e}")
            return False

class MeetingTTS:
    """
    Specialized TTS for meeting agent responses
    """
    
    def __init__(self):
        """Initialize meeting TTS"""
        self.tts = TextToSpeech(engine_type="pyttsx3")
        self.response_templates = {
            "greeting": "Hello! I'm Dexy, your meeting assistant. How can I help you today?",
            "summarizing": "I'm analyzing the meeting transcript and preparing a summary. Please wait a moment.",
            "summary_ready": "I've completed the meeting summary. Here's what I found:",
            "action_items": "Here are the action items from the meeting:",
            "no_action_items": "I didn't identify any specific action items in this meeting.",
            "error": "I'm sorry, I encountered an error. Please try again.",
            "goodbye": "Thank you for using Dexy. Have a great day!"
        }
    
    def speak_response(self, response_type: str, custom_text: str = None) -> bool:
        """
        Speak meeting agent response
        
        Args:
            response_type: Type of response from templates
            custom_text: Custom text to speak (overrides template)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if custom_text:
                text = custom_text
            else:
                text = self.response_templates.get(response_type, response_type)
            
            return self.tts.speak(text, async_mode=True)
            
        except Exception as e:
            logger.error(f"Error in meeting TTS response: {e}")
            return False
    
    def speak_summary(self, summary_text: str) -> bool:
        """
        Speak meeting summary
        
        Args:
            summary_text: Summary text to speak
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Announce summary
            self.speak_response("summary_ready")
            time.sleep(2)  # Brief pause
            
            # Speak the summary
            return self.tts.speak(summary_text, async_mode=True)
            
        except Exception as e:
            logger.error(f"Error speaking summary: {e}")
            return False
    
    def is_speaking(self) -> bool:
        """Check if TTS is currently speaking"""
        return self.tts.is_speaking
    
    def stop_speaking(self):
        """Stop current TTS playback"""
        self.tts.stop_speaking()

# Test function
def test_tts():
    """Test TTS functionality"""
    print("Testing Text-to-Speech...")
    
    # Test pyttsx3
    tts_pyttsx3 = TextToSpeech(engine_type="pyttsx3")
    print("Testing pyttsx3...")
    tts_pyttsx3.speak("Hello! This is a test of the pyttsx3 text to speech engine.")
    
    time.sleep(2)
    
    # Test gTTS
    try:
        tts_gtts = TextToSpeech(engine_type="gtts")
        print("Testing gTTS...")
        tts_gtts.speak("Hello! This is a test of the Google text to speech engine.")
    except Exception as e:
        print(f"gTTS test failed: {e}")
    
    # Test meeting TTS
    print("Testing meeting TTS...")
    meeting_tts = MeetingTTS()
    meeting_tts.speak_response("greeting")
    time.sleep(3)
    meeting_tts.speak_response("summarizing")
    time.sleep(2)
    meeting_tts.speak_summary("This is a sample meeting summary with key points and action items.")
    
    # Show available voices
    voices = tts_pyttsx3.get_available_voices()
    print(f"Available voices: {len(voices)}")
    for voice in voices[:3]:  # Show first 3 voices
        print(f"  - {voice['name']} ({voice['gender']})")

if __name__ == "__main__":
    test_tts()