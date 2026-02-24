import logging
import pyaudio
import struct
import threading
import time
from typing import Callable, Optional
import pvporcupine
from config.settings import PORCUPINE_ACCESS_KEY, SAMPLE_RATE, CHUNK_SIZE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WakeWordDetector:
    """
    Wake word detection using Picovoice Porcupine
    Detects wake words like "Hey Dexy" to activate the meeting agent
    """
    
    def __init__(self, 
                 access_key: str = PORCUPINE_ACCESS_KEY,
                 keywords: list = None,
                 callback: Optional[Callable] = None):
        """
        Initialize wake word detector
        
        Args:
            access_key: Porcupine access key
            keywords: List of wake words to detect
            callback: Function to call when wake word is detected
        """
        self.access_key = access_key
        self.keywords = keywords or ["hey google", "porcupine"]  # Default keywords
        self.callback = callback
        self.porcupine = None
        self.audio_stream = None
        self.is_listening = False
        self.listen_thread = None
        
        # Audio configuration
        self.sample_rate = SAMPLE_RATE
        self.frame_length = 512
        self.chunk_size = CHUNK_SIZE
        
        self._setup_porcupine()
    
    def _setup_porcupine(self):
        """Setup Porcupine wake word engine"""
        try:
            # Initialize Porcupine
            self.porcupine = pvporcupine.create(
                access_key=self.access_key,
                keywords=self.keywords,
                sensitivities=[0.5] * len(self.keywords)
            )
            
            # Verify audio configuration
            if self.porcupine.sample_rate != self.sample_rate:
                self.sample_rate = self.porcupine.sample_rate
                
            if self.porcupine.frame_length != self.frame_length:
                self.frame_length = self.porcupine.frame_length
                
            logger.info(f"Porcupine initialized with keywords: {self.keywords}")
            logger.info(f"Sample rate: {self.sample_rate}, Frame length: {self.frame_length}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Porcupine: {e}")
            raise
    
    def start_listening(self):
        """Start listening for wake words"""
        if self.is_listening:
            logger.warning("Already listening for wake words")
            return
        
        self.is_listening = True
        self.listen_thread = threading.Thread(target=self._listen_loop)
        self.listen_thread.daemon = True
        self.listen_thread.start()
        logger.info("Started listening for wake words")
    
    def stop_listening(self):
        """Stop listening for wake words"""
        self.is_listening = False
        if self.listen_thread:
            self.listen_thread.join(timeout=1.0)
        self._close_audio_stream()
        logger.info("Stopped listening for wake words")
    
    def _listen_loop(self):
        """Main listening loop"""
        try:
            # Initialize audio stream
            self._setup_audio_stream()
            
            while self.is_listening:
                try:
                    # Read audio data
                    audio_data = self.audio_stream.read(
                        self.frame_length,
                        exception_on_overflow=False
                    )
                    
                    # Convert to 16-bit PCM
                    pcm = struct.unpack_from("h" * self.frame_length, audio_data)
                    
                    # Process audio frame
                    keyword_index = self.porcupine.process(pcm)
                    
                    if keyword_index >= 0:
                        detected_keyword = self.keywords[keyword_index]
                        logger.info(f"Wake word detected: {detected_keyword}")
                        
                        # Call callback if provided
                        if self.callback:
                            self.callback(detected_keyword)
                        
                except Exception as e:
                    logger.error(f"Error in listening loop: {e}")
                    time.sleep(0.1)
                    
        except Exception as e:
            logger.error(f"Failed to start listening: {e}")
        finally:
            self._close_audio_stream()
    
    def _setup_audio_stream(self):
        """Setup audio input stream"""
        try:
            pa = pyaudio.PyAudio()
            
            self.audio_stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.frame_length
            )
            
            logger.info("Audio stream initialized")
            
        except Exception as e:
            logger.error(f"Failed to setup audio stream: {e}")
            raise
    
    def _close_audio_stream(self):
        """Close audio stream"""
        if self.audio_stream:
            self.audio_stream.close()
            self.audio_stream = None
    
    def __del__(self):
        """Cleanup resources"""
        self.stop_listening()
        if self.porcupine:
            self.porcupine.delete()

class SimpleFallbackWakeWordDetector:
    """
    Simple fallback wake word detector using basic audio analysis
    Used when Porcupine is not available
    """
    
    def __init__(self, callback: Optional[Callable] = None):
        self.callback = callback
        self.is_listening = False
        self.listen_thread = None
        self.wake_phrases = ["hey dexy", "dexy", "meeting agent"]
        
    def start_listening(self):
        """Start listening for wake words"""
        if self.is_listening:
            return
        
        self.is_listening = True
        self.listen_thread = threading.Thread(target=self._listen_loop)
        self.listen_thread.daemon = True
        self.listen_thread.start()
        logger.info("Started simple wake word detection")
    
    def stop_listening(self):
        """Stop listening for wake words"""
        self.is_listening = False
        if self.listen_thread:
            self.listen_thread.join(timeout=1.0)
        logger.info("Stopped simple wake word detection")
    
    def _listen_loop(self):
        """Simple listening loop - placeholder for actual implementation"""
        # This is a simplified version - in reality, you'd implement
        # basic audio processing or speech recognition here
        while self.is_listening:
            time.sleep(1)
            # Placeholder - in real implementation, you'd process audio
            # and check for wake words
    
    def manual_trigger(self, phrase: str = "hey dexy"):
        """Manually trigger wake word detection"""
        if self.callback:
            self.callback(phrase)
        logger.info(f"Manual wake word trigger: {phrase}")

def create_wake_word_detector(callback: Optional[Callable] = None) -> WakeWordDetector:
    """
    Factory function to create wake word detector with fallback
    
    Args:
        callback: Function to call when wake word is detected
        
    Returns:
        WakeWordDetector instance
    """
    try:
        # Try to create Porcupine detector
        if PORCUPINE_ACCESS_KEY:
            detector = WakeWordDetector(
                access_key=PORCUPINE_ACCESS_KEY,
                keywords=["hey google", "porcupine"],  # Use available keywords
                callback=callback
            )
            logger.info("Created Porcupine wake word detector")
            return detector
        else:
            logger.warning("Porcupine access key not found")
            
    except Exception as e:
        logger.error(f"Failed to create Porcupine detector: {e}")
    
    # Fallback to simple detector
    detector = SimpleFallbackWakeWordDetector(callback=callback)
    logger.info("Created fallback wake word detector")
    return detector

# Test function
def test_wake_word_detection():
    """Test wake word detection"""
    def on_wake_word(keyword):
        print(f"Wake word detected: {keyword}")
    
    detector = create_wake_word_detector(callback=on_wake_word)
    
    try:
        detector.start_listening()
        print("Listening for wake words... Press Ctrl+C to stop")
        
        # If using fallback detector, simulate wake word
        if isinstance(detector, SimpleFallbackWakeWordDetector):
            time.sleep(2)
            detector.manual_trigger("hey dexy")
        
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("Stopping wake word detection...")
    finally:
        detector.stop_listening()

if __name__ == "__main__":
    test_wake_word_detection()