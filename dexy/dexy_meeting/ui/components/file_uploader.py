import streamlit as st
import tempfile
import os
from pathlib import Path
import mimetypes
from typing import Optional, List, Dict, Any
import wave
import mutagen
from mutagen.mp3 import MP3
from mutagen.wave import WAVE
from mutagen.flac import FLAC

class FileUploader:
    """
    Advanced file uploader component for audio files with validation and processing
    """
    
    def __init__(self):
        self.supported_formats = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.wma']
        self.max_file_size = 200 * 1024 * 1024  # 200MB
        self.max_duration = 3600  # 1 hour in seconds
        
    def validate_audio_file(self, file_path: str) -> Dict[str, Any]:
        """
        Validate audio file and extract metadata
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Dictionary containing validation results and metadata
        """
        validation_result = {
            'is_valid': False,
            'error_message': '',
            'metadata': {}
        }
        
        try:
            # Check file extension
            file_ext = Path(file_path).suffix.lower()
            if file_ext not in self.supported_formats:
                validation_result['error_message'] = f"Unsupported file format: {file_ext}"
                return validation_result
                
            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size > self.max_file_size:
                validation_result['error_message'] = f"File too large: {file_size / (1024*1024):.1f}MB (max: {self.max_file_size / (1024*1024):.1f}MB)"
                return validation_result
                
            # Extract metadata based on file type
            metadata = self._extract_metadata(file_path, file_ext)
            
            # Check duration
            if metadata.get('duration', 0) > self.max_duration:
                validation_result['error_message'] = f"Audio too long: {metadata['duration']/60:.1f}min (max: {self.max_duration/60:.1f}min)"
                return validation_result
                
            validation_result['is_valid'] = True
            validation_result['metadata'] = metadata
            
        except Exception as e:
            validation_result['error_message'] = f"Error validating file: {str(e)}"
            
        return validation_result
        
    def _extract_metadata(self, file_path: str, file_ext: str) -> Dict[str, Any]:
        """
        Extract metadata from audio file
        
        Args:
            file_path: Path to the audio file
            file_ext: File extension
            
        Returns:
            Dictionary containing metadata
        """
        metadata = {
            'duration': 0,
            'bitrate': 0,
            'sample_rate': 0,
            'channels': 0,
            'format': file_ext,
            'size': os.path.getsize(file_path)
        }
        
        try:
            if file_ext == '.wav':
                with wave.open(file_path, 'rb') as wav_file:
                    frames = wav_file.getnframes()
                    sample_rate = wav_file.getframerate()
                    metadata['duration'] = frames / sample_rate
                    metadata['sample_rate'] = sample_rate
                    metadata['channels'] = wav_file.getnchannels()
                    metadata['bitrate'] = wav_file.getsampwidth() * 8 * sample_rate
                    
            elif file_ext == '.mp3':
                audio = MP3(file_path)
                metadata['duration'] = audio.info.length
                metadata['bitrate'] = audio.info.bitrate
                metadata['sample_rate'] = audio.info.sample_rate
                metadata['channels'] = audio.info.channels
                
            elif file_ext == '.flac':
                audio = FLAC(file_path)
                metadata['duration'] = audio.info.length
                metadata['bitrate'] = audio.info.bitrate
                metadata['sample_rate'] = audio.info.sample_rate
                metadata['channels'] = audio.info.channels
                
            else:
                # For other formats, use mutagen's generic approach
                audio = mutagen.File(file_path)
                if audio is not None:
                    metadata['duration'] = audio.info.length
                    metadata['bitrate'] = getattr(audio.info, 'bitrate', 0)
                    metadata['sample_rate'] = getattr(audio.info, 'sample_rate', 0)
                    metadata['channels'] = getattr(audio.info, 'channels', 0)
                    
        except Exception as e:
            st.warning(f"Could not extract metadata: {str(e)}")
            
        return metadata
        
    def render_file_info(self, metadata: Dict[str, Any]) -> None:
        """
        Render file information in a nice format
        
        Args:
            metadata: File metadata dictionary
        """
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Duration", f"{metadata['duration']//60:.0f}:{metadata['duration']%60:02.0f}")
            st.metric("Sample Rate", f"{metadata['sample_rate']:,} Hz")
            
        with col2:
            st.metric("Bitrate", f"{metadata['bitrate']:,} bps")
            st.metric("Channels", f"{metadata['channels']}")
            
        with col3:
            st.metric("File Size", f"{metadata['size'] / (1024*1024):.1f} MB")
            st.metric("Format", metadata['format'].upper())
            
    def render_upload_area(self) -> Optional[Any]:
        """
        Render the main file upload area
        
        Returns:
            Uploaded file object or None
        """
        st.markdown("### Upload Audio File")
        
        # File format information
        with st.expander("📋 Supported Formats & Limits"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Supported Formats:**")
                for fmt in self.supported_formats:
                    st.write(f"• {fmt.upper()}")
                    
            with col2:
                st.markdown("**Limits:**")
                st.write(f"• Max file size: {self.max_file_size / (1024*1024):.0f} MB")
                st.write(f"• Max duration: {self.max_duration / 60:.0f} minutes")
                st.write("• Audio quality: Any")
                
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=[fmt[1:] for fmt in self.supported_formats],  # Remove the dot
            help="Upload your meeting audio file for transcription and analysis"
        )
        
        return uploaded_file
        
    def render_drag_drop_area(self) -> Optional[Any]:
        """
        Render a drag-and-drop file upload area
        
        Returns:
            Uploaded file object or None
        """
        st.markdown("""
        <div style="
            border: 2px dashed #ccc;
            border-radius: 10px;
            padding: 2rem;
            text-align: center;
            margin: 1rem 0;
            background: #f9f9f9;
        ">
            <h3>📁 Drag & Drop Audio Files</h3>
            <p>Or click to browse and select files</p>
            <p style="color: #666; font-size: 0.9em;">
                Supported: WAV, MP3, FLAC, M4A, OGG, WMA
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        return st.file_uploader(
            "file_uploader",
            type=[fmt[1:] for fmt in self.supported_formats],
            label_visibility="collapsed"
        )
        
    def render_batch_uploader(self) -> List[Any]:
        """
        Render batch file uploader for multiple files
        
        Returns:
            List of uploaded file objects
        """
        st.markdown("### Batch Upload")
        
        uploaded_files = st.file_uploader(
            "Choose multiple audio files",
            type=[fmt[1:] for fmt in self.supported_formats],
            accept_multiple_files=True,
            help="Upload multiple meeting audio files for batch processing"
        )
        
        if uploaded_files:
            st.success(f"Selected {len(uploaded_files)} files")
            
            # Show file list
            with st.expander("📄 File List"):
                for i, file in enumerate(uploaded_files, 1):
                    st.write(f"{i}. {file.name} ({file.size / (1024*1024):.1f} MB)")
                    
        return uploaded_files or []
        
    def render_url_uploader(self) -> Optional[str]:
        """
        Render URL input for remote audio files
        
        Returns:
            URL string or None
        """
        st.markdown("### Upload from URL")
        
        audio_url = st.text_input(
            "Audio File URL",
            placeholder="https://example.com/meeting-audio.wav",
            help="Enter a direct URL to an audio file"
        )
        
        if audio_url:
            if st.button("🔗 Load from URL"):
                # Validate URL
                if not audio_url.startswith(('http://', 'https://')):
                    st.error("Please enter a valid URL starting with http:// or https://")
                    return None
                    
                # Check if URL ends with supported format
                url_ext = Path(audio_url).suffix.lower()
                if url_ext not in self.supported_formats:
                    st.error(f"URL must point to a supported audio format: {', '.join(self.supported_formats)}")
                    return None
                    
                st.info("URL validation passed. You can now process this file.")
                return audio_url
                
        return None
        
    def render_recording_uploader(self) -> Optional[bytes]:
        """
        Render audio recording interface (placeholder for WebRTC implementation)
        
        Returns:
            Recorded audio bytes or None
        """
        st.markdown("### Record Audio")
        
        st.info("🎤 Audio recording feature requires WebRTC implementation")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔴 Start Recording"):
                st.session_state.recording = True
                st.success("Recording started...")
                
        with col2:
            if st.button("⏸️ Pause"):
                st.session_state.recording = False
                st.warning("Recording paused")
                
        with col3:
            if st.button("⏹️ Stop"):
                st.session_state.recording = False
                st.info("Recording stopped")
                
        # Recording status
        if st.session_state.get('recording', False):
            st.markdown("🔴 **Recording in progress...**")
            
        return None
        
    def render_uploader(self) -> Optional[Any]:
        """
        Render the main file uploader interface with multiple options
        
        Returns:
            Uploaded file object or None
        """
        # Upload method selection
        upload_method = st.radio(
            "Choose upload method:",
            ["📁 File Upload", "🔗 URL Upload", "📁 Batch Upload", "🎤 Record Audio"],
            horizontal=True
        )
        
        if upload_method == "📁 File Upload":
            return self.render_upload_area()
        elif upload_method == "🔗 URL Upload":
            url = self.render_url_uploader()
            return url
        elif upload_method == "📁 Batch Upload":
            return self.render_batch_uploader()
        elif upload_method == "🎤 Record Audio":
            return self.render_recording_uploader()
            
        return None
        
    def process_uploaded_file(self, uploaded_file) -> Optional[str]:
        """
        Process uploaded file and return temporary file path
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Temporary file path or None if processing failed
        """
        if uploaded_file is None:
            return None
            
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
                
            # Validate the file
            validation_result = self.validate_audio_file(tmp_file_path)
            
            if not validation_result['is_valid']:
                st.error(f"File validation failed: {validation_result['error_message']}")
                os.unlink(tmp_file_path)
                return None
                
            # Display file information
            st.success("✅ File uploaded and validated successfully!")
            self.render_file_info(validation_result['metadata'])
            
            return tmp_file_path
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            return None
            
    def cleanup_temp_file(self, file_path: str) -> None:
        """
        Clean up temporary file
        
        Args:
            file_path: Path to temporary file
        """
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except Exception as e:
            st.warning(f"Could not clean up temporary file: {str(e)}")
            
    def render_file_preview(self, file_path: str) -> None:
        """
        Render audio file preview with playback controls
        
        Args:
            file_path: Path to audio file
        """
        st.markdown("### 🎵 Audio Preview")
        
        try:
            # Read audio file
            with open(file_path, 'rb') as audio_file:
                audio_bytes = audio_file.read()
                
            # Display audio player
            st.audio(audio_bytes, format='audio/wav')
            
            # Audio controls
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("⏮️ Skip to Start"):
                    st.info("Skip to start functionality")
                    
            with col2:
                if st.button("⏭️ Skip to End"):
                    st.info("Skip to end functionality")
                    
            with col3:
                playback_speed = st.selectbox("Playback Speed", [0.5, 0.75, 1.0, 1.25, 1.5, 2.0], index=2)
                
        except Exception as e:
            st.error(f"Could not preview audio: {str(e)}")
            
    def render_processing_options(self) -> Dict[str, Any]:
        """
        Render processing options for uploaded files
        
        Returns:
            Dictionary containing processing options
        """
        st.markdown("### ⚙️ Processing Options")
        
        with st.expander("Transcription Settings"):
            col1, col2 = st.columns(2)
            
            with col1:
                language = st.selectbox("Language", ["auto", "en", "es", "fr", "de", "it", "pt", "ru", "ja", "zh"], index=0)
                model_size = st.selectbox("Model Size", ["tiny", "base", "small", "medium", "large"], index=2)
                
            with col2:
                enable_timestamps = st.checkbox("Enable Timestamps", value=True)
                word_timestamps = st.checkbox("Word-level Timestamps", value=False)
                
        with st.expander("Summary Settings"):
            col1, col2 = st.columns(2)
            
            with col1:
                summary_length = st.selectbox("Summary Length", ["short", "medium", "long"], index=1)
                summary_style = st.selectbox("Summary Style", ["bullet_points", "paragraph", "structured"], index=0)
                
            with col2:
                include_action_items = st.checkbox("Extract Action Items", value=True)
                include_key_topics = st.checkbox("Extract Key Topics", value=True)
                
        with st.expander("Advanced Options"):
            col1, col2 = st.columns(2)
            
            with col1:
                noise_reduction = st.checkbox("Noise Reduction", value=True)
                speaker_diarization = st.checkbox("Speaker Diarization", value=False)
                
            with col2:
                confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.7)
                chunk_size = st.slider("Chunk Size (seconds)", 1, 60, 30)
                
        return {
            'language': language,
            'model_size': model_size,
            'enable_timestamps': enable_timestamps,
            'word_timestamps': word_timestamps,
            'summary_length': summary_length,
            'summary_style': summary_style,
            'include_action_items': include_action_items,
            'include_key_topics': include_key_topics,
            'noise_reduction': noise_reduction,
            'speaker_diarization': speaker_diarization,
            'confidence_threshold': confidence_threshold,
            'chunk_size': chunk_size
        }
        
    def render_upload_history(self) -> None:
        """
        Render upload history and recent files
        """
        st.markdown("### 📜 Recent Uploads")
        
        # Get recent uploads from session state
        recent_uploads = st.session_state.get('recent_uploads', [])
        
        if not recent_uploads:
            st.info("No recent uploads found")
            return
            
        # Display recent uploads
        for i, upload in enumerate(recent_uploads[-5:]):  # Show last 5
            with st.expander(f"📁 {upload['filename']} - {upload['timestamp']}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Size:** {upload['size']} MB")
                    st.write(f"**Duration:** {upload['duration']} seconds")
                    st.write(f"**Format:** {upload['format']}")
                    
                with col2:
                    st.write(f"**Status:** {upload['status']}")
                    if upload['status'] == 'processed':
                        if st.button(f"View Results {i}", key=f"view_{i}"):
                            st.session_state.selected_upload = upload
                            
                if st.button(f"Reprocess {i}", key=f"reprocess_{i}"):
                    st.info(f"Reprocessing {upload['filename']}...")
                    
    def save_upload_history(self, filename: str, metadata: Dict[str, Any], status: str = "uploaded") -> None:
        """
        Save upload to history
        
        Args:
            filename: Name of uploaded file
            metadata: File metadata
            status: Upload status
        """
        if 'recent_uploads' not in st.session_state:
            st.session_state.recent_uploads = []
            
        upload_record = {
            'filename': filename,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'size': metadata.get('size', 0) / (1024*1024),  # Convert to MB
            'duration': metadata.get('duration', 0),
            'format': metadata.get('format', 'unknown'),
            'status': status,
            'metadata': metadata
        }
        
        st.session_state.recent_uploads.append(upload_record)
        
        # Keep only last 20 uploads
        if len(st.session_state.recent_uploads) > 20:
            st.session_state.recent_uploads = st.session_state.recent_uploads[-20:]