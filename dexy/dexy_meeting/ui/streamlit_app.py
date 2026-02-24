import streamlit as st
import os
import sys
import asyncio
from datetime import datetime
import json
import tempfile
from pathlib import Path

# Add the parent directory to the path so we can import from core
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from core.transcriber import AudioTranscriber
from core.summarizer import MeetingSummarizer
from core.memory_manager import MemoryManager
from core.emailer import EmailNotifier
from core.wake_word import WakeWordDetector
from core.tts import TextToSpeech
from ui.components.file_uploader import FileUploader
from ui.components.dashboard import Dashboard
from config.settings import Settings

# Initialize settings
settings = Settings()

# Page configuration
st.set_page_config(
    page_title="Dexy Meeting Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-bottom: 2rem;
        border-radius: 10px;
    }
    .feature-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .error-message {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class DexyMeetingAgent:
    def __init__(self):
        self.transcriber = AudioTranscriber()
        self.summarizer = MeetingSummarizer()
        self.memory_manager = MemoryManager()
        self.emailer = EmailNotifier()
        self.wake_word_detector = WakeWordDetector()
        self.tts = TextToSpeech()
        self.file_uploader = FileUploader()
        self.dashboard = Dashboard()
        
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'meeting_history' not in st.session_state:
            st.session_state.meeting_history = []
        if 'current_transcript' not in st.session_state:
            st.session_state.current_transcript = ""
        if 'current_summary' not in st.session_state:
            st.session_state.current_summary = ""
        if 'memory_initialized' not in st.session_state:
            st.session_state.memory_initialized = False
            
    def render_header(self):
        """Render the main header"""
        st.markdown("""
        <div class="main-header">
            <h1>🤖 Dexy Meeting Agent</h1>
            <p>Your AI-powered meeting assistant for transcription, summarization, and insights</p>
        </div>
        """, unsafe_allow_html=True)
        
    def render_sidebar(self):
        """Render the sidebar with navigation and settings"""
        st.sidebar.title("Navigation")
        
        # Navigation menu
        pages = {
            "🏠 Dashboard": "dashboard",
            "📝 New Meeting": "new_meeting",
            "📁 Upload Audio": "upload_audio",
            "🎤 Live Recording": "live_recording",
            "📊 Analytics": "analytics",
            "⚙️ Settings": "settings"
        }
        
        selected_page = st.sidebar.selectbox("Select Page", list(pages.keys()))
        
        # Settings in sidebar
        st.sidebar.markdown("---")
        st.sidebar.subheader("Quick Settings")
        
        # Wake word settings
        st.sidebar.checkbox("Enable Wake Word Detection", value=True, key="wake_word_enabled")
        st.sidebar.text_input("Wake Word", value="Hey Dexy", key="wake_word")
        
        # Email settings
        st.sidebar.checkbox("Auto-send Email Summaries", value=False, key="auto_email")
        
        # TTS settings
        st.sidebar.checkbox("Enable Text-to-Speech", value=True, key="tts_enabled")
        
        return pages[selected_page]
        
    def render_dashboard(self):
        """Render the main dashboard"""
        st.subheader("📊 Meeting Dashboard")
        
        # Statistics cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Meetings", len(st.session_state.meeting_history))
            
        with col2:
            total_duration = sum([m.get('duration', 0) for m in st.session_state.meeting_history])
            st.metric("Total Duration", f"{total_duration:.1f} min")
            
        with col3:
            st.metric("This Month", len([m for m in st.session_state.meeting_history if m.get('date', '').startswith(datetime.now().strftime('%Y-%m'))]))
            
        with col4:
            st.metric("Memory Items", self.memory_manager.get_memory_count() if st.session_state.memory_initialized else 0)
            
        # Recent meetings
        st.subheader("Recent Meetings")
        if st.session_state.meeting_history:
            for meeting in st.session_state.meeting_history[-5:]:
                with st.expander(f"Meeting: {meeting.get('title', 'Untitled')} - {meeting.get('date', 'No date')}"):
                    st.write(f"**Duration:** {meeting.get('duration', 'Unknown')} minutes")
                    st.write(f"**Participants:** {', '.join(meeting.get('participants', []))}")
                    st.write(f"**Summary:** {meeting.get('summary', 'No summary available')[:200]}...")
        else:
            st.info("No meetings recorded yet. Start by uploading an audio file or recording a new meeting!")
            
    def render_new_meeting(self):
        """Render the new meeting interface"""
        st.subheader("📝 New Meeting")
        
        # Meeting details form
        with st.form("meeting_details"):
            col1, col2 = st.columns(2)
            
            with col1:
                meeting_title = st.text_input("Meeting Title", placeholder="Enter meeting title")
                meeting_date = st.date_input("Meeting Date", value=datetime.now())
                
            with col2:
                participants = st.text_area("Participants (one per line)", placeholder="john@example.com\njane@example.com")
                meeting_type = st.selectbox("Meeting Type", ["Team Meeting", "Client Call", "Interview", "Presentation", "Other"])
                
            submit_button = st.form_submit_button("Create Meeting")
            
            if submit_button and meeting_title:
                # Create new meeting session
                meeting_data = {
                    'title': meeting_title,
                    'date': meeting_date.strftime('%Y-%m-%d'),
                    'participants': [p.strip() for p in participants.split('\n') if p.strip()],
                    'type': meeting_type,
                    'status': 'active'
                }
                
                st.session_state.current_meeting = meeting_data
                st.success("Meeting created successfully! You can now start recording or upload audio.")
                
        # Live transcription interface
        if hasattr(st.session_state, 'current_meeting'):
            st.markdown("---")
            st.subheader("Live Transcription")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                if st.button("🎤 Start Recording"):
                    st.info("Recording functionality will be implemented with WebRTC")
                    
            with col2:
                if st.button("⏹️ Stop Recording"):
                    st.info("Recording stopped")
                    
            # Real-time transcript display
            transcript_placeholder = st.empty()
            with transcript_placeholder.container():
                st.text_area("Live Transcript", value=st.session_state.current_transcript, height=300, key="live_transcript")
                
    def render_upload_audio(self):
        """Render the audio upload interface"""
        st.subheader("📁 Upload Audio File")
        
        # File uploader
        uploaded_file = self.file_uploader.render_uploader()
        
        if uploaded_file is not None:
            # Display file information
            st.success(f"File uploaded: {uploaded_file.name}")
            st.write(f"File size: {uploaded_file.size / (1024*1024):.2f} MB")
            
            # Processing options
            col1, col2 = st.columns(2)
            
            with col1:
                process_button = st.button("🔄 Process Audio", type="primary")
                
            with col2:
                save_transcript = st.checkbox("Save transcript to file", value=True)
                
            if process_button:
                with st.spinner("Processing audio file..."):
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    try:
                        # Transcribe audio
                        transcript = self.transcriber.transcribe_audio(tmp_file_path)
                        st.session_state.current_transcript = transcript
                        
                        # Generate summary
                        summary = self.summarizer.generate_summary(transcript)
                        st.session_state.current_summary = summary
                        
                        # Store in memory
                        if not st.session_state.memory_initialized:
                            self.memory_manager.initialize_memory()
                            st.session_state.memory_initialized = True
                            
                        self.memory_manager.add_to_memory(transcript, summary)
                        
                        # Display results
                        st.success("Audio processed successfully!")
                        
                        # Create tabs for results
                        tab1, tab2, tab3 = st.tabs(["Transcript", "Summary", "Actions"])
                        
                        with tab1:
                            st.text_area("Transcript", value=transcript, height=300)
                            
                        with tab2:
                            st.text_area("Summary", value=summary, height=200)
                            
                        with tab3:
                            # Email sending
                            email_recipients = st.text_area("Email Recipients", placeholder="Enter email addresses (one per line)")
                            
                            if st.button("📧 Send Email Summary"):
                                if email_recipients:
                                    recipients = [email.strip() for email in email_recipients.split('\n') if email.strip()]
                                    try:
                                        self.emailer.send_summary(recipients, summary, transcript)
                                        st.success("Email sent successfully!")
                                    except Exception as e:
                                        st.error(f"Failed to send email: {str(e)}")
                                else:
                                    st.warning("Please enter at least one email address")
                                    
                            # TTS
                            if st.button("🔊 Listen to Summary"):
                                if st.session_state.tts_enabled:
                                    try:
                                        audio_data = self.tts.synthesize_speech(summary)
                                        st.audio(audio_data, format='audio/wav')
                                    except Exception as e:
                                        st.error(f"TTS failed: {str(e)}")
                                else:
                                    st.warning("Text-to-Speech is disabled")
                                    
                        # Save to meeting history
                        meeting_data = {
                            'title': uploaded_file.name,
                            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'transcript': transcript,
                            'summary': summary,
                            'duration': len(transcript.split()) * 0.5 / 60,  # Rough estimate
                            'participants': []
                        }
                        
                        st.session_state.meeting_history.append(meeting_data)
                        
                    except Exception as e:
                        st.error(f"Error processing audio: {str(e)}")
                    finally:
                        # Clean up temporary file
                        os.unlink(tmp_file_path)
                        
    def render_live_recording(self):
        """Render the live recording interface"""
        st.subheader("🎤 Live Recording")
        
        st.info("Live recording feature requires WebRTC integration. This is a placeholder for the full implementation.")
        
        # Recording controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🎤 Start Recording", type="primary"):
                st.session_state.recording = True
                st.success("Recording started...")
                
        with col2:
            if st.button("⏸️ Pause Recording"):
                st.session_state.recording = False
                st.warning("Recording paused")
                
        with col3:
            if st.button("⏹️ Stop Recording"):
                st.session_state.recording = False
                st.info("Recording stopped")
                
        # Wake word detection
        if st.session_state.get('wake_word_enabled', True):
            st.subheader("Wake Word Detection")
            
            if st.button("🎧 Start Listening for Wake Word"):
                st.info(f"Listening for '{st.session_state.get('wake_word', 'Hey Dexy')}'...")
                
        # Real-time transcript
        st.subheader("Real-time Transcript")
        transcript_container = st.container()
        with transcript_container:
            st.text_area("Live Transcript", value="", height=300, key="live_transcript_recording")
            
    def render_analytics(self):
        """Render the analytics interface"""
        st.subheader("📊 Analytics & Insights")
        
        if not st.session_state.meeting_history:
            st.info("No meeting data available for analytics. Record some meetings first!")
            return
            
        # Meeting statistics
        self.dashboard.render_meeting_stats(st.session_state.meeting_history)
        
        # Memory insights
        if st.session_state.memory_initialized:
            st.subheader("Memory Insights")
            memory_stats = self.memory_manager.get_memory_stats()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Memory Items", memory_stats.get('total_items', 0))
            with col2:
                st.metric("Average Similarity", f"{memory_stats.get('avg_similarity', 0):.2f}")
                
        # Trends and patterns
        st.subheader("Meeting Trends")
        self.dashboard.render_meeting_trends(st.session_state.meeting_history)
        
    def render_settings(self):
        """Render the settings interface"""
        st.subheader("⚙️ Settings")
        
        # API Keys
        st.subheader("API Configuration")
        with st.expander("API Keys"):
            openai_key = st.text_input("OpenAI API Key", type="password", value=settings.OPENAI_API_KEY or "")
            email_password = st.text_input("Email Password", type="password", value=settings.EMAIL_PASSWORD or "")
            
            if st.button("Save API Keys"):
                # In a real app, you'd save these securely
                st.success("API keys saved successfully!")
                
        # Audio Settings
        st.subheader("Audio Processing")
        with st.expander("Audio Settings"):
            audio_quality = st.selectbox("Audio Quality", ["Low", "Medium", "High"], index=1)
            chunk_duration = st.slider("Chunk Duration (seconds)", 1, 10, 5)
            
        # Memory Settings
        st.subheader("Memory Management")
        with st.expander("Memory Settings"):
            max_memory_items = st.number_input("Max Memory Items", min_value=100, max_value=10000, value=1000)
            memory_similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.8)
            
            if st.button("Clear Memory"):
                if st.button("Confirm Clear Memory"):
                    self.memory_manager.clear_memory()
                    st.success("Memory cleared!")
                    
        # Email Settings
        st.subheader("Email Configuration")
        with st.expander("Email Settings"):
            email_host = st.text_input("SMTP Host", value=settings.EMAIL_HOST or "")
            email_port = st.number_input("SMTP Port", value=settings.EMAIL_PORT or 587)
            email_user = st.text_input("Email Username", value=settings.EMAIL_USER or "")
            
        # Export/Import
        st.subheader("Data Management")
        with st.expander("Export/Import"):
            if st.button("Export Meeting History"):
                export_data = {
                    'meetings': st.session_state.meeting_history,
                    'export_date': datetime.now().isoformat()
                }
                st.download_button(
                    label="Download Export",
                    data=json.dumps(export_data, indent=2),
                    file_name=f"dexy_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
                
            uploaded_import = st.file_uploader("Import Meeting History", type=['json'])
            if uploaded_import:
                try:
                    import_data = json.load(uploaded_import)
                    st.session_state.meeting_history.extend(import_data.get('meetings', []))
                    st.success("Meeting history imported successfully!")
                except Exception as e:
                    st.error(f"Import failed: {str(e)}")
                    
    def run(self):
        """Main application runner"""
        self.initialize_session_state()
        self.render_header()
        
        # Get selected page from sidebar
        current_page = self.render_sidebar()
        
        # Render the selected page
        if current_page == "dashboard":
            self.render_dashboard()
        elif current_page == "new_meeting":
            self.render_new_meeting()
        elif current_page == "upload_audio":
            self.render_upload_audio()
        elif current_page == "live_recording":
            self.render_live_recording()
        elif current_page == "analytics":
            self.render_analytics()
        elif current_page == "settings":
            self.render_settings()
            
        # Footer
        st.markdown("---")
        st.markdown("*Dexy Meeting Agent v1.0.0 - Your AI Meeting Assistant*")

def main():
    """Main entry point"""
    app = DexyMeetingAgent()
    app.run()

if __name__ == "__main__":
    main()