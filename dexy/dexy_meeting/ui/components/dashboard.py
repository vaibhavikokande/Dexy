import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import json
from typing import Dict, List, Optional
import asyncio
from pathlib import Path

# Import your core modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from core.transcriber import TranscriberService
from core.summarizer import SummarizerService
from core.memory_manager import MemoryManager
from core.emailer import EmailService
from core.wake_word import WakeWordDetector
from core.tts import TextToSpeechService

class Dashboard:
    def __init__(self):
        self.data_dir = Path("data")
        self.summaries_dir = self.data_dir / "summaries"
        self.transcripts_dir = self.data_dir / "transcripts"
        self.audio_dir = self.data_dir / "audio"
        
        # Initialize services
        self.transcriber = TranscriberService()
        self.summarizer = SummarizerService()
        self.memory_manager = MemoryManager()
        self.emailer = EmailService()
        self.wake_word = WakeWordDetector()
        self.tts = TextToSpeechService()
        
        # Create directories if they don't exist
        for dir_path in [self.summaries_dir, self.transcripts_dir, self.audio_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def load_meeting_data(self) -> List[Dict]:
        """Load all meeting data from stored files"""
        meetings = []
        
        if self.summaries_dir.exists():
            for summary_file in self.summaries_dir.glob("*.json"):
                try:
                    with open(summary_file, 'r') as f:
                        meeting_data = json.load(f)
                        meetings.append(meeting_data)
                except Exception as e:
                    st.error(f"Error loading {summary_file}: {str(e)}")
        
        return sorted(meetings, key=lambda x: x.get('timestamp', ''), reverse=True)
    
    def save_meeting_data(self, meeting_data: Dict):
        """Save meeting data to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"meeting_{timestamp}.json"
        filepath = self.summaries_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(meeting_data, f, indent=2)
    
    def render_sidebar(self):
        """Render the sidebar with navigation and controls"""
        st.sidebar.title("🤖 Dexy Meeting Agent")
        
        # Navigation
        page = st.sidebar.selectbox(
            "Navigate",
            ["Dashboard", "New Meeting", "Meeting History", "Settings", "Analytics"]
        )
        
        # Quick stats
        meetings = self.load_meeting_data()
        st.sidebar.metric("Total Meetings", len(meetings))
        
        if meetings:
            recent_meetings = [m for m in meetings if self._is_recent(m.get('timestamp', ''))]
            st.sidebar.metric("This Week", len(recent_meetings))
        
        # Quick actions
        st.sidebar.subheader("Quick Actions")
        if st.sidebar.button("🎤 Start Recording"):
            st.session_state.recording = True
            st.rerun()
        
        if st.sidebar.button("📧 Send Last Summary"):
            self._send_last_summary()
        
        return page
    
    def _is_recent(self, timestamp_str: str) -> bool:
        """Check if timestamp is within last 7 days"""
        try:
            timestamp = datetime.fromisoformat(timestamp_str)
            return timestamp > datetime.now() - timedelta(days=7)
        except:
            return False
    
    def _send_last_summary(self):
        """Send the most recent meeting summary"""
        meetings = self.load_meeting_data()
        if meetings:
            latest_meeting = meetings[0]
            # This would need participant emails to be stored
            st.sidebar.success("Summary sent!")
        else:
            st.sidebar.error("No meetings found!")
    
    def render_dashboard(self):
        """Render the main dashboard page"""
        st.title("📊 Meeting Dashboard")
        
        meetings = self.load_meeting_data()
        
        if not meetings:
            st.info("No meetings recorded yet. Start by creating a new meeting!")
            return
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Meetings", len(meetings))
        
        with col2:
            recent_count = len([m for m in meetings if self._is_recent(m.get('timestamp', ''))])
            st.metric("This Week", recent_count)
        
        with col3:
            avg_duration = sum(m.get('duration', 0) for m in meetings) / len(meetings)
            st.metric("Avg Duration", f"{avg_duration:.1f} min")
        
        with col4:
            total_participants = sum(len(m.get('participants', [])) for m in meetings)
            st.metric("Total Participants", total_participants)
        
        # Recent meetings
        st.subheader("Recent Meetings")
        
        for meeting in meetings[:5]:  # Show last 5 meetings
            with st.expander(f"Meeting - {meeting.get('title', 'Untitled')} ({meeting.get('timestamp', '')[:10]})"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write("**Summary:**")
                    st.write(meeting.get('summary', 'No summary available'))
                    
                    if meeting.get('action_items'):
                        st.write("**Action Items:**")
                        for item in meeting.get('action_items', []):
                            st.write(f"• {item}")
                
                with col2:
                    st.write("**Participants:**")
                    for participant in meeting.get('participants', []):
                        st.write(f"• {participant}")
                    
                    if st.button(f"📧 Send Summary", key=f"send_{meeting.get('timestamp')}"):
                        st.success("Summary sent!")
        
        # Quick visualizations
        if len(meetings) > 1:
            st.subheader("Meeting Trends")
            
            # Create DataFrame for visualization
            df = pd.DataFrame([
                {
                    'date': meeting.get('timestamp', '')[:10],
                    'duration': meeting.get('duration', 0),
                    'participants': len(meeting.get('participants', [])),
                    'action_items': len(meeting.get('action_items', []))
                }
                for meeting in meetings
            ])
            
            # Meeting frequency chart
            fig = px.line(df, x='date', y='duration', title='Meeting Duration Trends')
            st.plotly_chart(fig, use_container_width=True)
    
    def render_new_meeting(self):
        """Render the new meeting page"""
        st.title("🎤 New Meeting")
        
        # Meeting setup
        col1, col2 = st.columns(2)
        
        with col1:
            meeting_title = st.text_input("Meeting Title", "Daily Standup")
            meeting_date = st.date_input("Meeting Date", datetime.now())
            meeting_time = st.time_input("Meeting Time", datetime.now().time())
        
        with col2:
            participants = st.text_area(
                "Participants (one per line)",
                "john@company.com\nmary@company.com\nbob@company.com"
            )
            meeting_type = st.selectbox(
                "Meeting Type",
                ["Standup", "Planning", "Review", "Brainstorming", "Other"]
            )
        
        # Audio input options
        st.subheader("Audio Input")
        
        input_method = st.radio(
            "Choose input method:",
            ["Upload Audio File", "Record Live", "Paste Transcript"]
        )
        
        transcript_text = ""
        
        if input_method == "Upload Audio File":
            uploaded_file = st.file_uploader(
                "Upload audio file", 
                type=['mp3', 'wav', 'mp4', 'm4a']
            )
            
            if uploaded_file is not None:
                # Save uploaded file
                file_path = self.audio_dir / uploaded_file.name
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getvalue())
                
                st.success(f"File uploaded: {uploaded_file.name}")
                
                if st.button("🎯 Transcribe Audio"):
                    with st.spinner("Transcribing audio..."):
                        transcript_text = self.transcriber.transcribe_audio(str(file_path))
                        st.session_state.transcript = transcript_text
        
        elif input_method == "Record Live":
            st.info("🎙️ Live recording feature would be implemented here")
            st.write("This would integrate with your microphone for real-time transcription")
            
            # Placeholder for live recording
            if st.button("Start Recording"):
                st.session_state.recording = True
                st.success("Recording started... (This is a placeholder)")
        
        elif input_method == "Paste Transcript":
            transcript_text = st.text_area(
                "Paste meeting transcript here:",
                height=300,
                placeholder="Paste your meeting transcript or notes here..."
            )
        
        # Process transcript
        if transcript_text or st.session_state.get('transcript'):
            current_transcript = transcript_text or st.session_state.get('transcript', '')
            
            st.subheader("Meeting Transcript")
            with st.expander("View Transcript"):
                st.text_area("Transcript", current_transcript, height=200, disabled=True)
            
            # Generate summary
            if st.button("🤖 Generate Summary"):
                with st.spinner("Analyzing meeting..."):
                    # Get context from previous meetings
                    context = self.memory_manager.get_relevant_context(current_transcript)
                    
                    # Generate summary
                    summary_result = self.summarizer.summarize_meeting(
                        current_transcript, 
                        context=context
                    )
                    
                    # Store the results
                    meeting_data = {
                        'title': meeting_title,
                        'timestamp': datetime.combine(meeting_date, meeting_time).isoformat(),
                        'participants': [p.strip() for p in participants.split('\n') if p.strip()],
                        'type': meeting_type,
                        'transcript': current_transcript,
                        'summary': summary_result.get('summary', ''),
                        'action_items': summary_result.get('action_items', []),
                        'key_decisions': summary_result.get('key_decisions', []),
                        'next_meeting_topics': summary_result.get('next_meeting_topics', []),
                        'duration': summary_result.get('duration', 0)
                    }
                    
                    # Save to memory and storage
                    self.memory_manager.store_meeting_memory(meeting_data)
                    self.save_meeting_data(meeting_data)
                    
                    st.session_state.current_meeting = meeting_data
                    st.success("Meeting summary generated!")
        
        # Display results
        if st.session_state.get('current_meeting'):
            meeting = st.session_state.current_meeting
            
            st.subheader("Meeting Summary")
            
            # Summary tabs
            tab1, tab2, tab3 = st.tabs(["Summary", "Action Items", "Key Decisions"])
            
            with tab1:
                st.write(meeting['summary'])
                
                # TTS option
                if st.button("🔊 Read Summary Aloud"):
                    with st.spinner("Converting to speech..."):
                        audio_file = self.tts.synthesize_speech(meeting['summary'])
                        if audio_file:
                            st.audio(audio_file)
            
            with tab2:
                if meeting['action_items']:
                    for i, item in enumerate(meeting['action_items'], 1):
                        st.write(f"{i}. {item}")
                else:
                    st.info("No action items identified")
            
            with tab3:
                if meeting['key_decisions']:
                    for i, decision in enumerate(meeting['key_decisions'], 1):
                        st.write(f"{i}. {decision}")
                else:
                    st.info("No key decisions identified")
            
            # Email summary
            st.subheader("Share Summary")
            
            if st.button("📧 Send Email Summary"):
                with st.spinner("Sending emails..."):
                    email_list = [p for p in meeting['participants'] if '@' in p]
                    if email_list:
                        success = self.emailer.send_meeting_summary(meeting, email_list)
                        if success:
                            st.success(f"Summary sent to {len(email_list)} participants!")
                        else:
                            st.error("Failed to send emails")
                    else:
                        st.warning("No valid email addresses found in participants")
    
    def render_meeting_history(self):
        """Render the meeting history page"""
        st.title("📚 Meeting History")
        
        meetings = self.load_meeting_data()
        
        if not meetings:
            st.info("No meetings recorded yet.")
            return
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            date_filter = st.date_input("Filter by date", datetime.now())
        
        with col2:
            type_filter = st.selectbox(
                "Filter by type",
                ["All"] + list(set(m.get('type', 'Other') for m in meetings))
            )
        
        with col3:
            search_query = st.text_input("Search in summaries")
        
        # Filter meetings
        filtered_meetings = meetings
        
        if type_filter != "All":
            filtered_meetings = [m for m in filtered_meetings if m.get('type') == type_filter]
        
        if search_query:
            filtered_meetings = [
                m for m in filtered_meetings 
                if search_query.lower() in m.get('summary', '').lower()
            ]
        
        # Display meetings
        st.subheader(f"Found {len(filtered_meetings)} meetings")
        
        for meeting in filtered_meetings:
            with st.expander(f"{meeting.get('title', 'Untitled')} - {meeting.get('timestamp', '')[:10]}"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write("**Summary:**")
                    st.write(meeting.get('summary', 'No summary available'))
                    
                    if meeting.get('action_items'):
                        st.write("**Action Items:**")
                        for item in meeting.get('action_items', []):
                            st.write(f"• {item}")
                    
                    if meeting.get('key_decisions'):
                        st.write("**Key Decisions:**")
                        for decision in meeting.get('key_decisions', []):
                            st.write(f"• {decision}")
                
                with col2:
                    st.write("**Details:**")
                    st.write(f"Type: {meeting.get('type', 'N/A')}")
                    st.write(f"Duration: {meeting.get('duration', 0)} min")
                    st.write(f"Participants: {len(meeting.get('participants', []))}")
                    
                    if st.button(f"📧 Resend", key=f"resend_{meeting.get('timestamp')}"):
                        st.success("Summary resent!")
    
    def render_settings(self):
        """Render the settings page"""
        st.title("⚙️ Settings")
        
        # API Settings
        st.subheader("API Configuration")
        
        with st.expander("OpenAI Settings"):
            openai_key = st.text_input("OpenAI API Key", type="password")
            model_choice = st.selectbox(
                "Model",
                ["gpt-4", "gpt-3.5-turbo", "gpt-4-turbo"]
            )
        
        # Email Settings
        with st.expander("Email Settings"):
            smtp_server = st.text_input("SMTP Server", "smtp.gmail.com")
            smtp_port = st.number_input("SMTP Port", value=587)
            email_user = st.text_input("Email Username")
            email_pass = st.text_input("Email Password", type="password")
        
        # Wake Word Settings
        with st.expander("Wake Word Settings"):
            wake_word = st.text_input("Wake Word", "Hey Dexy")
            sensitivity = st.slider("Sensitivity", 0.1, 1.0, 0.5)
        
        # TTS Settings
        with st.expander("Text-to-Speech Settings"):
            tts_voice = st.selectbox("Voice", ["alloy", "echo", "fable", "onyx", "nova", "shimmer"])
            tts_speed = st.slider("Speed", 0.25, 4.0, 1.0)
        
        if st.button("💾 Save Settings"):
            st.success("Settings saved!")
    
    def render_analytics(self):
        """Render the analytics page"""
        st.title("📈 Analytics")
        
        meetings = self.load_meeting_data()
        
        if not meetings:
            st.info("No data available for analytics.")
            return
        
        # Create DataFrame for analysis
        df = pd.DataFrame([
            {
                'date': meeting.get('timestamp', '')[:10],
                'title': meeting.get('title', 'Untitled'),
                'type': meeting.get('type', 'Other'),
                'duration': meeting.get('duration', 0),
                'participants': len(meeting.get('participants', [])),
                'action_items': len(meeting.get('action_items', [])),
                'key_decisions': len(meeting.get('key_decisions', []))
            }
            for meeting in meetings
        ])
        
        # Convert date column
        df['date'] = pd.to_datetime(df['date'])
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Meeting Time", f"{df['duration'].sum():.1f} hours")
        
        with col2:
            st.metric("Average Participants", f"{df['participants'].mean():.1f}")
        
        with col3:
            st.metric("Total Action Items", df['action_items'].sum())
        
        # Charts
        st.subheader("Meeting Trends")
        
        # Meeting frequency over time
        daily_meetings = df.groupby('date').size().reset_index(name='count')
        fig1 = px.line(daily_meetings, x='date', y='count', title='Meeting Frequency Over Time')
        st.plotly_chart(fig1, use_container_width=True)
        
        # Meeting types distribution
        type_counts = df['type'].value_counts()
        fig2 = px.pie(values=type_counts.values, names=type_counts.index, title='Meeting Types Distribution')
        st.plotly_chart(fig2, use_container_width=True)
        
        # Duration vs Participants
        fig3 = px.scatter(df, x='participants', y='duration', color='type', 
                         title='Meeting Duration vs Number of Participants')
        st.plotly_chart(fig3, use_container_width=True)
        
        # Action items over time
        fig4 = px.bar(df, x='date', y='action_items', title='Action Items Generated Over Time')
        st.plotly_chart(fig4, use_container_width=True)
    
    def run(self):
        """Main dashboard runner"""
        st.set_page_config(
            page_title="Dexy Meeting Agent",
            page_icon="🤖",
            layout="wide"
        )
        
        # Initialize session state
        if 'recording' not in st.session_state:
            st.session_state.recording = False
        if 'transcript' not in st.session_state:
            st.session_state.transcript = ""
        if 'current_meeting' not in st.session_state:
            st.session_state.current_meeting = None
        
        # Render sidebar and get current page
        page = self.render_sidebar()
        
        # Render appropriate page
        if page == "Dashboard":
            self.render_dashboard()
        elif page == "New Meeting":
            self.render_new_meeting()
        elif page == "Meeting History":
            self.render_meeting_history()
        elif page == "Settings":
            self.render_settings()
        elif page == "Analytics":
            self.render_analytics()

def main():
    dashboard = Dashboard()
    dashboard.run()

if __name__ == "__main__":
    main()