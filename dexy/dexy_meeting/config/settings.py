import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

class Settings:
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    
    # Email Configuration
    EMAIL_HOST = os.getenv("EMAIL_HOST", "smtp.gmail.com")
    EMAIL_PORT = int(os.getenv("EMAIL_PORT", "587"))
    EMAIL_USER = os.getenv("EMAIL_USER")
    EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
    
    # Picovoice Configuration
    PICOVOICE_ACCESS_KEY = os.getenv("PICOVOICE_ACCESS_KEY")
    
    # Agent Configuration
    AGENT_NAME = os.getenv("AGENT_NAME", "Dexy")
    WAKE_WORD = os.getenv("WAKE_WORD", "hey dexy")
    
    # Directory Configuration
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    AUDIO_DIR = DATA_DIR / "audio"
    TRANSCRIPT_DIR = DATA_DIR / "transcripts"
    SUMMARY_DIR = DATA_DIR / "summaries"
    MEMORY_DIR = DATA_DIR / "memory"
    
    # Development
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # Audio Settings
    SAMPLE_RATE = 16000
    CHUNK_SIZE = 1024
    AUDIO_FORMAT = "wav"
    
    # Memory Settings
    MAX_MEMORY_ITEMS = 1000
    SIMILARITY_THRESHOLD = 0.7
    
    # Meeting Settings
    MAX_TRANSCRIPT_LENGTH = 50000
    SUMMARY_MAX_TOKENS = 1000
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        directories = [
            cls.DATA_DIR,
            cls.AUDIO_DIR,
            cls.TRANSCRIPT_DIR,
            cls.SUMMARY_DIR,
            cls.MEMORY_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
        # Create .gitkeep files
        gitkeep_files = [
            cls.AUDIO_DIR / ".gitkeep",
            cls.TRANSCRIPT_DIR / ".gitkeep",
            cls.SUMMARY_DIR / ".gitkeep",
            cls.MEMORY_DIR / ".gitkeep"
        ]
        
        for gitkeep_file in gitkeep_files:
            gitkeep_file.touch(exist_ok=True)
    
    @classmethod
    def validate_config(cls):
        """Validate essential configuration"""
        errors = []
        
        if not cls.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is required")
            
        if not cls.EMAIL_USER or not cls.EMAIL_PASSWORD:
            errors.append("EMAIL_USER and EMAIL_PASSWORD are required for email functionality")
            
        if errors:
            raise ValueError("Configuration errors: " + ", ".join(errors))
    
    @classmethod
    def get_summary_prompt(cls):
        """Get the prompt template for meeting summarization"""
        return """
        You are {agent_name}, an AI meeting assistant. Please analyze the following meeting transcript and provide a comprehensive summary.
        
        Meeting Transcript:
        {transcript}
        
        Previous Meeting Context (if available):
        {previous_context}
        
        Please provide a summary in the following format:
        
        ## Meeting Summary
        
        **Date:** {date}
        **Duration:** {duration}
        **Participants:** {participants}
        
        ### Key Discussion Points:
        - List the main topics discussed
        - Include important decisions made
        - Note any action items or follow-ups
        
        ### Action Items:
        - Clear, actionable tasks with responsible parties
        - Deadlines if mentioned
        
        ### Next Steps:
        - What should happen before the next meeting
        - Any preparations needed
        
        ### Key Insights:
        - Important observations or trends
        - Potential concerns or opportunities
        
        Keep the summary concise but comprehensive, focusing on actionable information.
        """
    
    @classmethod
    def get_wake_response_prompt(cls):
        """Get the prompt template for wake word responses"""
        return """
        You are {agent_name}, an AI meeting assistant. You've been called to attention during a meeting.
        
        Current meeting context:
        {current_context}
        
        User request: {user_request}
        
        Please provide a helpful, concise response. If asked to summarize, provide a brief overview of the current discussion.
        Keep your response natural and conversational as you'll be speaking it aloud.
        """

# Create directories on import
Settings.create_directories()