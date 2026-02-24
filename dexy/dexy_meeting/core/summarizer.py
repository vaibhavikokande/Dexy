import logging
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
import json

from langchain_community.llms import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config.settings import Settings

logger = logging.getLogger(__name__)

class MeetingSummarizer:
    def __init__(self):
        self.settings = Settings()
        self.chat_model = ChatOpenAI(
            model_name=self.settings.MODEL_NAME,
            temperature=0.3,
            openai_api_key=self.settings.OPENAI_API_KEY
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.MAX_TRANSCRIPT_LENGTH,
            chunk_overlap=200
        )
        
    def summarize_meeting(
        self, 
        transcript: str, 
        meeting_id: str,
        participants: List[str] = None,
        previous_context: str = "",
        duration: str = "Unknown"
    ) -> Dict[str, str]:
        """
        Summarize a meeting transcript
        
        Args:
            transcript: The meeting transcript
            meeting_id: Unique identifier for the meeting
            participants: List of participant names
            previous_context: Context from previous meetings
            duration: Meeting duration
            
        Returns:
            Dictionary containing summary components
        """
        try:
            logger.info(f"Summarizing meeting {meeting_id}")
            
            # Handle long transcripts by splitting
            if len(transcript) > self.settings.MAX_TRANSCRIPT_LENGTH:
                transcript = self._process_long_transcript(transcript)
                
            # Prepare prompt
            prompt_template = PromptTemplate(
                input_variables=["agent_name", "transcript", "previous_context", "date", "duration", "participants"],
                template=self.settings.get_summary_prompt()
            )
            
            # Create chain
            chain = LLMChain(llm=self.chat_model, prompt=prompt_template)
            
            # Generate summary
            summary = chain.run(
                agent_name=self.settings.AGENT_NAME,
                transcript=transcript,
                previous_context=previous_context,
                date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                duration=duration,
                participants=", ".join(participants) if participants else "Unknown"
            )
            
            # Extract components
            summary_components = self._extract_summary_components(summary)
            
            # Save summary
            self._save_summary(meeting_id, summary_components)
            
            logger.info(f"Meeting summary completed for {meeting_id}")
            return summary_components
            
        except Exception as e:
            logger.error(f"Error summarizing meeting {meeting_id}: {e}")
            return {"error": str(e)}
            
    def _process_long_transcript(self, transcript: str) -> str:
        """Process long transcripts by summarizing chunks"""
        try:
            # Split transcript into chunks
            chunks = self.text_splitter.split_text(transcript)
            
            if len(chunks) <= 1:
                return transcript
                
            # Summarize each chunk
            chunk_summaries = []
            for i, chunk in enumerate(chunks):
                logger.info(f"Processing chunk {i+1}/{len(chunks)}")
                
                prompt = f"""
                Please provide a concise summary of this meeting transcript section:
                
                {chunk}
                
                Focus on key points, decisions, and action items. Keep it detailed but concise.
                """
                
                chain = LLMChain(
                    llm=self.chat_model,
                    prompt=PromptTemplate(input_variables=["text"], template="{text}")
                )
                
                chunk_summary = chain.run(text=prompt)
                chunk_summaries.append(chunk_summary)
                
            # Combine chunk summaries
            combined_summary = "\n\n".join(chunk_summaries)
            
            # If still too long, summarize the summaries
            if len(combined_summary) > self.settings.MAX_TRANSCRIPT_LENGTH:
                final_prompt = f"""
                Please create a comprehensive summary from these meeting section summaries:
                
                {combined_summary}
                
                Create a unified summary that captures all key points, decisions, and action items.
                """
                
                chain = LLMChain(
                    llm=self.chat_model,
                    prompt=PromptTemplate(input_variables=["text"], template="{text}")
                )
                
                return chain.run(text=final_prompt)
                
            return combined_summary
            
        except Exception as e:
            logger.error(f"Error processing long transcript: {e}")
            return transcript[:self.settings.MAX_TRANSCRIPT_LENGTH]
            
    def _extract_summary_components(self, summary: str) -> Dict[str, str]:
        """Extract structured components from summary"""
        components = {
            "full_summary": summary,
            "key_points": "",
            "action_items": "",
            "next_steps": "",
            "insights": "",
            "participants": "",
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        try:
            # Simple parsing - in production, you might want more sophisticated extraction
            lines = summary.split('\n')
            current_section = None
            
            for line in lines:
                line = line.strip()
                
                if '### Key Discussion Points:' in line:
                    current_section = 'key_points'
                    continue
                elif '### Action Items:' in line:
                    current_section = 'action_items'
                    continue
                elif '### Next Steps:' in line:
                    current_section = 'next_steps'
                    continue
                elif '### Key Insights:' in line:
                    current_section = 'insights'
                    continue
                elif '**Participants:**' in line:
                    components['participants'] = line.replace('**Participants:**', '').strip()
                    continue
                    
                if current_section and line:
                    components[current_section] += line + '\n'
                    
        except Exception as e:
            logger.error(f"Error extracting summary components: {e}")
            
        return components
        
    def _save_summary(self, meeting_id: str, summary_components: Dict[str, str]):
        """Save summary to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_file = self.settings.SUMMARY_DIR / f"meeting_{meeting_id}_{timestamp}.json"
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_components, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Summary saved to: {summary_file}")
            
        except Exception as e:
            logger.error(f"Error saving summary: {e}")
            
    def generate_quick_summary(self, recent_transcript: str) -> str:
        """Generate a quick summary for real-time responses"""
        try:
            prompt = f"""
            Please provide a brief summary of what's currently being discussed in this meeting:
            
            {recent_transcript[-2000:]}  # Last 2000 characters
            
            Keep it concise and focus on the current topic. This will be spoken aloud.
            """
            
            chain = LLMChain(
                llm=self.chat_model,
                prompt=PromptTemplate(input_variables=["text"], template="{text}")
            )
            
            return chain.run(text=prompt)
            
        except Exception as e:
            logger.error(f"Error generating quick summary: {e}")
            return "I'm having trouble processing the current discussion. Please try again."
            
    def generate_action_items(self, transcript: str) -> List[str]:
        """Extract action items from transcript"""
        try:
            prompt = f"""
            Please extract all action items from this meeting transcript:
            
            {transcript}
            
            Format as a simple list, one action item per line.
            Include responsible person if mentioned.
            """
            
            chain = LLMChain(
                llm=self.chat_model,
                prompt=PromptTemplate(input_variables=["text"], template="{text}")
            )
            
            result = chain.run(text=prompt)
            
            # Parse action items
            action_items = []
            for line in result.split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    # Clean up formatting
                    line = line.lstrip('-•* ')
                    if line:
                        action_items.append(line)
                        
            return action_items
            
        except Exception as e:
            logger.error(f"Error extracting action items: {e}")
            return []
            
    def generate_insights(self, transcript: str, previous_meetings: List[Dict] = None) -> str:
        """Generate insights from the meeting"""
        try:
            context = ""
            if previous_meetings:
                context = "\n\nPrevious meeting context:\n"
                for meeting in previous_meetings[-3:]:  # Last 3 meetings
                    context += f"- {meeting.get('date', 'Unknown date')}: {meeting.get('key_points', '')}\n"
                    
            prompt = f"""
            Please analyze this meeting transcript and provide insights:
            
            {transcript}
            
            {context}
            
            Focus on:
            1. Progress on ongoing projects
            2. Potential issues or blockers
            3. Team dynamics and collaboration
            4. Strategic implications
            5. Recommendations for improvement
            
            Keep insights actionable and specific.
            """
            
            chain = LLMChain(
                llm=self.chat_model,
                prompt=PromptTemplate(input_variables=["text"], template="{text}")
            )
            
            return chain.run(text=prompt)
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return "Unable to generate insights at this time."
            
    def respond_to_query(self, query: str, current_context: str) -> str:
        """Respond to user queries during meetings"""
        try:
            prompt_template = PromptTemplate(
                input_variables=["agent_name", "current_context", "user_request"],
                template=self.settings.get_wake_response_prompt()
            )
            
            chain = LLMChain(llm=self.chat_model, prompt=prompt_template)
            
            response = chain.run(
                agent_name=self.settings.AGENT_NAME,
                current_context=current_context,
                user_request=query
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error responding to query: {e}")
            return f"I'm sorry, I'm having trouble processing your request. Please try again."