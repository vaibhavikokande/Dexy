import logging
import json
import pickle
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from config.settings import Settings

logger = logging.getLogger(__name__)

class MemoryManager:
    def __init__(self):
        self.settings = Settings()
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.settings.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'}
        )
        self.vector_store = None
        self.memory_file = self.settings.MEMORY_DIR / "meeting_memory.json"
        self.vector_store_path = self.settings.MEMORY_DIR / "vector_store"
        
        # Load existing memory
        self._load_memory()
        
    def _load_memory(self):
        """Load existing memory from disk"""
        try:
            # Load vector store if exists
            if self.vector_store_path.exists():
                self.vector_store = FAISS.load_local(
                    str(self.vector_store_path),
                    self.embeddings
                )
                logger.info("Loaded existing vector store")
            else:
                # Initialize empty vector store
                self.vector_store = FAISS.from_texts(
                    ["Initial memory"], 
                    self.embeddings,
                    metadatas=[{"type": "system", "date": datetime.now().isoformat()}]
                )
                
        except Exception as e:
            logger.error(f"Error loading memory: {e}")
            # Create new vector store
            self.vector_store = FAISS.from_texts(
                ["Initial memory"], 
                self.embeddings,
                metadatas=[{"type": "system", "date": datetime.now().isoformat()}]
            )
            
    def save_memory(self):
        """Save memory to disk"""
        try:
            self.vector_store.save_local(str(self.vector_store_path))
            logger.info("Memory saved successfully")
        except Exception as e:
            logger.error(f"Error saving memory: {e}")
            
    def add_meeting_memory(
        self, 
        meeting_id: str, 
        transcript: str, 
        summary: Dict[str, str],
        participants: List[str] = None
    ):
        """Add meeting information to memory"""
        try:
            # Create memory documents
            documents = []
            metadata_base = {
                "meeting_id": meeting_id,
                "date": datetime.now().isoformat(),
                "participants": participants or [],
                "type": "meeting"
            }
            
            # Add transcript chunks
            transcript_chunks = self._chunk_text(transcript)
            for i, chunk in enumerate(transcript_chunks):
                metadata = metadata_base.copy()
                metadata.update({
                    "content_type": "transcript",
                    "chunk_id": i
                })
                documents.append(Document(page_content=chunk, metadata=metadata))
                
            # Add summary components
            for component_type, content in summary.items():
                if content and component_type != "full_summary":
                    metadata = metadata_base.copy()
                    metadata.update({
                        "content_type": component_type,
                        "chunk_id": 0
                    })
                    documents.append(Document(page_content=content, metadata=metadata))
                    
            # Add to vector store
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            
            self.vector_store.add_texts(texts, metadatas)
            
            # Save to disk
            self.save_memory()
            
            # Also save structured meeting data
            self._save_structured_meeting(meeting_id, {
                "transcript": transcript,
                "summary": summary,
                "participants": participants,
                "date": datetime.now().isoformat()
            })
            
            logger.info(f"Added meeting {meeting_id} to memory")
            
        except Exception as e:
            logger.error(f"Error adding meeting to memory: {e}")
            
    def _chunk_text(self, text: str, chunk_size: int = 1000) -> List[str]:
        """Split text into chunks for better retrieval"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) > chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
                current_length += len(word) + 1
                
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks
        
    def _save_structured_meeting(self, meeting_id: str, meeting_data: Dict):
        """Save structured meeting data"""
        try:
            meetings_file = self.settings.MEMORY_DIR / "meetings.json"
            
            # Load existing meetings
            meetings = {}
            if meetings_file.exists():
                with open(meetings_file, 'r', encoding='utf-8') as f:
                    meetings = json.load(f)
                    
            # Add new meeting
            meetings[meeting_id] = meeting_data
            
            # Save back
            with open(meetings_file, 'w', encoding='utf-8') as f:
                json.dump(meetings, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Error saving structured meeting: {e}")
            
    def get_relevant_context(self, query: str, k: int = 5) -> str:
        """Get relevant context from memory based on query"""
        try:
            # Search for relevant documents
            docs = self.vector_store.similarity_search(query, k=k)
            
            # Format context
            context_parts = []
            for doc in docs:
                metadata = doc.metadata
                content = doc.page_content
                
                # Add metadata context
                context_info = f"From meeting {metadata.get('meeting_id', 'unknown')}"
                if metadata.get('date'):
                    date_str = metadata['date'][:10]  # Just date part
                    context_info += f" on {date_str}"
                    
                context_parts.append(f"{context_info}:\n{content}")
                
            return "\n\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error getting relevant context: {e}")
            return ""
            
    def get_previous_meetings(self, days: int = 30) -> List[Dict]:
        """Get previous meetings from specified number of days"""
        try:
            meetings_file = self.settings.MEMORY_DIR / "meetings.json"
            
            if not meetings_file.exists():
                return []
                
            with open(meetings_file, 'r', encoding='utf-8') as f:
                meetings = json.load(f)
                
            # Filter meetings from last N days
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_meetings = []
            
            for meeting_id, meeting_data in meetings.items():
                try:
                    meeting_date = datetime.fromisoformat(meeting_data['date'])
                    if meeting_date >= cutoff_date:
                        recent_meetings.append({
                            'meeting_id': meeting_id,
                            'date': meeting_data['date'],
                            'summary': meeting_data.get('summary', {}),
                            'participants': meeting_data.get('participants', [])
                        })
                except Exception as e:
                    logger.warning(f"Error parsing meeting date for {meeting_id}: {e}")
                    
            # Sort by date
            recent_meetings.sort(key=lambda x: x['date'], reverse=True)
            
            return recent_meetings
            
        except Exception as e:
            logger.error(f"Error getting previous meetings: {e}")
            return []
            
    def search_meetings(self, query: str, limit: int = 10) -> List[Dict]:
        """Search meetings by content"""
        try:
            # Get relevant documents
            docs = self.vector_store.similarity_search(query, k=limit)
            
            # Group by meeting
            meeting_results = {}
            for doc in docs:
                meeting_id = doc.metadata.get('meeting_id', 'unknown')
                if meeting_id not in meeting_results:
                    meeting_results[meeting_id] = {
                        'meeting_id': meeting_id,
                        'date': doc.metadata.get('date', 'Unknown'),
                        'participants': doc.metadata.get('participants', []),
                        'relevant_content': []
                    }
                    
                meeting_results[meeting_id]['relevant_content'].append({
                    'content': doc.page_content,
                    'type': doc.metadata.get('content_type', 'unknown'),
                    'relevance_score': 0.0  # You could implement similarity scoring
                })
                
            return list(meeting_results.values())
            
        except Exception as e:
            logger.error(f"Error searching meetings: {e}")
            return []
            
    def get_meeting_participants(self, days: int = 30) -> Dict[str, int]:
        """Get meeting participants and their frequency"""
        try:
            recent_meetings = self.get_previous_meetings(days)
            participant_count = {}
            
            for meeting in recent_meetings:
                participants = meeting.get('participants', [])
                for participant in participants:
                    participant_count[participant] = participant_count.get(participant, 0) + 1
                    
            return participant_count
            
        except Exception as e:
            logger.error(f"Error getting meeting participants: {e}")
            return {}
            
    def get_action_items_status(self) -> List[Dict]:
        """Get status of action items from recent meetings"""
        try:
            recent_meetings = self.get_previous_meetings(7)  # Last week
            action_items = []
            
            for meeting in recent_meetings:
                summary = meeting.get('summary', {})
                meeting_action_items = summary.get('action_items', '')
                
                if meeting_action_items:
                    # Parse action items (simple implementation)
                    items = meeting_action_items.split('\n')
                    for item in items:
                        item = item.strip().lstrip('-•* ')
                        if item:
                            action_items.append({
                                'item': item,
                                'meeting_id': meeting['meeting_id'],
                                'date': meeting['date'],
                                'status': 'pending'  # You could track status changes
                            })
                            
            return action_items
            
        except Exception as e:
            logger.error(f"Error getting action items: {e}")
            return []
            
    def clear_old_memories(self, days: int = 90):
        """Clear memories older than specified days"""
        try:
            # This is a simplified implementation
            # In production, you'd want to rebuild the vector store without old documents
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # Clear structured meetings
            meetings_file = self.settings.MEMORY_DIR / "meetings.json"
            if meetings_file.exists():
                with open(meetings_file, 'r', encoding='utf-8') as f:
                    meetings = json.load(f)
                    
                # Filter out old meetings
                recent_meetings = {}
                for meeting_id, meeting_data in meetings.items():
                    try:
                        meeting_date = datetime.fromisoformat(meeting_data['date'])
                        if meeting_date >= cutoff_date:
                            recent_meetings[meeting_id] = meeting_data
                    except Exception as e:
                        logger.warning(f"Error parsing date for meeting {meeting_id}: {e}")
                        
                # Save filtered meetings
                with open(meetings_file, 'w', encoding='utf-8') as f:
                    json.dump(recent_meetings, f, indent=2, ensure_ascii=False)
                    
            logger.info(f"Cleared memories older than {days} days")
            
        except Exception as e:
            logger.error(f"Error clearing old memories: {e}")
            
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        try:
            stats = {
                'total_meetings': 0,
                'total_documents': 0,
                'recent_meetings': 0,
                'participants': {},
                'memory_size': 0
            }
            
            # Count meetings
            meetings_file = self.settings.MEMORY_DIR / "meetings.json"
            if meetings_file.exists():
                with open(meetings_file, 'r', encoding='utf-8') as f:
                    meetings = json.load(f)
                    stats['total_meetings'] = len(meetings)
                    
                    # Count recent meetings (last 7 days)
                    cutoff_date = datetime.now() - timedelta(days=7)
                    recent_count = 0
                    
                    for meeting_data in meetings.values():
                        try:
                            meeting_date = datetime.fromisoformat(meeting_data['date'])
                            if meeting_date >= cutoff_date:
                                recent_count += 1
                        except:
                            pass
                            
                    stats['recent_meetings'] = recent_count
                    
            # Vector store stats
            if self.vector_store:
                stats['total_documents'] = self.vector_store.index.ntotal
                
            # Memory size
            if self.vector_store_path.exists():
                stats['memory_size'] = sum(f.stat().st_size for f in self.vector_store_path.rglob('*') if f.is_file())
                
            return stats
            
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return {'error': str(e)}