"""Memory manager for Blender Assistant.

Handles loading, saving, and retrieving persistent memory (preferences, facts, etc.)
from a JSON file in the extension directory.
"""

import json
import os
from typing import Dict, List, Any, Optional

class MemoryManager:
    def __init__(self, memory_file: str = "blender_assistant_vectors.json"):
        import bpy
        
        # Use Blender's user config directory for persistence
        config_dir = bpy.utils.user_resource("CONFIG")
        self.file_path = os.path.join(config_dir, memory_file)
        
        # Initialize Vector Memory
        from pathlib import Path
        from .vector_memory import VectorMemory
        self.vector_store = VectorMemory(Path(self.file_path))

    def remember_preference(self, key: str, value: Any):
        """Store a user preference."""
        self.vector_store.set_preference(key, value)

    def remember_fact(self, fact: str, category: str = "general"):
        """Store a general fact."""
        self.vector_store.add(fact, {"category": category, "type": "fact"})

    def remember_learning(self, topic: str, insight: str):
        """Store a technical learning, pitfall, or version quirk."""
        content = f"[{topic}] {insight}"
        self.vector_store.add(content, {"category": "learning", "type": "learning", "topic": topic})

    def search_memory(self, query: str, limit: int = 5) -> List[str]:
        """Semantic search over memory."""
        results = self.vector_store.search(query, k=limit)
        return [doc["text"] for doc, score in results]

    def get_context(self) -> str:
        """Get a formatted string of relevant memory for the system prompt."""
        lines = []
        
        prefs = self.vector_store.preferences
        if prefs:
            lines.append("User Preferences:")
            for k, v in prefs.items():
                lines.append(f"- {k}: {v}")
        
        # Show recent learnings/facts from vector store to keep it fresh
        # We can just take the last 5 documents
        recent_docs = self.vector_store.documents[-5:] if self.vector_store.documents else []
        
        if recent_docs:
            lines.append("Recent Learnings & Facts:")
            for doc in recent_docs:
                lines.append(f"- {doc['text']}")
                
        return "\n".join(lines)

    def clear(self):
        """Clear all memory."""
        # Reset vector store
        self.vector_store.documents = []
        self.vector_store.preferences = {}
        self.vector_store.vectors = None
        self.vector_store.save()

    # --- Memory Abstracts ---
    
    def get_abstracts_path(self) -> str:
        import bpy
        config_dir = bpy.utils.user_resource("CONFIG")
        return os.path.join(config_dir, "blender_assistant_abstracts.json")

    def get_abstracts(self) -> List[str]:
        """Load all session abstracts."""
        path = self.get_abstracts_path()
        if not os.path.exists(path):
            return []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []

    def store_abstract(self, abstract: str):
        """Store a new abstract."""
        abstracts = self.get_abstracts()
        abstracts.append(abstract)
        path = self.get_abstracts_path()
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(abstracts, f, indent=2)
        except Exception as e:
            print(f"Failed to save abstract: {e}")

    def create_abstract(self, session_history: List[Dict[str, str]], llm_client: Any = None, model_name: str = "gpt-oss:20b") -> Optional[str]:
        """Generate an abstract from session history using LLM."""
        if not session_history:
            return None
            
        # Format history for summarization
        history_text = ""
        for msg in session_history:
            role = msg.get("role", "unknown")
            if role == "thinking":
                continue # Skip thinking in abstract
            content = msg.get("content", "")
            history_text += f"{role}: {content}\n"
            
        prompt = f"""Summarize the following conversation into a concise abstract (1-2 sentences).
Focus on key decisions, learnings, or user preferences.
Ignore trivial chit-chat.

Conversation:
{history_text}

Abstract:"""

        try:
             # Use injected client if available, else fallback
             if llm_client:
                 response = llm_client.chat_completion(
                    model_path=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3
                )
             else:
                from . import ollama_adapter
                response = ollama_adapter.chat_completion(
                    model_path=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=128
                )
             abstract = response.get("content", "").strip()
             if abstract:
                self.store_abstract(abstract)
                return abstract
        except Exception:
             pass
             
        return None

    def compact_history(self, history: List[Dict[str, str]], llm_client: Any, model_name: str = "gpt-oss:20b") -> List[Dict[str, str]]:
        """
        Compress context using 'Gradient Summarization'.
        
        Policy:
        - Trigger: If len(history) > 20.
        - Preserve: Index 0 (Initial User Request).
        - Compress: Index 1-11 (Oldest 10 messages after first).
        - Keep: Index 11+ (Recent context).
        """
        if len(history) <= 20:
            return history
            
        print("[MemoryManager] Compressing history (Gradient Summarization)...")
        
        first_msg = history[0]
        to_compress = history[1:11]
        remaining = history[11:]
        
        # Format for summarization
        chunk_text = ""
        for msg in to_compress:
            if msg.get('role') == "thinking":
                continue
            chunk_text += f"{msg.get('role')}: {msg.get('content')}\n"
            
        prompt = f"""Summarize this segment of the conversation. capture what was attempted and the outcome.
        
        Segment:
        {chunk_text}
        
        Summary:"""
        
        try:
            response = llm_client.chat_completion(
                model_path=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            summary = response.get("content", "").strip()
        except Exception as e:
            print(f"[MemoryManager] Compression Failed: {e}")
            return history # Fail safe
            
        summary_msg = {
            "role": "system", 
            "content": f"[Previous Context Summary]: {summary}"
        }
        
        # Return new history structure
        new_history = [first_msg, summary_msg] + remaining
        print(f"[MemoryManager] Compressed {len(history)} -> {len(new_history)} messages.")
        return new_history
