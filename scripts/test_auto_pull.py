import sys
import os
import json
from unittest.mock import MagicMock

# Add parent dir to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Mock bpy
sys.modules["bpy"] = MagicMock()

from blender_assistant_mcp.ollama_subprocess import OllamaSubprocess

def test_auto_pull():
    ollama = OllamaSubprocess()
    if not ollama.is_running():
        print("Ollama not running")
        return

    model = "nomic-embed-text"
    text = "Test embedding auto-pull"
    
    print(f"Generating embedding for '{text}' with {model}...")
    print("(This should trigger auto-pull if model is missing)")
    
    try:
        # This calls the updated generate_embedding which has auto-pull
        emb = ollama.generate_embedding(model, text)
        if emb:
            print(f"Success! Embedding length: {len(emb)}")
        else:
            print("Failed: returned None or empty")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_auto_pull()
