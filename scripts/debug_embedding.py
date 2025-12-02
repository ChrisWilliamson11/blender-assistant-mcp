import sys
import os
import json
import urllib.request

# Add parent dir to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Mock bpy
from unittest.mock import MagicMock
sys.modules["bpy"] = MagicMock()

from blender_assistant_mcp.ollama_subprocess import OllamaSubprocess

def debug_embedding():
    ollama = OllamaSubprocess()
    if not ollama.is_running():
        print("Ollama not running")
        return

    print("Listing models...")
    models = ollama.list_models()
    print(f"Models: {models}")
    
    model = "nomic-embed-text"
    if model not in models:
        print(f"WARNING: {model} not found in models list! Attempting to pull...")
        try:
            # Use the binary path from the instance
            binary = ollama.binary_path
            print(f"Pulling with {binary}...")
            subprocess.run([binary, "pull", model], check=True)
            print("Pull complete.")
        except Exception as e:
            print(f"Pull failed: {e}")

    text = "Test embedding"
    print(f"Generating embedding for '{text}' with {model}...")
    
    try:
        emb = ollama.generate_embedding(model, text)
        if emb:
            print(f"Success! Embedding length: {len(emb)}")
        else:
            print("Failed: returned None or empty")
            
        # Try raw request to see full response
        url = "http://127.0.0.1:11435/api/embed"
        payload = {"model": model, "input": text}
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"))
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req) as resp:
            print("Raw response:", resp.read().decode("utf-8"))
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_embedding()
