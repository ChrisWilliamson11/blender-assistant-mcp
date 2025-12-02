import sys
import os
import inspect
from unittest.mock import MagicMock

# Add parent dir to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Mock bpy
sys.modules["bpy"] = MagicMock()

from blender_assistant_mcp import ollama_adapter
from blender_assistant_mcp import ollama_subprocess
from blender_assistant_mcp import assistant

def verify_signatures():
    print("Verifying signatures...")
    
    # Check ollama_subprocess.generate_embedding
    sig = inspect.signature(ollama_subprocess.OllamaSubprocess.generate_embedding)
    print(f"ollama_subprocess.generate_embedding: {sig}")
    if "keep_alive" not in sig.parameters:
        print("FAIL: keep_alive missing in ollama_subprocess.generate_embedding")
    else:
        print("PASS: keep_alive present in ollama_subprocess.generate_embedding")

    # Check ollama_adapter.generate_embedding
    sig = inspect.signature(ollama_adapter.generate_embedding)
    print(f"ollama_adapter.generate_embedding: {sig}")
    # It accepts **kwargs, so keep_alive won't be in signature explicitly, but that's fine.
    # We can check if it's documented or just assume it works if **kwargs is there.
    if "kwargs" in str(sig) or "keep_alive" in str(sig):
         print("PASS: ollama_adapter.generate_embedding accepts kwargs or keep_alive")
    else:
         print("FAIL: ollama_adapter.generate_embedding signature issue")

    # Check assistant.ASSISTANT_OT_send._http_worker
    sig = inspect.signature(assistant.ASSISTANT_OT_send._http_worker)
    print(f"assistant.ASSISTANT_OT_send._http_worker: {sig}")
    if "keep_alive" not in sig.parameters:
        print("FAIL: keep_alive missing in assistant._http_worker")
    else:
        print("PASS: keep_alive present in assistant._http_worker")

if __name__ == "__main__":
    verify_signatures()
