
import sys
import os
import json
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from blender_assistant_mcp.memory import MemoryManager
except ImportError as e:
    # Mock bpy if needed (memory.py imports it inside __init__)
    import types
    sys.modules["bpy"] = MagicMock()
    sys.modules["bpy.utils"] = MagicMock()
    sys.modules["bpy.utils.user_resource"] = MagicMock(return_value=".")
    from blender_assistant_mcp.memory import MemoryManager

def test_compact_history():
    print("\n--- Testing Context Compression ---")
    
    # Mock LLM Client
    mock_client = MagicMock()
    mock_client.chat_completion.return_value = {"content": "SUMMARY OF OLD STUFF"}
    
    # Initialize MemoryManager (mocks vector store internally via try/except or just let it fail/pass)
    # MemoryManager internal imports might fail without bpy, but we mocked bpy above.
    # However, vector_memory imports might fail. 
    # Let's hope VectorMemory treats import errors gracefully or we mock it too.
    
    # We can patch VectorMemory
    sys.modules["blender_assistant_mcp.vector_memory"] = MagicMock()
    
    manager = MemoryManager("test_mem.json")
    
    # 1. Test Small History (Should NOT compress)
    small_history = [{"role": "user", "content": f"msg {i}"} for i in range(10)]
    print(f"Testing Small History ({len(small_history)})...")
    res = manager.compact_history(small_history, mock_client)
    assert len(res) == 10
    assert res == small_history
    print("PASS: Small history untouched.")
    
    # 2. Test Large History (Should compress)
    # 30 messages
    # Index 0: User Prompt
    # Index 1-10: Compress (10 items)
    # Index 11-29: Keep (19 items)
    # Result size: 1 (User) + 1 (Summary) + 19 (Recent) = 21 items.
    
    large_history = [{"role": "user", "content": f"msg {i}"} for i in range(30)]
    large_history[0] = {"role": "user", "content": "INITIAL PROMPT"}
    
    print(f"Testing Large History ({len(large_history)})...")
    
    res = manager.compact_history(large_history, mock_client)
    
    # Check Length
    # Expected: 1 + 1 + 19 = 21
    print(f"Result Length: {len(res)}")
    assert len(res) == 21, f"Expected 21, got {len(res)}"
    
    # Check Integrity
    assert res[0]["content"] == "INITIAL PROMPT", "Failed to preserve first message"
    assert res[1]["role"] == "system", "Summary message missing"
    assert "[Previous Context Summary]" in res[1]["content"], "Summary content format wrong"
    assert res[2]["content"] == "msg 11", "Failed to join recent history correctly"
    
    print("PASS: Large history compressed correctly.")
    
    # Verify LLM was called
    mock_client.chat_completion.assert_called_once()
    print("PASS: LLM Client called.")

if __name__ == "__main__":
    test_compact_history()
