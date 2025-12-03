import json
from pathlib import Path

TEMP_DIR = Path(r"h:\blender-assistant-mcp\temp_docs")
CACHE_FILE = TEMP_DIR / "chunks_cache.json"

def inspect():
    if not CACHE_FILE.exists():
        print(f"Cache file not found: {CACHE_FILE}")
        return

    data = json.loads(CACHE_FILE.read_text(encoding='utf-8'))
    print(f"Loaded {len(data)} chunks.")

    # Indices from log are 1-based
    indices = [7090, 7091, 7093]
    
    for idx in indices:
        i = idx - 1 # Convert to 0-based
        if 0 <= i < len(data):
            chunk = data[i]
            print(f"\n=== Chunk {idx} ===")
            print(f"File: {chunk['metadata'].get('file')}")
            print(f"Section: {chunk['metadata'].get('section')}")
            print(f"Length: {len(chunk['content'])}")
            print("--- Content Start ---")
            print(chunk['content'][:200])
            print("--- Content End ---")
            print(chunk['content'][-200:])
            print("===================")
        else:
            print(f"Chunk {idx} out of range.")

if __name__ == "__main__":
    inspect()
