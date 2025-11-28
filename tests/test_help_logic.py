
# Mock mcp_tools
class MockMCPTools:
    def get_tools_list(self):
        return []
    def register_tool(self, *args, **kwargs):
        pass

import sys
import types
sys.modules["blender_assistant_mcp.mcp_tools"] = MockMCPTools()

# Import the function (we need to extract it or mock the module import)
# Since I can't easily import the module without bpy, I'll just copy the function logic here for verification
# or rely on my code review. 

# Actually, I can just run a simplified version of the logic to prove the regex/string manipulation works.

def assistant_help_logic(tool, tools=None):
    queries = []
    if tools:
        queries.extend([str(t).strip() for t in tools if str(t).strip()])
    if tool:
        queries.append(str(tool).strip())
        
    sdk_docs = {
        "polyhaven.search": {"alias": "polyhaven.search"},
        "polyhaven.download": {"alias": "polyhaven.download"},
        "blender.create_object": {"alias": "blender.create_object"}
    }
    
    results = []
    for q in queries:
        clean_q = q.replace("assistant_sdk.", "").lower()
        sub_queries = clean_q.split("/")
        
        for sub_q in sub_queries:
            sub_q = sub_q.strip()
            if not sub_q: continue
            
            # Exact
            if sub_q in sdk_docs:
                results.append(sdk_docs[sub_q])
                continue
            # Namespace
            ns_matches = [k for k in sdk_docs.keys() if k.startswith(sub_q + ".")]
            if ns_matches:
                for k in ns_matches: results.append(sdk_docs[k])
                continue
            # Fuzzy
            fuzzy = [k for k in sdk_docs.keys() if sub_q in k]
            if fuzzy:
                for k in fuzzy: results.append(sdk_docs[k])
                continue
                
    return results

# Test cases
print("Test 1:", assistant_help_logic("assistant_sdk.polyhaven.search/download"))
print("Test 2:", assistant_help_logic("polyhaven"))
print("Test 3:", assistant_help_logic("create_object"))
