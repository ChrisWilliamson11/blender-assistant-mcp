
import bpy

def test_markdown_rendering():
    """Injects a rich markdown message into the Assistant chat for UI testing."""
    
    # Ensure window manager properties are initialized
    wm = bpy.context.window_manager
    
    # Create a session if none exists
    if not wm.assistant_chat_sessions:
        bpy.ops.assistant.new_chat()

    # Get active session
    if wm.assistant_active_chat_index < 0:
        wm.assistant_active_chat_index = 0
        
    session = wm.assistant_chat_sessions[wm.assistant_active_chat_index]

    # Add a message
    msg = session.messages.add()
    msg.role = "Assistant"
    msg.content = """# Markdown Rendering Test

This message tests the new **MarkdownRenderer** implementation.

## Headers
We support H1 through H6 (though visualized simply).

### H3 Header
#### H4 Header

## Lists
- Bullet point 1
- Bullet point 2 with **long text** to test wrapping. Use standard text wrapping logic to ensure this line fits within the panel width comfortably without clipping.

1. Ordered Item 1
2. Ordered Item 2
   - Nested bullets (might just render flat visually but let's see)

## Code Blocks
```python
def hello_world():
    print("Code highlighting is not here yet")
    print("But blocks should be collapsible!")
    return True
```

## Quotes
> This is a blockquote.
> It should appear in a box.

## Invalid Markdown
- [ ] Tasks are not supported yet
| Tables | Are | Not |
|--------|-----|-----|
| Supported| Yet | Sorry |

End of test.
"""
    
    # Select the new message (last one)
    wm.assistant_chat_message_index = len(session.messages) - 1
    
    # Force redraw
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            area.tag_redraw()
            
    print("✅ Test message injected successfully. Check the Assistant panel.")

if __name__ == "__main__":
    test_markdown_rendering()
