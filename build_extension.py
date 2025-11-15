"""Build script for Blender AI Assistant MCP extension.

This script packages the extension into a .zip file that can be installed in Blender 4.2+.
"""

import os
import zipfile
import shutil
from pathlib import Path


def build_extension():
    """Build the extension .zip file."""
    
    # Paths
    source_dir = Path("blender_assistant_mcp")
    output_file = Path("blender_assistant_mcp-1.0.0.zip")
    
    # Remove old build if exists
    if output_file.exists():
        print(f"Removing old build: {output_file}")
        output_file.unlink()
    
    # Create zip file
    print(f"Creating extension package: {output_file}")
    with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add all files from source directory
        for root, dirs, files in os.walk(source_dir):
            # Skip __pycache__ directories
            if '__pycache__' in root:
                continue
            
            for file in files:
                # Skip .pyc files
                if file.endswith('.pyc'):
                    continue
                
                file_path = Path(root) / file
                arcname = file_path.relative_to(source_dir.parent)
                
                print(f"  Adding: {arcname}")
                zipf.write(file_path, arcname)
    
    # Get file size
    size_mb = output_file.stat().st_size / (1024 * 1024)
    
    print(f"\n✓ Extension built successfully!")
    print(f"  File: {output_file}")
    print(f"  Size: {size_mb:.2f} MB")
    print(f"\nTo install:")
    print(f"  1. Open Blender 4.2+")
    print(f"  2. Edit → Preferences → Get Extensions")
    print(f"  3. Click dropdown (⌄) → Install from Disk")
    print(f"  4. Select: {output_file.absolute()}")
    print(f"  5. Enable the extension")


if __name__ == "__main__":
    build_extension()

