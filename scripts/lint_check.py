import ast
import os
import sys
from pathlib import Path

def get_imports_and_functions(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            tree = ast.parse(f.read(), filename=file_path)
        except SyntaxError:
            print(f"SyntaxError in {file_path}")
            return set(), []

    imports = set()
    empty_funcs = []

    for node in ast.walk(tree):
        # Collect imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split('.')[0])
            elif node.level > 0:
                # Relative import, ignore for wheel check
                pass

        # Collect empty functions
        if isinstance(node, ast.FunctionDef):
            is_empty = False
            if len(node.body) == 0:
                is_empty = True
            elif len(node.body) == 1:
                stmt = node.body[0]
                if isinstance(stmt, ast.Pass):
                    is_empty = True
                elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                    # Docstring only
                    is_empty = True
                elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Str): # Python < 3.8
                    is_empty = True
            
            if is_empty:
                empty_funcs.append(node.name)

    return imports, empty_funcs

def main():
    target_dir = Path("blender_assistant_mcp")
    all_imports = set()
    
    print("=== Scanning for Empty Functions ===")
    for root, _, files in os.walk(target_dir):
        for file in files:
            if file.endswith(".py"):
                path = Path(root) / file
                imports, empty_funcs = get_imports_and_functions(path)
                all_imports.update(imports)
                
                if empty_funcs:
                    rel_path = path.relative_to(target_dir)
                    print(f"\n{rel_path}:")
                    for func in empty_funcs:
                        print(f"  - {func}")

    print("\n=== Detected Top-Level Imports ===")
    for imp in sorted(all_imports):
        print(f"- {imp}")

if __name__ == "__main__":
    main()
