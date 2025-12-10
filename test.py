import os
from pathlib import Path

def print_tree(dir_path: Path, prefix: str = ""):
    # 제외할 폴더 목록
    ignore_dirs = {'.git', '__pycache__', 'venv', 'env', '.idea', '.vscode', 'node_modules', 'raw_data'}
    
    try:
        # 정렬된 파일 목록 가져오기
        entries = sorted(list(dir_path.iterdir()), key=lambda x: (x.is_file(), x.name))
        entries = [e for e in entries if e.name not in ignore_dirs]
        
        for i, entry in enumerate(entries):
            is_last = (i == len(entries) - 1)
            connector = "└── " if is_last else "├── "
            
            print(f"{prefix}{connector}{entry.name}")
            
            if entry.is_dir():
                extension = "    " if is_last else "│   "
                print_tree(entry, prefix + extension)
                
    except PermissionError:
        pass

if __name__ == "__main__":
    root = Path(".")
    print(f"📦 {root.resolve().name}")
    print_tree(root)