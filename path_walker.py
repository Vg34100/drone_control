import os
from datetime import datetime
import fnmatch
from pathlib import Path

def load_gitignore(path):
    """Load .gitignore patterns and convert them to proper glob patterns"""
    gitignore_patterns = set()
    gitignore_path = os.path.join(path, '.gitignore')

    if os.path.exists(gitignore_path):
        with open(gitignore_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    pattern = line.replace('\\', '/')
                    if pattern.endswith('/'):
                        pattern = f"**/{pattern}**"
                    else:
                        pattern = f"**/{pattern}"
                    gitignore_patterns.add(pattern)
                    if pattern.startswith('**/'):
                        gitignore_patterns.add(pattern[3:])

    return gitignore_patterns

def should_ignore(path, base_path, gitignore_patterns, custom_patterns=None):
    """Check if a path should be ignored using glob patterns"""
    try:
        rel_path = os.path.relpath(path, base_path).replace('\\', '/')
        filename = os.path.basename(path)
    except ValueError:
        return False

    if '.git' in Path(rel_path).parts:
        return True

    # Check custom patterns first
    if custom_patterns:
        for pattern in custom_patterns:
            if fnmatch.fnmatch(filename, pattern):
                return True

    # Then check gitignore patterns
    for pattern in gitignore_patterns:
        if fnmatch.fnmatch(rel_path, pattern):
            return True

        path_parts = Path(rel_path).parts
        for i in range(len(path_parts)):
            partial_path = '/'.join(path_parts[:i+1])
            if fnmatch.fnmatch(partial_path, pattern):
                return True
            if fnmatch.fnmatch(f"{partial_path}/", pattern):
                return True

    return False

def print_directory_structure(startpath, output_file, accepted=('.ts', '.tsx', '.py'), custom_ignore_patterns=None):
    """
    Print directory structure to a file.

    Args:
        startpath: Path to start scanning from
        output_file: Output file path
        accepted: Tuple of accepted file extensions
        custom_ignore_patterns: List of additional patterns to ignore (e.g., ['Lab*', 'test*'])
    """
    startpath = os.path.abspath(startpath)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Load gitignore patterns
    gitignore_patterns = load_gitignore(startpath)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Scan Report\n")
        f.write(f"===========\n")
        f.write(f"Date: {timestamp}\n")
        f.write(f"Source Directory: {startpath}\n")
        f.write(f"Output File: {os.path.abspath(output_file)}\n\n")

        if gitignore_patterns or custom_ignore_patterns:
            f.write("Ignored Patterns:\n")
            f.write("================\n")
            if gitignore_patterns:
                for pattern in sorted(gitignore_patterns):
                    f.write(f"- {pattern}\n")
            if custom_ignore_patterns:
                for pattern in sorted(custom_ignore_patterns):
                    f.write(f"- {pattern}\n")
            f.write("\n")

        f.write("Directory Structure:\n")
        f.write("===================\n\n")

        for root, dirs, files in os.walk(startpath):
            if should_ignore(root, startpath, gitignore_patterns, custom_ignore_patterns):
                dirs[:] = []
                continue

            try:
                level = len(Path(root).relative_to(Path(startpath)).parts)
            except ValueError:
                level = 0

            indent = '  ' * level
            f.write(f'{indent}{os.path.basename(root)}/\n')

            visible_files = []
            for file in sorted(files):
                full_path = os.path.join(root, file)
                if (file.endswith(accepted) and
                    not should_ignore(full_path, startpath, gitignore_patterns, custom_ignore_patterns)):
                    visible_files.append(file)

            subindent = '  ' * (level + 1)
            for file in visible_files:
                f.write(f'{subindent}{file}\n')

            i = 0
            while i < len(dirs):
                dir_path = os.path.join(root, dirs[i])
                if should_ignore(dir_path, startpath, gitignore_patterns, custom_ignore_patterns):
                    dirs.pop(i)
                else:
                    i += 1

        f.write("\n\nFile Contents:\n")
        f.write("=============\n\n")

        for root, dirs, files in os.walk(startpath):
            if should_ignore(root, startpath, gitignore_patterns, custom_ignore_patterns):
                continue

            for file in sorted(files):
                if not file.endswith(accepted):
                    continue

                full_path = os.path.join(root, file)
                if should_ignore(full_path, startpath, gitignore_patterns, custom_ignore_patterns):
                    continue

                rel_path = os.path.relpath(full_path, startpath)
                f.write(f"\n--- {rel_path} ---\n")
                try:
                    with open(full_path, 'r', encoding='utf-8') as file_content:
                        f.write(file_content.read())
                except Exception as e:
                    f.write(f"\nError reading file: {str(e)}\n")
                f.write("\n")

# Ensure docs/history folder exists
os.makedirs('docs/history', exist_ok=True)

# Add timestamp to the output file names
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f'docs/history/structure_{timestamp}.txt'

print_directory_structure(
    '',
    output_file,
    accepted=('.py'),
    custom_ignore_patterns=['*debug*', '*models*', 'path_walker.py', '*old*']  # Add any patterns you want to ignore
)
