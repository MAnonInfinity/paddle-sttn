import os
import re

def find_chinese_in_files(directory):
    chinese_re = re.compile(r'[\u4e00-\u9fa5]')
    results = []
    for root, dirs, files in os.walk(directory):
        if '.git' in dirs:
            dirs.remove('.git')
        if '.venv' in dirs:
            dirs.remove('.venv')
        if 'models' in dirs:
            dirs.remove('models')
            
        for file in files:
            if file.endswith(('.py', '.sh', '.md', '.toml', '.txt')):
                path = os.path.join(root, file)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        for i, line in enumerate(f):
                            if chinese_re.search(line):
                                results.append((path, i + 1, line.strip()))
                except Exception:
                    pass
    return results

if __name__ == "__main__":
    found = find_chinese_in_files('.')
    for path, line_no, content in found:
        print(f"{path}:{line_no}: {content}")
