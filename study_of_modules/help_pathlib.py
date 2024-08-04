from pathlib import Path

# 경로 생성
p = Path('/home/user/documents')

# 파일과 디렉토리 존재 여부 확인
print(p.exists())        # True if path exists
print(p.is_file())       # True if path is a file
print(p.is_dir())        # True if path is a directory

# 경로 조작
new_file = p / 'newfile.txt'  # Concatenate paths using /
print(new_file)               # /home/user/documents/newfile.txt

# 파일 쓰기
new_file.write_text("Hello, World!")

# 파일 읽기
content = new_file.read_text()
print(content)               # Hello, World!

# 디렉토리 내용 목록
for item in p.iterdir():
    print(item)

# 특정 패턴에 맞는 파일 찾기
for txt_file in p.glob('*.txt'):
    print(txt_file)

# 디렉토리와 하위 디렉토리에서 특정 패턴에 맞는 파일 찾기
for txt_file in p.rglob('*.txt'):
    print(txt_file)

# 디렉토리 생성 및 삭제
new_dir = p / 'new_dir'
new_dir.mkdir(exist_ok=True)  # Create directory
new_dir.rmdir()               # Remove directory

# 파일 삭제
new_file.unlink()             # Remove file

