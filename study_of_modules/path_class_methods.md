# Class Path

## Initiation

* p = Path(---/---)

## Extension

**`joinpath(*other)`** : 현재 경로에 추가 경로를 연결합니다.

**`/` 연산자** : 경로 연결을 위한 연산자입니다.

**`with_name(name)`** : 파일 이름을 변경합니다.

**`with_suffix(suffix)`** : 파일 확장자를 변경합니다.

**`relative_to(*other)`** : 다른 경로에 대한 상대 경로를 반환합니다.

**`exists()`** : 경로가 존재하는지 확인합니다.

**`is_dir()`** : 경로가 디렉토리인지 확인합니다.

 **`is_file()`** : 경로가 파일인지 확인합니다.

**`is_symlink()`** : 경로가 심볼릭 링크인지 확인합니다.

**`is_relative_to(*other)`** : 경로가 다른 경로에 대해 상대적일 경우 `True`를 반환합니다.

**`read_text(encoding=None, errors=None)`** : 파일 내용을 텍스트로 읽습니다.

**`write_text(data, encoding=None, errors=None)`** : 파일에 텍스트 데이터를 씁니다.

**`read_bytes()`** : 파일 내용을 바이트로 읽습니다.

**`write_bytes(data)`** : 파일에 바이트 데이터를 씁니다.

**`mkdir(mode=0o777, parents=False, exist_ok=False)`** : 디렉토리를 생성합니다.

**`rmdir()`** : 디렉토리를 삭제합니다.

**`iterdir()`** : 디렉토리 내용을 반환하는 이터레이터를 제공합니다.

**`glob(pattern)`** : 와일드카드 패턴에 맞는 경로를 찾습니다.

**`rglob(pattern)`** : 재귀적으로 와일드카드 패턴에 맞는 경로를 찾습니다.

**`rename(target)`** : 파일 또는 디렉토리 이름을 변경합니다.

**`replace(target)`** : 파일 또는 디렉토리를 대체합니다.

**`unlink(missing_ok=False)`** : 파일을 삭제합니다.

**`stat()`** : 파일이나 디렉토리의 상태 정보를 반환합니다.

**`absolute()`** : 경로의 절대 경로를 반환합니다.

**`resolve(strict=False)`** : 심볼릭 링크를 따라 실제 경로를 반환합니다.
