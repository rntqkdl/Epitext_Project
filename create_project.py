import os, pathlib

# 각 파일에 들어갈 내용 정의
init_content = """# 이 디렉터리는 Python 패키지로 인식되도록 하는 __init__.py 파일입니다.
"""

config_crawlers_content = """import pathlib, os

def find_project_root(current_path=__file__):
    current = pathlib.Path(current_path).resolve()
    for parent in current.parents:
        if (parent / "1_data").exists() and (parent / "3_model").exists():
            return parent
    return pathlib.Path(current_path).resolve().parent

PROJECT_ROOT = find_project_root()
# 크롤러 설정 예시 (config.py)
# - 크롤링 대상 URL 목록과 결과 저장 경로를 설정합니다.
URLS = [
    "https://example.com/data1.pdf",
    "https://example.com/data2.pdf"
]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "crawlers"
# 예시: Google Drive에 데이터가 있는 경우 (Colab 환경)
# from google.colab import drive; drive.mount('/content/drive')
# OUTPUT_DIR = pathlib.Path("/content/drive/MyDrive/내드라이브경로/outputs/crawlers")
"""

main_crawlers_content = """import os, logging
from datetime import datetime
import pathlib
import config

# 로그 설정: 콘솔 및 파일
module_rel_path = pathlib.Path(__file__).parent.relative_to(config.PROJECT_ROOT)
parts = list(module_rel_path.parts)
if parts and parts[0] in ("1_data", "2_notebooks", "3_model", "4_test_main", "5_docs"):
    parts = parts[1:]
log_dir = config.PROJECT_ROOT / "result" / "log" / pathlib.Path(*parts)
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f"{datetime.now().strftime('%Y-%m-%d')}.log"

logger = logging.getLogger()
if logger.hasHandlers():
    logger.handlers.clear()
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

logging.info("크롤러 모듈 시작")
logging.info(f"설정된 다운로드 경로: {config.OUTPUT_DIR}")
os.makedirs(config.OUTPUT_DIR, exist_ok=True)

# URL 목록을 순회하며 다운로드 (시뮬레이션)
for url in config.URLS:
    file_name = url.split('/')[-1] or "downloaded_file"
    save_path = config.OUTPUT_DIR / file_name
    logging.info(f"다운로드: {url} -> {save_path}")
    # 실제 다운로드 구현은 생략하고 빈 파일 생성으로 시뮬레이션
    try:
        with open(save_path, 'wb') as f:
            pass
        logging.info(f"파일 저장 완료: {save_path}")
    except Exception as e:
        logging.error(f"파일 저장 실패: {e}")

logging.info("크롤러 모듈 종료")
"""

readme_crawlers_content = """# 모듈: 데이터 수집 (크롤러)
이 모듈은 설정된 URL 목록에서 데이터를 다운로드하는 크롤러 예시입니다. PDF 등 파일을 외부 사이트에서 가져와 프로젝트 폴더에 저장합니다.

## 사용법
1. `config.py` 파일에서 `URLS`와 `OUTPUT_DIR` 등을 실제 수집 대상과 경로에 맞게 수정합니다.
2. 터미널에서 모듈의 `main.py`를 실행합니다:
"""

# 필요한 디렉터리 구조 생성
dirs_to_create = [
    "1_data/crawlers",
    "2_notebooks",
    "3_model",
    "4_test_main",
    "5_docs",
    "outputs/crawlers",
    "result/log"
]
for d in dirs_to_create:
    os.makedirs(d, exist_ok=True)

# 각 파일을 생성하고 내용 쓰기
with open("1_data/crawlers/__init__.py", "w", encoding="utf-8") as f:
    f.write(init_content)
with open("1_data/crawlers/config.py", "w", encoding="utf-8") as f:
    f.write(config_crawlers_content)
with open("1_data/crawlers/main.py", "w", encoding="utf-8") as f:
    f.write(main_crawlers_content)
with open("1_data/crawlers/README.md", "w", encoding="utf-8") as f:
    f.write(readme_crawlers_content)

print("Project structure created successfully.")
