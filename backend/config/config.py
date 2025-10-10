import os
from dotenv import load_dotenv

# data.env 파일 로드
try:
    possible_paths = [
        'data.env',
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data.env'),
        os.path.join(os.getcwd(), 'data.env'),
    ]
    
    for env_path in possible_paths:
        if os.path.exists(env_path):
            load_dotenv(env_path, encoding='utf-8')
            break
except:
    pass

# ========== 디렉토리 경로 설정 ==========
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
MODEL_DIR = os.path.join(BASE_DIR, "models")
CONFIG_DIR = os.path.join(BASE_DIR, "config")

# ========== 파일 경로 설정 ==========
# 불용어 파일 경로
STOPWORDS_PATH = os.path.join(CONFIG_DIR, "stopwords.txt")

# 대학교가 아닌 제외 단어 파일 경로
NON_UNIV_WORD_PATH = os.path.join(CONFIG_DIR, "non_university_words.txt")

# FastText 모델 경로
FASTTEXT_MODEL_PATH = os.path.join(MODEL_DIR, "cc.ko.300.model")

# ========== 데이터베이스 설정 ==========
DB_HOST = os.getenv('dbhost')
DB_PORT = int(os.getenv('dbport'))
DB_NAME = os.getenv('dbname')
DB_USER = os.getenv('dbuser')
DB_PASSWORD = os.getenv('dbpass')

# ========== 뉴스 데이터 설정 ==========
# 조회할 뉴스의 최대 기간 (일 단위)
NEWS_FRESHNESS_DAYS = 5