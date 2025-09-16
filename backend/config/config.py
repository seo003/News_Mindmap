import os
from dotenv import load_dotenv

# data.env 파일 로드
try:
    # load_dotenv('data.env', encoding='utf-8')
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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
MODEL_DIR = os.path.join(BASE_DIR, "models")
CONFIG_DIR = os.path.join(BASE_DIR, "config")

STOPWORDS_PATH = os.path.join(CONFIG_DIR, "stopwords.txt")
NON_UNIV_WORD_PATH = os.path.join(CONFIG_DIR, "non_university_words.txt")
FASTTEXT_MODEL_PATH = os.path.join(MODEL_DIR, "cc.ko.300.model")

# Database Configuration
DB_HOST = os.getenv('dbhost')
DB_PORT = int(os.getenv('dbport'))
DB_NAME = os.getenv('dbname')
DB_USER = os.getenv('dbuser')
DB_PASSWORD = os.getenv('dbpass')

# News freshness period (in days)
NEWS_FRESHNESS_DAYS = 5