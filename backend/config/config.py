import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
MODEL_DIR = os.path.join(BASE_DIR, "models")
CONFIG_DIR = os.path.join(BASE_DIR, "config")

STOPWORDS_PATH = os.path.join(CONFIG_DIR, "stopwords.txt")
FASTTEXT_MODEL_PATH = os.path.join(MODEL_DIR, "cc.ko.300.model")