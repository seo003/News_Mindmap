import os
import json
from konlpy.tag import Okt
from config.config import STOPWORDS_PATH

current_dir = os.path.dirname(os.path.abspath(__file__))
news_path = os.path.join(current_dir, "news.json")

def load_stopwords(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def extract_keywords():
    with open(news_path, "r", encoding="utf-8") as f:
        news_data = json.load(f)

    titles = [article["title"] for article in news_data]
    stopwords = load_stopwords(STOPWORDS_PATH)
    okt = Okt()

    tokenized_titles = []
    for title in titles:
        title_nouns = okt.nouns(title)
        filtered_nouns = [noun for noun in title_nouns if noun not in stopwords]
        unique_nouns = list(dict.fromkeys(filtered_nouns))
        tokenized_titles.append(unique_nouns)
        print(unique_nouns)

    print(f"✅ 키워드 추출 완료 (총 {len(tokenized_titles)}개)")    
    return titles, tokenized_titles
