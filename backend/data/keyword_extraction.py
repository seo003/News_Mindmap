import os
import json
from konlpy.tag import Okt
from config.config import STOPWORDS_PATH

current_dir = os.path.dirname(os.path.abspath(__file__))
news_path = os.path.join(current_dir, "news.json")

def load_stopwords(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def extract_nouns_full(title, stopwords):
    okt = Okt()
    tokens = okt.pos(title, stem=True)
    nouns = []

    for word, tag in tokens:
        if tag == "Noun" and word not in stopwords and len(word) > 1:
            nouns.append(word)

    return nouns

def extract_keywords():
    with open(news_path, "r", encoding="utf-8") as f:
        news_data = json.load(f)

    titles = [article["title"] for article in news_data]
    stopwords = load_stopwords(STOPWORDS_PATH)

    tokenized_titles = []
    for title in titles:
        nouns = extract_nouns_full(title, stopwords)
        unique_nouns = list(dict.fromkeys(nouns))
        tokenized_titles.append(unique_nouns)
        print(unique_nouns)

    print(f"키워드 추출 완료 (총 {len(tokenized_titles)}개)")    
    return news_data, tokenized_titles
