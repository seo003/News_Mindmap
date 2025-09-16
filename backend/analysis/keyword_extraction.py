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


def extract_keywords(crawl_data):
    news_data = crawl_data
    stopwords = load_stopwords(STOPWORDS_PATH)
    okt = Okt()

    print("뉴스 제목 토큰 추출 중...")
    for i, article in enumerate(news_data):
        title = article["title"]

        # 명사 추출
        nouns_only = [
            word for word, tag in okt.pos(title, stem=True)
            if tag == "Noun" and word not in stopwords and len(word) > 1
        ]
        article["tokens"] = list(dict.fromkeys(nouns_only))

        # print(f"\n--- 뉴스 {i+1} ---")
        # print(f"제목: '{title}'")
        # print(f"토큰: {article['tokens']}")

    print(f"토큰 추출 완료 (총 {len(news_data)}개 뉴스)")
    return news_data





# tf-idf 수정전---------
# def extract_keywords(crawl_data):
#     news_data = crawl_data
#     stopwords = load_stopwords(STOPWORDS_PATH)
#     okt = Okt()

#     print("뉴스 제목 토큰 추출 중...")
#     for i, article in enumerate(news_data):
#         title = article["title"]

#         # 1. 대학교 분류 등 명사 기반 매칭을 위한 명사만 추출
#         nouns_only = [
#             word for word, tag in okt.pos(title, stem=True)
#             if tag == "Noun" and word not in stopwords and len(word) > 1
#         ]
#         article["tokens_nouns_only"] = list(dict.fromkeys(nouns_only)) 

#         # 2. 벡터화 및 TF-IDF를 위한 명사, 동사, 형용사 추출
#         all_tokens = [
#             word for word, tag in okt.pos(title, stem=True)
#             if tag in ['Noun', 'Verb', 'Adjective'] and word not in stopwords and len(word) > 1
#         ]
#         article["tokens_for_vectors"] = all_tokens 

#         print(f"\n--- 뉴스 {i+1} ---")
#         print(f"제목: '{title}'")
#         print(f"  [명사만]: {article['tokens_nouns_only']}")
#         print(f"  [명사, 동사, 형용사]: {article['tokens_for_vectors']}")
        

#     print(f"토큰 추출 완료 (총 {len(news_data)}개 뉴스)")
#     return news_data


# tf-idf 수정후------
# def extract_keywords(crawl_data):
#     news_data = crawl_data
#     stopwords = load_stopwords(STOPWORDS_PATH)
#     okt = Okt()

#     print("뉴스 제목 토큰 추출 중...")
#     for i, article in enumerate(news_data):
#         title = article["title"]

#         all_tokens = [
#             word for word, tag in okt.pos(title, stem=True)
#             if tag in ['Noun', 'Verb', 'Adjective'] and word not in stopwords and len(word) > 1
#         ]
#         article["tokens"] = all_tokens 

#         print(f"\n--- 뉴스 {i+1} ---")
#         print(f"제목: '{title}'")
#         print(f"토큰: {article['tokens']}")
        

#     print(f"토큰 추출 완료 (총 {len(news_data)}개 뉴스)")
#     return news_data