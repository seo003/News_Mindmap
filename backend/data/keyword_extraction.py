import json
import os
from konlpy.tag import Okt
from collections import Counter
from gensim import corpora
from gensim.models import LdaModel

# 현재 파일 위치를 기준으로 절대 경로 계산
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "news.json")

# JSON 파일에서 뉴스 제목 가져오기
with open(file_path, "r", encoding="utf-8") as f:
    news_data = json.load(f)

# 제목 리스트 추출
titles = [article["title"] for article in news_data]

# 불용어 설정정
stopwords = ["보도자료", "뉴스", "기사", "속보", "발표", "대한", "한국"]

# 형태소 분석기
okt = Okt()

# 뉴스 제목에서 명사만 추출하고, 불용어를 제외
tokenized_titles = []
for title in titles:
    title_nouns = okt.nouns(title)
    filtered_nouns = [noun for noun in title_nouns if noun not in stopwords]
    tokenized_titles.append(filtered_nouns)
    print(filtered_nouns)

# # LDA를 위한 사전 만들기
# dictionary = corpora.Dictionary(tokenized_titles)
# corpus = [dictionary.doc2bow(text) for text in tokenized_titles]

# # LDA 모델 학습 (토픽 3개)
# lda_model = LdaModel(corpus, num_topics=3, id2word=dictionary, passes=15)

# # 각 뉴스의 토픽 예측
# for i, text in enumerate(corpus):
#     topic_probs = lda_model[text]
#     top_topic = max(topic_probs, key=lambda x: x[1])  # 확률이 가장 높은 토픽 선택
#     print(f"뉴스: {titles[i]} → 대분류: 토픽 {top_topic[0]}")
