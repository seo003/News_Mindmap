import os
from gensim import corpora
from gensim.models import LdaModel, FastText
from sklearn.cluster import KMeans
import numpy as np
from config.config import FASTTEXT_MODEL_PATH, EXCLUDED_UNIVERSITY_PATH
import re
from collections import defaultdict

def load_excluded_universities(filepath):
    if not os.path.exists(filepath):
        return set()
    with open(filepath, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f if line.strip())

def get_topic_vector(w2v_model, keywords):
    vectors = [w2v_model.wv[word] for word in keywords if word in w2v_model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(w2v_model.vector_size)

def extract_university_keywords(tokenized_titles):
    uni_keywords = defaultdict(list)
    other_titles = []

    # 제외할 대학교 유사 키워드 불러오기
    excluded_unis = load_excluded_universities(EXCLUDED_UNIVERSITY_PATH)

    for tokens in tokenized_titles:
        unis = [word for word in tokens if re.match(r'.+대$', word) and word not in excluded_unis]
        if unis:
            for uni in unis:
                related_keywords = [t for t in tokens if t != uni]
                uni_keywords[uni].extend(related_keywords)
        else:
            other_titles.append(tokens)

    return uni_keywords, other_titles


def analyze_keywords(titles, tokenized_titles):
    # === 대학교 기반 분류 ===
    uni_keywords, remaining_titles = extract_university_keywords(tokenized_titles)

    print("\n=== [1] 대학교 기반 대분류/소분류 ===")
    for uni, keywords in uni_keywords.items():
        print(f"\n[대분류: {uni}]")
        print("→ 소분류 키워드:", ', '.join(set(keywords)))

    # === 기타 뉴스 처리 ===
    if not remaining_titles:
        print("\n※ 모든 키워드가 대학 기반으로 분류됨")
        return

    dictionary = corpora.Dictionary(remaining_titles)
    corpus = [dictionary.doc2bow(text) for text in remaining_titles]
    lda_model = LdaModel(corpus, num_topics=10, id2word=dictionary, passes=15)

    topics = []
    print("\n=== [2] 기타 키워드 LDA 토픽 ===")
    for topic_id in range(lda_model.num_topics):
        keywords = lda_model.show_topic(topic_id, topn=5)
        keyword_list = [word for word, _ in keywords]
        if keyword_list:  # 비어있지 않은 경우만
            topics.append(keyword_list)
            print(f"토픽 {topic_id}: {', '.join(keyword_list)}")

    # === FastText 로드 및 클러스터링 ===
    w2v_model = FastText.load(FASTTEXT_MODEL_PATH)
    topic_vectors = [get_topic_vector(w2v_model, topic) for topic in topics]
    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(topic_vectors)

    # === 클러스터별 대분류/소분류 지정 ===
    print("\n=== [3] 기타 키워드 기반 대분류/소분류 ===")
    clustered_topics = defaultdict(list)

    for i, label in enumerate(labels):
        topic_keywords = topics[i]
        if not topic_keywords:
            continue
        main_keyword = topic_keywords[0]
        sub_keywords = topic_keywords[1:]
        clustered_topics[main_keyword].extend(sub_keywords)

    for main, subs in clustered_topics.items():
        print(f"\n[대분류: {main}]")
        print("→ 소분류 키워드:", ', '.join(set(subs)))
