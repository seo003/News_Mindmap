import os
from gensim import corpora
from gensim.models import LdaModel, FastText
from sklearn.cluster import KMeans
import numpy as np
from config.config import FASTTEXT_MODEL_PATH
import re
from collections import defaultdict

def get_topic_vector(w2v_model, keywords):
    vectors = [w2v_model.wv[word] for word in keywords if word in w2v_model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(w2v_model.vector_size)

def analyze_keywords(titles_with_links, tokenized_titles):
    result = defaultdict(lambda: {"소분류": set(), "뉴스": []})
    uni_pattern = re.compile(r".+대$")

    remaining_tokens = []
    remaining_title_info = []

    for idx, (item, tokens) in enumerate(zip(titles_with_links, tokenized_titles)):
        title = item["title"]
        link = item["link"]

        uni_kw = next((kw for kw in tokens if uni_pattern.match(kw)), None)

        if uni_kw:
            main_kw = uni_kw
        elif tokens:
            main_kw = tokens[0]
        else:
            continue  # 키워드 없는 뉴스는 건너뜀

        sub_keywords = [kw for kw in tokens if kw != main_kw]

        result[main_kw]["소분류"].update(sub_keywords)
        result[main_kw]["뉴스"].append({"title": title, "link": link})

        # 대학교 키워드 없을 경우 LDA 용으로 따로 저장
        if not uni_kw:
            remaining_tokens.append(tokens)
            remaining_title_info.append({"title": title, "link": link, "tokens": tokens})

    # LDA + KMeans 분석
    if remaining_tokens:
        dictionary = corpora.Dictionary(remaining_tokens)
        corpus = [dictionary.doc2bow(text) for text in remaining_tokens]
        lda_model = LdaModel(corpus, num_topics=10, id2word=dictionary, passes=15)

        topics = []
        for topic_id in range(lda_model.num_topics):
            topic = [word for word, _ in lda_model.show_topic(topic_id, topn=5)]
            if topic:
                topics.append(topic)

        w2v_model = FastText.load(FASTTEXT_MODEL_PATH)
        topic_vectors = [get_topic_vector(w2v_model, topic) for topic in topics]
        kmeans = KMeans(n_clusters=3, random_state=42)
        labels = kmeans.fit_predict(topic_vectors)

        clustered = defaultdict(list)
        for i, label in enumerate(labels):
            main = topics[i][0]
            subs = topics[i][1:]
            clustered[main].extend(subs)

        # 클러스터링 결과 등록
        for main, subs in clustered.items():
            result[main]["소분류"].update(subs)

        # 뉴스 분배
        for item in remaining_title_info:
            title, link, tokens = item["title"], item["link"], item["tokens"]
            matched_main = None

            for main in clustered:
                if any(tok in clustered[main] or tok == main for tok in tokens):
                    matched_main = main
                    break

            if matched_main:
                result[matched_main]["뉴스"].append({"title": title, "link": link})

    # set → list
    for val in result.values():
        val["소분류"] = list(val["소분류"])

    for main_category, data in result.items():
        print(f"\n 대분류: {main_category}")
        print(f"소분류: {', '.join(data['소분류'])}")
        print("관련 뉴스:")
        for news in data["뉴스"]:
            print(f"-  {news['title']}")
            print(f"   {news['link']}")
    return result