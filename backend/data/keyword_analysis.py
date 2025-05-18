import os
from gensim import corpora
from gensim.models import LdaModel, FastText
from sklearn.cluster import KMeans
import numpy as np
from config.config import FASTTEXT_MODEL_PATH
import re
from collections import defaultdict, Counter

# 뉴스 기사가 하나도 없는 대분류, 중분류 항목 제거 함수 
def remove_empty_categories(categories):
    removed_categories = []

    MIDDLE_LIST_KEY = "middleKeywords"
    RELATED_NEWS_KEY = "relatedNews"

    for main_category in categories:
        valid_middle_categories = []

        # 중분류에 뉴스가 있는지 확인하고 유효한 중분류만 남김
        for middle_category in main_category.get(MIDDLE_LIST_KEY, []):
            if middle_category.get(RELATED_NEWS_KEY): 
                valid_middle_categories.append(middle_category)

        main_category[MIDDLE_LIST_KEY] = valid_middle_categories

        if main_category.get(MIDDLE_LIST_KEY):
            removed_categories.append(main_category)

    return removed_categories


# 평균 벡터를 계산하는 함수
def get_document_vector(w2v_model, tokens):
    valid_words = [word for word in tokens if word in w2v_model.wv]

    if not valid_words:
        return np.zeros(w2v_model.vector_size)
     
    return np.mean(w2v_model.wv[valid_words], axis=0)

def analyze_keywords(titles_with_links, tokenized_titles, num_final_topics=10, num_middle_keywords=5):
    print("키워드 분류 중...")
    MAJOR_KEY = "majorKeyword"
    MIDDLE_LIST_KEY = "middleKeywords"
    MIDDLE_ITEM_KEY = "middleKeyword"
    RELATED_NEWS_KEY = "relatedNews"
    OTHER_NEWS_KEY = "otherNews"

    final_categorized_results = [] 

    # 대학교 이름 기반 분리
    university_news_by_name = defaultdict(list)
    clustering_candidates_info = [] 

    uni_pattern = re.compile(r".+대$") 

    for idx, (item, tokens) in enumerate(zip(titles_with_links, tokenized_titles)):
        news_info = {"title": item["title"], "link": item["link"], "tokens": tokens}

        uni_kw = next((kw for kw in tokens if uni_pattern.match(kw)), None)
        if not uni_kw and "KAIST" in tokens: uni_kw = "KAIST" 

        if uni_kw:
            university_news_by_name[uni_kw].append(news_info)
        else:
            clustering_candidates_info.append(news_info)


    # 대학교 외 뉴스 KMeans 클러스터링
    clustered_news_by_label = defaultdict(list) 
    cluster_labels = []

    if clustering_candidates_info: 
        try:
            w2v_model = FastText.load(FASTTEXT_MODEL_PATH)
            print(f"FastText model loaded for clustering from {FASTTEXT_MODEL_PATH}")

            clustering_candidate_vectors = [get_document_vector(w2v_model, item["tokens"]) for item in clustering_candidates_info]
            clustering_candidate_vectors = np.array(clustering_candidate_vectors) 

            num_clusters_for_kmeans = num_final_topics - (1 if university_news_by_name else 0)
            num_clusters_for_kmeans = max(1, min(num_clusters_for_kmeans, len(clustering_candidates_info))) 

            if num_clusters_for_kmeans >= 1 and len(clustering_candidates_info) >= num_clusters_for_kmeans: 
                kmeans = KMeans(n_clusters=num_clusters_for_kmeans, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(clustering_candidate_vectors)

                for i, label in enumerate(cluster_labels):
                    clustered_news_by_label[label].append(clustering_candidates_info[i])

            else:
                print(f"Skipping KMeans: Not enough clustering candidates ({len(clustering_candidates_info)}) for {num_clusters_for_kmeans} clusters or num_clusters < 1.")
        except FileNotFoundError:
            print(f"FastText model file not found at {FASTTEXT_MODEL_PATH}. Skipping clustering.")
        except Exception as e:
            print(f"Error during FastText loading or KMeans: {e}. Skipping clustering.")
            clustered_news_by_label = {} 


    # 최종 대분류 구조 생성 및 뉴스 할당
    assigned_news_links_across_all_categories = set() 

    # '대학교' 대분류 추가
    if university_news_by_name:
        university_major_structure = {
            MAJOR_KEY: "대학교", 
             MIDDLE_LIST_KEY: [],
            OTHER_NEWS_KEY: []
        }
        all_uni_news = [item for sublist in university_news_by_name.values() for item in sublist]
        university_major_structure[OTHER_NEWS_KEY].extend(all_uni_news) 

        final_categorized_results.append(university_major_structure)


    # KMeans 클러스터링 기반 대분류 추가
    if clustered_news_by_label:
        sorted_cluster_labels = sorted(clustered_news_by_label.keys(), key=lambda k: len(clustered_news_by_label[k]), reverse=True)

        for cluster_label in sorted_cluster_labels:
            news_list = clustered_news_by_label[cluster_label]
            if not news_list: continue 

            major_name = f"클러스터 {cluster_label + 1}" 
            all_tokens_in_cluster = [token for item in news_list for token in item["tokens"]]
            token_counts = Counter(all_tokens_in_cluster)
            candidate_names = [word for word, count in token_counts.most_common(20) if len(word) > 1] 
            if candidate_names:
                for name_candidate in candidate_names:
                    if not any(res.get(MAJOR_KEY) == name_candidate for res in final_categorized_results) and name_candidate != "대학교":
                        major_name = name_candidate
                        break 

            cluster_major_structure = {
                MAJOR_KEY: major_name, 
                MIDDLE_LIST_KEY: [],
                OTHER_NEWS_KEY: []
            }
            cluster_major_structure[OTHER_NEWS_KEY].extend(news_list) 

            final_categorized_results.append(cluster_major_structure) 


    # 각 대분류 내에서 중분류 설정 및 뉴스 할당 
    for category in final_categorized_results:
        major_name = category.get(MAJOR_KEY)
        news_items_in_major = category.get(OTHER_NEWS_KEY, []) 
        category[MIDDLE_LIST_KEY] = [] 
        category[OTHER_NEWS_KEY] = [] 

        if not news_items_in_major: continue 

        # 중분류 후보 키워드 결정
        middle_candidate_kws = []
        if major_name == "대학교":
            middle_candidate_kws = list(university_news_by_name.keys())
        else:
            all_tokens_in_major = [token for item in news_items_in_major for token in item["tokens"]]
            token_counts = Counter(all_tokens_in_major)
            candidate_kws_from_tokens = [word for word, count in token_counts.most_common(num_middle_keywords * 3 + 5) if len(word) > 1 and word != major_name] 

            middle_candidate_kws = candidate_kws_from_tokens[:num_middle_keywords * 2] 


        # 중분류 구조 생성 및 뉴스 할당 
        middle_candidates_structure = []
        for middle_kw in middle_candidate_kws:
            middle_candidates_structure.append({
                MIDDLE_ITEM_KEY: middle_kw,
                RELATED_NEWS_KEY: []
            })

        unassigned_news_in_major = []
        for news_item in news_items_in_major: 
            news_link = news_item["link"]
            if news_link in assigned_news_links_across_all_categories:
                continue 

            assigned_to_middle_in_major = False
            for middle_item in middle_candidates_structure:
                middle_kw = middle_item.get(MIDDLE_ITEM_KEY)
                if middle_kw and middle_kw in news_item["tokens"]:
                    middle_item[RELATED_NEWS_KEY].append(news_item)
                    assigned_to_middle_in_major = True
                    assigned_news_links_across_all_categories.add(news_link) 
                    break 

            if not assigned_to_middle_in_major:
                unassigned_news_in_major.append(news_item)
                assigned_news_links_across_all_categories.add(news_link)


        # 중분류 필터링 
        valid_middle_categories = []
        news_from_removed_singletons = [] 

        for middle_item in middle_candidates_structure:
            if len(middle_item.get(RELATED_NEWS_KEY, [])) >= 2: # 뉴스가 2개 이상인 경우만 유효
                valid_middle_categories.append(middle_item)
            else: 
                news_from_removed_singletons.extend(middle_item.get(RELATED_NEWS_KEY, []))

        # 제거된 중분류의 뉴스들을 기타 목록에 추가
        unassigned_news_in_major.extend(news_from_removed_singletons)

        # 중분류 정렬
        sorted_middle_list_by_news_count = sorted(
            valid_middle_categories,
            key=lambda item: len(item.get(RELATED_NEWS_KEY, [])),
            reverse=True
        )

        # 중분류 개수 제한(10개)
        actual_num_middle_keywords = min(num_middle_keywords, 10)
        top_middle_categories = sorted_middle_list_by_news_count[:actual_num_middle_keywords] 

        # 그 외 뉴스를 기타 뉴스로 이동
        removed_middle_categories = sorted_middle_list_by_news_count[actual_num_middle_keywords:]
        for removed_item in removed_middle_categories:
            unassigned_news_in_major.extend(removed_item.get(RELATED_NEWS_KEY, []))

        # 최종 중분류 확정
        category[MIDDLE_LIST_KEY] = top_middle_categories

        # 기타 뉴스 목록 확정
        seen_other_news_links = set()
        final_other_news = []
        for news_item in unassigned_news_in_major:
            if news_item.get("link") and news_item["link"] not in seen_other_news_links:
                final_other_news.append(news_item)
                seen_other_news_links.add(news_item["link"])

        category[OTHER_NEWS_KEY] = final_other_news

    # 뉴스 기사가 없는 분류 항목 제거
    final_categorized_results = remove_empty_categories(final_categorized_results)

    # 결과 출력 
    print(f"\n--- 최종 분석 결과 (총 {len(final_categorized_results)}개 대분류) ---")
    if not final_categorized_results:
        print("뉴스 기사가 있는 분류 항목이 없습니다.")
        return []

    for category in final_categorized_results:
        main_cat_name = category.get(MAJOR_KEY, "알수없음")
        all_news_links_in_major_set = set()
        for news in category.get(OTHER_NEWS_KEY, []):
            if news and news.get("link"): all_news_links_in_major_set.add(news["link"])
        for m in category.get(MIDDLE_LIST_KEY, []):
            for news in m.get(RELATED_NEWS_KEY, []):
                if news and news.get("link"): all_news_links_in_major_set.add(news["link"])
        total_news_in_major = len(all_news_links_in_major_set)

        print(f"\n=== 대분류: {main_cat_name} (총 뉴스 {total_news_in_major}개) ===")

        if category.get(MIDDLE_LIST_KEY):
            print(" 하위_분류 (중분류):")
            for middle_item in category[MIDDLE_LIST_KEY]:
                middle_name = middle_item.get(MIDDLE_ITEM_KEY, "알수없음")
                middle_news = middle_item.get(RELATED_NEWS_KEY, [])
                middle_news_count = len(middle_news)
                if middle_news_count > 0:
                    print(f"   - 중분류: {middle_name} (뉴스 {middle_news_count}개)")
                    for news in middle_news[: min(3, middle_news_count)]:
                        if news and news.get('title'): 
                            print(f"     - {news['title']}")
                        else:
                            print(f"     - 유효하지 않은 뉴스 항목")


        if category.get(OTHER_NEWS_KEY):
            other_news_count = len(category[OTHER_NEWS_KEY])
            if other_news_count > 0:
                print(f"    기타 뉴스 ({other_news_count}개):")
                for news in category[OTHER_NEWS_KEY][: min(5, other_news_count)]:
                    if news and news.get('title'): 
                        print(f"- {news['title']}")
                    else:
                        print(f"- 유효하지 않은 뉴스 항목")

    return final_categorized_results
