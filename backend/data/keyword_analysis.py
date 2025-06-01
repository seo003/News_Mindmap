import re
from collections import defaultdict, Counter
import numpy as np
from gensim.models import FastText
from sklearn.cluster import KMeans
from config.config import FASTTEXT_MODEL_PATH, NON_UNIV_WORD_PATH

MAJOR_KEY = "majorKeyword"
MIDDLE_LIST_KEY = "middleKeywords"
MIDDLE_ITEM_KEY = "middleKeyword"
RELATED_NEWS_KEY = "relatedNews"
OTHER_NEWS_KEY = "otherNews"

"""토큰 리스트의 평균 벡터 계산"""
def get_document_vector(w2v_model, tokens):
    valid_words = [w for w in tokens if w in w2v_model.wv]
    if not valid_words:
        return np.zeros(w2v_model.vector_size)
    return np.mean(w2v_model.wv[valid_words], axis=0)


"""예외 키워드 목록을 텍스트 파일에서 불러오기"""
def load_exclude_words(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        return set(line.strip() for line in file if line.strip())


"""대학 이름 기반 뉴스 분류, '대'로 끝나는 단어와 예외 KAIST 처리"""
def split_news_by_uni_name(titles_with_links, tokenized_titles):
    univ_news = defaultdict(list)
    other_news = []
    # '대'로 끝나는 단어 정규표현식
    uni_pattern = re.compile(r".+대$")
    # 대학교 명이 아닌 제외단어 리스트
    exclude_words = load_exclude_words(NON_UNIV_WORD_PATH)

    for item, tokens in zip(titles_with_links, tokenized_titles):
        # news_info = {제목, 링크, 토큰}
        news_info = {"title": item["title"], "link": item["link"], "tokens": tokens}
        uni_kw = next((kw for kw in tokens if uni_pattern.match(kw) and kw not in exclude_words), None)

        if not uni_kw and "KAIST" in tokens:
            uni_kw = "KAIST"

        # 대학교 명이 있으면 univ_news {00대: [title, link, tokens]}
        if uni_kw:
            univ_news[uni_kw].append(news_info)
        else:
            # 없으면 other_news {title, link, tokens}
            other_news.append(news_info)
    return univ_news, other_news


"""FastText 임베딩으로 KMeans 클러스터링 실행"""
def cluster_news(kmeans_num, news_list, w2v_model):
    try:
        # 뉴스 tokens 벡터전환 
        vectors = np.array([get_document_vector(w2v_model, item["tokens"]) for item in news_list])
        # 클러스터 수 설정
        n_clusters = max(1, min(kmeans_num, len(news_list)))
        # kmeans 클러스터링
        if n_clusters > 0:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(vectors)
            clustered = defaultdict(list)
            for item, label in zip(news_list, labels):
                clustered[label].append(item)
            return clustered
        else:
            print(f"KMeans 클러스터링 스킵: 뉴스 수({len(news_list)})가 클러스터 수({kmeans_num})보다 적거나 0임")
            return {}
    except FileNotFoundError:
        print(f"FastText 모델을 찾을 수 없음: {FASTTEXT_MODEL_PATH}")
    except Exception as e:
        print(f"KMeans 또는 FastText 에러: {e}")
    return {}

""" TF-IDF 점수 계산 """
def calculate_tfidf_scores(all_tokens_in_clusters):
    print(f"\n총 클러스터 수: {len(all_tokens_in_clusters)}")
    # 문서-단어 행렬 생성
    doc_word_counts = {} # {클러스터_ID: Counter(단어빈도), ...}
    for label, tokens_list in all_tokens_in_clusters.items():
        doc_word_counts[label] = Counter(tokens_list)
    
    num_documents = len(doc_word_counts)
    
    # IDF 계산
    idf_scores = defaultdict(lambda: 0.0) # IDF 점수
    document_frequency = defaultdict(int) # 해당 단어를 포함하는 클러스터 수

    # 모든 단어에 대해 단어가 포함된 문서 수를 계산
    all_unique_tokens = set()
    for _, counts in doc_word_counts.items():
        for token in counts:
            all_unique_tokens.add(token)
            document_frequency[token] += 1
    
    print(f"총 고유 단어 수: {len(all_unique_tokens)}")
    for token in list(all_unique_tokens)[:5]: # 상위 5개 단어의 IDF 점수만 샘플 출력
        print(f"  {token}'의 DF: {document_frequency[token]}, IDF: {np.log(num_documents / (document_frequency[token] + 1)):.4f}")

    for token in all_unique_tokens:
        # 단어의 IDF 점수 계산
        idf_scores[token] = np.log(num_documents / (document_frequency[token] + 1)) # +1은 ZeroDivisionError 방지

    # 3. TF-IDF 점수 계산
    tfidf_results = defaultdict(dict)
    for label, word_counts in doc_word_counts.items():
        total_words_in_doc = sum(word_counts.values())
        if total_words_in_doc == 0: continue

        for word, count in word_counts.items():
            tf = count / total_words_in_doc
            tfidf_results[label][word] = tf * idf_scores[word]
            
    # tfidf_results: {클러스터_ID: {단어: tfidf_score, ...}, ...} 
    return tfidf_results

"""대분류 카테고리 생성 (대학교 및 클러스터)"""
def create_major_categories(uni_news, clustered_news):
    results = []
    
    # TF-IDF 계산을 위한 전체 클러스터별 토큰 모음
    all_tokens_in_clusters = defaultdict(list)
    for label, news_list in clustered_news.items():
        for news in news_list:
            all_tokens_in_clusters[label].extend(news["tokens"])

    # all_tokens_in_clusters: {클러스터_ID: [단어1, 단어2, ...], ...}
    tfidf_scores_by_cluster = calculate_tfidf_scores(all_tokens_in_clusters)

    if uni_news:
        uni_category = {
            MAJOR_KEY: "대학교",
            MIDDLE_LIST_KEY: [],
            OTHER_NEWS_KEY: []
        }
        for uni_name, news_list in uni_news.items():
            if not news_list:
                continue
            uni_category[MIDDLE_LIST_KEY].append({
                MIDDLE_ITEM_KEY: uni_name,
                RELATED_NEWS_KEY: news_list
            })
        results.append(uni_category)

    if clustered_news:
        sorted_labels = sorted(clustered_news.keys(), key=lambda k: len(clustered_news[k]), reverse=True)
        for label in sorted_labels:
            news_list = clustered_news[label]
            if not news_list:
                continue
            
            major_name = f"클러스터 {label + 1}" # 기본 대분류 이름
            print(f"\n 클러스터 {label} 처리 중 (뉴스 {len(news_list)}개)")

            # 현재 클러스터의 TF-IDF 점수 가져오기
            cluster_tfidf_scores = tfidf_scores_by_cluster.get(label, {})
            
            # TF-IDF 점수가 높은 단어들을 후보로 선정 (기존 대분류와 겹치지 않는 단어)
            sorted_tfidf_words = sorted(cluster_tfidf_scores.items(), key=lambda item: item[1], reverse=True)
            print(f" TF-IDF 상위 5개 단어: {sorted_tfidf_words[:5]}")
            
            # 여기서 TF-IDF 점수가 높은 상위 N개의 단어를 대분류 이름 후보로 사용
            candidate_words = [
                word for word, score in sorted_tfidf_words
                if len(word) > 1 and word not in load_exclude_words(NON_UNIV_WORD_PATH) and word != "대학교"
            ]
            print(f"대분류 이름 후보 (상위 5개): {candidate_words[:5]}")

            # 최종 대분류 이름 선정
            for cand in candidate_words:
                if not any(cat.get(MAJOR_KEY) == cand for cat in results):
                    major_name = cand
                    break
            print(f"'{major_name}' 대분류 확정")
            results.append({
                MAJOR_KEY: major_name,
                MIDDLE_LIST_KEY: [],
                OTHER_NEWS_KEY: news_list
            })
    return results

""" 중분류 개수 제한 및 기타 뉴스 분리 처리 """
def trim_middle_categories(category, num_middle_keywords):
   
    middle_categories = category.get(MIDDLE_LIST_KEY, [])
    unassigned_news = []

    # 뉴스 수 기준 정렬
    middle_categories = sorted(middle_categories, key=lambda c: len(c[RELATED_NEWS_KEY]), reverse=True)
    top_middle = middle_categories[:min(num_middle_keywords, 10)]
    removed = middle_categories[min(num_middle_keywords, 10):]

    # 제거된 중분류 뉴스들을 기타 뉴스로 이동
    for rem in removed:
        unassigned_news.extend(rem.get(RELATED_NEWS_KEY, []))

    # 기타 뉴스 중복 제거
    seen_links = set()
    final_others = []
    for news in unassigned_news:
        link = news.get("link")
        if link and link not in seen_links:
            final_others.append(news)
            seen_links.add(link)

    # 최종 반영
    category[MIDDLE_LIST_KEY] = top_middle
    category[OTHER_NEWS_KEY] = final_others


"""각 대분류 내에서 중분류 키워드 선정 및 뉴스 할당"""
def assign_middle_categories(category, num_middle_keywords, w2v_model):
    major_name = category.get(MAJOR_KEY)

    # '대학교' 대분류는 중분류 생성 없이 트리밍만 수행
    if major_name == "대학교":
        trim_middle_categories(category, num_middle_keywords)
        return
    
    news_list = category.get(OTHER_NEWS_KEY, [])
    category[MIDDLE_LIST_KEY] = []
    category[OTHER_NEWS_KEY] = []

    if not news_list:
        return

    # 클러스터링을 위한 뉴스 목록
    news_for_clustering = news_list

    # 중분류 클러스터 생성
    clustered_middle_news = cluster_news(num_middle_keywords, news_for_clustering, w2v_model)
    print(f"중분류를 위해 {len(news_for_clustering)}개의 뉴스를 {len(clustered_middle_news)}개 클러스터로 분류 완료.")

     # 클러스터링 실패 시 기타 뉴스로 저장
    if not clustered_middle_news:
        category[OTHER_NEWS_KEY] = news_list
        return

    # 각 클러스터터 TF-IDF 점수 계산
    all_tokens_in_middle_clusters = defaultdict(list)
    for label, news_items in clustered_middle_news.items():
        for news in news_items:
            all_tokens_in_middle_clusters[label].extend(news["tokens"])

    tfidf_scores_by_middle_cluster = calculate_tfidf_scores(all_tokens_in_middle_clusters)

    middle_categories_result = []
    assigned_links = set()

    # 각 클러스터 별 중분류 키워드 선정
    for label in sorted(clustered_middle_news.keys(), key=lambda k: len(clustered_middle_news[k]), reverse=True):
        cluster_news_items = clustered_middle_news[label]
        if not cluster_news_items:
            continue

        middle_name = f"클러스터 {label + 1}"
        middle_cluster_tfidf_scores = tfidf_scores_by_middle_cluster.get(label, {})
        
        sorted_tfidf_words = sorted(middle_cluster_tfidf_scores.items(), key=lambda item: item[1], reverse=True)
        print(f"    - TF-IDF 상위 5개 단어: {sorted_tfidf_words[:5]}")
        
        # 중분류 이름 후보 선정
        candidate_words = [
            word for word, score in sorted_tfidf_words
            if len(word) > 1 and word not in load_exclude_words(NON_UNIV_WORD_PATH) and word != major_name
        ]
        print(f"    - 중분류 이름 후보 (상위 5개): {candidate_words[:5]}")

        # 최종 중분류 결정
        for cand in candidate_words:
            if not any(mid_cat.get(MIDDLE_ITEM_KEY) == cand for mid_cat in middle_categories_result):
                middle_name = cand
                break
        print(f"    - '{middle_name}' 중분류 확정")

        middle_categories_result.append({
            MIDDLE_ITEM_KEY: middle_name,
            RELATED_NEWS_KEY: cluster_news_items
        })
        for news in cluster_news_items:
            assigned_links.add(news.get("link"))

    # 중분류에 할당되지 않은 뉴스
    unassigned_news = [news for news in news_list if news.get("link") not in assigned_links]

    category[MIDDLE_LIST_KEY] = middle_categories_result
    category[OTHER_NEWS_KEY] = unassigned_news
    
    print(f"대분류 '{major_name}'")
    print(f"  중분류 할당 결과: {len(category[MIDDLE_LIST_KEY])}개 중분류, {len(category[OTHER_NEWS_KEY])}개 기타 뉴스")


    # 중분류 개수 제한 및 기타 뉴스 분리 처리
    trim_middle_categories(category, num_middle_keywords)
    print(f"최종 중분류 수: {len(category[MIDDLE_LIST_KEY])}")


"""중분류가 없거나 뉴스가 없는 대분류 제거"""
def remove_empty_categories(categories):
    cleaned = []
    for major in categories:
        valid_middles = [m for m in major.get(MIDDLE_LIST_KEY, []) if m.get(RELATED_NEWS_KEY)]
        major[MIDDLE_LIST_KEY] = valid_middles
        if major.get(MIDDLE_LIST_KEY):
            cleaned.append(major)
    return cleaned
    
def print_analysis_results(final_results):
    """최종 결과 출력 함수"""
    print(f"\n--- 최종 분석 결과 (총 {len(final_results)}개 대분류) ---")
    if not final_results:
        print("뉴스 기사가 있는 분류 항목이 없습니다.")
        return

    for category in final_results:
        major_name = category.get(MAJOR_KEY, "알수없음")
        all_news_links = set()
        for news in category.get(OTHER_NEWS_KEY, []):
            if news and news.get("link"):
                all_news_links.add(news["link"])
        for mid_cat in category.get(MIDDLE_LIST_KEY, []):
            for news in mid_cat.get(RELATED_NEWS_KEY, []):
                if news and news.get("link"):
                    all_news_links.add(news["link"])

        print(f"\n=== 대분류: {major_name} (총 뉴스 {len(all_news_links)}개) ===")

        if category.get(MIDDLE_LIST_KEY):
            print(" 하위_분류 (중분류):")
            for mid_cat in category[MIDDLE_LIST_KEY]:
                mid_name = mid_cat.get(MIDDLE_ITEM_KEY, "알수없음")
                mid_news = mid_cat.get(RELATED_NEWS_KEY, [])
                if len(mid_news) > 0:
                    print(f"   - 중분류: {mid_name} (뉴스 {len(mid_news)}개)")
                    for news in mid_news[:3]:
                        print(f"     - {news.get('title', '유효하지 않은 뉴스 항목')}")

        if category.get(OTHER_NEWS_KEY):
            print(f" 기타 뉴스 (총 {len(category[OTHER_NEWS_KEY])}개)")
            for news in category[OTHER_NEWS_KEY][:3]:
                print(f"   - {news.get('title', '유효하지 않은 뉴스 항목')}")


def analyze_keywords(titles_with_links, tokenized_titles, num_final_topics=10, num_middle_keywords=5):
    print("키워드 분류 중...")

    # FastText 모델을 여기서 한 번만 로드
    w2v_model = None
    try:
        w2v_model = FastText.load(FASTTEXT_MODEL_PATH)
        print(f"FastText 모델 로드 성공: {FASTTEXT_MODEL_PATH}")
    except FileNotFoundError:
        print(f"FastText 모델을 찾을 수 없음: {FASTTEXT_MODEL_PATH}")
    except Exception as e:
        print(f"FastText 모델 로딩 중 오류 발생: {e}")


    # 1. 대학 이름 기반 분류와 기타 뉴스 분리
    uni_news, other_news = split_news_by_uni_name(titles_with_links, tokenized_titles)
    print("대학교 및 기타 뉴스 분리 완료...")

    # 2. 기타 뉴스 KMeans 클러스터링
    clustered_news = {}
    if w2v_model:
        print("기타 뉴스 KMeans 클러스터링 시작")
        clustered_news = cluster_news(num_final_topics, other_news, w2v_model)
    else:
        print("FastText 모델 로드 실패 KMeans 클러스터링을 건너뜁니다.")

    # 3. 대분류 카테고리 생성
    major_categories = create_major_categories(uni_news, clustered_news)
    print("대분류 생성 완료...")

    # 4. 중분류 할당 및 기타 뉴스 분리
    print("중분류 할당 및 뉴스 분리 시작")
    for major in major_categories:
        assign_middle_categories(major, num_middle_keywords, w2v_model) 
    print("중분류 할당 완료")

    # 5. 중분류 없는 대분류 제거
    final_results = remove_empty_categories(major_categories)
    print("대분류 걸러내기 완료...")

    # 6. 결과 출력
    print_analysis_results(final_results)

    return final_results
