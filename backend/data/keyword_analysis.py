import re
from collections import defaultdict, Counter
import numpy as np
from gensim.models import FastText
from sklearn.cluster import KMeans
from config.config import FASTTEXT_MODEL_PATH

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


"""대학 이름 기반 뉴스 분류, '대'로 끝나는 단어와 예외 KAIST 처리"""
def split_news_by_uni_name(titles_with_links, tokenized_titles):
    univ_news = defaultdict(list)
    other_news = []
    # '대'로 끝나는 단어 정규표현식
    uni_pattern = re.compile(r".+대$")
    for item, tokens in zip(titles_with_links, tokenized_titles):
        # news_info = {제목, 링크, 토큰}
        news_info = {"title": item["title"], "link": item["link"], "tokens": tokens}
        uni_kw = next((kw for kw in tokens if uni_pattern.match(kw)), None)
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
def cluster_news(kmeans_num, news_list):
    try:
        # FastText 모델 불러오기기
        w2v_model = FastText.load(FASTTEXT_MODEL_PATH)
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


"""대분류 카테고리 생성 (대학교 및 클러스터)"""
def create_major_categories(uni_news, clustered_news):
    results = []
    # '대학교' 대분류 구조 생성
    if uni_news:
        uni_category = {
            MAJOR_KEY: "대학교",
            MIDDLE_LIST_KEY: [],
            OTHER_NEWS_KEY: []
        }
        # all_uni_news: [{title, link, tokens}, {}]
        all_uni_news = [item for news_list in uni_news.values() for item in news_list]
        # 기타뉴스에 저장
        uni_category[OTHER_NEWS_KEY].extend(all_uni_news)
        results.append(uni_category)

    # 근데 이걸 그냥 all_uni_news로 하지 말고 이미 '학교명' : title, link, tokens 있으니까 그대로 중분류 넣으면 되는거 아님?

    # KMeans 클러스터링 기반 대분류 추가
    if clustered_news:
        sorted_labels = sorted(clustered_news.keys(), key=lambda k: len(clustered_news[k]), reverse=True)
        # 클러스터 별 대분류 생성
        for label in sorted_labels:
            # 뉴스 없는 클러스터는 제외
            # news_list: [{title, link, tokens}, {}]
            news_list = clustered_news[label]
            if not news_list:
                continue
            
            # 기본 대분류 이름: 클러스터 n
            major_name = f"클러스터 {label + 1}"
            # 클러스터 내 모든 tokens 수집 및 빈도 계산
            all_tokens = [token for news in news_list for token in news["tokens"]]
            token_counts = Counter(all_tokens)
            # 후보 이름 선정
            candidates = [word for word, _ in token_counts.most_common(20) if len(word) > 1]

            # 기존 대분류 이름과 겹치지 않는 단어 사용 
            for cand in candidates:
                if cand != "대학교" and not any(cat.get(MAJOR_KEY) == cand for cat in results):
                    major_name = cand
                    break

            # 대분류 구조 생성 및 할당
            results.append({
                MAJOR_KEY: major_name,
                MIDDLE_LIST_KEY: [],
                OTHER_NEWS_KEY: news_list
            })

    return results


"""각 대분류 내에서 중분류 키워드 선정 및 뉴스 할당"""
def assign_middle_categories(category, university_news_by_name, num_middle_keywords):
    # 대분류 이름, 해당 뉴스 가져오기 및 초기화 
    major_name = category.get(MAJOR_KEY)
    news_list = category.get(OTHER_NEWS_KEY, [])
    category[MIDDLE_LIST_KEY] = []
    category[OTHER_NEWS_KEY] = []

    # 뉴스 없으면 건너뛰기
    if not news_list:
        return

    # 중분류 후보 키워드 선정
    if major_name == "대학교":
        # 대학교 일 경우 key('00대')를 중분류로 사용
        middle_keywords = list(university_news_by_name.keys())
    else:
        # tokens의 빈도수 기반으로 중분류 설정
        all_tokens = [token for news in news_list for token in news["tokens"]]
        token_counts = Counter(all_tokens)
        candidates = [w for w, _ in token_counts.most_common(num_middle_keywords * 3 + 5) if len(w) > 1 and w != major_name]
        middle_keywords = candidates[:num_middle_keywords * 2]

    # 중분류 구조 생성
    middle_categories = [{
        MIDDLE_ITEM_KEY: kw,
        RELATED_NEWS_KEY: []
    } for kw in middle_keywords]

    # 기타 뉴스 저장 리스트
    unassigned_news = []
    assigned_links = set()

    for news in news_list:
        # 다른 중분류에 할당된 뉴스 건너뛰기(중복 방지)
        news_link = news.get("link")
        if news_link in assigned_links:
            continue
        
        # 중분류 키워드 매칭
        assigned = False
        for mid_cat in middle_categories:
            # tokens 안에 중분류 키워드가 있다면 해당 뉴스를 추가
            if mid_cat[MIDDLE_ITEM_KEY] in news["tokens"]:
                mid_cat[RELATED_NEWS_KEY].append(news)
                assigned = True
                assigned_links.add(news_link)
                break
        
        # 해당하지 않으면 기타 목록에 뉴스 추가
        if not assigned:
            unassigned_news.append(news)
            assigned_links.add(news_link)

    # 중분류 필터링 
    # valid_middle_categories = []
    # news_from_removed_singletons = [] 

    # for middle_item in middle_candidates_structure:
    #     if len(middle_item.get(RELATED_NEWS_KEY, [])) >= 2: # 뉴스가 2개 이상인 경우만 유효
    #         valid_middle_categories.append(middle_item)
    #     else: 
    #         news_from_removed_singletons.extend(middle_item.get(RELATED_NEWS_KEY, []))

    # # 제거된 중분류의 뉴스들을 기타 목록에 추가
    # unassigned_news_in_major.extend(news_from_removed_singletons)


    # 중분류 정렬(뉴스 수 기준) 및 상위 중분류 유지, 나머지는 기타로 이동
    middle_categories = sorted(middle_categories, key=lambda c: len(c[RELATED_NEWS_KEY]), reverse=True)
    top_middle = middle_categories[:min(num_middle_keywords, 10)]
    removed = middle_categories[min(num_middle_keywords, 10):]

    for rem in removed:
        unassigned_news.extend(rem[RELATED_NEWS_KEY])

    # 최종 중분류 확정정
    category[MIDDLE_LIST_KEY] = top_middle

    # 기타 뉴스 중복 제거 후 저장
    seen_links = set()
    final_others = []
    for news in unassigned_news:
        link = news.get("link")
        if link and link not in seen_links:
            final_others.append(news)
            seen_links.add(link)

    category[OTHER_NEWS_KEY] = final_others


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

    # 1. 대학 이름 기반 분류와 기타 뉴스 분리
    uni_news, other_news = split_news_by_uni_name(titles_with_links, tokenized_titles)

    # 2. 기타 뉴스 KMeans 클러스터링
    clustered_news = cluster_news(num_final_topics, other_news)

    # 3. 대분류 카테고리 생성
    major_categories = create_major_categories(uni_news, clustered_news)

    # 4. 중분류 할당 및 기타 뉴스 분리
    for major in major_categories:
        assign_middle_categories(major, uni_news, num_middle_keywords)

    # 5. 중분류 없는 대분류 제거
    final_results = remove_empty_categories(major_categories)

    # 6. 결과 출력
    print_analysis_results(final_results)

    return final_results
