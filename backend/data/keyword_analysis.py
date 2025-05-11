import os
from gensim import corpora
from gensim.models import LdaModel
from sklearn.cluster import KMeans
import numpy as np
from config.config import FASTTEXT_MODEL_PATH
import re
from collections import defaultdict, Counter

def remove_empty_categories(categories):
    """
    뉴스 기사가 하나도 없는 대분류, 중분류 항목 제거
    """
    removed_categories = []

    MAJOR_KEY = "majorKeyword"
    MIDDLE_LIST_KEY = "middleKeywords"
    RELATED_NEWS_KEY = "relatedNews"
    OTHER_NEWS_KEY = "otherNews"

    for daefunryu in categories:
        removed_middle_categories = []

        for trungfunryu in daefunryu.get(MIDDLE_LIST_KEY, []):
            if trungfunryu.get(RELATED_NEWS_KEY):
                 removed_middle_categories.append(trungfunryu)

        daefunryu[MIDDLE_LIST_KEY] = removed_middle_categories

        if daefunryu.get(OTHER_NEWS_KEY) or daefunryu.get(MIDDLE_LIST_KEY):
            removed_categories.append(daefunryu)

    return removed_categories


def analyze_keywords(titles_with_links, tokenized_titles, num_final_topics=10, num_middle_keywords=5):
    MAJOR_KEY = "majorKeyword"
    MIDDLE_LIST_KEY = "middleKeywords"
    MIDDLE_ITEM_KEY = "middleKeyword"
    RELATED_NEWS_KEY = "relatedNews"
    OTHER_NEWS_KEY = "otherNews"

    final_categorized_results = []

    primary_topic_groups = defaultdict(lambda: {"newsItems": [], "tokens": [], "type": "topic", "ldaId": None})

    uni_pattern = re.compile(r".+대$")
    news_info_by_index = {}
    lda_analysis_tokens_list = []
    non_university_news_indices = []


    # 대학교 이름 기반 키워드 분리
    for idx, (item, tokens) in enumerate(zip(titles_with_links, tokenized_titles)):
        news_info_by_index[idx] = {"title": item["title"], "link": item["link"], "tokens": tokens}

        uni_kw = next((kw for kw in tokens if uni_pattern.match(kw)), None)
        if not uni_kw and "KAIST" in tokens: uni_kw = "KAIST"

        if uni_kw:
            primary_topic_groups[uni_kw]["newsItems"].append(news_info_by_index[idx])
            primary_topic_groups[uni_kw]["tokens"].append(tokens)
            primary_topic_groups[uni_kw]["type"] = "university"
        else:
            lda_analysis_tokens_list.append(tokens)
            non_university_news_indices.append(idx)


    # 대학교 외 키워드 LDA 분석
    lda_topics_keywords = []
    lda_topic_distributions = []
    lda_model = None

    if lda_analysis_tokens_list and len(lda_analysis_tokens_list) >= 2:
        lda_dictionary = corpora.Dictionary(lda_analysis_tokens_list)
        min_no_below = max(1, len(lda_analysis_tokens_list) // 100)
        lda_dictionary.filter_extremes(no_below=min_no_below, no_above=0.5)
        lda_corpus = [lda_dictionary.doc2bow(text) for text in lda_analysis_tokens_list]

        if lda_corpus:
            num_lda_topics_to_extract = min(len(lda_corpus), max(2, num_final_topics * 2))
            if num_lda_topics_to_extract >= 2:
                 try:
                    lda_model = LdaModel(lda_corpus, num_topics=num_lda_topics_to_extract, id2word=lda_dictionary, passes=20, random_state=42)

                    num_candidate_middle_keywords = max(num_middle_keywords * 4, 20)
                    for topic_id in range(lda_model.num_topics):
                        topic_words = [word for word, _ in lda_model.show_topic(topic_id, topn=num_candidate_middle_keywords + 5)]
                        if topic_words:
                            lda_topics_keywords.append(topic_words)
                            topic_name = topic_words[0]
                            if topic_name not in primary_topic_groups or primary_topic_groups[topic_name]["type"] != "university":
                                primary_topic_groups[topic_name]["type"] = "topic"
                                primary_topic_groups[topic_name]["ldaId"] = topic_id


                    lda_topic_distributions = [lda_model.get_document_topics(doc, minimum_probability=0.1) for doc in lda_corpus]

                 except ValueError as e:
                     print(f"LDA 모델 학습 중 오류 발생: {e}. LDA 분석 결과를 사용하지 않습니다.")
                     lda_topics_keywords = []
                     lda_topic_distributions = []
                     lda_model = None
                 except Exception as e:
                     print(f"LDA 처리 중 예상치 못한 오류 발생: {e}. LDA 분석 결과를 사용하지 않습니다.")
                     lda_topics_keywords = []
                     lda_topic_distributions = []
                     lda_model = None
            else:
                 print(f"뉴스 문서 수가 부족하거나 설정된 토픽 수가 너무 적습니다. 분석을 건너뜁니다.")
        else:
             print(f"키워드 목록이 비어있습니다. 분석을 건너뜁니다.")
    else:
         print(f"LDA 분석 대상 뉴스가 없거나 너무 적습니다 (need >= 2). LDA 분석을 건너뜁니다.")


    # 가장 적합한 LDA 토픽 할당
    if lda_topic_distributions:
         for i, news_idx in enumerate(non_university_news_indices):
             topic_dist = lda_topic_distributions[i]
             best_lda_topic_id = -1
             max_lda_prob = 0

             for topic_id, prob in topic_dist:
                 if lda_topics_keywords and topic_id < len(lda_topics_keywords) and lda_topics_keywords[topic_id] and prob > max_lda_prob:
                      max_lda_prob = prob
                      best_lda_topic_id = topic_id

             if best_lda_topic_id != -1 and max_lda_prob > 0.2:
                 topic_group_name = lda_topics_keywords[best_lda_topic_id][0]
                 if topic_group_name in primary_topic_groups and primary_topic_groups[topic_group_name]["type"] == "university":
                     pass
                 else:
                     primary_topic_groups[topic_group_name]["newsItems"].append(news_info_by_index[news_idx])
                     primary_topic_groups[topic_group_name]["tokens"].append(news_info_by_index[news_idx]["tokens"])
                     primary_topic_groups[topic_group_name]["type"] = "topic"
                     primary_topic_groups[topic_group_name]["ldaId"] = best_lda_topic_id


    # 뉴스 개수 기준으로 대분류 설정
    group_news_counts = sorted([
        (group_name, len(data["newsItems"]), data["type"], data.get("ldaId"))
        for group_name, data in primary_topic_groups.items() if data["newsItems"]
    ], key=lambda item: item[1], reverse=True)

    university_aggregate_name = "대학교"
    all_universities_with_news = [name for name, data in primary_topic_groups.items() if data["type"] == "university" and data["newsItems"]]
    total_uni_news_count = sum(len(primary_topic_groups[name]["newsItems"]) for name in all_universities_with_news)

    top_groups_info = []
    if all_universities_with_news:
        top_groups_info.append((university_aggregate_name, total_uni_news_count, "university_aggregate", None))

    other_groups_info = [info for info in group_news_counts if info[2] != "university"]
    top_groups_info.extend(other_groups_info)

    top_groups_info = sorted(top_groups_info, key=lambda item: item[1], reverse=True)
    top_groups_info = top_groups_info[:min(num_final_topics, len(top_groups_info))]


    # 대분류 아래에 중분류 설정
    processed_group_names_for_structure = set()

    for group_name, news_count, group_type, lda_id in top_groups_info:

        if group_name in processed_group_names_for_structure:
            continue

        daefunryu_structure = {
            MAJOR_KEY: group_name,
            MIDDLE_LIST_KEY: [],
            OTHER_NEWS_KEY: []
        }

        used_keywords_in_topic = {group_name}

        # --- '대학교' 대분류: 모든 대학교 이름을 중분류 후보로 추가 ---
        if group_name == university_aggregate_name:
            if all_universities_with_news:
                for uni_name in all_universities_with_news:
                    if uni_name in primary_topic_groups:
                         daefunryu_structure[MIDDLE_LIST_KEY].append({
                              MIDDLE_ITEM_KEY: uni_name,
                              RELATED_NEWS_KEY: []
                          })
                         used_keywords_in_topic.add(uni_name)

                final_categorized_results.append(daefunryu_structure)
                processed_group_names_for_structure.add(group_name)


        # --- '대학교' 외 다른 대분류 (LDA 토픽 기반): 상위 LDA 키워드를 중분류 후보로 추가 ---
        elif group_type == "topic":
            if group_name in primary_topic_groups:
                group_data = primary_topic_groups[group_name]
                lda_topic_words = lda_topics_keywords[lda_id] if lda_id is not None and lda_id < len(lda_topics_keywords) else []

                candidate_middle_keywords = [
                     kw for kw in lda_topic_words
                     if kw not in used_keywords_in_topic and len(kw) > 1 and not re.fullmatch(r'\d+', kw)
                ]

                middle_category_list = []
                for middle_kw in candidate_middle_keywords:
                    middle_category_list.append({
                         MIDDLE_ITEM_KEY: middle_kw,
                         RELATED_NEWS_KEY: []
                     })
                    used_keywords_in_topic.add(middle_kw)

                daefunryu_structure[MIDDLE_LIST_KEY] = middle_category_list

                final_categorized_results.append(daefunryu_structure)
                processed_group_names_for_structure.add(group_name)


    # 뉴스 할당 및 뉴스 수 기준으로 중분류 설정
    assigned_news_links_across_all_categories = set()

    for category in final_categorized_results:
        main_cat_name = category.get(MAJOR_KEY, "알수없음")
        all_news_for_this_daefunryu = []

        if main_cat_name == "대학교":
            for uni_name in all_universities_with_news:
                if uni_name in primary_topic_groups:
                    all_news_for_this_daefunryu.extend(primary_topic_groups[uni_name]["newsItems"])
        else:
            if main_cat_name in primary_topic_groups:
                all_news_for_this_daefunryu = primary_topic_groups[main_cat_name]["newsItems"]

        # 뉴스 할당
        initial_middle_list = category.get(MIDDLE_LIST_KEY, [])

        for news_item in all_news_for_this_daefunryu:
            news_link = news_item["link"]
            news_tokens = news_item["tokens"]
            assigned_to_middle_candidate = False

            if news_link in assigned_news_links_across_all_categories:
                continue

            for middle_item in initial_middle_list:
                middle_kw = middle_item.get(MIDDLE_ITEM_KEY)
                if middle_kw and middle_kw in news_tokens:
                    middle_item[RELATED_NEWS_KEY].append(news_item)
                    assigned_to_middle_candidate = True
                    break

            if not assigned_to_middle_candidate:
                category[OTHER_NEWS_KEY].append(news_item)

            assigned_news_links_across_all_categories.add(news_link)


        # 중분류 설정
        sorted_middle_list_by_news_count = sorted(
            initial_middle_list,
            key=lambda item: len(item.get(RELATED_NEWS_KEY, [])),
            reverse=True
        )

        top_middle_categories = sorted_middle_list_by_news_count[:num_middle_keywords]

        removed_middle_categories = sorted_middle_list_by_news_count[num_middle_keywords:]
        for removed_item in removed_middle_categories:
            category[OTHER_NEWS_KEY].extend(removed_item.get(RELATED_NEWS_KEY, []))

        # 최종 중분류 업데이트
        category[MIDDLE_LIST_KEY] = top_middle_categories


    # 뉴스 기사가 없는 분류 항목 제거
    final_categorized_results = remove_empty_categories(final_categorized_results)


    # 결과 출력
    print(f"\n--- 최종 분석 결과 (총 {len(final_categorized_results)}개 대분류) ---")
    if not final_categorized_results:
        print("뉴스 기사가 있는 분류 항목이 없습니다.")
        return []

    for category in final_categorized_results:
        main_cat_name = category.get(MAJOR_KEY, "알수없음")
        total_news_in_major = len(category.get(OTHER_NEWS_KEY, [])) + sum(len(m.get(RELATED_NEWS_KEY, [])) for m in category.get(MIDDLE_LIST_KEY, []))
        print(f"\n=== 대분류: {main_cat_name} (총 뉴스 {total_news_in_major}개) ===")

        if category.get(MIDDLE_LIST_KEY):
            print(" 하위_분류 (중분류):")
            for middle_item in category[MIDDLE_LIST_KEY]:
                middle_name = middle_item.get(MIDDLE_ITEM_KEY, "알수없음")
                middle_news_count = len(middle_item.get(RELATED_NEWS_KEY, []))
                if middle_news_count > 0:
                    print(f"   - 중분류: {middle_name} (뉴스 {middle_news_count}개)")

                    for news in middle_item[RELATED_NEWS_KEY][: min(3, middle_news_count)]:
                         print(f"       - {news['title']}")


        if category.get(OTHER_NEWS_KEY):
            other_news_count = len(category[OTHER_NEWS_KEY])
            if other_news_count > 0:
                print(f" 대분류 '{main_cat_name}' 기타 뉴스 ({other_news_count}개):")
                for news in category[OTHER_NEWS_KEY][: min(5, other_news_count)]:
                    print(f"- {news['title']}")

    return final_categorized_results
