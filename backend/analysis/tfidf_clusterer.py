#!/usr/bin/env python3
"""
TF-IDF 기반 뉴스 클러스터링 모듈

뉴스 데이터를 TF-IDF 코사인 유사도와 키워드 빈도를 기반으로 클러스터링을 수행
"""

import logging
from collections import Counter, defaultdict
from konlpy.tag import Okt
import re
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

class TfidfClusterer:
    """
    TF-IDF 기반 뉴스 클러스터링 클래스
    """
    
    def __init__(self):
        """
        초기화
        """
        # KoNLPy 형태소 분석기 초기화
        self.okt = Okt()
        
        # 클러스터링 파라미터 
        self.MIN_TITLE_LENGTH = 10          # 최소 제목 길이 
        self.MIN_WORD_LENGTH = 2            # 최소 단어(명사) 길이 
        self.MIN_NEWS_COUNT = 5             # 분석에 필요한 최소 뉴스 개수
        
        # 필터링 기준
        self.MIN_UNIV_NEWS_COUNT = 2        # 대학교로 분류되기 위한 최소 뉴스 개수
        self.MIN_CLUSTER_NEWS_COUNT = 3     # 클러스터로 분류되기 위한 최소 뉴스 개수
        self.MIN_MINOR_NEWS_COUNT = 2       # 중분류로 분류되기 위한 최소 뉴스 개수
        self.MAX_UNIV_DISPLAY = 5           # 표시할 최대 대학교 개수
        self.MIN_MIDDLE_KEYWORDS_COUNT = 1  # 대분류로 표시되기 위한 최소 중분류 개수
        
        # ===== 정규표현식 패턴 (컴파일된 패턴으로 성능 최적화) =====
        self.uni_pattern = re.compile(r".+대$")                    # 대학교 패턴 (예: 서울대, 연세대)
        self.bracket_pattern = re.compile(r'\[.*?\]')              # 대괄호 제거용
        self.parenthesis_pattern = re.compile(r'\(.*?\)')          # 소괄호 제거용
        self.html_tag_pattern = re.compile(r'<.*?>')               # HTML 태그 제거용
        self.special_char_pattern = re.compile(r'[^\w\s가-힣]')     # 특수문자 제거용
        self.whitespace_pattern = re.compile(r'\s+')               # 연속 공백 정리용
        
        # 파일 경로 설정
        config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")
        STOPWORDS_PATH = os.path.join(config_dir, "stopwords.txt")
        NON_UNIV_WORD_PATH = os.path.join(config_dir, "non_university_words.txt")
        
        # 불용어와 제외 단어 로드
        self.stopwords = self._load_text_file_as_set(STOPWORDS_PATH, "불용어")
        self.exclude_words = self._load_text_file_as_set(NON_UNIV_WORD_PATH, "제외 단어")
        
        logger.info("✅ TfidfClusterer 초기화 완료")
    
    def _load_text_file_as_set(self, file_path, file_description):
        """
        텍스트 파일을 읽어 set으로 반환하는 공통 메서드
        
        Args:
            file_path (str): 읽을 파일 경로
            file_description (str): 파일 설명 (에러 메시지용)
            
        Returns:
            set: 파일 내용을 줄 단위로 나눈 집합
            
        Raises:
            FileNotFoundError: 파일이 존재하지 않을 때
            PermissionError: 파일 접근 권한이 없을 때
            UnicodeDecodeError: 파일 인코딩 오류
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return {line.strip() for line in f if line.strip()}
        except FileNotFoundError:
            logger.error(f"{file_description} 파일을 찾을 수 없습니다: {file_path}")
            return set()
        except PermissionError:
            logger.error(f"{file_description} 파일 접근 권한이 없습니다: {file_path}")
            return set()
        except UnicodeDecodeError as e:
            logger.error(f"{file_description} 파일 인코딩 오류: {e}")
            return set()
        except Exception as e:
            logger.error(f"{file_description} 로드 중 예상치 못한 오류: {e}")
            return set()
    
    def extract_nouns(self, text):
        """
        KoNLPy의 Okt를 사용하여 형태소 분석 후 명사 추출 
        """
        tokens = self.okt.pos(text, stem=True)
        nouns = [word for word, tag in tokens 
                if tag == "Noun" and word not in self.stopwords and len(word) >= self.MIN_WORD_LENGTH]
        return nouns
    
    def _extract_university_keyword(self, nouns):
        """
        명사 리스트에서 대학교 키워드 추출
        """
        university_keyword = next(
            (kw for kw in nouns if self.uni_pattern.match(kw) and kw not in self.exclude_words), 
            None
        )
        
        if not university_keyword and "KAIST" in nouns:
            return "KAIST"
        
        return university_keyword
    
    def preprocess_titles(self, news_data):
        """
        뉴스 제목 전처리
        """
        processed_titles = []
        
        for item in news_data:
            title = item["title"]
            
            title = self.bracket_pattern.sub('', title)
            title = self.parenthesis_pattern.sub('', title)
            title = self.html_tag_pattern.sub('', title)
            title = self.special_char_pattern.sub(' ', title)
            title = self.whitespace_pattern.sub(' ', title).strip()
            
            if len(title) > self.MIN_TITLE_LENGTH:
                processed_titles.append({
                    "original": item,
                    "cleaned_title": title
                })
        
        unique_titles = []
        seen_titles = set()
        
        for item in processed_titles:
            title = item["cleaned_title"]
            if title not in seen_titles:
                seen_titles.add(title)
                unique_titles.append(item)
        
        return unique_titles
    
    def split_news_by_uni_name(self, processed_data):
        """
        대학교 이름으로 뉴스 분류 
        """
        university_news = defaultdict(list)
        other_news = []
        
        for item in processed_data:
            title = item["cleaned_title"]
            nouns = self.extract_nouns(title)
            university_keyword = self._extract_university_keyword(nouns)
            
            news_info = {
                "original": item["original"],
                "cleaned_title": title,
                "nouns": nouns
            }
            
            if university_keyword:
                university_news[university_keyword].append(news_info)
            else:
                other_news.append(news_info)
        
        return university_news, other_news
    
    def cluster_by_tfidf_cosine(self, news_data, min_keyword_count=5, score_threshold=0.2):
        """
        TF-IDF 코사인 유사도 기반 클러스터링
        
        이 메서드는 다음과 같은 과정으로 클러스터링을 수행합니다:
        1. 모든 뉴스에서 키워드 추출 및 빈도 계산
        2. 상위 키워드 선택 (min_keyword_count 이상)
        3. 각 키워드별 토픽 벡터 생성 (TF-IDF)
        4. 각 뉴스와 토픽 벡터 간 코사인 유사도 계산
        5. 가장 높은 유사도를 가진 토픽에 단일 할당 (score_threshold 이상)
        """
        logger.info(f"🎯 TF-IDF 코사인 유사도 클러스터링 시작: {len(news_data)}개 뉴스")
        
        # 1단계: 모든 뉴스에서 키워드 추출
        all_keywords = []
        for item in news_data:
            nouns = self.extract_nouns(item.get("cleaned_title", ""))
            all_keywords.extend(nouns)
        
        # 2단계: 키워드 빈도 계산 및 상위 키워드 선택
        keyword_counts = Counter(all_keywords)
        top_keywords = [kw for kw, count in keyword_counts.most_common() if count >= min_keyword_count]
        
        logger.info(f"📊 상위 키워드: {len(top_keywords)}개")
        
        if not top_keywords:
            logger.warning("상위 키워드가 없습니다")
            return {}
        
        # 3단계: 각 키워드에 해당하는 뉴스들 수집 (성능 개선)
        logger.info("📊 토픽별 뉴스 수집 중...")
        topic_news = {}
        
        # 미리 모든 뉴스의 명사를 추출 (한 번만)
        news_nouns = []
        for item in news_data:
            nouns = self.extract_nouns(item.get("cleaned_title", ""))
            news_nouns.append(nouns)
        
        # 각 키워드에 대해 빠르게 매칭
        for keyword in top_keywords:
            topic_news[keyword] = []
            for i, nouns in enumerate(news_nouns):
                if keyword in nouns:
                    topic_news[keyword].append(news_data[i])
            
            logger.info(f"   '{keyword}': {len(topic_news[keyword])}개 뉴스")
        
        # 4단계: TF-IDF 벡터화
        logger.info("📊 TF-IDF 벡터화 중...")
        all_texts = [item.get("cleaned_title", "") for item in news_data]
        
        try:
            # TF-IDF 벡터화 (명사 기반)
            tfidf = TfidfVectorizer(
                max_features=500,  # 1000 → 500으로 축소
                tokenizer=lambda x: self.extract_nouns(x),
                token_pattern=None,
                min_df=3,  # 2 → 3으로 증가 (더 엄격)
                max_df=0.7  # 0.8 → 0.7로 감소 (더 엄격)
            )
            tfidf_matrix = tfidf.fit_transform(all_texts)
            logger.info(f"✅ TF-IDF 벡터화 완료: {tfidf_matrix.shape}")
            
            # 5단계: 각 토픽에 대한 코사인 유사도 계산 (최적화)
            logger.info("📊 코사인 유사도 계산 중...")
            clusters = defaultdict(list)
            assigned_count = 0
            
            # 토픽별 TF-IDF 벡터를 미리 계산 (한 번만)
            topic_vectors = {}
            for keyword in top_keywords:
                if keyword not in topic_news or len(topic_news[keyword]) < min_keyword_count:
                    logger.info(f"   '{keyword}' 제외: 뉴스 수 부족 ({len(topic_news.get(keyword, []))}개)")
                    continue
                
                # 토픽 문서 생성
                topic_texts = [news_item.get("cleaned_title", "") for news_item in topic_news[keyword]]
                topic_doc = " ".join(topic_texts)
                
                # 토픽 벡터 미리 계산
                topic_vector = tfidf.transform([topic_doc])
                if topic_vector.nnz > 0:
                    topic_vectors[keyword] = topic_vector
                    logger.info(f"   '{keyword}' 토픽 벡터 생성 성공 (nnz: {topic_vector.nnz})")
                else:
                    logger.info(f"   '{keyword}' 토픽 벡터 생성 실패 (nnz: 0)")
            
            logger.info(f"✅ 토픽 벡터 계산 완료: {len(topic_vectors)}개")
            
            # 진행 상황 표시를 위한 카운터
            processed_count = 0
            total_news = len(news_data)
            
            for i, item in enumerate(news_data):
                best_topic = None
                best_score = 0.0
                
                news_vector = tfidf_matrix[i:i+1]
                if news_vector.nnz == 0:
                    continue
                
                for keyword, topic_vector in topic_vectors.items():
                    # 코사인 유사도 계산
                    similarity = cosine_similarity(news_vector, topic_vector)[0][0]
                    
                    if similarity > best_score:
                        best_score = similarity
                        best_topic = keyword
                
                # 임계값 이상이면 할당
                if best_topic and best_score >= score_threshold:
                    clusters[best_topic].append(item)
                    assigned_count += 1
                
                # 진행 상황 표시
                processed_count += 1
                if processed_count % 50 == 0:  # 100 → 50으로 변경 (더 자주 표시)
                    logger.info(f"   진행률: {processed_count}/{total_news} ({processed_count/total_news*100:.1f}%)")
            
            # 6단계: 클러스터 정리
            final_clusters = {}
            cluster_id = 0
            
            for keyword, cluster_news in clusters.items():
                if len(cluster_news) >= min_keyword_count:
                    final_clusters[cluster_id] = {
                        "keyword": keyword,
                        "news": cluster_news,
                        "size": len(cluster_news)
                    }
                    logger.info(f"   클러스터 {cluster_id}: '{keyword}' ({len(cluster_news)}개 뉴스)")
                    cluster_id += 1
            
            unassigned_count = len(news_data) - assigned_count
            logger.info(f"📊 할당 완료: {assigned_count}개, 미할당: {unassigned_count}개")
            logger.info(f"🎉 TF-IDF 코사인 클러스터링 완료: {len(final_clusters)}개 클러스터")
            
            return final_clusters
            
        except Exception as e:
            logger.error(f"❌ TF-IDF 코사인 클러스터링 실패: {e}")
            # Fallback: 빈도 기반 클러스터링
            return self.cluster_by_keyword_frequency(news_data, min_keyword_count)
    
    def cluster_by_keyword_frequency(self, news_data, min_keyword_count=5):
        logger.info(f"🔢 빈도수 기반 클러스터링 시작: {len(news_data)}개 뉴스")
        
        # 1단계: 모든 뉴스에서 키워드 추출
        all_keywords = []
        for item in news_data:
            nouns = self.extract_nouns(item.get("cleaned_title", ""))
            all_keywords.extend(nouns)
        
        # 2단계: 키워드 빈도 계산
        keyword_counts = Counter(all_keywords)
        
        # 3단계: 빈도가 높은 키워드로 대분류 생성
        clusters = {}
        used_news_indices = set()
        cluster_id = 0
        
        for keyword, count in keyword_counts.most_common():
            if count < min_keyword_count:  # 최소 빈도 이하는 제외
                break
            
            # 이 키워드를 포함하는 뉴스들 찾기
            cluster_news = []
            for i, item in enumerate(news_data):
                if i in used_news_indices:
                    continue
                
                nouns = self.extract_nouns(item.get("cleaned_title", ""))
                if keyword in nouns:
                    cluster_news.append(item)
                    used_news_indices.add(i)
            
            if len(cluster_news) >= min_keyword_count:
                clusters[cluster_id] = {
                    "keyword": keyword,
                    "news": cluster_news,
                    "size": len(cluster_news)
                }
                logger.info(f"   클러스터 {cluster_id}: '{keyword}' ({len(cluster_news)}개 뉴스)")
                cluster_id += 1
        
        # 4단계: 사용되지 않은 뉴스들은 제외 (기타 클러스터 생성 안함)
        unused_news = [news_data[i] for i in range(len(news_data)) if i not in used_news_indices]
        if unused_news:
            logger.info(f"   미분류 뉴스: {len(unused_news)}개 (제외)")
        
        logger.info(f"🎉 빈도수 기반 클러스터링 완료: {len(clusters)}개 클러스터")
        
        return clusters
    
    def create_subcategories_tfidf(self, cluster_news, max_subcategories=5):
        """
        TF-IDF 기반 중분류 생성
        - 해당 키워드를 포함하는 뉴스만 중분류에 포함
        - TF-IDF로 중요 키워드 추출
        """
        if len(cluster_news) < 6:  # 뉴스가 너무 적으면 중분류 생성 안함
            return []
        
        # 작은 클러스터(10개 이하)는 빈도 기반으로 바로 전환
        if len(cluster_news) <= 10:
            return self.create_subcategories(cluster_news, max_subcategories)
        
        # TF-IDF로 중요 키워드 추출
        texts = [item.get("cleaned_title", "") for item in cluster_news]
        
        try:
            # 클러스터 크기에 따라 min_df 동적 조정
            # 작은 클러스터는 min_df=1, 큰 클러스터는 min_df=2
            min_df_value = 1 if len(cluster_news) <= 15 else 2
            
            # TF-IDF 벡터화
            tfidf = TfidfVectorizer(
                max_features=50,
                tokenizer=lambda x: self.extract_nouns(x),
                token_pattern=None,
                min_df=min_df_value,  # 동적 조정
                max_df=0.9  # 더 관대하게 (0.8 -> 0.9)
            )
            tfidf_matrix = tfidf.fit_transform(texts)
            feature_names = tfidf.get_feature_names_out()
            
            if len(feature_names) == 0:
                # 특성이 없으면 빈도 기반으로 전환
                return self.create_subcategories(cluster_news, max_subcategories)
            
            # TF-IDF 점수 합산
            tfidf_scores = np.array(tfidf_matrix.sum(axis=0)).flatten()
            top_indices = tfidf_scores.argsort()[::-1][:max_subcategories]
            
            # 상위 키워드로 중분류 생성
            subcategories = []
            used_news_indices = set()
            
            for idx in top_indices:
                keyword = feature_names[idx]
                
                # 이 키워드를 포함하는 뉴스들만 찾기
                subcategory_news = []
                for i, item in enumerate(cluster_news):
                    if i in used_news_indices:
                        continue
                    
                    nouns = self.extract_nouns(item.get("cleaned_title", ""))
                    if keyword in nouns:
                        subcategory_news.append(item)
                        used_news_indices.add(i)
                
                if len(subcategory_news) >= 2:
                    subcategories.append({
                        "keyword": keyword,
                        "news": subcategory_news,
                        "size": len(subcategory_news)
                    })
            
            # 중분류가 2개 미만이면 빈도 기반으로 재시도
            if len(subcategories) < 2:
                return self.create_subcategories(cluster_news, max_subcategories)
            
            return subcategories
            
        except Exception as e:
            # TF-IDF 실패 시 빈도 기반으로 전환 (경고 로그 제거)
            return self.create_subcategories(cluster_news, max_subcategories)
    
    def create_subcategories(self, cluster_news, max_subcategories=5):
        """
        클러스터 내에서 중분류 생성 (키워드 빈도 기반, fallback용)
        """
        if len(cluster_news) < 6:  # 뉴스가 너무 적으면 중분류 생성 안함
            return []
        
        # 클러스터 내 모든 키워드 수집
        all_keywords = []
        for item in cluster_news:
            nouns = self.extract_nouns(item.get("cleaned_title", ""))
            all_keywords.extend(nouns)
        
        # 키워드 빈도 계산
        keyword_counts = Counter(all_keywords)
        
        # 상위 키워드로 중분류 생성
        subcategories = []
        used_news_indices = set()
        
        for keyword, count in keyword_counts.most_common(max_subcategories):
            if count < 2:  # 최소 2개 이상의 뉴스가 있어야 중분류로 인정
                break
            
            # 이 키워드를 포함하는 뉴스들 찾기
            subcategory_news = []
            for i, item in enumerate(cluster_news):
                if i in used_news_indices:
                    continue
                
                nouns = self.extract_nouns(item.get("cleaned_title", ""))
                
                if keyword in nouns:
                    subcategory_news.append(item)
                    used_news_indices.add(i)
            
            if len(subcategory_news) >= 2:
                subcategories.append({
                    "keyword": keyword,
                    "news": subcategory_news,
                    "size": len(subcategory_news)
                })
        
        return subcategories
    
    def normalize_keyword(self, keyword):
        """
        키워드의 공백을 하이픈으로 변경 (news_analyzer.py와 동일)
        """
        return keyword.replace(" ", "-")
    
    def _format_news_item(self, news):
        """
        뉴스 데이터를 프론트엔드 형식으로 변환 (news_analyzer.py와 동일)
        """
        return {
            "title": news["original"].get("title", news.get("cleaned_title", "Unknown")), 
            "link": news["original"].get("link", "")
        }
    
    def analyze_news(self, news_data):
        """
        뉴스 분석 메인 함수
        
        1. 뉴스 제목 전처리 (괄호 제거, 특수문자 정리)
        2. 대학교 뉴스 분리 (정규표현식 기반)
        3. 기타 뉴스 TF-IDF 코사인 유사도 클러스터링
        4. 각 클러스터 내 TF-IDF 기반 중분류 생성
        5. 프론트엔드 형식으로 결과 변환
        """
        logger.info(f"🚀 TF-IDF 기반 뉴스 분석 시작: {len(news_data)}개 뉴스")
        
        # 뉴스 제목 전처리
        processed_data = self.preprocess_titles(news_data)
        
        if len(processed_data) < 10:  # 최소 뉴스 수 체크
            logger.warning("분석 가능한 뉴스가 부족합니다")
            return None
        
        # 2단계: 대학교 뉴스 분리
        university_news, other_news = self.split_news_by_uni_name(processed_data)
        
        logger.info(f"📊 분류 완료: 대학교 {len(university_news)}개 그룹, 기타 {len(other_news)}개")
        
        # 3단계: 기타 뉴스 클러스터링 (TF-IDF 코사인 유사도 기반)
        clusters = self.cluster_by_tfidf_cosine(other_news, min_keyword_count=5, score_threshold=0.4)
        
        # 4단계: 프론트엔드 형식으로 변환
        result = []
        
        # 대학교 뉴스 처리
        if university_news:
            # 대학교 데이터 필터링 및 정렬 (뉴스 수 기준)
            filtered_universities = {
                university_name: news_list 
                for university_name, news_list in university_news.items() 
                if len(news_list) >= self.MIN_UNIV_NEWS_COUNT
            }
            
            # 뉴스 수가 많은 순으로 정렬하고 상위 N개만 선택
            sorted_items = sorted(filtered_universities.items(), key=lambda x: len(x[1]), reverse=True)
            sorted_universities = dict(sorted_items[:self.MAX_UNIV_DISPLAY])
            
            if sorted_universities:
                # 대학교별 중분류 생성
                univ_middle_keywords = [
                    {
                        "middleKeyword": self.normalize_keyword(uni_name),
                        "relatedNews": [self._format_news_item(news) for news in news_list]
                    }
                    for uni_name, news_list in sorted_universities.items()
                ]
                
                # 최소 중분류 수 확인 후 대분류 추가
                if len(univ_middle_keywords) >= self.MIN_MIDDLE_KEYWORDS_COUNT:
                    result.append({
                        "majorKeyword": self.normalize_keyword("대학교"),
                        "middleKeywords": univ_middle_keywords,
                        "otherNews": []
                    })
        
        # 일반 클러스터 처리
        for cluster_id, cluster_data in clusters.items():
            keyword = cluster_data["keyword"]
            news = cluster_data["news"]
            
            # 중분류 생성 (TF-IDF 기반)
            subcategories = self.create_subcategories_tfidf(news)
            
            # 중분류가 있으면 중분류로, 없으면 기타 뉴스로
            if subcategories and len(subcategories) > 1:  # 중분류가 2개 이상이어야 함
                middle_keywords = []
                other_news_in_cluster = []
                
                for subcat in subcategories:
                    if subcat["keyword"] == "기타":
                        other_news_in_cluster = [self._format_news_item(news) for news in subcat["news"]]
                    else:
                        middle_keywords.append({
                            "middleKeyword": subcat["keyword"],
                            "relatedNews": [self._format_news_item(news) for news in subcat["news"]]
                        })
                
                # 중분류가 있고 기타 뉴스가 있으면 추가
                if middle_keywords or other_news_in_cluster:
                    result.append({
                        "majorKeyword": keyword,
                        "middleKeywords": middle_keywords,
                        "otherNews": other_news_in_cluster
                    })
            else:
                # 중분류가 없거나 1개뿐이면 해당 대분류는 제외
                logger.info(f"   대분류 '{keyword}' 제외: 중분류 {len(subcategories) if subcategories else 0}개 (최소 2개 필요)")
        
        logger.info(f"✅ 분석 완료: {len(result)}개 대분류 생성")
        
        return result

# 메인 실행 부분
if __name__ == "__main__":
    test_tfidf_clusterer()
