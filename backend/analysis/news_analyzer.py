import os
import re
import logging
from collections import defaultdict
import numpy as np
from sentence_transformers import SentenceTransformer
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from konlpy.tag import Okt
from keybert import KeyBERT
from config.config import STOPWORDS_PATH, NON_UNIV_WORD_PATH

logger = logging.getLogger(__name__)


class NewsAnalyzer:
    """
    뉴스 분석 클래스
    
    뉴스 전처리, 클러스터링, 키워드 추출 등의 기능을 제공합니다.
    """
    
    # ========== 상수 정의 ==========
    # 텍스트 처리 관련
    MIN_TITLE_LENGTH = 10          # 최소 제목 길이
    MIN_WORD_LENGTH = 2            # 최소 단어(명사) 길이
    MIN_NEWS_COUNT = 5             # 분석에 필요한 최소 뉴스 개수
    
    # 필터링 기준
    MIN_UNIV_NEWS_COUNT = 2        # 대학교로 분류되기 위한 최소 뉴스 개수
    MIN_CLUSTER_NEWS_COUNT = 4     # 클러스터로 분류되기 위한 최소 뉴스 개수
    MIN_MINOR_NEWS_COUNT = 2       # 중분류로 분류되기 위한 최소 뉴스 개수
    MAX_UNIV_DISPLAY = 5           # 표시할 최대 대학교 개수
    MIN_MIDDLE_KEYWORDS_COUNT = 2  # 대분류로 표시되기 위한 최소 중분류 개수
    
    # 클러스터링 관련
    HDBSCAN_MIN_CLUSTER_SIZE = 2           # HDBSCAN 최소 클러스터 크기
    HDBSCAN_MIN_SAMPLES = 1                # HDBSCAN 최소 샘플 수
    HDBSCAN_EPSILON = 0.05                 # HDBSCAN 클러스터 선택 엡실론
    CLUSTER_DUPLICATE_THRESHOLD = 0.5      # 클러스터 중복 비율 임계값
    MIN_NOISE_FOR_RECLUSTERING = 5         # 재클러스터링을 위한 최소 노이즈 개수
    
    # K-Means 클러스터링 파라미터
    KMEANS_SMALL_DATA_THRESHOLD = 20       # 소규모 데이터 임계값
    KMEANS_MEDIUM_DATA_THRESHOLD = 100     # 중규모 데이터 임계값
    KMEANS_SMALL_MAX_CLUSTERS = 3          # 소규모 데이터 최대 클러스터 수
    KMEANS_SMALL_DIVISOR = 5               # 소규모 데이터 클러스터 수 계산 제수
    KMEANS_MEDIUM_MAX_CLUSTERS = 8         # 중규모 데이터 최대 클러스터 수
    KMEANS_MEDIUM_DIVISOR = 10             # 중규모 데이터 클러스터 수 계산 제수
    KMEANS_LARGE_MAX_CLUSTERS = 15         # 대규모 데이터 최대 클러스터 수
    KMEANS_LARGE_DIVISOR = 15              # 대규모 데이터 클러스터 수 계산 제수
    KMEANS_RANDOM_STATE = 42               # K-Means 랜덤 시드
    KMEANS_N_INIT = 10                     # K-Means 초기화 횟수
    
    # TF-IDF 파라미터
    TFIDF_NGRAM_MIN = 1            # TF-IDF n-gram 최소값
    TFIDF_NGRAM_MAX = 3            # TF-IDF n-gram 최대값
    TFIDF_MAX_FEATURES = 5000      # TF-IDF 최대 특성 수
    TFIDF_MIN_DF = 1               # TF-IDF 최소 문서 빈도
    TFIDF_MAX_DF = 0.9             # TF-IDF 최대 문서 빈도
    
    # 키워드 추출 관련
    TOP_KEYWORDS_COUNT = 5         # 추출할 키워드 개수
    KEYBERT_TOP_K = 3              # KeyBERT 키워드 개수
    KEYBERT_NGRAM_MIN = 1          # KeyBERT n-gram 최소값
    KEYBERT_NGRAM_MAX = 3          # KeyBERT n-gram 최대값
    KEYBERT_DIVERSITY = 0.5        # KeyBERT MMR 다양성 파라미터
    TFIDF_CATEGORY_KEYWORDS = 20   # 카테고리 분류용 TF-IDF 키워드 개수
    
    def __init__(self):
        """뉴스 분석기 초기화"""
        self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.kw_model = KeyBERT(model='paraphrase-multilingual-MiniLM-L12-v2')
        self.okt = Okt()
        
        self.stopwords = self.load_stopwords()
        self.exclude_words = self.load_exclude_words()
        
        self.uni_pattern = re.compile(r".+대$")
        self.bracket_pattern = re.compile(r'\[.*?\]')
        self.parenthesis_pattern = re.compile(r'\(.*?\)')
        self.html_tag_pattern = re.compile(r'<.*?>')
        self.special_char_pattern = re.compile(r'[^\w\s가-힣]')
        self.whitespace_pattern = re.compile(r'\s+')
        
    def _load_text_file_as_set(self, file_path, file_description):
        """
        텍스트 파일을 읽어 set으로 반환하는 공통 메서드
        
        Args:
            file_path (str): 파일 경로
            file_description (str): 파일 설명 (로깅용)
            
        Returns:
            set: 파일 내용을 set으로 변환한 결과
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
    
    def load_stopwords(self):
        """
        불용어 파일 로드
        
        Returns:
            set: 불용어 집합
        """
        return self._load_text_file_as_set(STOPWORDS_PATH, "불용어")
    
    def load_exclude_words(self):
        """
        대학교가 아닌 제외 단어 목록 로드
        
        Returns:
            set: 제외 단어 집합
        """
        return self._load_text_file_as_set(NON_UNIV_WORD_PATH, "제외 단어")
    
    def extract_nouns(self, text):
        """
        KoNLPy의 Okt를 사용하여 형태소 분석 후 명사 추출
        
        Args:
            text (str): 분석할 텍스트
            
        Returns:
            list: 추출된 명사 리스트
        """
        tokens = self.okt.pos(text, stem=True)
        nouns = [word for word, tag in tokens 
                if tag == "Noun" and word not in self.stopwords and len(word) >= self.MIN_WORD_LENGTH]
        return nouns
    
    def _extract_university_keyword(self, nouns):
        """
        명사 리스트에서 대학교 키워드 추출
        
        Args:
            nouns (list): 명사 리스트
            
        Returns:
            str or None: 대학교 키워드 (없으면 None)
        """
        university_keyword = next(
            (kw for kw in nouns if self.uni_pattern.match(kw) and kw not in self.exclude_words), 
            None
        )
        
        if not university_keyword and "KAIST" in nouns:
            return "KAIST"
        
        return university_keyword
    
    def split_news_by_uni_name(self, processed_data):
        """
        대학교 이름으로 뉴스 분류
        
        Args:
            processed_data (list): 전처리된 뉴스 데이터
            
        Returns:
            tuple: (대학교별 뉴스 딕셔너리, 기타 뉴스 리스트)
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
    
    def preprocess_titles(self, news_data):
        """
        뉴스 제목 전처리 (특수문자 제거, 중복 제거)
        
        Args:
            news_data (list): 원본 뉴스 데이터
            
        Returns:
            list: 전처리된 뉴스 데이터
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
    
    def extract_keywords_with_konlpy_tfidf(self, texts, topn=5):
        """
        KoNLPy + TF-IDF 기반 키워드 추출
        
        Args:
            texts (list): 텍스트 리스트
            topn (int): 추출할 키워드 개수
            
        Returns:
            list: 추출된 키워드 리스트
        """
        noun_texts = [" ".join(self.extract_nouns(text)) for text in texts]
        
        vectorizer = TfidfVectorizer(
            ngram_range=(self.TFIDF_NGRAM_MIN, self.TFIDF_NGRAM_MAX),
            max_features=self.TFIDF_MAX_FEATURES,
            min_df=self.TFIDF_MIN_DF,
            max_df=self.TFIDF_MAX_DF
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(noun_texts)
            feature_names = vectorizer.get_feature_names_out()
            
            mean_scores = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
            top_indices = mean_scores.argsort()[-topn:][::-1]
            
            return [feature_names[i] for i in top_indices]
        except ValueError as e:
            logger.warning(f"TF-IDF 벡터화 실패 (데이터 부족): {e}")
            return []
        except Exception as e:
            logger.error(f"TF-IDF 계산 중 예상치 못한 오류: {e}")
            return []
    
    def calculate_kmeans_clusters(self, n_data):
        """
        데이터 개수에 따라 적절한 K-Means 클러스터 수 계산
        
        Args:
            n_data (int): 데이터 개수
            
        Returns:
            int: 적절한 클러스터 개수
        """
        if n_data < self.KMEANS_SMALL_DATA_THRESHOLD:
            return min(self.KMEANS_SMALL_MAX_CLUSTERS, n_data // self.KMEANS_SMALL_DIVISOR)
        elif n_data < self.KMEANS_MEDIUM_DATA_THRESHOLD:
            return min(self.KMEANS_MEDIUM_MAX_CLUSTERS, n_data // self.KMEANS_MEDIUM_DIVISOR)
        else:
            return min(self.KMEANS_LARGE_MAX_CLUSTERS, n_data // self.KMEANS_LARGE_DIVISOR)
    
    def _perform_hdbscan_clustering(self, embeddings):
        """
        HDBSCAN 클러스터링 수행
        
        Args:
            embeddings: 뉴스 임베딩 벡터
            
        Returns:
            array: 클러스터 레이블
        """
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.HDBSCAN_MIN_CLUSTER_SIZE,
            min_samples=self.HDBSCAN_MIN_SAMPLES,
            metric="euclidean",
            cluster_selection_method='leaf',
            cluster_selection_epsilon=self.HDBSCAN_EPSILON,
            prediction_data=True
        )
        return clusterer.fit_predict(embeddings)
    
    def _is_meaningful_cluster(self, cluster_news):
        """
        클러스터가 의미있는지 판단
        
        Args:
            cluster_news (list): 클러스터 내 뉴스 리스트
            
        Returns:
            tuple: (의미있는지 여부, 사유)
        """
        if len(cluster_news) < self.MIN_CLUSTER_NEWS_COUNT:
            return False, f"크기 부족 ({len(cluster_news)}개)"
        
        titles = [news['cleaned_title'] for news in cluster_news]
        unique_titles = set(titles)
        duplicate_ratio = (len(titles) - len(unique_titles)) / len(titles)
        
        if duplicate_ratio >= self.CLUSTER_DUPLICATE_THRESHOLD:
            return False, f"중복 비율 높음 ({duplicate_ratio:.1%})"
        
        return True, "정상"
    
    def _filter_meaningful_clusters(self, hdbscan_clusters, titles_data, embeddings):
        """
        의미있는 클러스터만 필터링하고 의미없는 것들은 노이즈로 이동
        
        Args:
            hdbscan_clusters (dict): HDBSCAN 클러스터 딕셔너리
            titles_data (list): 전체 뉴스 데이터
            embeddings: 전체 임베딩 벡터
            
        Returns:
            tuple: (의미있는 클러스터, 노이즈 데이터, 노이즈 임베딩)
        """
        meaningful_clusters = {}
        noise_data = []
        noise_embeddings = []
        
        item_to_idx = {id(item): idx for idx, item in enumerate(titles_data)}
        
        for cluster_id, cluster_news in hdbscan_clusters.items():
            is_meaningful, reason = self._is_meaningful_cluster(cluster_news)
            
            if is_meaningful:
                meaningful_clusters[cluster_id] = cluster_news
            else:
                noise_data.extend(cluster_news)
                for item in cluster_news:
                    item_idx = item_to_idx.get(id(item))
                    if item_idx is not None:
                        noise_embeddings.append(embeddings[item_idx])
        
        return meaningful_clusters, noise_data, noise_embeddings
    
    def _perform_kmeans_clustering(self, data, embeddings):
        """
        K-Means 클러스터링 수행
        
        Args:
            data (list): 클러스터링할 데이터
            embeddings: 임베딩 벡터
            
        Returns:
            dict: 클러스터 딕셔너리
        """
        n_clusters = self.calculate_kmeans_clusters(len(data))
        
        if n_clusters < 2:
            return {}
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.KMEANS_RANDOM_STATE, n_init=self.KMEANS_N_INIT)
        kmeans_labels = kmeans.fit_predict(embeddings)
        
        clusters = defaultdict(list)
        for label, item in zip(kmeans_labels, data):
            clusters[label].append(item)
        
        return dict(clusters)
    
    def _recluster_noise_with_kmeans(self, noise_data, noise_embeddings, existing_clusters):
        """
        노이즈 데이터를 K-Means로 재클러스터링
        
        Args:
            noise_data (list): 노이즈 데이터
            noise_embeddings: 노이즈 임베딩
            existing_clusters (dict): 기존 클러스터
            
        Returns:
            dict: 업데이트된 클러스터 딕셔너리
        """
        if len(noise_data) <= self.MIN_NOISE_FOR_RECLUSTERING:
            logger.info(f"노이즈 데이터가 적어 K-Means 재분류 생략 (데이터 개수: {len(noise_data)})")
            return existing_clusters
        
        n_clusters = self.calculate_kmeans_clusters(len(noise_data))
        
        if n_clusters < 2:
            logger.info(f"노이즈 데이터가 너무 적어 K-Means 재분류 생략 (클러스터 수: {n_clusters})")
            return existing_clusters
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.KMEANS_RANDOM_STATE, n_init=self.KMEANS_N_INIT)
        kmeans_labels = kmeans.fit_predict(noise_embeddings)
        
        max_cluster_id = max(existing_clusters.keys()) if existing_clusters else -1
        
        updated_clusters = existing_clusters.copy()
        for label, item in zip(kmeans_labels, noise_data):
            new_cluster_id = max_cluster_id + 1 + label
            if new_cluster_id not in updated_clusters:
                updated_clusters[new_cluster_id] = []
            updated_clusters[new_cluster_id].append(item)
        
        return updated_clusters
    
    def enhanced_cluster_news(self, titles_data):
        """
        HDBSCAN + K-Means 하이브리드 클러스터링
        
        1단계에서 HDBSCAN으로 밀도 기반 클러스터링을 수행하고,
        2단계에서 노이즈 데이터를 K-Means로 재분류합니다.
        
        Args:
            titles_data (list): 클러스터링할 뉴스 데이터
            
        Returns:
            tuple: (클러스터 딕셔너리, 노이즈 리스트)
        """
        titles = [item["cleaned_title"] for item in titles_data]
        n_data = len(titles_data)
        
        embeddings = self.embedding_model.encode(titles, normalize_embeddings=True)
        
        hdbscan_labels = self._perform_hdbscan_clustering(embeddings)
        
        hdbscan_clusters = defaultdict(list)
        noise_data = []
        noise_embeddings = []
        
        if hdbscan_labels is not None:
            for label, item, embedding in zip(hdbscan_labels, titles_data, embeddings):
                if label == -1:
                    noise_data.append(item)
                    noise_embeddings.append(embedding)
                else:
                    hdbscan_clusters[label].append(item)
            
            meaningful_clusters, filtered_noise, filtered_noise_emb = self._filter_meaningful_clusters(
                hdbscan_clusters, titles_data, embeddings
            )
            noise_data.extend(filtered_noise)
            noise_embeddings.extend(filtered_noise_emb)
            hdbscan_clusters = meaningful_clusters
        else:
            noise_data = list(titles_data)
            noise_embeddings = list(embeddings)
            logger.warning(f"HDBSCAN 실패: 모든 {len(noise_data)}개 데이터를 노이즈로 처리")
        
        if len(hdbscan_clusters) == 0:
            logger.info("HDBSCAN이 클러스터를 찾지 못했습니다. K-Means로 대체 클러스터링을 수행합니다.")
            hdbscan_clusters = self._perform_kmeans_clustering(titles_data, embeddings)
            
            if not hdbscan_clusters:
                logger.warning(f"데이터가 너무 적어 클러스터링을 수행할 수 없습니다. (데이터 개수: {len(titles_data)})")
        else:
            hdbscan_clusters = self._recluster_noise_with_kmeans(
                noise_data, noise_embeddings, hdbscan_clusters
            )
        
        return hdbscan_clusters, []
    
    def extract_keywords_with_keybert(self, text, top_k=3):
        """
        KeyBERT 기반 키워드 추출
        
        Args:
            text (str): 텍스트
            top_k (int): 추출할 키워드 개수
            
        Returns:
            list: 추출된 키워드 리스트
        """
        try:
            keybert_keywords = self.kw_model.extract_keywords(
                text, 
                keyphrase_ngram_range=(self.KEYBERT_NGRAM_MIN, self.KEYBERT_NGRAM_MAX),
                stop_words=list(self.stopwords),
                top_n=top_k,
                use_mmr=True,
                diversity=self.KEYBERT_DIVERSITY
            )
            
            return [kw for kw, score in keybert_keywords]
        except ValueError as e:
            logger.warning(f"KeyBERT 키워드 추출 실패 (데이터 부족 또는 형식 오류): {e}")
            return []
        except Exception as e:
            logger.error(f"KeyBERT 계산 중 예상치 못한 오류: {e}")
            return []

    def generate_cluster_labels(self, clusters):
        """
        클러스터별 키워드 라벨 생성
        
        Args:
            clusters (dict): 클러스터 딕셔너리
            
        Returns:
            dict: 클러스터별 라벨 정보 (major_category, keywords 포함)
        """
        cluster_labels = {}
        
        for cluster_id, news_list in clusters.items():
            titles = [item["cleaned_title"] for item in news_list]
            combined_text = " ".join(titles)
            
            tfidf_keywords = self.extract_keywords_with_konlpy_tfidf(
                titles, topn=self.TOP_KEYWORDS_COUNT
            )
            
            keybert_keywords = self.extract_keywords_with_keybert(
                combined_text, top_k=self.KEYBERT_TOP_K
            )
            
            combined_keywords = tfidf_keywords.copy()
            for kw in keybert_keywords:
                if kw not in combined_keywords:
                    combined_keywords.append(kw)
            
            major_category = self.determine_major_category(combined_keywords, titles)
            
            cluster_labels[cluster_id] = {
                "major_category": major_category,
                "keywords": combined_keywords[:self.TOP_KEYWORDS_COUNT],
                "tfidf_keywords": tfidf_keywords,
                "keybert_keywords": keybert_keywords
            }
        
        return cluster_labels
    
    def determine_major_category(self, keywords, titles):
        """
        키워드와 제목을 분석하여 대분류 카테고리 결정
        
        Args:
            keywords (list): 키워드 리스트
            titles (list): 제목 리스트
            
        Returns:
            str: 대분류 카테고리 이름
        """
        category_mapping = {
            "정치": ["대통령", "정부", "국회", "정치", "선거", "여야", "정책", "국정", "정당"],
            "경제": ["경제", "투자", "기업", "금융", "주식", "시장", "수출", "수입", "GDP", "금리"],
            "사회": ["사회", "교육", "복지", "보건", "환경", "교통", "주택", "노동", "고용"],
            "국제": ["국제", "외교", "미국", "중국", "일본", "러시아", "유럽", "트럼프", "푸틴"],
            "법무": ["법무", "법원", "검찰", "경찰", "재판", "형사", "민사", "법률", "사법"],
            "문화": ["문화", "예술", "스포츠", "연예", "영화", "음악", "축제", "전시"],
            "기술": ["기술", "AI", "인공지능", "디지털", "스마트", "IT", "소프트웨어", "하드웨어"],
            "교육": ["교육", "대학", "학교", "학생", "교수", "연구", "학술", "입시"],
            "의료": ["의료", "병원", "의사", "치료", "건강", "질병", "의약", "보건"],
            "환경": ["환경", "기후", "에너지", "재생", "친환경", "대기", "수질", "폐기물"]
        }
        
        category_scores = {}
        all_texts = titles + [f"{' '.join(keywords)}"]
        tfidf_keywords = self.extract_keywords_with_konlpy_tfidf(all_texts, topn=self.TFIDF_CATEGORY_KEYWORDS)
        
        for category, words in category_mapping.items():
            score = 0
            for word in words:
                if word in tfidf_keywords:
                    try:
                        rank = tfidf_keywords.index(word)
                        weight = 1.0 / (rank + 1)
                        score += weight
                    except ValueError:
                        pass
            category_scores[category] = score
        
        if category_scores:
            best_category = max(category_scores, key=category_scores.get)
            if category_scores[best_category] > 0:
                return best_category
        
        return tfidf_keywords[0] if tfidf_keywords else "기타"
    
    def normalize_keyword(self, keyword):
        """
        키워드의 공백을 하이픈으로 변경 (노드 ID 매칭용)
        
        Args:
            keyword (str): 키워드
            
        Returns:
            str: 정규화된 키워드
        """
        return keyword.replace(" ", "-")
    
    def _format_news_item(self, news):
        """
        뉴스 데이터를 프론트엔드 형식으로 변환
        
        Args:
            news (dict): 뉴스 데이터
            
        Returns:
            dict: {"title": ..., "link": ...} 형식의 뉴스 데이터
        """
        return {
            "title": news['cleaned_title'], 
            "link": news["original"].get("link", "")
        }
    
    def _filter_and_sort_universities(self, univ_news):
        """
        대학교 데이터 필터링 및 정렬
        
        Args:
            univ_news (dict): 대학교별 뉴스
            
        Returns:
            dict: 필터링 및 정렬된 대학교 데이터
        """
        filtered_universities = {
            university_name: news_list 
            for university_name, news_list in univ_news.items() 
            if len(news_list) >= self.MIN_UNIV_NEWS_COUNT
        }
        
        sorted_items = sorted(filtered_universities.items(), key=lambda x: len(x[1]), reverse=True)
        return dict(sorted_items[:self.MAX_UNIV_DISPLAY])
    
    def _assign_news_to_minor_categories(self, news_list, keywords):
        """
        뉴스를 중분류 키워드에 할당 (각 뉴스는 하나의 중분류에만 할당)
        
        Args:
            news_list (list): 클러스터 내 뉴스 리스트
            keywords (list): 키워드 리스트
            
        Returns:
            dict: 키워드별 뉴스 딕셔너리
        """
        minor_category_news = {keyword: [] for keyword in keywords[:self.TOP_KEYWORDS_COUNT]}
        assigned_news = set()
        
        for keyword in keywords[:self.TOP_KEYWORDS_COUNT]:
            for news in news_list:
                news_link = news["original"].get("link", "")
                
                if news_link in assigned_news:
                    continue
                
                if keyword in news['cleaned_title']:
                    minor_category_news[keyword].append(news)
                    assigned_news.add(news_link)
        
        return minor_category_news
    
    def _build_minor_categories_data(self, minor_category_news):
        """
        중분류 데이터 구조 생성
        
        Args:
            minor_category_news (dict): 키워드별 뉴스 딕셔너리
            
        Returns:
            list: 중분류 데이터 리스트
        """
        minor_categories_data = []
        
        for minor_cat, minor_news in minor_category_news.items():
            if len(minor_news) < self.MIN_MINOR_NEWS_COUNT:
                continue
            
            minor_categories_data.append({
                "name": minor_cat,
                "news_count": len(minor_news),
                "news": [self._format_news_item(news) for news in minor_news]
            })
        
        return minor_categories_data
    
    def _extract_cluster_info(self, cluster_info, cluster_id):
        """
        클러스터 라벨 정보에서 카테고리와 키워드 추출
        
        Args:
            cluster_info: 클러스터 라벨 정보 (dict 또는 list)
            cluster_id (int): 클러스터 ID
            
        Returns:
            tuple: (대분류 카테고리, 키워드 리스트)
        """
        if isinstance(cluster_info, dict):
            major_category = cluster_info.get("major_category", f"클러스터 {cluster_id}")
            keywords = cluster_info.get("keywords", [])
            return major_category, keywords
        
        major_category = f"클러스터 {cluster_id}"
        keywords = cluster_info if isinstance(cluster_info, list) else []
        return major_category, keywords
    
    def _build_university_data(self, univ_news):
        """
        대학교 데이터 구조 생성
        
        Args:
            univ_news (dict): 대학교별 뉴스
            
        Returns:
            list: 대학교 데이터 리스트
        """
        if not univ_news:
            return []
        
        sorted_universities = self._filter_and_sort_universities(univ_news)
        universities = []
        
        for university_name, news_list in sorted_universities.items():
            universities.append({
                "name": university_name,
                "news_count": len(news_list),
                "news": [self._format_news_item(news) for news in news_list]
            })
        
        return universities
    
    def _build_cluster_data(self, clusters, cluster_labels):
        """
        클러스터 데이터 구조 생성
        
        Args:
            clusters (dict): 클러스터 딕셔너리
            cluster_labels (dict): 클러스터 라벨 정보
            
        Returns:
            list: 클러스터 데이터 리스트
        """
        clusters_data = []
        
        for cluster_id, news_list in clusters.items():
            if len(news_list) < self.MIN_CLUSTER_NEWS_COUNT:
                continue
            
            cluster_info = cluster_labels.get(cluster_id, {})
            major_category, keywords = self._extract_cluster_info(cluster_info, cluster_id)
            
            minor_category_news = self._assign_news_to_minor_categories(news_list, keywords)
            minor_categories_data = self._build_minor_categories_data(minor_category_news)
            
            clusters_data.append({
                "cluster_id": cluster_id,
                "major_category": major_category,
                "news_count": len(news_list),
                "minor_categories": minor_categories_data
            })
        
        return clusters_data
    
    def _convert_to_major_keyword_format(self, universities, clusters):
        """
        대학교와 클러스터 데이터를 majorKeyword 형식으로 변환
        
        Args:
            universities (list): 대학교 데이터
            clusters (list): 클러스터 데이터
            
        Returns:
            list: 변환된 데이터 리스트
        """
        converted_data = []
        
        if universities:
            univ_middle_keywords = [
                {
                    "middleKeyword": self.normalize_keyword(univ['name']),
                    "relatedNews": univ['news']
                }
                for univ in universities
            ]
            
            if len(univ_middle_keywords) >= self.MIN_MIDDLE_KEYWORDS_COUNT:
                converted_data.append({
                    "majorKeyword": self.normalize_keyword("대학교"),
                    "middleKeywords": univ_middle_keywords,
                    "otherNews": []
                })
        
        for cluster in clusters:
            cluster_middle_keywords = [
                {
                    "middleKeyword": self.normalize_keyword(minor_cat['name']),
                    "relatedNews": minor_cat['news']
                }
                for minor_cat in cluster['minor_categories']
            ]
            
            if len(cluster_middle_keywords) < self.MIN_MIDDLE_KEYWORDS_COUNT:
                continue
            
            converted_data.append({
                "majorKeyword": self.normalize_keyword(cluster['major_category']),
                "middleKeywords": cluster_middle_keywords,
                "otherNews": []
            })
        
        return converted_data
    
    def create_frontend_data(self, univ_news, clusters, cluster_labels, noise_news):
        """
        프론트엔드용 데이터 구조 생성
        
        Args:
            univ_news (dict): 대학교별 뉴스
            clusters (dict): 클러스터 딕셔너리
            cluster_labels (dict): 클러스터 라벨 정보
            noise_news (list): 노이즈 뉴스 리스트
            
        Returns:
            list: 프론트엔드 형식의 분석 결과
        """
        universities = self._build_university_data(univ_news)
        clusters_data = self._build_cluster_data(clusters, cluster_labels)
        return self._convert_to_major_keyword_format(universities, clusters_data)
    
    def analyze_from_db(self, news_data):
        """
        뉴스 제목 분석 파이프라인
        
        전처리, 대학교 분류, 클러스터링, 키워드 추출을 순차적으로 수행합니다.
        
        Args:
            news_data (list): 원본 뉴스 데이터
            
        Returns:
            list: 프론트엔드 형식의 분석 결과
        """
        processed_data = self.preprocess_titles(news_data)
        
        if len(processed_data) < self.MIN_NEWS_COUNT:
            return None
        
        university_news, other_news = self.split_news_by_uni_name(processed_data)
        
        if other_news:
            clusters, noise_news = self.enhanced_cluster_news(other_news)
        else:
            clusters, noise_news = {}, []
        
        cluster_labels = self.generate_cluster_labels(clusters)
        
        frontend_data = self.create_frontend_data(university_news, clusters, cluster_labels, noise_news)
        
        return frontend_data

