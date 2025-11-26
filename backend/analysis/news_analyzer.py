import os
import re
import logging
from collections import defaultdict
import numpy as np
from sentence_transformers import SentenceTransformer
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
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
    MIN_CLUSTER_NEWS_COUNT = 3     # 클러스터로 분류되기 위한 최소 뉴스 개수 (8→3으로 완화)
    MIN_MINOR_NEWS_COUNT = 2       # 중분류로 분류되기 위한 최소 뉴스 개수 (4→2로 완화)
    MAX_UNIV_DISPLAY = 5           # 표시할 최대 대학교 개수
    MIN_MIDDLE_KEYWORDS_COUNT = 1  # 대분류로 표시되기 위한 최소 중분류 개수 (2→1로 완화)
    
    # 클러스터링 방법 설정 ('graph_based', 'frequency_based', 'advanced')
    CLUSTERING_METHOD = 'frequency_based'   # 메인 마인드맵 기본값: 빈도 기반 클러스터링
    
    # 클러스터링 관련 (적절한 클러스터 수를 위한 조정)
    HDBSCAN_MIN_CLUSTER_SIZE = 20          # HDBSCAN 최소 클러스터 크기 (25→20으로 완화)
    HDBSCAN_MIN_SAMPLES = 12               # HDBSCAN 최소 샘플 수 (15→12로 완화)
    HDBSCAN_EPSILON = 0.3                 # HDBSCAN 클러스터 선택 엡실론 (0.15→0.3으로 완화)
    CLUSTER_DUPLICATE_THRESHOLD = 0.5      # 클러스터 중복 비율 임계값
    MIN_NOISE_FOR_RECLUSTERING = 5         # 재클러스터링을 위한 최소 노이즈 개수
    
    # K-Means 클러스터링 파라미터 (적절한 클러스터 수를 위한 조정)
    KMEANS_SMALL_DATA_THRESHOLD = 20       # 소규모 데이터 임계값
    KMEANS_MEDIUM_DATA_THRESHOLD = 100     # 중규모 데이터 임계값
    KMEANS_SMALL_MAX_CLUSTERS = 3          # 소규모 데이터 최대 클러스터 수 (2→3)
    KMEANS_SMALL_DIVISOR = 8               # 소규모 데이터 클러스터 수 계산 제수 (10→8)
    KMEANS_MEDIUM_MAX_CLUSTERS = 6         # 중규모 데이터 최대 클러스터 수 (3→6)
    KMEANS_MEDIUM_DIVISOR = 20             # 중규모 데이터 클러스터 수 계산 제수 (30→20)
    KMEANS_LARGE_MAX_CLUSTERS = 10         # 대규모 데이터 최대 클러스터 수 (5→10)
    KMEANS_LARGE_DIVISOR = 30              # 대규모 데이터 클러스터 수 계산 제수 (50→30)
    KMEANS_RANDOM_STATE = 42               # K-Means 랜덤 시드
    KMEANS_N_INIT = 10                     # K-Means 초기화 횟수
    
    # TF-IDF 파라미터 (더욱 완화된 설정)
    TFIDF_NGRAM_MIN = 1            # TF-IDF n-gram 최소값
    TFIDF_NGRAM_MAX = 3            # TF-IDF n-gram 최대값
    TFIDF_MAX_FEATURES = 3000      # TF-IDF 최대 특성 수 (5000→3000으로 감소)
    TFIDF_MIN_DF = 1               # TF-IDF 최소 문서 빈도 (1로 유지)
    TFIDF_MAX_DF = 0.99            # TF-IDF 최대 문서 빈도 (0.95→0.99로 더 완화)
    
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
        KoNLPy + TF-IDF 기반 키워드 추출 (원본 복원)
        
        Args:
            texts (list): 텍스트 리스트
            topn (int): 추출할 키워드 개수
            
        Returns:
            list: 추출된 키워드 리스트
        """
        if not texts or len(texts) == 0:
            return []
        
        noun_texts = [" ".join(self.extract_nouns(text)) for text in texts]
        # 빈 텍스트 필터링
        noun_texts = [text for text in noun_texts if text.strip()]
        
        if not noun_texts:
            return []
        
        # 뉴스 수가 적으면 (5개 이하) TF-IDF 대신 빈도 기반 사용
        if len(noun_texts) <= 5:
            from collections import Counter
            all_nouns = []
            for text in noun_texts:
                if text.strip():
                    all_nouns.extend(text.split())
            
            if not all_nouns:
                return []
            
            noun_counts = Counter(all_nouns)
            top_keywords = [word for word, count in noun_counts.most_common(topn)]
            return top_keywords
        
        vectorizer = TfidfVectorizer(
            ngram_range=(self.TFIDF_NGRAM_MIN, self.TFIDF_NGRAM_MAX),
            max_features=self.TFIDF_MAX_FEATURES,
            min_df=self.TFIDF_MIN_DF,
            max_df=self.TFIDF_MAX_DF
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(noun_texts)
            feature_names = vectorizer.get_feature_names_out()
            
            if len(feature_names) == 0:
                # 빈도 기반으로 fallback
                return self._extract_keywords_by_frequency(noun_texts, topn)
            
            mean_scores = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
            top_indices = mean_scores.argsort()[-topn:][::-1]
            
            return [feature_names[i] for i in top_indices]
        except ValueError as e:
            # TF-IDF 실패 시 빈도 기반으로 fallback (경고 로그 제거)
            return self._extract_keywords_by_frequency(noun_texts, topn)
    
    def _extract_keywords_by_frequency(self, noun_texts, topn=5):
        """
        빈도 기반 키워드 추출 (fallback 메서드)
        
        Args:
            noun_texts (list): 명사 텍스트 리스트
            topn (int): 추출할 키워드 개수
            
        Returns:
            list: 추출된 키워드 리스트
        """
        from collections import Counter
        all_nouns = []
        for text in noun_texts:
            if text.strip():
                all_nouns.extend(text.split())
        
        if not all_nouns:
            return []
        
        noun_counts = Counter(all_nouns)
        top_keywords = [word for word, count in noun_counts.most_common(topn)]
        return top_keywords
    
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
            metric="euclidean",  # cosine → euclidean으로 되돌림 (HDBSCAN이 cosine을 지원하지 않음)
            cluster_selection_method='eom',  # 더 관대한 클러스터 선택
            cluster_selection_epsilon=self.HDBSCAN_EPSILON,
            prediction_data=True,
            alpha=1.0,  # 클러스터 선택을 더 관대하게
            allow_single_cluster=True  # 단일 클러스터 허용
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
        
        logger.info(f"🔄 K-Means 재분류 시작: {len(noise_data)}개 노이즈 → {n_clusters}개 클러스터")
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.KMEANS_RANDOM_STATE, n_init=self.KMEANS_N_INIT)
        kmeans_labels = kmeans.fit_predict(noise_embeddings)
        
        max_cluster_id = max(existing_clusters.keys()) if existing_clusters else -1
        
        updated_clusters = existing_clusters.copy()
        new_clusters = defaultdict(list)
        
        for label, item in zip(kmeans_labels, noise_data):
            new_cluster_id = max_cluster_id + 1 + label
            if new_cluster_id not in updated_clusters:
                updated_clusters[new_cluster_id] = []
            updated_clusters[new_cluster_id].append(item)
            new_clusters[label].append(item)
        
        # 새로 생성된 클러스터별 뉴스 제목 로그
        logger.info("🆕 K-Means로 새로 생성된 클러스터:")
        for label, cluster_items in new_clusters.items():
            new_cluster_id = max_cluster_id + 1 + label
            logger.info(f"   📁 클러스터 {new_cluster_id}: {len(cluster_items)}개 뉴스")
            for i, item in enumerate(cluster_items[:3]):  # 최대 3개만 표시
                title = item.get('cleaned_title', 'Unknown')[:50] + "..." if len(item.get('cleaned_title', '')) > 50 else item.get('cleaned_title', 'Unknown')
                logger.info(f"      {i+1}. {title}")
            if len(cluster_items) > 3:
                logger.info(f"      ... 외 {len(cluster_items) - 3}개")
        
        logger.info(f"✅ K-Means 재분류 완료: {len(new_clusters)}개 새 클러스터 생성")
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
        
        logger.info(f"🔍 클러스터링 시작: {n_data}개 뉴스")
        
        embeddings = self.embedding_model.encode(titles, normalize_embeddings=True)
        
        logger.info("📊 1단계: HDBSCAN 클러스터링 수행 중...")
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
            
            # HDBSCAN 결과 로그
            n_hdbscan_clusters = len(hdbscan_clusters)
            n_hdbscan_noise = len(noise_data)
            logger.info(f"✅ HDBSCAN 완료: {n_hdbscan_clusters}개 클러스터, {n_hdbscan_noise}개 노이즈")
            
            # 클러스터별 뉴스 제목 로그
            for cluster_id, cluster_items in hdbscan_clusters.items():
                logger.info(f"   📁 클러스터 {cluster_id}: {len(cluster_items)}개 뉴스")
                for i, item in enumerate(cluster_items[:3]):  # 최대 3개만 표시
                    title = item.get('cleaned_title', 'Unknown')[:50] + "..." if len(item.get('cleaned_title', '')) > 50 else item.get('cleaned_title', 'Unknown')
                    logger.info(f"      {i+1}. {title}")
                if len(cluster_items) > 3:
                    logger.info(f"      ... 외 {len(cluster_items) - 3}개")
            
            meaningful_clusters, filtered_noise, filtered_noise_emb = self._filter_meaningful_clusters(
                hdbscan_clusters, titles_data, embeddings
            )
            noise_data.extend(filtered_noise)
            noise_embeddings.extend(filtered_noise_emb)
            hdbscan_clusters = meaningful_clusters
            
            logger.info(f"🔍 의미있는 클러스터 필터링 후: {len(hdbscan_clusters)}개 클러스터, {len(noise_data)}개 노이즈")
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
            logger.info("📊 2단계: K-Means로 노이즈 재분류 수행 중...")
            hdbscan_clusters = self._recluster_noise_with_kmeans(
                noise_data, noise_embeddings, hdbscan_clusters
            )
            
            # K-Means 재분류 결과 로그
            n_final_clusters = len(hdbscan_clusters)
            n_final_noise = len(noise_data)
            logger.info(f"✅ K-Means 재분류 완료: 총 {n_final_clusters}개 클러스터, {n_final_noise}개 노이즈")
            
            # 최종 클러스터별 뉴스 제목 로그
            logger.info("📋 최종 클러스터링 결과:")
            for cluster_id, cluster_items in hdbscan_clusters.items():
                logger.info(f"   📁 클러스터 {cluster_id}: {len(cluster_items)}개 뉴스")
                for i, item in enumerate(cluster_items[:3]):  # 최대 3개만 표시
                    title = item.get('cleaned_title', 'Unknown')[:50] + "..." if len(item.get('cleaned_title', '')) > 50 else item.get('cleaned_title', 'Unknown')
                    logger.info(f"      {i+1}. {title}")
                if len(cluster_items) > 3:
                    logger.info(f"      ... 외 {len(cluster_items) - 3}개")
        
        logger.info(f"🎉 클러스터링 완료: {len(hdbscan_clusters)}개 클러스터, {len(noise_data)}개 노이즈")
        return hdbscan_clusters, []
    
    def kmeans_only_cluster_news(self, titles_data):
        """
        K-Means만 사용한 클러스터링 (HDBSCAN 대신)
        
        Args:
            titles_data (list): 클러스터링할 뉴스 데이터
            
        Returns:
            tuple: (클러스터 딕셔너리, 노이즈 리스트)
        """
        titles = [item["cleaned_title"] for item in titles_data]
        n_data = len(titles_data)
        
        logger.info(f"🔍 K-Means 클러스터링 시작: {n_data}개 뉴스")
        
        embeddings = self.embedding_model.encode(titles, normalize_embeddings=True)
        
        # K-Means 클러스터 수 계산
        n_clusters = self.calculate_kmeans_clusters(n_data)
        
        if n_clusters < 2:
            logger.warning(f"데이터가 너무 적어 클러스터링을 수행할 수 없습니다. (클러스터 수: {n_clusters})")
            return {}, titles_data
        
        logger.info(f"📊 K-Means 클러스터링 수행: {n_data}개 뉴스 → {n_clusters}개 클러스터")
        
        # K-Means 클러스터링 수행
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.KMEANS_RANDOM_STATE, n_init=self.KMEANS_N_INIT)
        kmeans_labels = kmeans.fit_predict(embeddings)
        
        # 클러스터 딕셔너리 생성
        clusters = defaultdict(list)
        for label, item in zip(kmeans_labels, titles_data):
            clusters[label].append(item)
        
        # 클러스터별 뉴스 제목 로그
        logger.info("📋 K-Means 클러스터링 결과:")
        for cluster_id, cluster_items in clusters.items():
            logger.info(f"   📁 클러스터 {cluster_id}: {len(cluster_items)}개 뉴스")
            for i, item in enumerate(cluster_items[:3]):  # 최대 3개만 표시
                title = item.get('cleaned_title', 'Unknown')[:50] + "..." if len(item.get('cleaned_title', '')) > 50 else item.get('cleaned_title', 'Unknown')
                logger.info(f"      {i+1}. {title}")
            if len(cluster_items) > 3:
                logger.info(f"      ... 외 {len(cluster_items) - 3}개")
        
        logger.info(f"🎉 K-Means 클러스터링 완료: {len(clusters)}개 클러스터")
        return clusters, []
    
    def _reduce_dimensions_with_umap(self, embeddings, n_components=15, n_neighbors=25, metric='cosine'):
        """
        UMAP을 사용한 차원 축소
        
        Args:
            embeddings: 원본 임베딩 (384차원)
            n_components: 축소할 차원 수
            n_neighbors: UMAP 이웃 수
            metric: 거리 메트릭
            
        Returns:
            축소된 임베딩
        """
        try:
            import umap
            logger.info(f"🔽 UMAP 차원 축소 시작: {embeddings.shape[1]}D → {n_components}D")
            logger.info(f"   파라미터: n_neighbors={n_neighbors}, metric={metric}")
            
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                metric=metric,
                n_jobs=1,  # 멀티프로세싱 비활성화로 경고 제거
                verbose=False
            )
            
            reduced_embeddings = reducer.fit_transform(embeddings)
            logger.info(f"✅ UMAP 차원 축소 완료: {reduced_embeddings.shape}")
            
            return reduced_embeddings
            
        except ImportError:
            logger.warning("⚠️ UMAP이 설치되지 않음. 원본 임베딩 사용")
            return embeddings
        except Exception as e:
            logger.error(f"❌ UMAP 차원 축소 실패: {e}. 원본 임베딩 사용")
            return embeddings
    
    def _tuned_hdbscan_clustering(self, embeddings, min_cluster_size=30, min_samples=12, 
                                 cluster_selection_epsilon=0.6, probability_threshold=0.2):
        """
        튜닝된 HDBSCAN 클러스터링 (확률 기반 필터 포함)
        
        Args:
            embeddings: 차원 축소된 임베딩
            min_cluster_size: 최소 클러스터 크기
            min_samples: 최소 샘플 수
            cluster_selection_epsilon: 클러스터 선택 엡실론
            probability_threshold: 확률 임계값 (이하 노이즈 처리)
            
        Returns:
            tuple: (클러스터 라벨, 확률, 클러스터러 객체)
        """
        try:
            import hdbscan
            logger.info(f"🔍 튜닝된 HDBSCAN 클러스터링 시작")
            logger.info(f"   파라미터: min_cluster_size={min_cluster_size}, min_samples={min_samples}")
            logger.info(f"   cluster_selection_epsilon={cluster_selection_epsilon}, probability_threshold={probability_threshold}")
            
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric="euclidean",  # UMAP 출력은 유클리드로 처리
                cluster_selection_method='eom',
                cluster_selection_epsilon=cluster_selection_epsilon,
                allow_single_cluster=False,  # 단일 대클러스터 방지
                prediction_data=True
            )
            
            cluster_labels = clusterer.fit_predict(embeddings)
            probabilities = clusterer.probabilities_
            
            # 확률 기반 필터 적용
            original_labels = cluster_labels.copy()
            low_prob_mask = probabilities < probability_threshold
            cluster_labels[low_prob_mask] = -1
            
            n_original_clusters = len(set(original_labels)) - (1 if -1 in original_labels else 0)
            n_filtered_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_noise = list(cluster_labels).count(-1)
            
            logger.info(f"✅ HDBSCAN 완료: {n_original_clusters}개 → {n_filtered_clusters}개 클러스터")
            logger.info(f"   노이즈: {n_noise}개 (확률 임계값 {probability_threshold} 적용)")
            
            return cluster_labels, probabilities, clusterer
            
        except ImportError:
            logger.warning("⚠️ HDBSCAN이 설치되지 않음. K-Means 사용")
            return None, None, None
        except Exception as e:
            logger.error(f"❌ HDBSCAN 클러스터링 실패: {e}")
            return None, None, None
    
    def _merge_similar_clusters(self, clusters, embeddings, similarity_threshold=0.8):
        """
        유사한 클러스터 병합 (센트로이드 코사인 유사도 기반)
        
        Args:
            clusters: 클러스터 딕셔너리
            embeddings: 원본 임베딩
            similarity_threshold: 유사도 임계값
            
        Returns:
            병합된 클러스터 딕셔너리
        """
        if len(clusters) <= 1:
            return clusters
        
        logger.info(f"🔗 클러스터 병합 시작: {len(clusters)}개 클러스터")
        logger.info(f"   유사도 임계값: {similarity_threshold}")
        
        # 각 클러스터의 센트로이드 계산
        cluster_centroids = {}
        for cluster_id, cluster_items in clusters.items():
            cluster_indices = [i for i, item in enumerate(cluster_items)]
            if cluster_indices:
                centroid = np.mean(embeddings[cluster_indices], axis=0)
                cluster_centroids[cluster_id] = centroid
        
        # 코사인 유사도 계산
        cluster_ids = list(cluster_centroids.keys())
        centroids_matrix = np.array([cluster_centroids[cid] for cid in cluster_ids])
        similarity_matrix = cosine_similarity(centroids_matrix)
        
        # 병합할 클러스터 찾기
        merged_clusters = clusters.copy()
        merge_count = 0
        
        for i, cluster_id1 in enumerate(cluster_ids):
            if cluster_id1 not in merged_clusters:
                continue
                
            for j, cluster_id2 in enumerate(cluster_ids[i+1:], i+1):
                if cluster_id2 not in merged_clusters:
                    continue
                    
                similarity = similarity_matrix[i, j]
                if similarity >= similarity_threshold:
                    # 클러스터 병합
                    merged_clusters[cluster_id1].extend(merged_clusters[cluster_id2])
                    del merged_clusters[cluster_id2]
                    merge_count += 1
                    logger.info(f"   병합: 클러스터 {cluster_id1} + {cluster_id2} (유사도: {similarity:.3f})")
                    break  # 한 번에 하나씩만 병합
        
        logger.info(f"✅ 클러스터 병합 완료: {len(clusters)}개 → {len(merged_clusters)}개 ({merge_count}번 병합)")
        return merged_clusters
    
    def _assign_noise_to_clusters(self, clusters, noise_data, noise_embeddings, similarity_threshold=0.75):
        """
        노이즈 데이터를 기존 클러스터에 편승
        
        Args:
            clusters: 기존 클러스터
            noise_data: 노이즈 데이터
            noise_embeddings: 노이즈 임베딩
            similarity_threshold: 유사도 임계값
            
        Returns:
            tuple: (업데이트된 클러스터, 남은 노이즈)
        """
        if not noise_data or not clusters:
            return clusters, noise_data
        
        logger.info(f"🎯 노이즈 편승 시작: {len(noise_data)}개 노이즈")
        logger.info(f"   유사도 임계값: {similarity_threshold}")
        
        # 각 클러스터의 센트로이드 계산
        cluster_centroids = {}
        for cluster_id, cluster_items in clusters.items():
            if cluster_items:
                # 클러스터 아이템의 임베딩 인덱스 찾기 (실제 구현에서는 더 정확한 방법 필요)
                centroid = np.mean([noise_embeddings[0]], axis=0)  # 임시 구현
                cluster_centroids[cluster_id] = centroid
        
        updated_clusters = clusters.copy()
        remaining_noise = []
        assigned_count = 0
        
        for i, noise_item in enumerate(noise_data):
            noise_embedding = noise_embeddings[i]
            best_similarity = 0
            best_cluster_id = None
            
            # 가장 유사한 클러스터 찾기
            for cluster_id, centroid in cluster_centroids.items():
                similarity = cosine_similarity([noise_embedding], [centroid])[0][0]
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_cluster_id = cluster_id
            
            # 임계값 이상이면 편승
            if best_similarity >= similarity_threshold:
                updated_clusters[best_cluster_id].append(noise_item)
                assigned_count += 1
            else:
                remaining_noise.append(noise_item)
        
        logger.info(f"✅ 노이즈 편승 완료: {assigned_count}개 편승, {len(remaining_noise)}개 남음")
        return updated_clusters, remaining_noise
    
    def _spherical_kmeans_clustering(self, noise_data, noise_embeddings, n_clusters=None):
        """
        Spherical K-Means 클러스터링 (L2 정규화 + 일반 K-Means 근사)
        
        Args:
            noise_data: 노이즈 데이터
            noise_embeddings: 노이즈 임베딩
            n_clusters: 클러스터 수 (None이면 자동 계산)
            
        Returns:
            클러스터 딕셔너리
        """
        if not noise_data or len(noise_data) < 2:
            return {}
        
        if n_clusters is None:
            n_clusters = self.calculate_kmeans_clusters(len(noise_data))
        
        if n_clusters < 2:
            logger.warning(f"⚠️ 노이즈 데이터가 너무 적어 Spherical K-Means 생략: {len(noise_data)}개")
            return {}
        
        logger.info(f"🌐 Spherical K-Means 시작: {len(noise_data)}개 노이즈 → {n_clusters}개 클러스터")
        
        # L2 정규화
        from sklearn.preprocessing import normalize
        normalized_embeddings = normalize(noise_embeddings, norm='l2')
        
        # 일반 K-Means 사용 (MiniBatchKMeans로 속도 향상)
        from sklearn.cluster import MiniBatchKMeans
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=self.KMEANS_RANDOM_STATE,
            n_init='auto',
            batch_size=min(100, len(noise_data))
        )
        
        kmeans_labels = kmeans.fit_predict(normalized_embeddings)
        
        # 클러스터 딕셔너리 생성
        clusters = defaultdict(list)
        for label, item in zip(kmeans_labels, noise_data):
            clusters[label].append(item)
        
        # 클러스터별 뉴스 제목 로그
        logger.info("📋 Spherical K-Means 결과:")
        for cluster_id, cluster_items in clusters.items():
            logger.info(f"   📁 클러스터 {cluster_id}: {len(cluster_items)}개 뉴스")
            for i, item in enumerate(cluster_items[:3]):  # 최대 3개만 표시
                title = item.get('cleaned_title', 'Unknown')[:50] + "..." if len(item.get('cleaned_title', '')) > 50 else item.get('cleaned_title', 'Unknown')
                logger.info(f"      {i+1}. {title}")
            if len(cluster_items) > 3:
                logger.info(f"      ... 외 {len(cluster_items) - 3}개")
        
        logger.info(f"✅ Spherical K-Means 완료: {len(clusters)}개 클러스터")
        return clusters
    
    def graph_based_cluster_news(self, titles_data):
        """
        그래프 기반 클러스터링: TF-IDF 코사인 유사도 + 연결요소로 토픽 그룹화
        
        Args:
            titles_data (list): 클러스터링할 뉴스 데이터
            
        Returns:
            tuple: (클러스터 딕셔너리, 노이즈 리스트)
        """
        titles = [item["cleaned_title"] for item in titles_data]
        n_data = len(titles_data)
        
        logger.info(f"🔗 그래프 기반 클러스터링 시작: {n_data}개 뉴스")
        
        # 1단계: 문자 n-gram TF-IDF 벡터화 (띄어쓰기/형태소 변동에 강함)
        logger.info("📊 문자 n-gram TF-IDF 벡터화 중...")
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
            
            # 한글 제목에 최적화된 TF-IDF 설정
            vectorizer = TfidfVectorizer(
                analyzer='char',           # 문자 단위 분석
                ngram_range=(3, 5),        # 3~5글자 n-gram
                min_df=3,                  # 최소 문서 빈도
                max_df=0.6,                # 최대 문서 빈도
                sublinear_tf=True,         # 로그 스케일링
                norm='l2'                  # L2 정규화
            )
            
            tfidf_matrix = vectorizer.fit_transform(titles)
            logger.info(f"✅ TF-IDF 벡터화 완료: {tfidf_matrix.shape}")
            
        except Exception as e:
            logger.warning(f"⚠️ TF-IDF 벡터화 실패: {e}")
            # Fallback: 단순 키워드 기반 클러스터링
            return self._fallback_keyword_clustering(titles_data)
        
        # 2단계: 코사인 유사도 계산 및 간선 생성
        logger.info("🔗 코사인 유사도 계산 및 간선 생성 중...")
        similarity_threshold = 0.22  # τ=0.22±0.05
        
        # 상위 k-NN만 계산 (메모리 효율성)
        k_neighbors = min(15, n_data - 1)
        
        # 코사인 유사도 계산
        cosine_sim = cosine_similarity(tfidf_matrix)
        
        # 간선 생성: 상위 k개 + 임계값 이상
        edges = []
        for i in range(n_data):
            # 상위 k개 유사도 인덱스
            top_k_indices = np.argsort(cosine_sim[i])[-k_neighbors-1:-1]  # 자기 자신 제외
            
            for j in top_k_indices:
                if cosine_sim[i][j] >= similarity_threshold:
                    edges.append((i, j, cosine_sim[i][j]))
        
        logger.info(f"✅ 간선 생성 완료: {len(edges)}개 간선 (임계값: {similarity_threshold})")
        
        # 3단계: 연결요소(Connected Components) 찾기
        logger.info("🔍 연결요소 탐색 중...")
        clusters = self._find_connected_components(n_data, edges)
        
        # 4단계: 클러스터 후처리 및 라벨링
        logger.info("🏷️ 클러스터 후처리 및 라벨링 중...")
        processed_clusters = {}
        cluster_id = 0
        
        for component in clusters:
            if len(component) >= self.MIN_CLUSTER_NEWS_COUNT:
                cluster_items = [titles_data[idx] for idx in component]
                processed_clusters[cluster_id] = cluster_items
                cluster_id += 1
                logger.info(f"   클러스터 {cluster_id-1}: {len(component)}개 뉴스")
        
        # 5단계: 노이즈 처리
        used_indices = set()
        for cluster in clusters:
            used_indices.update(cluster)
        noise_news = [titles_data[i] for i in range(n_data) if i not in used_indices]
        
        logger.info(f"🎉 그래프 기반 클러스터링 완료: {len(processed_clusters)}개 클러스터, {len(noise_news)}개 노이즈")
        
        return processed_clusters, noise_news
    
    def _find_connected_components(self, n_nodes, edges):
        """
        연결요소 찾기 (Union-Find 알고리즘)
        """
        parent = list(range(n_nodes))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # 간선으로 노드들 연결
        for i, j, _ in edges:
            union(i, j)
        
        # 연결요소별로 그룹화
        components = {}
        for i in range(n_nodes):
            root = find(i)
            if root not in components:
                components[root] = []
            components[root].append(i)
        
        return list(components.values())
    
    def _fallback_keyword_clustering(self, titles_data):
        """
        TF-IDF 실패 시 단순 키워드 기반 클러스터링
        """
        logger.info("🔄 Fallback: 단순 키워드 기반 클러스터링")
        
        titles = [item["cleaned_title"] for item in titles_data]
        n_data = len(titles_data)
        
        # 단순 키워드 추출 (명사만)
        news_keywords = []
        for title in titles:
            try:
                nouns = self.okt.nouns(title)
                # 길이 2 이상인 명사만
                keywords = [noun for noun in nouns if len(noun) >= 2]
                news_keywords.append(keywords[:5])  # 상위 5개만
            except:
                news_keywords.append([])
        
        # Jaccard 유사도 기반 클러스터링
        clusters = {}
        cluster_id = 0
        used_indices = set()
        
        for i, keywords_i in enumerate(news_keywords):
            if i in used_indices or not keywords_i:
                continue
                
            similar_news = [i]
            used_indices.add(i)
            
            for j, keywords_j in enumerate(news_keywords):
                if j in used_indices or not keywords_j:
                    continue
                    
                # Jaccard 유사도
                intersection = len(set(keywords_i) & set(keywords_j))
                union = len(set(keywords_i) | set(keywords_j))
                similarity = intersection / union if union > 0 else 0
                
                if similarity >= 0.3:  # 30% 이상 유사
                    similar_news.append(j)
                    used_indices.add(j)
            
            if len(similar_news) >= self.MIN_CLUSTER_NEWS_COUNT:
                clusters[cluster_id] = [titles_data[idx] for idx in similar_news]
                cluster_id += 1
        
        noise_news = [titles_data[i] for i in range(n_data) if i not in used_indices]
        
        logger.info(f"✅ Fallback 클러스터링 완료: {len(clusters)}개 클러스터, {len(noise_news)}개 노이즈")
        
        return clusters, noise_news
    
    def frequency_based_cluster_news(self, titles_data):
        """
        빈도 기반 클러스터링: TF-IDF 키워드 유사도로 뉴스 그룹화
        
        Args:
            titles_data (list): 클러스터링할 뉴스 데이터
            
        Returns:
            tuple: (클러스터 딕셔너리, 노이즈 리스트)
        """
        titles = [item["cleaned_title"] for item in titles_data]
        n_data = len(titles_data)
        
        logger.info(f"🔢 빈도 기반 클러스터링 시작: {n_data}개 뉴스")
        
        # 1단계: TF-IDF 키워드 추출
        logger.info("📊 TF-IDF 키워드 추출 중...")
        all_keywords = []
        news_keywords = []
        
        for i, title in enumerate(titles):
            keywords = self.extract_keywords_with_konlpy_tfidf([title], topn=5)
            news_keywords.append(keywords)
            all_keywords.extend(keywords)
        
        logger.info(f"✅ 키워드 추출 완료: 평균 {sum(len(kw) for kw in news_keywords) / len(news_keywords):.1f}개/뉴스")
        
        # 2단계: 키워드 유사도 기반 클러스터링
        logger.info("🔗 키워드 유사도 기반 클러스터링 중...")
        clusters = {}
        cluster_id = 0
        used_indices = set()
        
        # 각 뉴스에 대해 유사한 뉴스 찾기
        for i, keywords_i in enumerate(news_keywords):
            if i in used_indices:
                continue
                
            # 현재 뉴스와 유사한 뉴스들 찾기
            similar_news = [i]
            used_indices.add(i)
            
            for j, keywords_j in enumerate(news_keywords):
                if j in used_indices:
                    continue
                    
                # 키워드 유사도 계산 (Jaccard 유사도)
                if keywords_i and keywords_j:
                    intersection = len(set(keywords_i) & set(keywords_j))
                    union = len(set(keywords_i) | set(keywords_j))
                    similarity = intersection / union if union > 0 else 0
                    
                    # 유사도 임계값 (30% 이상)
                    if similarity >= 0.3:
                        similar_news.append(j)
                        used_indices.add(j)
            
            # 클러스터 크기가 충분한 경우만 저장
            if len(similar_news) >= self.MIN_CLUSTER_NEWS_COUNT:
                clusters[cluster_id] = [titles_data[idx] for idx in similar_news]
                cluster_id += 1
                logger.info(f"   클러스터 {cluster_id-1}: {len(similar_news)}개 뉴스 (키워드: {keywords_i[:3]})")
        
        # 3단계: 노이즈 처리
        noise_news = [titles_data[i] for i in range(n_data) if i not in used_indices]
        
        logger.info(f"🎉 빈도 기반 클러스터링 완료: {len(clusters)}개 클러스터, {len(noise_news)}개 노이즈")
        
        return clusters, noise_news
    
    def advanced_cluster_news(self, titles_data, embeddings=None):
        """
        고급 클러스터링 파이프라인: UMAP → 튜닝된 HDBSCAN → 병합 → 노이즈 편승 → Spherical K-Means
        
        Args:
            titles_data (list): 클러스터링할 뉴스 데이터
            embeddings: 미리 생성된 임베딩 (None이면 새로 생성)
            
        Returns:
            tuple: (클러스터 딕셔너리, 노이즈 리스트)
        """
        titles = [item["cleaned_title"] for item in titles_data]
        n_data = len(titles_data)
        
        logger.info(f"🚀 고급 클러스터링 파이프라인 시작: {n_data}개 뉴스")
        
        # 1단계: 임베딩 생성 (없는 경우에만)
        if embeddings is None:
            logger.info("🤖 임베딩 생성 중...")
            embeddings = self.embedding_model.encode(titles, normalize_embeddings=True)
            logger.info(f"✅ 임베딩 생성 완료: {embeddings.shape}")
        else:
            logger.info(f"♻️ 기존 임베딩 재사용: {embeddings.shape}")
        
        # 2단계: UMAP 차원 축소
        reduced_embeddings = self._reduce_dimensions_with_umap(
            embeddings, 
            n_components=15, 
            n_neighbors=25, 
            metric='cosine'
        )
        
        # 3단계: 튜닝된 HDBSCAN 클러스터링
        cluster_labels, probabilities, clusterer = self._tuned_hdbscan_clustering(
            reduced_embeddings,
            min_cluster_size=30,
            min_samples=12,
            cluster_selection_epsilon=0.6,
            probability_threshold=0.2
        )
        
        if cluster_labels is None:
            logger.warning("⚠️ HDBSCAN 실패. K-Means로 대체")
            return self.kmeans_only_cluster_news(titles_data)
        
        # 4단계: 클러스터 딕셔너리 생성
        clusters = defaultdict(list)
        noise_data = []
        noise_embeddings = []
        
        for label, item, embedding in zip(cluster_labels, titles_data, embeddings):
            if label == -1:
                noise_data.append(item)
                noise_embeddings.append(embedding)
            else:
                clusters[label].append(item)
        
        logger.info(f"📊 HDBSCAN 결과: {len(clusters)}개 클러스터, {len(noise_data)}개 노이즈")
        
        # 5단계: 클러스터 병합
        if len(clusters) > 1:
            clusters = self._merge_similar_clusters(clusters, embeddings, similarity_threshold=0.8)
        
        # 6단계: 노이즈 편승
        if noise_data and clusters:
            clusters, remaining_noise_data = self._assign_noise_to_clusters(
                clusters, noise_data, noise_embeddings, similarity_threshold=0.75
            )
            noise_data = remaining_noise_data
        
        # 7단계: 남은 노이즈를 Spherical K-Means로 처리
        final_noise = []
        if noise_data:
            spherical_clusters = self._spherical_kmeans_clustering(noise_data, noise_embeddings)
            
            # Spherical K-Means 결과를 기존 클러스터에 추가
            max_cluster_id = max(clusters.keys()) if clusters else -1
            for label, cluster_items in spherical_clusters.items():
                new_cluster_id = max_cluster_id + 1 + label
                clusters[new_cluster_id] = cluster_items
            
            logger.info(f"🔄 Spherical K-Means로 {len(spherical_clusters)}개 추가 클러스터 생성")
        
        logger.info(f"🎉 고급 클러스터링 완료: 총 {len(clusters)}개 클러스터, {len(final_noise)}개 최종 노이즈")
        
        # 최종 클러스터별 뉴스 제목 로그
        logger.info("📋 최종 클러스터링 결과:")
        for cluster_id, cluster_items in clusters.items():
            logger.info(f"   📁 클러스터 {cluster_id}: {len(cluster_items)}개 뉴스")
            for i, item in enumerate(cluster_items[:3]):  # 최대 3개만 표시
                title = item.get('cleaned_title', 'Unknown')[:50] + "..." if len(item.get('cleaned_title', '')) > 50 else item.get('cleaned_title', 'Unknown')
                logger.info(f"      {i+1}. {title}")
            if len(cluster_items) > 3:
                logger.info(f"      ... 외 {len(cluster_items) - 3}개")
        
        return clusters, final_noise
    
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
        클러스터별 키워드 라벨 생성 (원본 복원)
        
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
        클러스터 라벨 정보에서 카테고리와 키워드 추출 (원본 복원)
        
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
    
    def analyze_from_db(self, news_data, embeddings=None, clustering_method=None):
        """
        뉴스 제목 분석 파이프라인
        
        전처리, 대학교 분류, 클러스터링, 키워드 추출을 순차적으로 수행합니다.
        
        Args:
            news_data (list): 원본 뉴스 데이터
            embeddings: 미리 생성된 임베딩 (None이면 새로 생성)
            clustering_method (str): 클러스터링 방법 ('graph_based', 'frequency_based', 'advanced')
                                     None이면 self.CLUSTERING_METHOD 사용
            
        Returns:
            list: 프론트엔드 형식의 분석 결과
        """
        processed_data = self.preprocess_titles(news_data)
        
        if len(processed_data) < self.MIN_NEWS_COUNT:
            return None
        
        university_news, other_news = self.split_news_by_uni_name(processed_data)
        
        if other_news:
            # 클러스터링 방법 선택 (파라미터가 없으면 기본값 사용)
            if clustering_method is None:
                clustering_method = self.CLUSTERING_METHOD
            
            if clustering_method == 'graph_based':
                logger.info("🔗 그래프 기반 클러스터링 사용 (TF-IDF 코사인 유사도 + 연결요소)")
                clusters, noise_news = self.graph_based_cluster_news(other_news)
            elif clustering_method == 'frequency_based':
                logger.info("🔢 빈도 기반 클러스터링 사용 (TF-IDF 키워드 유사도)")
                clusters, noise_news = self.frequency_based_cluster_news(other_news)
            elif clustering_method == 'advanced':
                logger.info("🚀 고급 클러스터링 사용 (UMAP + HDBSCAN + K-Means)")
                clusters, noise_news = self.advanced_cluster_news(other_news, embeddings)
            else:
                logger.info("🔢 기본 빈도 기반 클러스터링 사용")
                clusters, noise_news = self.frequency_based_cluster_news(other_news)
        else:
            clusters, noise_news = {}, []
        
        cluster_labels = self.generate_cluster_labels(clusters)
        
        frontend_data = self.create_frontend_data(university_news, clusters, cluster_labels, noise_news)
        
        return frontend_data

