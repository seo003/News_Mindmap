import os
import re
from collections import defaultdict, Counter
import numpy as np
from sentence_transformers import SentenceTransformer
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from konlpy.tag import Okt
from keybert import KeyBERT
from config.config import STOPWORDS_PATH, NON_UNIV_WORD_PATH


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
    
    # 클러스터링 관련
    HDBSCAN_MIN_CLUSTER_SIZE = 2           # HDBSCAN 최소 클러스터 크기
    HDBSCAN_MIN_SAMPLES = 1                # HDBSCAN 최소 샘플 수
    HDBSCAN_EPSILON = 0.05                 # HDBSCAN 클러스터 선택 엡실론
    CLUSTER_DUPLICATE_THRESHOLD = 0.5      # 클러스터 중복 비율 임계값
    MIN_NOISE_FOR_RECLUSTERING = 5         # 재클러스터링을 위한 최소 노이즈 개수
    
    # 키워드 추출 관련
    TOP_KEYWORDS_COUNT = 5         # 추출할 키워드 개수
    KEYBERT_TOP_K = 3              # KeyBERT 키워드 개수
    TFIDF_CATEGORY_KEYWORDS = 20   # 카테고리 분류용 TF-IDF 키워드 개수
    
    def __init__(self):
        """뉴스 분석기 초기화"""
        # 한글 지원 멀티링구얼 모델
        self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.kw_model = KeyBERT(model='paraphrase-multilingual-MiniLM-L12-v2')
        self.okt = Okt()
        
        # 파일에서 불용어와 제외 단어를 로드하여 캐싱
        self.stopwords = self.load_stopwords()
        self.exclude_words = self.load_exclude_words()
        
    def load_stopwords(self):
        """
        불용어 파일 로드
        
        Returns:
            set: 불용어 집합
        """
        try:
            with open(STOPWORDS_PATH, "r", encoding="utf-8") as f:
                return set([line.strip() for line in f if line.strip()])
        except FileNotFoundError:
            print(f"불용어 파일을 찾을 수 없습니다: {STOPWORDS_PATH}")
            return set()
        except Exception as e:
            print(f"불용어 로드 오류: {e}")
            return set()
    
    def load_exclude_words(self):
        """
        대학교가 아닌 제외 단어 목록 로드
        
        Returns:
            set: 제외 단어 집합
        """
        try:
            with open(NON_UNIV_WORD_PATH, "r", encoding="utf-8") as f:
                return set([line.strip() for line in f if line.strip()])
        except FileNotFoundError:
            print(f"제외 단어 파일을 찾을 수 없습니다: {NON_UNIV_WORD_PATH}")
            return set()
        except Exception as e:
            print(f"제외 단어 로드 오류: {e}")
            return set()
    
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
    
    def split_news_by_uni_name(self, processed_data):
        """
        대학교 이름으로 뉴스 분류
        """
        univ_news = defaultdict(list)
        other_news = []
        
        # '대'로 끝나는 단어 정규표현식
        uni_pattern = re.compile(r".+대$")
        
        for item in processed_data:
            title = item["cleaned_title"]
            
            # 명사 추출
            nouns = self.extract_nouns(title)
            
            # 대학교 이름 찾기 (캐시된 exclude_words 사용)
            uni_kw = next((kw for kw in nouns if uni_pattern.match(kw) and kw not in self.exclude_words), None)
            
            if not uni_kw and "KAIST" in nouns:
                uni_kw = "KAIST"
            
            # 결과 저장
            news_info = {
                "original": item["original"],
                "cleaned_title": title,
                "nouns": nouns
            }
            
            if uni_kw:
                univ_news[uni_kw].append(news_info)
            else:
                other_news.append(news_info)
        
        return univ_news, other_news
    
    def preprocess_titles(self, news_data):
        """
        뉴스 제목 전처리 (특수문자 제거, 중복 제거)
        """
        # 특수문자 제거 및 기본 정제
        processed_titles = []
        
        for item in news_data:
            title = item["title"]
            
            # 특수문자 및 괄호 내용 제거
            title = re.sub(r'\[.*?\]', '', title)  # 대괄호 내용 제거
            title = re.sub(r'\(.*?\)', '', title)  # 소괄호 내용 제거
            title = re.sub(r'<.*?>', '', title)    # HTML 태그 제거
            title = re.sub(r'[^\w\s가-힣]', ' ', title)  # 특수문자를 공백으로
            title = re.sub(r'\s+', ' ', title).strip()  # 연속 공백 정리
            
            if len(title) > self.MIN_TITLE_LENGTH:
                processed_titles.append({
                    "original": item,
                    "cleaned_title": title
                })
        
        # 중복 뉴스 제거
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
        # 1단계: 각 텍스트에서 명사 추출
        noun_texts = []
        for text in texts:
            nouns = self.extract_nouns(text)
            noun_texts.append(" ".join(nouns))
        
        # 2단계: TF-IDF 벡터화
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),  # 1-3그램 사용
            max_features=5000,   # 최대 특성 수
            min_df=1,           # 최소 문서 빈도
            max_df=0.9          # 최대 문서 빈도
        )
        
        try:
            X = vectorizer.fit_transform(noun_texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # 평균 TF-IDF 점수로 상위 키워드 선택
            mean_scores = np.asarray(X.mean(axis=0)).ravel()
            top_indices = mean_scores.argsort()[-topn:][::-1]
            
            return [feature_names[i] for i in top_indices]
        except Exception as e:
            print(f"TF-IDF 계산 오류: {e}")
            return []
    
    def calculate_kmeans_clusters(self, n_data):
        """
        데이터 개수에 따라 적절한 K-Means 클러스터 수 계산
        
        Args:
            n_data (int): 데이터 개수
            
        Returns:
            int: 적절한 클러스터 개수
        """
        if n_data < 20:
            return min(3, n_data // 5)
        elif n_data < 100:
            return min(8, n_data // 10)
        else:
            return min(15, n_data // 15)
    
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
        
        # 임베딩 생성
        embeddings = self.embedding_model.encode(titles, normalize_embeddings=True)
        
        # HDBSCAN 클러스터링 설정
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.HDBSCAN_MIN_CLUSTER_SIZE,
            min_samples=self.HDBSCAN_MIN_SAMPLES,
            metric="euclidean",
            cluster_selection_method='leaf',
            cluster_selection_epsilon=self.HDBSCAN_EPSILON,
            prediction_data=True
        )
        hdbscan_labels = clusterer.fit_predict(embeddings)
        
        # HDBSCAN 결과 분석
        hdbscan_clusters = defaultdict(list)
        noise_data = []
        noise_embeddings = []
        
        if hdbscan_labels is not None:
            # 레이블을 이용해 클러스터 딕셔너리 구성
            for label, item, embedding in zip(hdbscan_labels, titles_data, embeddings):
                if label == -1:
                    # 노이즈 데이터 수집
                    noise_data.append(item)
                    noise_embeddings.append(embedding)
                else:
                    hdbscan_clusters[label].append(item)
            
            # 의미있는 클러스터만 추출 (크기 4개 이상, 중복 뉴스가 아닌 클러스터)
            meaningful_clusters = {}
            meaningless_clusters = []
            
            for cluster_id, cluster_news in hdbscan_clusters.items():
                # 클러스터 크기가 최소 개수 이상인지 확인
                if len(cluster_news) >= self.MIN_CLUSTER_NEWS_COUNT:
                    # 중복 뉴스가 있는지 확인
                    titles = [news['cleaned_title'] for news in cluster_news]
                    unique_titles = set(titles)
                    
                    # 중복 비율이 임계값 미만인 경우만 의미있는 클러스터로 간주
                    duplicate_ratio = (len(titles) - len(unique_titles)) / len(titles)
                    if duplicate_ratio < self.CLUSTER_DUPLICATE_THRESHOLD:
                        meaningful_clusters[cluster_id] = cluster_news
                    else:
                        meaningless_clusters.append((cluster_id, cluster_news, f"중복 비율 높음 ({duplicate_ratio:.1%})"))
                else:
                    meaningless_clusters.append((cluster_id, cluster_news, f"크기 부족 ({len(cluster_news)}개)"))
            
            # 의미없는 클러스터들을 노이즈로 이동
            for cluster_id, cluster_news, _ in meaningless_clusters:
                noise_data.extend(cluster_news)
                # 해당 클러스터의 임베딩도 노이즈로 이동
                for item in cluster_news:
                    item_idx = titles_data.index(item)
                    noise_embeddings.append(embeddings[item_idx])
            
            # 의미있는 클러스터만 남김
            hdbscan_clusters = meaningful_clusters
        else:
            # HDBSCAN이 실패한 경우 모든 데이터를 노이즈로 처리
            noise_data = titles_data.copy()
            noise_embeddings = embeddings.copy()
            print(f"HDBSCAN 실패: 모든 {len(noise_data)}개 데이터를 노이즈로 처리")
        
        # HDBSCAN이 클러스터를 찾지 못한 경우 K-Means로 대체
        if len(hdbscan_clusters) == 0:
            print("\nHDBSCAN이 클러스터를 찾지 못했습니다. K-Means로 대체 클러스터링을 수행합니다.")
            
            # K-Means 클러스터 수 결정
            n_clusters = self.calculate_kmeans_clusters(n_data)
            
            if n_clusters >= 2:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                kmeans_labels = kmeans.fit_predict(embeddings)
                
                # K-Means 결과를 클러스터로 변환
                for label, item in zip(kmeans_labels, titles_data):
                    hdbscan_clusters[label].append(item)
            else:
                print("데이터가 너무 적어 클러스터링을 수행할 수 없습니다.")
        
        # 2단계: 노이즈 데이터를 K-Means로 재분류
        elif len(noise_data) > self.MIN_NOISE_FOR_RECLUSTERING:
            # 노이즈 데이터에 대한 K-Means 클러스터 수 결정
            n_clusters = self.calculate_kmeans_clusters(len(noise_data))
            
            if n_clusters >= 2:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                kmeans_labels = kmeans.fit_predict(noise_embeddings)
                
                # K-Means 결과를 기존 클러스터에 추가
                max_cluster_id = max(hdbscan_clusters.keys()) if hdbscan_clusters else -1
                for label, item in zip(kmeans_labels, noise_data):
                    new_cluster_id = max_cluster_id + 1 + label
                    if new_cluster_id not in hdbscan_clusters:
                        hdbscan_clusters[new_cluster_id] = []
                    hdbscan_clusters[new_cluster_id].append(item)
            else:
                print("노이즈 데이터가 너무 적어 K-Means 재분류 생략")
        else:
            print("노이즈 데이터가 적어 K-Means 재분류 생략")
        
        # 최종 결과
        final_clusters = hdbscan_clusters
        final_noise = []
        
        return final_clusters, final_noise
    
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
            # KeyBERT로 키워드 추출
            keybert_keywords = self.kw_model.extract_keywords(
                text, 
                keyphrase_ngram_range=(1, 3),  # 1-3그램 키워드
                stop_words=list(self.stopwords),  # 불용어 제거
                top_n=top_k,
                use_mmr=True,  # MMR 알고리즘 사용으로 다양성 확보
                diversity=0.5   # 다양성 파라미터
            )
            
            return [kw for kw, score in keybert_keywords]
        except Exception as e:
            print(f"KeyBERT 계산 오류: {e}")
            return []

    def generate_cluster_labels(self, clusters):
        """
        클러스터별 키워드 라벨 생성
        
        TF-IDF와 KeyBERT를 결합하여 각 클러스터의 대표 키워드를 추출합니다.
        
        Args:
            clusters (dict): 클러스터 딕셔너리
            
        Returns:
            dict: 클러스터별 라벨 정보
        """
        cluster_labels = {}
        
        for cluster_id, news_list in clusters.items():
            titles = [item["cleaned_title"] for item in news_list]
            combined_text = " ".join(titles)
            
            # 1단계: TF-IDF 기반 키워드 추출
            tfidf_keywords = self.extract_keywords_with_konlpy_tfidf(
                titles, topn=self.TOP_KEYWORDS_COUNT
            )
            
            # 2단계: KeyBERT 키워드 추출
            keybert_keywords = self.extract_keywords_with_keybert(
                combined_text, top_k=self.KEYBERT_TOP_K
            )
            
            # 3단계: 키워드 결합 및 중복 제거
            combined_keywords = tfidf_keywords.copy()  # TF-IDF 우선
            for kw in keybert_keywords:
                if kw not in combined_keywords:  # 중복 제거
                    combined_keywords.append(kw)
            
            # 4단계: 대분류 카테고리 결정
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
        
        미리 정의된 카테고리와 TF-IDF 키워드를 매칭하여 가장 적합한 카테고리를 선택합니다.
        
        Args:
            keywords (list): 키워드 리스트
            titles (list): 제목 리스트
            
        Returns:
            str: 대분류 카테고리 이름
        """
        # 대분류 카테고리 매핑
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
        
        # TF-IDF 기반 카테고리 점수 계산
        category_scores = {}
        
        # 전체 텍스트에서 TF-IDF 계산
        all_texts = titles + [f"{' '.join(keywords)}"]  # 키워드도 하나의 문서로 취급
        tfidf_keywords = self.extract_keywords_with_konlpy_tfidf(all_texts, topn=self.TFIDF_CATEGORY_KEYWORDS)
        
        # TF-IDF 키워드와 카테고리 매핑 비교
        for category, words in category_mapping.items():
            score = 0
            for word in words:
                if word in tfidf_keywords:
                    # TF-IDF 키워드 리스트에서의 순위에 따른 가중치 적용
                    try:
                        rank = tfidf_keywords.index(word)
                        weight = 1.0 / (rank + 1)  # 순위가 높을수록 높은 가중치
                        score += weight
                    except ValueError:
                        pass
            category_scores[category] = score
        
        # 가장 높은 점수의 카테고리 선택
        if category_scores:
            best_category = max(category_scores, key=category_scores.get)
            if category_scores[best_category] > 0:
                return best_category
        
        # 매칭되는 카테고리가 없으면 첫 번째 TF-IDF 키워드 사용
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
        # 테스트 코드와 동일한 구조로 생성
        frontend_data = {
            "universities": [],
            "clusters": []
        }
        
        # 1. 대학교 데이터 추가
        if univ_news:
            # 최소 개수 이상인 대학교만 필터링하고 뉴스 수가 많은 순으로 정렬하여 최대 개수만 선택
            filtered_univ_news = {uni_name: news_list for uni_name, news_list in univ_news.items() 
                                 if len(news_list) >= self.MIN_UNIV_NEWS_COUNT}
            sorted_univ_news = dict(sorted(filtered_univ_news.items(), key=lambda x: len(x[1]), reverse=True)[:self.MAX_UNIV_DISPLAY])
            
            for uni_name, news_list in sorted_univ_news.items():
                frontend_data["universities"].append({
                    "name": uni_name,
                    "news_count": len(news_list),
                    "news": [{"title": news['cleaned_title'], "link": news["original"].get("link", "")} for news in news_list]
                })
        
        # 2. 클러스터 데이터 추가
        for cluster_id, news_list in clusters.items():
            if len(news_list) >= self.MIN_CLUSTER_NEWS_COUNT:
                cluster_info = cluster_labels.get(cluster_id, {})
                if isinstance(cluster_info, dict):
                    major_category = cluster_info.get("major_category", f"클러스터 {cluster_id}")
                    keywords = cluster_info.get("keywords", [])
                else:
                    major_category = f"클러스터 {cluster_id}"
                    keywords = cluster_info if isinstance(cluster_info, list) else []
                
                # 중분류별 뉴스 분류 (각 뉴스는 하나의 중분류에만 할당)
                minor_categories = []
                minor_category_news = {}
                assigned_news = set()  # 이미 할당된 뉴스 추적
                
                for keyword in keywords[:self.TOP_KEYWORDS_COUNT]:
                    minor_categories.append(keyword)
                    minor_category_news[keyword] = []
                    
                    # 해당 키워드가 제목에 포함되고 아직 할당되지 않은 뉴스만 필터링
                    for news in news_list:
                        news_link = news["original"].get("link", "")
                        if keyword in news['cleaned_title'] and news_link not in assigned_news:
                            minor_category_news[keyword].append(news)
                            assigned_news.add(news_link)  # 할당됨으로 표시
                
                # 중분류별 뉴스 데이터 변환
                minor_categories_data = []
                for minor_cat in minor_categories:
                    minor_news = minor_category_news[minor_cat]
                    if len(minor_news) >= self.MIN_MINOR_NEWS_COUNT:
                        minor_categories_data.append({
                            "name": minor_cat,
                            "news_count": len(minor_news),
                            "news": [{"title": news['cleaned_title'], "link": news["original"].get("link", "")} for news in minor_news]
                        })
                
                frontend_data["clusters"].append({
                    "cluster_id": cluster_id,
                    "major_category": major_category,
                    "news_count": len(news_list),
                    "minor_categories": minor_categories_data
                })
        
        # 프론트엔드가 기대하는 형식으로 변환
        converted_data = []
        
        # 1. 대학교 데이터를 majorKeyword 형식으로 변환
        if frontend_data['universities']:
            univ_middle_keywords = []
            univ_other_news = []
            
            for univ in frontend_data['universities']:
                univ_middle_keywords.append({
                    "middleKeyword": self.normalize_keyword(univ['name']),
                    "relatedNews": univ['news']
                })
                univ_other_news.extend(univ['news'])
            
            converted_data.append({
                "majorKeyword": self.normalize_keyword("대학교"),
                "middleKeywords": univ_middle_keywords,
                "otherNews": univ_other_news
            })
        
        # 2. 클러스터 데이터를 majorKeyword 형식으로 변환
        for cluster in frontend_data['clusters']:
            cluster_middle_keywords = []
            cluster_other_news = []
            
            for minor_cat in cluster['minor_categories']:
                cluster_middle_keywords.append({
                    "middleKeyword": self.normalize_keyword(minor_cat['name']),
                    "relatedNews": minor_cat['news']
                })
                cluster_other_news.extend(minor_cat['news'])
            
            # 뉴스가 있는 경우에만 대분류 추가
            if cluster_middle_keywords or cluster_other_news:
                converted_data.append({
                    "majorKeyword": self.normalize_keyword(cluster['major_category']),
                    "middleKeywords": cluster_middle_keywords,
                    "otherNews": cluster_other_news
                })
        
        return converted_data
    
    def analyze_from_db(self, news_data):
        """
        뉴스 제목 분석 파이프라인
        
        전처리, 대학교 분류, 클러스터링, 키워드 추출을 순차적으로 수행합니다.
        
        Args:
            news_data (list): 원본 뉴스 데이터
            
        Returns:
            list: 프론트엔드 형식의 분석 결과
        """
        # 1단계: 전처리 (특수문자 제거)
        processed_data = self.preprocess_titles(news_data)
        
        if len(processed_data) < self.MIN_NEWS_COUNT:
            return None
        
        # 2단계: 대학교 이름으로 분류
        univ_news, other_news = self.split_news_by_uni_name(processed_data)
        
        # 3단계: 클러스터링 (HDBSCAN + K-Means)
        if other_news:
            clusters, noise_news = self.enhanced_cluster_news(other_news)
        else:
            clusters, noise_news = {}, []
        
        # 4단계: 키워드 추출 (KonlPy + TF-IDF + KeyBERT)
        cluster_labels = self.generate_cluster_labels(clusters)
        
        # 5단계: 마인드맵 데이터 생성
        frontend_data = self.create_frontend_data(univ_news, clusters, cluster_labels, noise_news)
        
        return frontend_data

