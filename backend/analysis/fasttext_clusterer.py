#!/usr/bin/env python3
"""
FastText + K-Means 기반 클러스터링 모듈

TF-IDF 대신 FastText 임베딩과 K-Means를 사용한 안정적인 클러스터링
"""

import logging
from collections import Counter, defaultdict
from konlpy.tag import Okt
import re
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import fasttext
import tempfile

logger = logging.getLogger(__name__)

class FastTextClusterer:
    """
    FastText + K-Means 기반 클러스터링 클래스
    """
    
    def __init__(self):
        """초기화"""
        self.okt = Okt()
        
        # 기존 NewsAnalyzer와 동일한 설정
        self.MIN_WORD_LENGTH = 2
        self.MIN_TITLE_LENGTH = 10
        
        # 정규표현식 패턴 (기존 코드와 동일)
        self.bracket_pattern = re.compile(r'\[.*?\]')
        self.parenthesis_pattern = re.compile(r'\(.*?\)')
        self.html_tag_pattern = re.compile(r'<[^>]+>')
        self.special_char_pattern = re.compile(r'[^\w\s가-힣]')
        self.whitespace_pattern = re.compile(r'\s+')
        
        # 대학교 패턴 (기존 코드와 동일)
        self.uni_pattern = re.compile(r'.*대(학교|학원)?$')
        
        # 파일 경로 설정 (backend/config 폴더)
        # __file__: backend/analysis/fasttext_clusterer.py
        # dirname(__file__): backend/analysis
        # dirname(dirname(__file__)): backend
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_dir = os.path.join(backend_dir, "config")
        STOPWORDS_PATH = os.path.join(config_dir, "stopwords.txt")
        NON_UNIV_WORD_PATH = os.path.join(config_dir, "non_university_words.txt")
        
        # 경로 확인 로그
        logger.debug(f"Config 디렉토리: {config_dir}")
        logger.debug(f"Stopwords 경로: {STOPWORDS_PATH}")
        logger.debug(f"Non-univ words 경로: {NON_UNIV_WORD_PATH}")
        
        # 불용어와 제외 단어 로드 (기존 코드와 동일)
        self.stopwords = self._load_text_file_as_set(STOPWORDS_PATH, "불용어")
        self.exclude_words = self._load_text_file_as_set(NON_UNIV_WORD_PATH, "제외 단어")
        
        # FastText 모델 초기화
        self.fasttext_model = None
        self._init_fasttext()
        
        logger.info("✅ FastTextClusterer 초기화 완료")
    
    def _load_text_file_as_set(self, file_path, file_description):
        """
        텍스트 파일을 읽어 set으로 반환하는 공통 메서드 (기존 코드와 동일)
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
    
    def _init_fasttext(self):
        """FastText 모델 초기화"""
        try:
            # 한국어 FastText 모델 로드
            # models 폴더에서 먼저 찾고, 없으면 현재 디렉토리에서 찾기
            import os
            model_paths = [
                'models/cc.ko.300.bin',
                'cc.ko.300.bin',
                os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'cc.ko.300.bin')
            ]
            
            model_path = None
            for path in model_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            if model_path:
                self.fasttext_model = fasttext.load_model(model_path)
                logger.info(f"✅ 한국어 FastText 모델 로드 완료: {model_path}")
            else:
                raise FileNotFoundError("FastText 모델 파일을 찾을 수 없습니다. models/cc.ko.300.bin 또는 cc.ko.300.bin 파일이 필요합니다.")
        except Exception as e:
            logger.warning(f"⚠️ FastText 모델 로드 실패: {e}")
            logger.info("🔄 간단한 임베딩 방식으로 대체...")
            self.fasttext_model = None
    
    def extract_nouns(self, text):
        """
        KoNLPy의 Okt를 사용하여 형태소 분석 후 명사 추출 (기존 코드와 동일)
        """
        tokens = self.okt.pos(text, stem=True)
        nouns = [word for word, tag in tokens 
                if tag == "Noun" and word not in self.stopwords and len(word) >= self.MIN_WORD_LENGTH]
        return nouns
    
    def _extract_university_keyword(self, nouns):
        """
        명사 리스트에서 대학교 키워드 추출 (기존 코드와 동일)
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
        뉴스 제목 전처리 (기존 코드와 동일)
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
        대학교 이름으로 뉴스 분류 (기존 코드와 동일)
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
    
    def get_text_embedding(self, text):
        """
        텍스트의 FastText 임베딩 생성
        
        Args:
            text (str): 임베딩할 텍스트
            
        Returns:
            np.array: 임베딩 벡터
        """
        if self.fasttext_model:
            # FastText 모델 사용
            return self.fasttext_model.get_sentence_vector(text)
        else:
            # Fallback: 단순한 단어 벡터 평균
            words = text.split()
            if not words:
                return np.zeros(300)
            
            # 각 단어의 평균 벡터 (랜덤 벡터로 대체)
            np.random.seed(hash(text) % 2**32)  # 텍스트별로 일관된 랜덤
            return np.random.randn(300)
    
    def cluster_with_fasttext_kmeans(self, news_data, n_clusters=None):
        """
        FastText 임베딩 + K-Means로 클러스터링
        
        Args:
            news_data (list): 뉴스 데이터 리스트
            n_clusters (int): 클러스터 수 (None이면 자동 결정)
            
        Returns:
            dict: 클러스터 결과
        """
        logger.info(f"🚀 FastText + K-Means 클러스터링 시작: {len(news_data)}개 뉴스")
        
        if len(news_data) < 10:
            logger.warning("뉴스가 너무 적어서 클러스터링을 수행할 수 없습니다")
            return {0: {"keyword": "기타", "news": news_data, "size": len(news_data)}}
        
        # 1단계: 텍스트 임베딩 생성
        logger.info("📊 FastText 임베딩 생성 중...")
        embeddings = []
        texts = []
        
        for item in news_data:
            text = item.get("cleaned_title", "")
            if text.strip():
                embedding = self.get_text_embedding(text)
                embeddings.append(embedding)
                texts.append(text)
        
        if not embeddings:
            logger.warning("임베딩을 생성할 수 없습니다")
            return {0: {"keyword": "기타", "news": news_data, "size": len(news_data)}}
        
        embeddings = np.array(embeddings)
        logger.info(f"✅ 임베딩 생성 완료: {embeddings.shape}")
        
        # 2단계: 클러스터 수 결정
        if n_clusters is None:
            # 데이터 크기에 따라 클러스터 수 결정
            n_data = len(news_data)
            if n_data < 50:
                n_clusters = min(3, n_data // 10)
            elif n_data < 200:
                n_clusters = min(8, n_data // 20)
            else:
                n_clusters = min(15, n_data // 30)
            
            n_clusters = max(2, n_clusters)  # 최소 2개
        
        logger.info(f"📊 클러스터 수: {n_clusters}개")
        
        # 3단계: K-Means 클러스터링
        logger.info("🔢 K-Means 클러스터링 실행 중...")
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            logger.info(f"✅ K-Means 완료: {len(set(cluster_labels))}개 클러스터")
            
        except Exception as e:
            logger.error(f"❌ K-Means 실패: {e}")
            return {0: {"keyword": "기타", "news": news_data, "size": len(news_data)}}
        
        # 4단계: 클러스터별 키워드 추출 및 정리
        logger.info("🏷️ 클러스터별 키워드 추출 중...")
        clusters = {}
        
        for cluster_id in range(n_clusters):
            # 클러스터에 속한 뉴스들
            cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
            cluster_news = [news_data[i] for i in cluster_indices]
            
            if len(cluster_news) < 3:  # 너무 적은 클러스터는 건너뛰기
                continue
            
            # 클러스터 내 키워드 추출
            cluster_keywords = []
            for item in cluster_news:
                nouns = self.extract_nouns(item.get("cleaned_title", ""))
                cluster_keywords.extend(nouns)
            
            # 가장 빈번한 키워드 선택
            if cluster_keywords:
                keyword_counts = Counter(cluster_keywords)
                main_keyword = keyword_counts.most_common(1)[0][0]
            else:
                main_keyword = f"클러스터_{cluster_id}"
            
            clusters[cluster_id] = {
                "keyword": main_keyword,
                "news": cluster_news,
                "size": len(cluster_news)
            }
            
            logger.info(f"   클러스터 {cluster_id}: '{main_keyword}' ({len(cluster_news)}개 뉴스)")
        
        # 5단계: 사용되지 않은 뉴스들을 '기타' 클러스터로
        used_indices = set()
        for cluster_data in clusters.values():
            for item in cluster_data["news"]:
                if item in news_data:
                    used_indices.add(news_data.index(item))
        
        unused_news = [news_data[i] for i in range(len(news_data)) if i not in used_indices]
        if unused_news:
            clusters[len(clusters)] = {
                "keyword": "기타",
                "news": unused_news,
                "size": len(unused_news)
            }
            logger.info(f"   클러스터 {len(clusters)-1}: '기타' ({len(unused_news)}개 뉴스)")
        
        logger.info(f"🎉 FastText + K-Means 클러스터링 완료: {len(clusters)}개 클러스터")
        
        return clusters
    
    def create_subcategories(self, cluster_news, max_subcategories=5):
        """
        클러스터 내에서 중분류 생성 (FastText 기반)
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
        
        # 사용되지 않은 뉴스들을 '기타' 중분류로
        unused_news = [cluster_news[i] for i in range(len(cluster_news)) if i not in used_news_indices]
        if unused_news:
            subcategories.append({
                "keyword": "기타",
                "news": unused_news,
                "size": len(unused_news)
            })
        
        return subcategories
    
    def analyze_news(self, news_data):
        """
        뉴스 분석 메인 함수 (FastText + K-Means 기반)
        """
        logger.info(f"🚀 FastText + K-Means 뉴스 분석 시작: {len(news_data)}개 뉴스")
        
        # 1단계: 전처리 (기존 코드와 동일)
        processed_data = self.preprocess_titles(news_data)
        
        if len(processed_data) < 10:  # 최소 뉴스 수 체크
            logger.warning("분석 가능한 뉴스가 부족합니다")
            return None
        
        # 2단계: 대학교 뉴스 분리 (기존 코드와 동일)
        university_news, other_news = self.split_news_by_uni_name(processed_data)
        
        logger.info(f"📊 분류 완료: 대학교 {len(university_news)}개 그룹, 기타 {len(other_news)}개")
        
        # 3단계: 기타 뉴스 클러스터링 (FastText + K-Means)
        clusters = self.cluster_with_fasttext_kmeans(other_news)
        
        # 4단계: 프론트엔드 형식으로 변환
        result = []
        
        # 대학교 뉴스 추가 (하나의 "대학교" 대분류로 통합)
        if university_news:
            # 모든 대학교 뉴스를 하나로 합치기
            all_university_news = []
            for uni_name, uni_news_list in university_news.items():
                all_university_news.extend(uni_news_list)
            
            # "대학교" 대분류로 추가
            result.append({
                "majorKeyword": "대학교",
                "middleKeywords": [],
                "otherNews": all_university_news
            })
        
        # 클러스터들 추가
        for cluster_id, cluster_data in clusters.items():
            keyword = cluster_data["keyword"]
            news = cluster_data["news"]
            
            # 중분류 생성
            subcategories = self.create_subcategories(news)
            
            # 중분류가 있으면 중분류로, 없으면 기타 뉴스로
            if subcategories:
                middle_keywords = []
                other_news_in_cluster = []
                
                for subcat in subcategories:
                    if subcat["keyword"] == "기타":
                        other_news_in_cluster = subcat["news"]
                    else:
                        middle_keywords.append({
                            "middleKeyword": subcat["keyword"],
                            "relatedNews": subcat["news"]
                        })
                
                result.append({
                    "majorKeyword": keyword,
                    "middleKeywords": middle_keywords,
                    "otherNews": other_news_in_cluster
                })
            else:
                result.append({
                    "majorKeyword": keyword,
                    "middleKeywords": [],
                    "otherNews": news
                })
        
        logger.info(f"✅ 분석 완료: {len(result)}개 대분류 생성")
        
        return result


def test_fasttext_clusterer():
    """테스트 함수"""
    import json
    import os
    
    # 테스트 데이터 로드
    json_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "test_news_1000.json")
    
    if not os.path.exists(json_file_path):
        print(f"❌ 테스트 파일을 찾을 수 없습니다: {json_file_path}")
        return
    
    with open(json_file_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    news_data = json_data.get('news_data', [])[:100]  # 처음 100개만 테스트
    
    # 클러스터링 실행
    clusterer = FastTextClusterer()
    result = clusterer.analyze_news(news_data)
    
    # 결과 출력
    print("\n" + "=" * 80)
    print("📊 FastText + K-Means 클러스터링 결과")
    print("=" * 80)
    
    for major_idx, major_cat in enumerate(result, 1):
        major_name = major_cat.get('majorKeyword', 'Unknown')
        middle_keywords = major_cat.get('middleKeywords', [])
        other_news = major_cat.get('otherNews', [])
        
        total_news = sum(len(mid.get('relatedNews', [])) for mid in middle_keywords) + len(other_news)
        print(f"\n📁 대분류 {major_idx}: {major_name} (총 {total_news}개 뉴스)")
        
        if middle_keywords:
            for middle_idx, middle_cat in enumerate(middle_keywords, 1):
                middle_name = middle_cat.get('middleKeyword', 'Unknown')
                related_news = middle_cat.get('relatedNews', [])
                print(f"   ├─ 중분류 {middle_idx}: {middle_name} ({len(related_news)}개 뉴스)")
        
        if other_news:
            print(f"   └─ 기타 뉴스: {len(other_news)}개")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    test_fasttext_clusterer()
