#!/usr/bin/env python3
"""
간단한 빈도 기반 클러스터링 모듈

복잡한 TF-IDF, HDBSCAN, K-Means 대신
순수 빈도 기반으로 대분류/중분류를 구하는 간단한 방법
"""

import logging
from collections import Counter, defaultdict
from konlpy.tag import Okt
import re
import os

logger = logging.getLogger(__name__)

class SimpleClusterer:
    """
    간단한 빈도 기반 클러스터링 클래스
    기존 NewsAnalyzer의 대학교 분류 로직을 재사용
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
        # __file__: backend/analysis/simple_clusterer.py
        # dirname(__file__): backend/analysis
        # dirname(dirname(__file__)): backend
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_dir = os.path.join(backend_dir, "config")
        STOPWORDS_PATH = os.path.join(config_dir, "stopwords.txt")
        NON_UNIV_WORD_PATH = os.path.join(config_dir, "non_university_words.txt")
        
        # 불용어와 제외 단어 로드 (기존 코드와 동일)
        self.stopwords = self._load_text_file_as_set(STOPWORDS_PATH, "불용어")
        self.exclude_words = self._load_text_file_as_set(NON_UNIV_WORD_PATH, "제외 단어")
        
        logger.info("✅ SimpleClusterer 초기화 완료 (기존 NewsAnalyzer 로직 재사용)")
    
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
    
    def extract_keywords(self, text, min_length=2):
        """
        텍스트에서 키워드 추출 (명사만) - 기존 NewsAnalyzer 로직 사용
        """
        return self.extract_nouns(text)
    
    def cluster_by_keyword_frequency(self, news_data, min_cluster_size=3):
        """
        키워드 빈도 기반으로 뉴스 클러스터링
        
        Args:
            news_data (list): 뉴스 데이터 리스트
            min_cluster_size (int): 최소 클러스터 크기
            
        Returns:
            dict: 클러스터 결과
        """
        logger.info(f"🔢 키워드 빈도 기반 클러스터링 시작: {len(news_data)}개 뉴스")
        
        # 1단계: 모든 뉴스에서 키워드 추출
        all_keywords = []
        news_keywords = []
        
        for item in news_data:
            title = item.get("title", "")
            keywords = self.extract_keywords(title)
            news_keywords.append(keywords)
            all_keywords.extend(keywords)
        
        logger.info(f"✅ 키워드 추출 완료: 총 {len(all_keywords)}개 키워드")
        
        # 2단계: 키워드 빈도 계산
        keyword_counts = Counter(all_keywords)
        logger.info(f"📊 고유 키워드 수: {len(keyword_counts)}개")
        
        # 3단계: 상위 키워드로 클러스터 생성
        clusters = {}
        cluster_id = 0
        used_news_indices = set()
        
        # 빈도 순으로 정렬된 키워드들
        top_keywords = keyword_counts.most_common(50)  # 상위 50개 키워드만 사용
        
        for keyword, count in top_keywords:
            if count < min_cluster_size:
                break  # 최소 클러스터 크기 미만이면 중단
            
            # 이 키워드를 포함하는 뉴스들 찾기
            cluster_news = []
            for i, keywords in enumerate(news_keywords):
                if i in used_news_indices:
                    continue
                
                if keyword in keywords:
                    cluster_news.append(news_data[i])
                    used_news_indices.add(i)
            
            # 클러스터 크기가 충분한 경우만 저장
            if len(cluster_news) >= min_cluster_size:
                clusters[cluster_id] = {
                    "keyword": keyword,
                    "news": cluster_news,
                    "size": len(cluster_news)
                }
                cluster_id += 1
                logger.info(f"   클러스터 {cluster_id-1}: '{keyword}' ({len(cluster_news)}개 뉴스)")
        
        # 4단계: 사용되지 않은 뉴스들을 '기타' 클러스터로
        unused_news = [news_data[i] for i in range(len(news_data)) if i not in used_news_indices]
        
        if unused_news:
            clusters[cluster_id] = {
                "keyword": "기타",
                "news": unused_news,
                "size": len(unused_news)
            }
            logger.info(f"   클러스터 {cluster_id}: '기타' ({len(unused_news)}개 뉴스)")
        
        logger.info(f"🎉 클러스터링 완료: {len(clusters)}개 클러스터")
        
        return clusters
    
    def create_subcategories(self, cluster_news, max_subcategories=5):
        """
        클러스터 내에서 중분류 생성
        
        Args:
            cluster_news (list): 클러스터 내 뉴스들
            max_subcategories (int): 최대 중분류 개수
            
        Returns:
            list: 중분류 리스트
        """
        if len(cluster_news) < 6:  # 뉴스가 너무 적으면 중분류 생성 안함
            return []
        
        # 클러스터 내 모든 키워드 수집
        all_keywords = []
        for item in cluster_news:
            title = item.get("title", "")
            keywords = self.extract_keywords(title)
            all_keywords.extend(keywords)
        
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
                
                title = item.get("title", "")
                keywords = self.extract_keywords(title)
                
                if keyword in keywords:
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
        뉴스 분석 메인 함수 (기존 NewsAnalyzer 로직 재사용)
        
        Args:
            news_data (list): 뉴스 데이터 리스트
            
        Returns:
            list: 프론트엔드 형식의 분석 결과
        """
        logger.info(f"🚀 간단한 뉴스 분석 시작: {len(news_data)}개 뉴스")
        
        # 1단계: 전처리 (기존 코드와 동일)
        processed_data = self.preprocess_titles(news_data)
        
        if len(processed_data) < 10:  # 최소 뉴스 수 체크
            logger.warning("분석 가능한 뉴스가 부족합니다")
            return None
        
        # 2단계: 대학교 뉴스 분리 (기존 코드와 동일)
        university_news, other_news = self.split_news_by_uni_name(processed_data)
        
        logger.info(f"📊 분류 완료: 대학교 {len(university_news)}개 그룹, 기타 {len(other_news)}개")
        
        # 3단계: 기타 뉴스 클러스터링 (간단한 빈도 기반)
        clusters = self.cluster_by_keyword_frequency(other_news)
        
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


def test_simple_clusterer():
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
    clusterer = SimpleClusterer()
    result = clusterer.analyze_news(news_data)
    
    # 결과 출력
    print("\n" + "=" * 80)
    print("📊 간단한 클러스터링 결과")
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
    test_simple_clusterer()
