import os
import re
import json
from collections import defaultdict, Counter
import numpy as np
from sentence_transformers import SentenceTransformer
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from konlpy.tag import Okt
from keybert import KeyBERT
from config.config import STOPWORDS_PATH, NON_UNIV_WORD_PATH

class NewsAnalyzer:
    def __init__(self):
        # 한글 지원 멀티링구얼 모델
        self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.kw_model = KeyBERT(model='paraphrase-multilingual-MiniLM-L12-v2')
        self.okt = Okt()
        self.stopwords = self.load_stopwords()
        
    def load_stopwords(self):
        """불용어 로드"""
        try:
            with open(STOPWORDS_PATH, "r", encoding="utf-8") as f:
                return set([line.strip() for line in f if line.strip()])
        except:
            return set()
    
    def load_exclude_words(self):
        """대학교가 아닌 제외 단어 로드"""
        try:
            with open(NON_UNIV_WORD_PATH, "r", encoding="utf-8") as f:
                return set([line.strip() for line in f if line.strip()])
        except:
            return set()
    
    def split_news_by_uni_name(self, processed_data):
        """대학교 이름으로 뉴스 분류"""
        
        univ_news = defaultdict(list)
        other_news = []
        
        # '대'로 끝나는 단어 정규표현식
        uni_pattern = re.compile(r".+대$")
        exclude_words = self.load_exclude_words()
        
        for i, item in enumerate(processed_data):
            title = item["cleaned_title"]
            
            # KoNLPy로 명사 추출
            tokens = self.okt.pos(title, stem=True)
            nouns = [word for word, tag in tokens 
                    if tag == "Noun" and word not in self.stopwords and len(word) > 1]
            
            # 대학교 이름 찾기
            uni_kw = next((kw for kw in nouns if uni_pattern.match(kw) and kw not in exclude_words), None)
            
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
                # if i < 20:  # 20개씩 출력
                #     print(f"{i+1:2d}. 대학: {uni_kw} - {title}")
            else:
                other_news.append(news_info)
                # if i < 20:  # 20개씩 출력
                #     print(f"{i+1:2d}. 기타: {title}")
        
        # print(f"\n대학교 분류 완료:")
        # print(f"- 대학교 뉴스: {len(univ_news)}개 대학, {sum(len(news_list) for news_list in univ_news.values())}개 뉴스")
        # print(f"- 기타 뉴스: {len(other_news)}개")
        
        # 대학교별 뉴스 수 출력
        # for uni_name, news_list in sorted(univ_news.items(), key=lambda x: len(x[1]), reverse=True):
        #     print(f"  {uni_name}: {len(news_list)}개")
        
        return univ_news, other_news
    
    def preprocess_titles(self, news_data):
        """전처리: 특수문자 제거"""
        # print("뉴스 제목 전처리 중...")
        
        # 특수문자 제거 및 기본 정제
        processed_titles = []
        # print(f"\n특수문자 제거 (총 {len(news_data)}개)")
        
        for i, item in enumerate(news_data):
            title = item["title"]
            original_title = title
            
            # 특수문자 및 괄호 내용 제거
            title = re.sub(r'\[.*?\]', '', title)  # 대괄호 내용 제거
            title = re.sub(r'\(.*?\)', '', title)  # 소괄호 내용 제거
            title = re.sub(r'<.*?>', '', title)    # HTML 태그 제거
            title = re.sub(r'[^\w\s가-힣]', ' ', title)  # 특수문자를 공백으로
            title = re.sub(r'\s+', ' ', title).strip()  # 연속 공백 정리
            
            # 20개씩 출력
            # if i < 20:
            #     print(f"{i+1:2d}. 원본: {original_title}")
            #     print(f"    정제: {title}")
            #     print()
            
            if len(title) > 10:
                processed_titles.append({
                    "original": item,
                    "cleaned_title": title
                })
        
        # print(f"특수문자 제거 완료: {len(news_data)} → {len(processed_titles)}개")
        
        # 중복 뉴스 제거
        unique_titles = []
        seen_titles = set()
        duplicate_count = 0
        
        for item in processed_titles:
            title = item["cleaned_title"]
            if title not in seen_titles:
                seen_titles.add(title)
                unique_titles.append(item)
            else:
                duplicate_count += 1
        return unique_titles
    
    
    def extract_nouns_with_konlpy(self, processed_data):
        """KoNLPy로 명사 추출"""
        
        noun_extracted_data = []
        
        for i, item in enumerate(processed_data):
            title = item["cleaned_title"]
            
            # KoNLPy Okt로 형태소 분석
            tokens = self.okt.pos(title, stem=True)
            
            # 명사만 추출 (불용어 제외, 1글자 이상)
            nouns = [word for word, tag in tokens 
                    if tag == "Noun" and word not in self.stopwords and len(word) > 1]
            
            # 결과 저장
            item_with_nouns = {
                "original": item["original"],
                "cleaned_title": title,
                "nouns": nouns,
                "noun_text": " ".join(nouns)
            }
            noun_extracted_data.append(item_with_nouns)
            
            # 20개씩 출력
            # if i < 20:
            #     print(f"{i+1:2d}. 제목: {title}")
            #     print(f"    형태소: {tokens}")
            #     print(f"    명사: {nouns}")
            #     print()
        
        # print(f"명사 추출 완료: {len(processed_data)}개")
        return noun_extracted_data

    def extract_keywords_with_konlpy_tfidf(self, texts, topn=5, show_details=False):
        """KonlPy + TF-IDF 키워드 추출 (이미지 요구사항)"""
        
        # 1단계: KonlPy로 명사 추출
        noun_texts = []
        for i, text in enumerate(texts):
            # KoNLPy Okt로 형태소 분석
            tokens = self.okt.pos(text, stem=True)
            
            # 명사만 추출 (불용어 제외, 1글자 이상)
            nouns = [word for word, tag in tokens 
                    if tag == "Noun" and word not in self.stopwords and len(word) > 1]
            
            noun_texts.append(" ".join(nouns))
            
            # 상세 출력
            # if show_details and i < 10:
            #     print(f"{i+1:2d}. 원본: {text}")
            #     print(f"    형태소: {tokens}")
            #     print(f"    명사: {nouns}")
            #     print()
        
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
            
            # TF-IDF 점수와 함께 출력
            # print(f"\nTF-IDF 상위 {topn}개 키워드:")
            # for i, idx in enumerate(top_indices):
            #     print(f"{i+1:2d}. {feature_names[idx]} (TF-IDF: {mean_scores[idx]:.4f})")
            
            return [feature_names[i] for i in top_indices]
        except Exception as e:
            print(f"TF-IDF 계산 오류: {e}")
            return []
    
    def enhanced_cluster_news(self, titles_data, use_hdbscan=False, use_hybrid=True):
        """HDBSCAN + K-Means"""
        
        titles = [item["cleaned_title"] for item in titles_data]
        # print(f"클러스터링 대상: {len(titles)}개 뉴스")
        
        # 1단계: HDBSCAN으로 밀도 기반 클러스터링 
        # HDBSCAN 최적화된 파라미터 설정
        n_data = len(titles_data)
        if n_data < 20:
            min_cluster_size = 2
            min_samples = 1
        elif n_data < 50:
            min_cluster_size = 2
            min_samples = 1
        elif n_data < 100:
            min_cluster_size = max(2, n_data // 20)  # 더 작은 클러스터 허용
            min_samples = 1  # 항상 1로 고정 (핵심!)
        else:
            min_cluster_size = max(2, n_data // 30)  # 더 작은 클러스터 허용
            min_samples = 1  # 항상 1로 고정 (핵심!)
        
        # print(f"  - 최소 클러스터 크기: {min_cluster_size}")
        # print(f"  - 최소 샘플 수: {min_samples}")
        
        # 임베딩 생성(정규화 OK)
        # print("임베딩 생성 중...")
        embeddings = self.embedding_model.encode(titles, normalize_embeddings=True)
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=2,
            min_samples=1,
            metric="euclidean",
            cluster_selection_method='leaf',
            cluster_selection_epsilon=0.05,
            prediction_data=True
        )
        hdbscan_labels = clusterer.fit_predict(embeddings)
        n_clusters = len(set(hdbscan_labels) - {-1})
        n_noise = list(hdbscan_labels).count(-1)

        if hdbscan_labels is None:
            print("HDBSCAN: 유의미한 클러스터를 찾지 못했습니다(전부 노이즈).")
            # 아래 그래프 폴백으로 진행
        else:
            pass
        
        # HDBSCAN 결과 분석
        hdbscan_clusters = defaultdict(list)
        noise_data = []
        noise_embeddings = []
        
        if hdbscan_labels is not None:
            # 레이블을 이용해 클러스터 딕셔너리 구성
            for i, (label, item, embedding) in enumerate(zip(hdbscan_labels, titles_data, embeddings)):
                if label == -1:
                    # 노이즈 데이터 수집
                    noise_data.append(item)
                    noise_embeddings.append(embedding)
                else:
                    hdbscan_clusters[label].append(item)
            
            # print(f"HDBSCAN 결과: {len(hdbscan_clusters)}개 클러스터, 노이즈 {len(noise_data)}개")
            
            # 의미있는 클러스터만 추출 (크기 4개 이상, 중복 뉴스가 아닌 클러스터)
            meaningful_clusters = {}
            meaningless_clusters = []
            
            for cluster_id, cluster_news in hdbscan_clusters.items():
                # 클러스터 크기가 4개 이상인지 확인
                if len(cluster_news) >= 4:
                    # 중복 뉴스가 있는지 확인
                    titles = [news['cleaned_title'] for news in cluster_news]
                    unique_titles = set(titles)
                    
                    # 중복 비율이 50% 미만인 경우만 의미있는 클러스터로 간주
                    duplicate_ratio = (len(titles) - len(unique_titles)) / len(titles)
                    if duplicate_ratio < 0.5:
                        meaningful_clusters[cluster_id] = cluster_news
                    else:
                        meaningless_clusters.append((cluster_id, cluster_news, f"중복 비율 높음 ({duplicate_ratio:.1%})"))
                else:
                    meaningless_clusters.append((cluster_id, cluster_news, f"크기 부족 ({len(cluster_news)}개)"))
            
            # print(f"뉴스 3개 이상 클러스터: {len(meaningful_clusters)}개")
            # print(f"그 외 클러스터: {len(meaningless_clusters)}개")
            
            # 의미없는 클러스터들을 노이즈로 이동
            for cluster_id, cluster_news, reason in meaningless_clusters:
                noise_data.extend(cluster_news)
                # 해당 클러스터의 임베딩도 노이즈로 이동
                for item in cluster_news:
                    item_idx = titles_data.index(item)
                    noise_embeddings.append(embeddings[item_idx])
            
            # 의미있는 클러스터만 남김
            hdbscan_clusters = meaningful_clusters
            
            # HDBSCAN 클러스터별 결과 출력 (의미있는 클러스터만)
            # print(f"\n=== HDBSCAN 클러스터링 결과 ===")
            # for cluster_id, cluster_news in sorted(hdbscan_clusters.items()):
            #     print(f"\n--- 클러스터 {cluster_id} ({len(cluster_news)}개 뉴스) ---")
            #     for i, news in enumerate(cluster_news[:5]):  # 상위 5개만 출력
            #         print(f"  {i+1}. {news['cleaned_title']}")
            #     if len(cluster_news) > 5:
            #         print(f"  ... 그 외 {len(cluster_news) - 5}개")
        else:
            # HDBSCAN이 실패한 경우 모든 데이터를 노이즈로 처리
            noise_data = titles_data.copy()
            noise_embeddings = embeddings.copy()
            print(f"HDBSCAN 실패: 모든 {len(noise_data)}개 데이터를 노이즈로 처리")
        
        # HDBSCAN이 클러스터를 찾지 못한 경우 K-Means로 대체
        if len(hdbscan_clusters) == 0:
            print("\nHDBSCAN이 클러스터를 찾지 못했습니다. K-Means로 대체 클러스터링을 수행합니다.")
            
            # K-Means 클러스터 수 결정
            if n_data < 20:
                n_clusters = min(3, n_data // 5)
            elif n_data < 100:
                n_clusters = min(8, n_data // 10)
            else:
                n_clusters = min(15, n_data // 15)
            
            # print(f"K-Means 클러스터 수: {n_clusters}")
            
            if n_clusters >= 2:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                kmeans_labels = kmeans.fit_predict(embeddings)
                
                # K-Means 결과를 클러스터로 변환
                for i, (label, item) in enumerate(zip(kmeans_labels, titles_data)):
                    hdbscan_clusters[label].append(item)
                
                # print(f"K-Means 클러스터링 완료: {n_clusters}개 클러스터 생성")
            else:
                print("데이터가 너무 적어 클러스터링을 수행할 수 없습니다.")
        
        # 2단계: 노이즈 데이터를 K-Means로 재분류
        elif len(noise_data) > 5:
            # print(f"\n2단계: 노이즈 {len(noise_data)}개를 K-Means로 재분류...")
            
            # 노이즈 데이터에 대한 K-Means 클러스터 수 결정
            n_noise = len(noise_data)
            if n_noise > 50:
                n_clusters = min(5, n_noise // 10)
            elif n_noise > 20:
                n_clusters = min(3, n_noise // 7)
            else:
                n_clusters = min(2, n_noise // 5)
            
            # print(f"  - K-Means 클러스터 수: {n_clusters}")
            
            if n_clusters >= 2:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                kmeans_labels = kmeans.fit_predict(noise_embeddings)
                
                # K-Means 결과를 기존 클러스터에 추가
                max_cluster_id = max(hdbscan_clusters.keys()) if hdbscan_clusters else -1
                for i, (label, item) in enumerate(zip(kmeans_labels, noise_data)):
                    new_cluster_id = max_cluster_id + 1 + label
                    if new_cluster_id not in hdbscan_clusters:
                        hdbscan_clusters[new_cluster_id] = []
                    hdbscan_clusters[new_cluster_id].append(item)
                
                # print(f"K-Means 재분류 완료: {n_clusters}개 추가 클러스터 생성")
            else:
                print("노이즈 데이터가 너무 적어 K-Means 재분류 생략")
        else:
            print("노이즈 데이터가 적어 K-Means 재분류 생략")
        
        # 최종 결과
        final_clusters = hdbscan_clusters
        final_noise = []
        
        # print(f"총 클러스터 수: {len(final_clusters)}개")
        # print(f"최종 노이즈: {len(final_noise)}개")
        
        # 클러스터별 뉴스 수 출력
        # for cluster_id, news_list in sorted(final_clusters.items(), key=lambda x: len(x[1]), reverse=True):
        #     print(f"  클러스터 {cluster_id}: {len(news_list)}개 뉴스")
        
        return final_clusters, final_noise
    
    def extract_keywords_with_keybert(self, text, top_k=3, show_details=False):
        """KeyBERT 키워드 추출 (이미지 요구사항)"""
        
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
            
            # if show_details:
            #     print(f"KeyBERT 상위 {top_k}개 키워드:")
            #     for i, (keyword, score) in enumerate(keybert_keywords):
            #         print(f"{i+1:2d}. {keyword} (점수: {score:.4f})")
            
            return [kw for kw, score in keybert_keywords]
        except Exception as e:
            print(f"KeyBERT 계산 오류: {e}")
            return []

    def generate_cluster_labels(self, clusters):
        """클러스터별 키워드 라벨 생성 (이미 추출된 명사 사용 + KeyBERT)"""
        cluster_labels = {}
        cluster_count = 0
        
        for cluster_id, news_list in clusters.items():
            cluster_count += 1
            titles = [item["cleaned_title"] for item in news_list]
            combined_text = " ".join(titles)
            
            # 1단계: TF-IDF 기반 키워드 추출
            tfidf_keywords = self.extract_keywords_with_konlpy_tfidf(
                titles, topn=5, show_details=False
            )
            
            # 2단계: KeyBERT 키워드 추출
            keybert_keywords = self.extract_keywords_with_keybert(
                combined_text, top_k=3, show_details=False
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
                "keywords": combined_keywords[:5],  # 상위 5개만
                "tfidf_keywords": tfidf_keywords,
                "keybert_keywords": keybert_keywords
            }
        
        return cluster_labels
    
    def determine_major_category(self, keywords, titles):
        """키워드와 제목을 분석해서 대분류 단어 결정 (TF-IDF 기반)"""
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
        tfidf_keywords = self.extract_keywords_with_konlpy_tfidf(all_texts, topn=20, show_details=False)
        
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
    
    
    def create_mindmap_data(self, clusters, cluster_labels, noise_news):
        """마인드맵용 그래프 데이터 생성"""
        print("마인드맵 데이터 생성 중...")
        
        nodes = []
        links = []
        node_id_counter = 0
        
        def add_node(node_type, label, size=20, url=None, cluster_id=None):
            nonlocal node_id_counter
            node = {
                "id": f"{node_type}_{node_id_counter}",
                "type": node_type,
                "label": label,
                "size": size,
                "level": 0 if node_type == "cluster" else 1 if node_type == "keyword" else 2
            }
            if url:
                node["url"] = url
            if cluster_id is not None:
                node["cluster_id"] = cluster_id
            nodes.append(node)
            node_id_counter += 1
            return node["id"]
        
        # 1. 클러스터 노드 추가
        cluster_nodes = {}
        for cluster_id, news_list in clusters.items():
            keywords = cluster_labels.get(cluster_id, [])
            cluster_label = " / ".join(keywords[:3]) if keywords else f"클러스터 {cluster_id}"
            
            cluster_node_id = add_node(
                "cluster", 
                cluster_label, 
                size=30 + len(news_list) * 2,  # 뉴스 수에 비례한 크기
                cluster_id=cluster_id
            )
            cluster_nodes[cluster_id] = cluster_node_id
        
        # 2. 키워드 노드 추가
        keyword_nodes = {}
        for cluster_id, keywords in cluster_labels.items():
            if cluster_id not in cluster_nodes:
                continue
                
            cluster_node_id = cluster_nodes[cluster_id]
            for keyword in keywords[:3]:  # 상위 3개 키워드만
                keyword_node_id = add_node("keyword", keyword, size=15)
                keyword_nodes[f"{cluster_id}_{keyword}"] = keyword_node_id
                
                # 클러스터 → 키워드 연결
                links.append({
                    "source": cluster_node_id,
                    "target": keyword_node_id
                })
        
        # 3. 뉴스 제목 노드 추가
        for cluster_id, news_list in clusters.items():
            if cluster_id not in cluster_nodes:
                continue
                
            cluster_node_id = cluster_nodes[cluster_id]
            for news in news_list:
                title = news["cleaned_title"][:50] + "..." if len(news["cleaned_title"]) > 50 else news["cleaned_title"]
                title_node_id = add_node(
                    "title", 
                    title, 
                    size=8, 
                    url=news["original"].get("link", "")
                )
                
                # 클러스터 → 제목 연결
                links.append({
                    "source": cluster_node_id,
                    "target": title_node_id
                })
        
        # 4. 노이즈 뉴스 처리 (기타 카테고리)
        if noise_news:
            noise_cluster_id = add_node("cluster", "기타 뉴스", size=20 + len(noise_news) * 2)
            for news in noise_news:
                title = news["cleaned_title"][:50] + "..." if len(news["cleaned_title"]) > 50 else news["cleaned_title"]
                title_node_id = add_node(
                    "title", 
                    title, 
                    size=8, 
                    url=news["original"].get("link", "")
                )
                links.append({
                    "source": noise_cluster_id,
                    "target": title_node_id
                })
        
        return {
            "nodes": nodes,
            "links": links,
            "clusters": len(clusters),
            "total_news": sum(len(news_list) for news_list in clusters.values()) + len(noise_news)
        }
    
    def create_mindmap_data_with_university(self, univ_news, clusters, cluster_labels, noise_news):
        """대학교와 클러스터를 포함한 마인드맵용 그래프 데이터 생성"""
        print("대학교 + 클러스터 마인드맵 데이터 생성 중...")
        
        nodes = []
        links = []
        node_id_counter = 0
        
        def add_node(node_type, label, size=20, url=None, cluster_id=None, university_name=None):
            nonlocal node_id_counter
            node = {
                "id": f"{node_type}_{node_id_counter}",
                "type": node_type,
                "label": label,
                "size": size,
                "level": 0 if node_type == "major" else 1 if node_type == "middle" else 2
            }
            if url:
                node["url"] = url
            if cluster_id is not None:
                node["cluster_id"] = cluster_id
            if university_name is not None:
                node["university_name"] = university_name
            nodes.append(node)
            node_id_counter += 1
            return node["id"]
        
        # 1. 대학교 대분류 노드 추가
        if univ_news:
            # 대학교별 키워드 추출
            university_keywords = []
            for uni_name, news_list in univ_news.items():
                if len(news_list) >= 2:  # 2개 이상인 경우만 키워드 추출
                    titles = [news["cleaned_title"] for news in news_list]
                    tfidf_keywords = self.extract_keywords_with_kiwi(titles, topn=3, show_details=False)
                    university_keywords.extend(tfidf_keywords)
            
            # 대학교 이름들을 키워드로 추가
            university_names = list(univ_news.keys())
            all_university_keywords = list(set(university_keywords + university_names))
            
            major_university_id = add_node("major", "대학", size=50)
            
            # 대학교별 중분류 노드 추가
            for uni_name, news_list in univ_news.items():
                uni_node_id = add_node(
                    "middle", 
                    uni_name, 
                    size=30 + len(news_list) * 2,
                    university_name=uni_name
                )
                
                # 대분류 → 중분류 연결
                links.append({
                    "source": major_university_id,
                    "target": uni_node_id
                })
                
                # 중분류 → 뉴스 제목 연결
                for news in news_list:
                    title = news["cleaned_title"][:50] + "..." if len(news["cleaned_title"]) > 50 else news["cleaned_title"]
                    title_node_id = add_node(
                        "title", 
                        title, 
                        size=8, 
                        url=news["original"].get("link", "")
                    )
                    links.append({
                        "source": uni_node_id,
                        "target": title_node_id
                    })
        
        # 2. 클러스터 대분류 노드 추가
        for cluster_id, news_list in clusters.items():
            cluster_info = cluster_labels.get(cluster_id, {})
            if isinstance(cluster_info, dict):
                major_category = cluster_info.get("major_category", f"클러스터 {cluster_id}")
                keywords = cluster_info.get("keywords", [])
            else:
                # 기존 형식 호환성
                major_category = f"클러스터 {cluster_id}"
                keywords = cluster_info if isinstance(cluster_info, list) else []
            
            cluster_label = f"{major_category} ({len(news_list)}개)"
            
            cluster_node_id = add_node(
                "major", 
                cluster_label, 
                size=30 + len(news_list) * 2, 
                cluster_id=cluster_id
            )
            
            # 클러스터 → 뉴스 제목 연결
            for news in news_list:
                title = news["cleaned_title"][:50] + "..." if len(news["cleaned_title"]) > 50 else news["cleaned_title"]
                title_node_id = add_node(
                    "title", 
                    title, 
                    size=8, 
                    url=news["original"].get("link", "")
                )
                links.append({
                    "source": cluster_node_id,
                    "target": title_node_id
                })
        
        # 3. 노이즈 뉴스 처리 (기타 카테고리)
        if noise_news:
            noise_cluster_id = add_node("major", "기타 뉴스", size=20 + len(noise_news) * 2)
            for news in noise_news:
                title = news["cleaned_title"][:50] + "..." if len(news["cleaned_title"]) > 50 else news["cleaned_title"]
                title_node_id = add_node(
                    "title", 
                    title, 
                    size=8, 
                    url=news["original"].get("link", "")
                )
                links.append({
                    "source": noise_cluster_id,
                    "target": title_node_id
                })
        
        total_univ_news = sum(len(news_list) for news_list in univ_news.values())
        total_cluster_news = sum(len(news_list) for news_list in clusters.values())
        
        return {
            "nodes": nodes,
            "links": links,
            "universities": len(univ_news),
            "clusters": len(clusters),
            "total_news": total_univ_news + total_cluster_news + len(noise_news)
        }
    
    def normalize_keyword(self, keyword):
        """키워드의 공백을 하이픈으로 변경 (노드 ID 매칭용)"""
        return keyword.replace(" ", "-")
    
    def create_frontend_data(self, univ_news, clusters, cluster_labels, noise_news):
        """프론트엔드용 데이터 구조 생성 (테스트 코드와 동일한 구조)"""
        # 테스트 코드와 동일한 구조로 생성
        frontend_data = {
            "universities": [],
            "clusters": []
        }
        
        # 1. 대학교 데이터 추가 (3개 이상, 상위 5개만)
        if univ_news:
            # 3개 이상인 대학교만 필터링하고 뉴스 수가 많은 순으로 정렬하여 최대 5개만 선택
            filtered_univ_news = {uni_name: news_list for uni_name, news_list in univ_news.items() if len(news_list) >= 3}
            sorted_univ_news = dict(sorted(filtered_univ_news.items(), key=lambda x: len(x[1]), reverse=True)[:5])
            
            for uni_name, news_list in sorted_univ_news.items():
                frontend_data["universities"].append({
                    "name": uni_name,
                    "news_count": int(len(news_list)),
                    "news": [{"title": news['cleaned_title'], "link": news["original"].get("link", "")} for news in news_list]
                })
        
        # 2. 클러스터 데이터 추가 (4개 이상 뉴스만)
        for cluster_id, news_list in clusters.items():
            if len(news_list) >= 4:  # 4개 이상 뉴스만 포함
                cluster_info = cluster_labels.get(cluster_id, {})
                if isinstance(cluster_info, dict):
                    major_category = cluster_info.get("major_category", f"클러스터 {cluster_id}")
                    keywords = cluster_info.get("keywords", [])
                else:
                    major_category = f"클러스터 {cluster_id}"
                    keywords = cluster_info if isinstance(cluster_info, list) else []
                
                # 중분류별 뉴스 분류 (상위 5개 키워드)
                minor_categories = []
                minor_category_news = {}
                
                for keyword in keywords[:5]:  # 상위 5개 키워드
                    minor_categories.append(keyword)
                    minor_category_news[keyword] = []
                    
                    # 해당 키워드가 제목에 포함된 뉴스만 필터링
                    for news in news_list:
                        if keyword in news['cleaned_title']:
                            minor_category_news[keyword].append(news)
                
                # 중분류별 뉴스 데이터 변환 (2개 이상인 경우만)
                minor_categories_data = []
                for minor_cat in minor_categories:
                    minor_news = minor_category_news[minor_cat]
                    if len(minor_news) >= 2:  # 2개 이상인 경우만
                        minor_categories_data.append({
                            "name": minor_cat,
                            "news_count": int(len(minor_news)),
                            "news": [{"title": news['cleaned_title'], "link": news["original"].get("link", "")} for news in minor_news]
                        })
                
                frontend_data["clusters"].append({
                    "cluster_id": int(cluster_id),
                    "major_category": major_category,
                    "news_count": int(len(news_list)),
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
    
    def analyze_from_db(self, news_data, use_hdbscan=False, use_hybrid=True):
        """뉴스 제목 분석 파이프라인 (이미지 요구사항)"""
        # 1단계: 전처리 (특수문자 제거)
        processed_data = self.preprocess_titles(news_data)
        
        if len(processed_data) < 5:
            return None
        
        # 2단계: 대학교 이름으로 분류
        univ_news, other_news = self.split_news_by_uni_name(processed_data)
        
        # 3단계: 클러스터링 (HDBSCAN + K-Means)
        if other_news:
            clusters, noise_news = self.enhanced_cluster_news(other_news, use_hdbscan=use_hdbscan, use_hybrid=use_hybrid)
        else:
            clusters, noise_news = {}, []
        
        # 4단계: 키워드 추출 (KonlPy + TF-IDF + KeyBERT)
        cluster_labels = self.generate_cluster_labels(clusters)
        
        # 5단계: 마인드맵 데이터 생성
        frontend_data = self.create_frontend_data(univ_news, clusters, cluster_labels, noise_news)
        
        return frontend_data

