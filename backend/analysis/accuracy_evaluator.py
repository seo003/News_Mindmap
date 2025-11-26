# -*- coding: utf-8 -*-

import logging
import time
import numpy as np
from collections import defaultdict, Counter
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from database.news_fetcher import fetch_news_from_db
from analysis.news_analyzer import NewsAnalyzer

# 로깅 설정
def setup_logging():
    """로깅 설정 - 콘솔과 파일 모두에 출력"""
    import os
    from datetime import datetime
    
    # 로그 파일 경로 설정
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%m-%d_%H%M")
    log_file = os.path.join(log_dir, f"accuracy_{timestamp}.log")
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # 콘솔 출력
            logging.FileHandler(log_file, encoding='utf-8')  # 파일 출력
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"📝 로그 파일 저장 위치: {log_file}")
    return logger

logger = setup_logging()


class AccuracyEvaluator:
    """
    뉴스 분석 정확도 평가 클래스
    
    클러스터링 품질, 키워드 추출 정확도, 대학교 분류 정확도 등을 측정합니다.
    """
    
    def __init__(self):
        """정확도 평가기 초기화"""
        logger.info("🔧 AccuracyEvaluator 초기화 시작...")
        
        # NewsAnalyzer 인스턴스 생성 (이미 모델이 로딩되어 있음)
        self.news_analyzer = NewsAnalyzer()
        
        # NewsAnalyzer의 모델을 재사용 (중복 로딩 방지)
        self.embedding_model = self.news_analyzer.embedding_model
        self.keybert_model = self.news_analyzer.kw_model  # KeyBERT 모델 재사용
        logger.info("✅ SentenceTransformer 모델 재사용 (NewsAnalyzer에서 로딩된 모델)")
        logger.info("✅ KeyBERT 모델 재사용 (NewsAnalyzer에서 로딩된 모델)")
        
        # 평가용 기준 데이터
        self.university_keywords = {
            "인하공전", "인하대", "항공대", "KAIST", "서울대", "연세대", "고려대",
            "성균관대", "한양대", "중앙대", "경희대", "동국대", "홍익대", "국민대"
        }
        
        self.category_keywords = {
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
        
        logger.info(f"📚 평가 기준 데이터 설정 완료 (대학교: {len(self.university_keywords)}개, 카테고리: {len(self.category_keywords)}개)")
        logger.info("🎉 AccuracyEvaluator 초기화 완료!")
    
    def evaluate_clustering_quality(self, news_data, limit=1000, embeddings=None, clusterer=None):
        """
        클러스터링 품질 평가
        
        Args:
            news_data (list): 뉴스 데이터
            limit (int): 분석할 최대 뉴스 개수
            embeddings: 미리 생성된 임베딩 (None이면 새로 생성)
            
        Returns:
            dict: 클러스터링 품질 지표
        """
        try:
            logger.info("=" * 60)
            logger.info("📊 클러스터링 품질 평가 시작")
            logger.info("=" * 60)
            logger.info(f"📰 입력 뉴스 데이터: {len(news_data)}개")
            logger.info(f"🔢 분석 제한: {limit}개")
            
            # 뉴스 제목 추출
            logger.info("📝 뉴스 제목 추출 중...")
            titles = [item["title"] for item in news_data[:limit]]
            logger.info(f"✅ 제목 추출 완료: {len(titles)}개")
            
            if len(titles) < 10:
                logger.warning(f"⚠️ 분석할 뉴스가 부족합니다: {len(titles)}개 (최소 10개 필요)")
                return {"error": "분석할 뉴스가 부족합니다 (최소 10개 필요)"}
            
            # 임베딩 생성 또는 재사용
            if embeddings is None:
                logger.info("🤖 SentenceTransformer로 임베딩 생성 중...")
                start_time = time.time()
                embeddings = self.embedding_model.encode(titles, normalize_embeddings=True)
                embedding_time = time.time() - start_time
                logger.info(f"✅ 임베딩 생성 완료: {embeddings.shape} (소요시간: {embedding_time:.2f}초)")
            else:
                logger.info(f"♻️ 기존 임베딩 재사용: {embeddings.shape}")
            
            # K-Means 클러스터링 (중복 클러스터링 방지를 위해 주석 처리)
            # logger.info("🔍 클러스터링 시작...")
            # from sklearn.cluster import KMeans
            
            # 클러스터 수 계산 (NewsAnalyzer와 동일한 방식) - 주석 처리
            # n_clusters = self.news_analyzer.calculate_kmeans_clusters(len(titles))
            
            # if n_clusters < 2:
            #     logger.warning(f"⚠️ 클러스터 수가 부족합니다: {n_clusters}개 (최소 2개 필요)")
            #     return {"error": "클러스터 수가 부족합니다 (최소 2개 필요)"}
            
            # kmeans = KMeans(
            #     n_clusters=n_clusters, 
            #     random_state=self.news_analyzer.KMEANS_RANDOM_STATE, 
            #     n_init=self.news_analyzer.KMEANS_N_INIT
            # )
            
            # clustering_start = time.time()
            # cluster_labels = kmeans.fit_predict(embeddings)
            # clustering_time = time.time() - clustering_start
            # logger.info(f"✅ K-Means 클러스터링 완료 (소요시간: {clustering_time:.2f}초)")
            
            # NewsAnalyzer 분석 결과에서 클러스터링 정보 추출
            logger.info("♻️ NewsAnalyzer 분석 결과에서 클러스터링 정보 추출...")
            
            # 품질 지표 초기화
            quality_metrics = {}
            
            # 선택한 클러스터러를 사용하여 실제 분석 수행 (임베딩 재사용)
            if clusterer is None:
                clusterer = self.news_analyzer
            
            analysis_start_time = time.time()
            try:
                logger.info("♻️ 기존 임베딩을 재사용하여 뉴스 분석 수행...")
                # 클러스터러 타입에 따라 다른 메서드 호출
                if hasattr(clusterer, 'analyze_from_db'):
                    # NewsAnalyzer
                    analysis_result = clusterer.analyze_from_db(news_data[:limit], embeddings)
                elif hasattr(clusterer, 'analyze_news'):
                    # SimpleClusterer, TfidfClusterer, FastTextClusterer
                    analysis_result = clusterer.analyze_news(news_data[:limit])
                else:
                    logger.error(f"❌ 지원하지 않는 클러스터러 타입: {type(clusterer)}")
                    return {"error": f"지원하지 않는 클러스터러 타입: {type(clusterer)}"}
                analysis_time = time.time() - analysis_start_time
                if analysis_result:
                    logger.info(f"✅ 뉴스 분석 완료 (소요시간: {analysis_time:.2f}초) - 임베딩 재사용으로 시간 단축")
                    # 분석 시간을 메트릭에 추가
                    quality_metrics["analysis_time"] = analysis_time
                else:
                    logger.warning("⚠️ 뉴스 분석 실패")
            except Exception as e:
                logger.error(f"❌ 뉴스 분석 중 오류: {e}")
            
            # 분석 결과에서 클러스터링 통계 추출
            if analysis_result:
                n_clusters = len(analysis_result)
                n_noise = 0  # NewsAnalyzer는 모든 데이터를 클러스터에 할당
                
                # 실제로 클러스터에 할당된 뉴스 수 계산
                clustered_news_count = 0
                for major_category in analysis_result:
                    middle_keywords = major_category.get('middleKeywords', [])
                    other_news = major_category.get('otherNews', [])
                    # 중분류에 포함된 뉴스 수
                    for middle_cat in middle_keywords:
                        related_news = middle_cat.get('relatedNews', [])
                        clustered_news_count += len(related_news)
                    # 기타 뉴스 수
                    clustered_news_count += len(other_news)
                
                logger.info(f"🔢 클러스터 수: {n_clusters}개")
                logger.info(f"🔇 노이즈 수: {n_noise}개")
                logger.info(f"📈 노이즈 비율: {(n_noise / len(titles) * 100):.1f}%")
                logger.info(f"📊 실제 클러스터된 뉴스: {clustered_news_count}개")
                
                quality_metrics.update({
                    "total_news": len(titles),
                    "n_clusters": n_clusters,
                    "n_noise": n_noise,
                    "noise_ratio": n_noise / len(titles) if len(titles) > 0 else 0,
                    "avg_cluster_size": clustered_news_count / n_clusters if n_clusters > 0 else 0
                })
                
                logger.info(f"📊 평균 클러스터 크기: {quality_metrics['avg_cluster_size']:.1f}")
                
                # 실루엣 점수는 NewsAnalyzer의 클러스터링 결과를 기반으로 계산
                if n_clusters > 1:
                    logger.info("📏 실루엣 점수 계산 중...")
                    try:
                        # 분석 결과에서 클러스터 라벨 재구성 (기타 뉴스만 대상)
                        cluster_labels = []
                        total_news_count = 0
                        
                        # NewsAnalyzer가 처리한 기타 뉴스 수 계산
                        other_news_count = 0
                        for major_category in analysis_result:
                            middle_keywords = major_category.get('middleKeywords', [])
                            other_news = major_category.get('otherNews', [])
                            other_news_count += len(other_news)
                            for middle_cat in middle_keywords:
                                related_news = middle_cat.get('relatedNews', [])
                                other_news_count += len(related_news)
                        
                        logger.info(f"🔍 클러스터 라벨 재구성 시작: 전체 {len(titles)}개 제목 중 기타 뉴스 {other_news_count}개")
                        
                        for i, major_category in enumerate(analysis_result):
                            major_name = major_category.get('majorKeyword', f'cluster_{i}')
                            middle_keywords = major_category.get('middleKeywords', [])
                            other_news = major_category.get('otherNews', [])
                            
                            logger.info(f"   대분류 {i}: {major_name} (중분류 {len(middle_keywords)}개, 기타 {len(other_news)}개)")
                            
                            # 중분류별로 클러스터 라벨 할당
                            for j, middle_cat in enumerate(middle_keywords):
                                related_news = middle_cat.get('relatedNews', [])
                                cluster_labels.extend([i] * len(related_news))
                                total_news_count += len(related_news)
                                logger.info(f"     중분류 {j}: {len(related_news)}개 뉴스")
                            
                            # 기타 뉴스도 같은 클러스터에 할당
                            cluster_labels.extend([i] * len(other_news))
                            total_news_count += len(other_news)
                        
                        logger.info(f"📊 클러스터 라벨 재구성 완료: {len(cluster_labels)}개 라벨, {total_news_count}개 뉴스")
                        logger.info(f"📊 임베딩 수: {len(embeddings)}개")
                        logger.info(f"📊 클러스터 라벨 분포: {dict(zip(*np.unique(cluster_labels, return_counts=True)))}")
                        
                        # 기타 뉴스에 해당하는 임베딩만 사용 (대학교 뉴스 제외)
                        # NewsAnalyzer는 대학교 뉴스를 먼저 분리하므로, 기타 뉴스는 뒤쪽에 위치
                        university_news_count = len(titles) - other_news_count
                        other_embeddings = embeddings[university_news_count:]
                        
                        logger.info(f"📊 대학교 뉴스: {university_news_count}개, 기타 뉴스: {len(other_embeddings)}개")
                        logger.info(f"📊 클러스터 수: {len(set(cluster_labels))}개")
                        
                        if len(cluster_labels) == len(other_embeddings) and len(set(cluster_labels)) > 1:
                            from sklearn.metrics import silhouette_score
                            silhouette_avg = silhouette_score(other_embeddings, cluster_labels)
                            quality_metrics["silhouette_score"] = silhouette_avg
                            logger.info(f"✅ 실루엣 점수: {silhouette_avg:.4f}")
                        else:
                            logger.warning(f"⚠️ 클러스터 라벨과 기타 뉴스 임베딩 수가 일치하지 않음: {len(cluster_labels)} vs {len(other_embeddings)}")
                            logger.warning(f"⚠️ 클러스터 수: {len(set(cluster_labels))}개")
                            quality_metrics["silhouette_score"] = None
                    except Exception as e:
                        logger.warning(f"⚠️ 실루엣 점수 계산 실패: {e}")
                        quality_metrics["silhouette_score"] = None
                else:
                    logger.warning("⚠️ 클러스터가 1개 이하로 실루엣 점수 계산 불가")
                    quality_metrics["silhouette_score"] = None
                
                # Davies-Bouldin Index 계산
                if n_clusters > 1 and len(cluster_labels) == len(other_embeddings):
                    logger.info("📏 Davies-Bouldin Index 계산 중...")
                    try:
                        db_index = davies_bouldin_score(other_embeddings, cluster_labels)
                        quality_metrics["davies_bouldin_index"] = db_index
                        logger.info(f"✅ Davies-Bouldin Index: {db_index:.4f} (낮을수록 좋음)")
                    except Exception as e:
                        logger.warning(f"⚠️ Davies-Bouldin Index 계산 실패: {e}")
                        quality_metrics["davies_bouldin_index"] = None
                else:
                    logger.warning("⚠️ 클러스터가 1개 이하이거나 데이터 수가 일치하지 않아 Davies-Bouldin Index 계산 불가")
                    quality_metrics["davies_bouldin_index"] = None
                
                # Calinski-Harabasz Index 계산 (추가된 내부 평가 지표)
                if n_clusters > 1 and len(cluster_labels) == len(other_embeddings):
                    logger.info("📏 Calinski-Harabasz Index 계산 중...")
                    try:
                        ch_index = calinski_harabasz_score(other_embeddings, cluster_labels)
                        quality_metrics["calinski_harabasz_index"] = ch_index
                        logger.info(f"✅ Calinski-Harabasz Index: {ch_index:.4f} (높을수록 좋음)")
                    except Exception as e:
                        logger.warning(f"⚠️ Calinski-Harabasz Index 계산 실패: {e}")
                        quality_metrics["calinski_harabasz_index"] = None
                else:
                    logger.warning("⚠️ 클러스터가 1개 이하이거나 데이터 수가 일치하지 않아 Calinski-Harabasz Index 계산 불가")
                    quality_metrics["calinski_harabasz_index"] = None
                
                # 클러스터별 통계
                logger.info("📋 클러스터별 통계 생성 중...")
                cluster_stats = {}
                for i, major_category in enumerate(analysis_result):
                    major_name = major_category.get('majorKeyword', f'cluster_{i}')
                    middle_keywords = major_category.get('middleKeywords', [])
                    other_news = major_category.get('otherNews', [])
                    
                    cluster_size = len(other_news) + sum(len(middle.get('relatedNews', [])) for middle in middle_keywords)
                    
                    cluster_stats[f"cluster_{i}"] = {
                        "name": major_name,
                        "size": cluster_size,
                        "middle_categories": len(middle_keywords)
                    }
                
                quality_metrics["cluster_details"] = cluster_stats
                quality_metrics["analysis_result"] = analysis_result  # 분석 결과도 함께 반환
            else:
                logger.error("❌ 분석 결과를 가져올 수 없습니다")
                return {"error": "분석 결과를 가져올 수 없습니다"}
            
            logger.info("🎉 클러스터링 품질 평가 완료!")
            logger.info("=" * 60)
            return quality_metrics
            
        except Exception as e:
            logger.error(f"클러스터링 품질 평가 중 오류: {e}")
            return {"error": str(e)}
    
    def evaluate_keyword_extraction(self, news_data, limit=1000, embeddings=None, clusterer=None):
        """
        키워드 추출 정확도 평가
        
        Args:
            news_data (list): 뉴스 데이터
            limit (int): 분석할 최대 뉴스 개수
            embeddings: 미리 생성된 임베딩 (None이면 새로 생성)
            
        Returns:
            dict: 키워드 추출 정확도 지표
        """
        try:
            logger.info("=" * 60)
            logger.info("🔑 키워드 추출 정확도 평가 시작")
            logger.info("=" * 60)
            logger.info(f"📰 입력 뉴스 데이터: {len(news_data)}개")
            logger.info(f"🔢 분석 제한: {limit}개")
            
            # 사용할 클러스터러 결정
            if clusterer is None:
                clusterer = self.news_analyzer
            
            # 뉴스 분석 실행
            # 전처리는 NewsAnalyzer의 메서드를 사용 (모든 클러스터러가 동일한 전처리 로직 사용)
            logger.info("📊 뉴스 전처리 중...")
            if hasattr(clusterer, 'preprocess_titles'):
                processed_data = clusterer.preprocess_titles(news_data[:limit])
            elif hasattr(self.news_analyzer, 'preprocess_titles'):
                processed_data = self.news_analyzer.preprocess_titles(news_data[:limit])
            else:
                processed_data = news_data[:limit]
            logger.info(f"✅ 전처리 완료: {len(processed_data)}개")
            
            if len(processed_data) < 5:
                logger.warning(f"⚠️ 분석할 뉴스가 부족합니다: {len(processed_data)}개 (최소 5개 필요)")
                return {"error": "분석할 뉴스가 부족합니다 (최소 5개 필요)"}
            
            # 대학교별 분류 (NewsAnalyzer의 메서드 사용)
            logger.info("🏫 대학교별 뉴스 분류 중...")
            if hasattr(clusterer, 'split_news_by_uni_name'):
                university_news, other_news = clusterer.split_news_by_uni_name(processed_data)
            elif hasattr(self.news_analyzer, 'split_news_by_uni_name'):
                university_news, other_news = self.news_analyzer.split_news_by_uni_name(processed_data)
            else:
                # 대학교 분류가 없는 경우 모든 뉴스를 기타 뉴스로 처리
                university_news = {}
                other_news = processed_data
            logger.info(f"✅ 대학교 뉴스: {len(university_news)}개 그룹")
            logger.info(f"✅ 기타 뉴스: {len(other_news)}개")
            
            # 클러스터링 - 선택한 클러스터러로 전체 분석 수행
            if other_news:
                logger.info("🔍 선택한 클러스터러로 전체 분석 수행...")
                # 전체 뉴스 데이터로 분석 수행 (클러스터러 타입에 따라 다른 메서드 호출)
                if embeddings is not None and hasattr(clusterer, 'analyze_from_db'):
                    logger.info("♻️ 기존 임베딩을 재사용하여 분석...")
                    analysis_result = clusterer.analyze_from_db(news_data[:limit], embeddings)
                elif hasattr(clusterer, 'analyze_news'):
                    logger.info("♻️ 선택한 클러스터러로 분석...")
                    analysis_result = clusterer.analyze_news(news_data[:limit])
                else:
                    logger.error(f"❌ 지원하지 않는 클러스터러 타입: {type(clusterer)}")
                    return {"error": f"지원하지 않는 클러스터러 타입: {type(clusterer)}"}
                
                # 분석 결과에서 클러스터 정보 추출
                if analysis_result:
                    clusters = {}
                    noise_news = []
                    # 분석 결과를 클러스터 형식으로 변환
                    for major_category in analysis_result:
                        middle_keywords = major_category.get('middleKeywords', [])
                        other_news_list = major_category.get('otherNews', [])
                        # 중분류를 클러스터로 변환
                        for middle_cat in middle_keywords:
                            related_news = middle_cat.get('relatedNews', [])
                            if related_news:
                                cluster_id = len(clusters)
                                clusters[cluster_id] = related_news
                        # 기타 뉴스도 클러스터로 추가
                        if other_news_list:
                            cluster_id = len(clusters)
                            clusters[cluster_id] = other_news_list
                    logger.info(f"✅ 클러스터 수: {len(clusters)}개")
                    logger.info(f"✅ 노이즈 뉴스: {len(noise_news)}개")
                else:
                    clusters, noise_news = {}, []
            else:
                logger.info("ℹ️ 기타 뉴스가 없어 클러스터링 건너뜀")
                clusters, noise_news = {}, []
            
            # 키워드 추출 정확도 평가
            logger.info("📊 키워드 추출 정확도 지표 계산 중...")
            keyword_metrics = {
                "total_processed_news": len(processed_data),
                "university_news_count": len(university_news),
                "clustered_news_count": sum(len(cluster) for cluster in clusters.values()),
                "noise_news_count": len(noise_news)
            }
            
            logger.info(f"📈 처리된 뉴스: {keyword_metrics['total_processed_news']}개")
            logger.info(f"🏫 대학교 뉴스: {keyword_metrics['university_news_count']}개")
            logger.info(f"🔍 클러스터된 뉴스: {keyword_metrics['clustered_news_count']}개")
            logger.info(f"🔇 노이즈 뉴스: {keyword_metrics['noise_news_count']}개")
            
            # 대학교 키워드 추출 정확도
            logger.info("🏫 대학교 키워드 추출 정확도 평가 중...")
            university_accuracy = self._evaluate_university_keywords(university_news)
            keyword_metrics.update(university_accuracy)
            logger.info(f"✅ 대학교 키워드 정확도: {university_accuracy.get('university_keyword_accuracy', 0):.1%}")
            
            # 클러스터 키워드 추출 정확도
            logger.info("🔍 클러스터 키워드 추출 정확도 평가 중...")
            cluster_accuracy = self._evaluate_cluster_keywords(clusters)
            keyword_metrics.update(cluster_accuracy)
            logger.info(f"✅ 클러스터 키워드 정확도: {cluster_accuracy.get('cluster_keyword_accuracy', 0):.1%}")
            
            logger.info("🎉 키워드 추출 정확도 평가 완료!")
            logger.info("=" * 60)
            return keyword_metrics
            
        except Exception as e:
            logger.error(f"키워드 추출 정확도 평가 중 오류: {e}")
            return {"error": str(e)}
    
    def evaluate_topic_consistency(self, news_data, analysis_result, limit=1000):
        """
        Topic Consistency 평가 (ChatGPT 제안)
        
        각 클러스터의 대표 키워드와 뉴스 본문 KeyBERT 키워드 간 유사도를 측정합니다.
        
        Args:
            news_data (list): 뉴스 데이터
            analysis_result (list): 분석 결과
            limit (int): 분석할 최대 뉴스 개수
            
        Returns:
            dict: Topic Consistency 지표
        """
        try:
            logger.info("=" * 60)
            logger.info("📊 Topic Consistency 평가 시작")
            logger.info("=" * 60)
            
            if not analysis_result:
                logger.warning("⚠️ 분석 결과가 없어 Topic Consistency 평가 불가")
                return {"error": "분석 결과가 없습니다"}
            
            consistency_scores = []
            cluster_details = []
            
            # 각 클러스터별로 평가
            for major_idx, major_category in enumerate(analysis_result):
                major_keyword = major_category.get('majorKeyword', '')
                middle_keywords = major_category.get('middleKeywords', [])
                other_news = major_category.get('otherNews', [])
                
                # 중분류별로 평가
                for middle_cat in middle_keywords:
                    middle_keyword = middle_cat.get('middleKeyword', '')
                    related_news = middle_cat.get('relatedNews', [])
                    
                    if not related_news:
                        continue
                    
                    # 클러스터 대표 키워드 (중분류 키워드 사용)
                    cluster_keyword = middle_keyword if middle_keyword else major_keyword
                    
                    # 해당 클러스터의 모든 뉴스 본문 수집
                    cluster_texts = []
                    for news in related_news:
                        # 원본 뉴스 데이터에서 본문 찾기
                        news_id = news.get('id') or news.get('title', '')
                        original_news = next(
                            (item for item in news_data[:limit] 
                             if item.get('id') == news_id or item.get('title') == news.get('title', '')),
                            None
                        )
                        if original_news:
                            # 본문이 있으면 사용, 없으면 제목 사용
                            text = original_news.get('content', '') or original_news.get('title', '')
                            if text:
                                cluster_texts.append(text)
                    
                    if not cluster_texts:
                        continue
                    
                    # KeyBERT로 클러스터 전체의 키워드 추출
                    combined_text = ' '.join(cluster_texts[:20])  # 최대 20개 뉴스만 사용
                    try:
                        keybert_keywords = self.keybert_model.extract_keywords(
                            combined_text,
                            keyphrase_ngram_range=(1, 3),
                            top_n=5,
                            use_mmr=True,
                            diversity=0.5
                        )
                        keybert_keyword_list = [kw for kw, score in keybert_keywords]
                    except Exception as e:
                        logger.warning(f"⚠️ KeyBERT 키워드 추출 실패: {e}")
                        keybert_keyword_list = []
                    
                    # 클러스터 대표 키워드와 KeyBERT 키워드 간 유사도 계산
                    if keybert_keyword_list and cluster_keyword:
                        try:
                            # 키워드들을 임베딩으로 변환
                            keywords_to_compare = [cluster_keyword] + keybert_keyword_list[:3]  # 상위 3개만
                            keyword_embeddings = self.embedding_model.encode(
                                keywords_to_compare, 
                                normalize_embeddings=True
                            )
                            
                            # 클러스터 대표 키워드와 KeyBERT 키워드들의 코사인 유사도 계산
                            cluster_keyword_emb = keyword_embeddings[0]
                            keybert_embs = keyword_embeddings[1:]
                            
                            similarities = cosine_similarity(
                                [cluster_keyword_emb], 
                                keybert_embs
                            )[0]
                            
                            avg_similarity = float(np.mean(similarities)) if len(similarities) > 0 else 0.0
                            consistency_scores.append(avg_similarity)
                            
                            cluster_details.append({
                                "cluster_id": f"major_{major_idx}_middle_{len(cluster_details)}",
                                "major_keyword": major_keyword,
                                "middle_keyword": middle_keyword,
                                "cluster_keyword": cluster_keyword,
                                "keybert_keywords": keybert_keyword_list[:3],
                                "similarity": avg_similarity,
                                "news_count": len(related_news)
                            })
                            
                        except Exception as e:
                            logger.warning(f"⚠️ 유사도 계산 실패: {e}")
                            continue
                
                # 기타 뉴스도 평가
                if other_news and major_keyword:
                    # 기타 뉴스의 본문 수집
                    other_texts = []
                    for news in other_news[:10]:  # 최대 10개만
                        news_id = news.get('id') or news.get('title', '')
                        original_news = next(
                            (item for item in news_data[:limit] 
                             if item.get('id') == news_id or item.get('title') == news.get('title', '')),
                            None
                        )
                        if original_news:
                            text = original_news.get('content', '') or original_news.get('title', '')
                            if text:
                                other_texts.append(text)
                    
                    if other_texts:
                        combined_text = ' '.join(other_texts)
                        try:
                            keybert_keywords = self.keybert_model.extract_keywords(
                                combined_text,
                                keyphrase_ngram_range=(1, 3),
                                top_n=5,
                                use_mmr=True,
                                diversity=0.5
                            )
                            keybert_keyword_list = [kw for kw, score in keybert_keywords]
                        except Exception as e:
                            logger.warning(f"⚠️ KeyBERT 키워드 추출 실패: {e}")
                            keybert_keyword_list = []
                        
                        if keybert_keyword_list and major_keyword:
                            try:
                                keywords_to_compare = [major_keyword] + keybert_keyword_list[:3]
                                keyword_embeddings = self.embedding_model.encode(
                                    keywords_to_compare,
                                    normalize_embeddings=True
                                )
                                
                                cluster_keyword_emb = keyword_embeddings[0]
                                keybert_embs = keyword_embeddings[1:]
                                
                                similarities = cosine_similarity(
                                    [cluster_keyword_emb],
                                    keybert_embs
                                )[0]
                                
                                avg_similarity = float(np.mean(similarities)) if len(similarities) > 0 else 0.0
                                consistency_scores.append(avg_similarity)
                                
                                cluster_details.append({
                                    "cluster_id": f"major_{major_idx}_other",
                                    "major_keyword": major_keyword,
                                    "middle_keyword": "",
                                    "cluster_keyword": major_keyword,
                                    "keybert_keywords": keybert_keyword_list[:3],
                                    "similarity": avg_similarity,
                                    "news_count": len(other_news)
                                })
                            except Exception as e:
                                logger.warning(f"⚠️ 유사도 계산 실패: {e}")
            
            # 전체 평균 계산
            avg_consistency = float(np.mean(consistency_scores)) if consistency_scores else 0.0
            
            logger.info(f"✅ Topic Consistency 평가 완료")
            logger.info(f"📊 평가된 클러스터 수: {len(consistency_scores)}개")
            logger.info(f"📊 평균 Topic Consistency: {avg_consistency:.4f}")
            logger.info("=" * 60)
            
            return {
                "topic_consistency_score": avg_consistency,
                "evaluated_clusters": len(consistency_scores),
                "cluster_details": cluster_details,
                "all_scores": consistency_scores
            }
            
        except Exception as e:
            logger.error(f"Topic Consistency 평가 중 오류: {e}")
            return {"error": str(e)}
    
    def _evaluate_university_keywords(self, university_news):
        """대학교 키워드 추출 정확도 평가"""
        if not university_news:
            return {"university_keyword_accuracy": 0, "university_keyword_details": {}}
        
        correct_classifications = 0
        total_classifications = 0
        details = {}
        
        for univ_name, news_list in university_news.items():
            total_classifications += len(news_list)
            
            # 실제 대학교명이 제목에 포함되어 있는지 확인
            correct_count = 0
            for news in news_list:
                title = news['cleaned_title']
                if univ_name in title or any(keyword in title for keyword in self.university_keywords):
                    correct_count += 1
                    correct_classifications += 1
            
            details[univ_name] = {
                "total_news": len(news_list),
                "correct_classifications": correct_count,
                "accuracy": correct_count / len(news_list) if len(news_list) > 0 else 0
            }
        
        return {
            "university_keyword_accuracy": correct_classifications / total_classifications if total_classifications > 0 else 0,
            "university_keyword_details": details
        }
    
    def _evaluate_cluster_keywords(self, clusters):
        """클러스터 키워드 추출 정확도 평가"""
        if not clusters:
            return {"cluster_keyword_accuracy": 0, "cluster_keyword_details": {}}
        
        # 클러스터 데이터 형식 확인 및 변환
        # TF-IDF 클러스터러는 _format_news_item으로 포맷팅된 데이터를 반환하므로
        # generate_cluster_labels를 사용하기 전에 원본 형식으로 변환 필요
        converted_clusters = {}
        for cluster_id, news_list in clusters.items():
            converted_news = []
            for news in news_list:
                # 이미 포맷팅된 데이터인 경우 (title, link만 있는 경우)
                if "title" in news and "cleaned_title" not in news:
                    # title을 cleaned_title로 사용
                    converted_news.append({
                        "cleaned_title": news.get("title", ""),
                        "original": news  # 원본 데이터 보존
                    })
                # 원본 형식 데이터인 경우
                elif "cleaned_title" in news:
                    converted_news.append(news)
                else:
                    # title도 없는 경우 건너뛰기
                    logger.warning(f"⚠️ 클러스터 {cluster_id}의 뉴스에 title 또는 cleaned_title이 없습니다: {news}")
                    continue
            if converted_news:
                converted_clusters[cluster_id] = converted_news
        
        if not converted_clusters:
            logger.warning("⚠️ 변환된 클러스터가 없습니다")
            return {"cluster_keyword_accuracy": 0, "cluster_keyword_details": {}}
        
        try:
            cluster_labels = self.news_analyzer.generate_cluster_labels(converted_clusters)
        except Exception as e:
            logger.error(f"❌ generate_cluster_labels 실패: {e}")
            # Fallback: 간단한 키워드 추출
            cluster_labels = {}
            for cluster_id, news_list in converted_clusters.items():
                titles = [item.get("cleaned_title", item.get("title", "")) for item in news_list]
                # 간단한 키워드 추출 (첫 번째 제목의 첫 단어 사용)
                major_category = titles[0].split()[0] if titles else "Unknown"
                cluster_labels[cluster_id] = {
                    "major_category": major_category,
                    "keywords": []
                }
        
        total_clusters = len(clusters)
        meaningful_clusters = 0
        details = {}
        
        for cluster_id, cluster_info in cluster_labels.items():
            major_category = cluster_info.get("major_category", "")
            keywords = cluster_info.get("keywords", [])
            
            # 카테고리 분류 정확도 확인
            category_match = False
            for category, category_words in self.category_keywords.items():
                if any(word in major_category for word in category_words):
                    category_match = True
                    break
            
            if category_match:
                meaningful_clusters += 1
            
            details[f"cluster_{cluster_id}"] = {
                "major_category": major_category,
                "keywords": keywords,
                "news_count": len(clusters.get(cluster_id, [])),
                "category_match": category_match
            }
        
        return {
            "cluster_keyword_accuracy": meaningful_clusters / total_clusters if total_clusters > 0 else 0,
            "cluster_keyword_details": details
        }
    
    def evaluate_performance(self, news_data, limit=1000, total_time=None):
        """
        성능 벤치마크 평가
        
        Args:
            news_data (list): 뉴스 데이터
            limit (int): 분석할 최대 뉴스 개수
            
        Returns:
            dict: 성능 지표
        """
        try:
            logger.info("=" * 60)
            logger.info("⚡ 성능 벤치마크 평가 시작")
            logger.info("=" * 60)
            logger.info(f"📰 입력 뉴스 데이터: {len(news_data)}개")
            logger.info(f"🔢 분석 제한: {limit}개")
            
            performance_metrics = {}
            
            # 전체 시간이 제공되면 재사용, 아니면 측정
            if total_time is not None:
                processing_time = total_time
                logger.info(f"⏱️ 전체 평가 시간 재사용: {processing_time:.2f}초")
            else:
                # 전체 분석 시간 측정 (fallback)
                logger.info("🚀 전체 뉴스 분석 실행 중...")
                start_time = time.time()
                result = self.news_analyzer.analyze_from_db(news_data[:limit])
                end_time = time.time()
                
                processing_time = end_time - start_time
                logger.info(f"✅ 분석 완료! 소요시간: {processing_time:.2f}초")
            
            performance_metrics.update({
                "total_processing_time": processing_time,
                "news_count": len(news_data[:limit]),
                "throughput": len(news_data[:limit]) / processing_time if processing_time > 0 else 0,
                "analysis_success": True
            })
            
            logger.info(f"📊 처리량: {performance_metrics['throughput']:.1f} 뉴스/초")
            logger.info("🎉 성능 벤치마크 완료!")
            
            # 메모리 사용량 추정 (간단한 방법)
            import sys
            memory_usage = sys.getsizeof(news_data[:limit]) / (1024 * 1024)
            performance_metrics["estimated_memory_usage_mb"] = memory_usage
            logger.info(f"💾 추정 메모리 사용량: {memory_usage:.1f}MB")
            
            logger.info("🎉 성능 벤치마크 완료!")
            logger.info("=" * 60)
            return performance_metrics
            
        except Exception as e:
            logger.error(f"성능 벤치마크 중 오류: {e}")
            return {"error": str(e)}
    
    def comprehensive_evaluation(self, limit=1000, use_json_file=True, json_file_path="test_news_1000.json", method='news_analyzer', clusterer=None):
        """
        종합 정확도 평가
        
        Args:
            limit (int): 분석할 최대 뉴스 개수
            use_json_file (bool): JSON 파일 사용 여부
            json_file_path (str): JSON 파일 경로
            method (str): 클러스터링 방법 ID
            clusterer: 사용할 클러스터러 객체 (None이면 self.news_analyzer 사용)
            
        Returns:
            dict: 종합 평가 결과
        """
        try:
            logger.info("🎯" * 30)
            logger.info("🎯 종합 정확도 평가 시작")
            logger.info("🎯" * 30)
            logger.info(f"📊 평가 제한: {limit}개 뉴스")
            logger.info(f"⏰ 시작 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # 뉴스 데이터 가져오기
            if use_json_file and json_file_path:
                # JSON 파일 경로를 절대 경로로 변환
                import os
                if not os.path.isabs(json_file_path):
                    # 상대 경로인 경우 backend/data 폴더에서 찾기
                    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    data_dir = os.path.join(backend_dir, "data")
                    
                    # 파일명만 있는 경우 (예: "test_news_1000.json")
                    if os.path.dirname(json_file_path) == "" or os.path.dirname(json_file_path) == ".":
                        json_file_path = os.path.join(data_dir, os.path.basename(json_file_path))
                    # "data/"로 시작하는 경우
                    elif json_file_path.startswith("data/"):
                        json_file_path = os.path.join(backend_dir, json_file_path)
                    # 그 외의 경우는 backend_dir 기준
                    else:
                        json_file_path = os.path.join(backend_dir, json_file_path)
                
                logger.info(f"📁 JSON 파일에서 뉴스 데이터 가져오는 중: {json_file_path}")
                news_data = self._load_news_from_json(json_file_path, limit)
            else:
                logger.info("📰 데이터베이스에서 뉴스 데이터 가져오는 중...")
                news_data = fetch_news_from_db(limit=limit)
            
            if not news_data:
                logger.error("❌ 뉴스 데이터를 가져올 수 없습니다")
                return {"error": "뉴스 데이터를 가져올 수 없습니다"}
            
            logger.info(f"✅ 뉴스 데이터 로드 완료: {len(news_data)}개")
            
            # 전체 평가 시간 측정 시작
            evaluation_start_time = time.time()
            
            # 임베딩 한 번만 생성 (모든 평가에서 재사용)
            logger.info("🤖 전체 평가용 임베딩 생성 중...")
            titles = [item["title"] for item in news_data[:limit]]
            embeddings_start_time = time.time()
            embeddings = self.embedding_model.encode(titles, normalize_embeddings=True)
            embeddings_time = time.time() - embeddings_start_time
            logger.info(f"✅ 전체 평가용 임베딩 생성 완료: {embeddings.shape} (소요시간: {embeddings_time:.2f}초)")
            
            evaluation_results = {
                "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "data_info": {
                    "total_news": len(news_data),
                    "limit": limit,
                    "embeddings_shape": embeddings.shape,
                    "embeddings_time": embeddings_time
                }
            }
            
            # 사용할 클러스터러 결정
            if clusterer is None:
                clusterer = self.news_analyzer
                logger.info(f"📌 기본 클러스터러 사용: NewsAnalyzer")
            else:
                logger.info(f"📌 선택한 클러스터러 사용: {method}")
            
            # 1. 클러스터링 품질 평가 (임베딩 재사용)
            logger.info("\n" + "📊" * 20)
            logger.info("1️⃣ 클러스터링 품질 평가 시작")
            clustering_results = self.evaluate_clustering_quality(news_data, limit, embeddings, clusterer=clusterer)
            evaluation_results["clustering_quality"] = clustering_results
            
            # 분석 결과 추출 (중복 분석 방지용)
            analysis_result = clustering_results.get("analysis_result")
            
            # 2. 키워드 추출 정확도 평가 (임베딩 재사용)
            logger.info("\n" + "🔑" * 20)
            logger.info("2️⃣ 키워드 추출 정확도 평가 시작")
            keyword_results = self.evaluate_keyword_extraction(news_data, limit, embeddings, clusterer=clusterer)
            evaluation_results["keyword_extraction"] = keyword_results
            
            # 2-1. Topic Consistency 평가 (ChatGPT 제안 - 새로운 평가 지표)
            logger.info("\n" + "📊" * 20)
            logger.info("2️⃣-1️⃣ Topic Consistency 평가 시작")
            topic_consistency_results = self.evaluate_topic_consistency(news_data, analysis_result, limit)
            evaluation_results["topic_consistency"] = topic_consistency_results
            
            # 3. 성능 벤치마크 (클러스터링 품질 평가의 분석 시간 사용)
            logger.info("\n" + "⚡" * 20)
            logger.info("3️⃣ 성능 벤치마크 시작")
            
            # 클러스터링 품질 평가에서 분석 시간 가져오기
            analysis_time = clustering_results.get("analysis_time", None)
            if analysis_time:
                logger.info(f"⏱️ 클러스터링 품질 평가의 분석 시간 재사용: {analysis_time:.2f}초")
                performance_results = self.evaluate_performance(news_data, limit, analysis_time)
            else:
                logger.warning("⚠️ 분석 시간을 찾을 수 없어 전체 평가 시간 사용")
                total_evaluation_time = time.time() - evaluation_start_time
                performance_results = self.evaluate_performance(news_data, limit, total_evaluation_time)
            
            evaluation_results["performance"] = performance_results
            
            # 4. 종합 점수 계산
            logger.info("\n" + "🏆" * 20)
            logger.info("4️⃣ 종합 점수 계산 시작")
            overall_score = self._calculate_overall_score(
                clustering_results, keyword_results, performance_results, topic_consistency_results
            )
            evaluation_results["overall_score"] = overall_score
            
            logger.info(f"🎉 종합 정확도 평가 완료!")
            logger.info(f"🏆 최종 점수: {overall_score['score']:.2f}/100 ({overall_score['grade']})")
            logger.info("🎯" * 30)
            
            # 평가 결과 요약을 로그 파일에 저장 (분석 결과 전달)
            self._save_evaluation_summary(evaluation_results, overall_score, analysis_result)
            
            # JSON 직렬화를 위해 float32를 float로 변환
            return self._convert_to_json_serializable(evaluation_results)
            
        except Exception as e:
            logger.error(f"종합 정확도 평가 중 오류: {e}")
            return {"error": str(e)}
    
    def _save_evaluation_summary(self, evaluation_results, overall_score, analysis_result=None):
        """
        평가 결과 요약을 별도 로그 파일에 저장 (상세 정보 포함)
        
        Args:
            evaluation_results (dict): 전체 평가 결과
            overall_score (dict): 종합 점수
        """
        try:
            import os
            from datetime import datetime
            
            # 요약 로그 파일 경로
            log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")
            os.makedirs(log_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%m-%d_%H%M")
            summary_file = os.path.join(log_dir, f"summary_{timestamp}.txt")
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("📊 정확도 평가 결과 상세 보고서\n")
                f.write("=" * 80 + "\n")
                f.write(f"평가 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"분석 뉴스 수: {evaluation_results.get('data_info', {}).get('total_news', 'N/A')}개\n")
                f.write(f"제한: {evaluation_results.get('data_info', {}).get('limit', 'N/A')}개\n\n")
                
                # 종합 점수
                f.write("🏆 종합 점수\n")
                f.write("-" * 40 + "\n")
                f.write(f"총점: {overall_score.get('score', 0):.2f}/100\n")
                f.write(f"등급: {overall_score.get('grade', 'N/A')}\n\n")
                
                # 세부 점수
                components = overall_score.get('components', {})
                f.write("📊 세부 점수\n")
                f.write("-" * 40 + "\n")
                f.write(f"클러스터링 품질: {components.get('clustering', 0):.1f}/30\n")
                f.write(f"키워드 추출: {components.get('keyword_extraction', 0):.1f}/40\n")
                f.write(f"성능: {components.get('performance', 0):.1f}/30\n\n")
                
                # 클러스터링 품질 상세
                clustering = evaluation_results.get('clustering_quality', {})
                f.write("🔍 클러스터링 품질 상세\n")
                f.write("-" * 40 + "\n")
                f.write(f"클러스터 수: {clustering.get('n_clusters', 0)}개\n")
                f.write(f"노이즈 수: {clustering.get('n_noise', 0)}개\n")
                f.write(f"노이즈 비율: {clustering.get('noise_ratio', 0):.1%}\n")
                f.write(f"평균 클러스터 크기: {clustering.get('avg_cluster_size', 0):.1f}\n")
                if clustering.get('silhouette_score'):
                    f.write(f"실루엣 점수: {clustering['silhouette_score']:.4f}\n")
                if clustering.get('davies_bouldin_index'):
                    f.write(f"Davies-Bouldin Index: {clustering['davies_bouldin_index']:.4f}\n\n")
                
                # HDBSCAN 설정값 저장
                f.write("⚙️ HDBSCAN 설정값\n")
                f.write("-" * 40 + "\n")
                f.write(f"MIN_CLUSTER_SIZE: {self.news_analyzer.HDBSCAN_MIN_CLUSTER_SIZE}\n")
                f.write(f"MIN_SAMPLES: {self.news_analyzer.HDBSCAN_MIN_SAMPLES}\n")
                f.write(f"EPSILON: {self.news_analyzer.HDBSCAN_EPSILON}\n\n")
                
                # 키워드 추출 상세
                keyword = evaluation_results.get('keyword_extraction', {})
                f.write("🔑 키워드 추출 상세\n")
                f.write("-" * 40 + "\n")
                f.write(f"대학교 키워드 정확도: {keyword.get('university_keyword_accuracy', 0):.1%}\n")
                f.write(f"클러스터 키워드 정확도: {keyword.get('cluster_keyword_accuracy', 0):.1%}\n\n")
                
                # 성능 상세
                performance = evaluation_results.get('performance', {})
                f.write("⚡ 성능 상세\n")
                f.write("-" * 40 + "\n")
                f.write(f"총 처리 시간: {performance.get('total_processing_time', 0):.2f}초\n")
                f.write(f"처리량: {performance.get('throughput', 0):.1f} 뉴스/초\n\n")
                
                # 실제 분석 결과 저장
                f.write("📋 실제 분석 결과\n")
                f.write("=" * 80 + "\n")
                
                # 클러스터링 과정 로그 저장
                f.write("🔍 클러스터링 과정 상세\n")
                f.write("-" * 60 + "\n")
                
                # NewsAnalyzer를 사용하여 실제 분석 수행 및 결과 저장
                try:
                    # JSON 파일에서 뉴스 데이터 다시 로드 (backend/data 폴더에서 찾기)
                    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    data_dir = os.path.join(backend_dir, "data")
                    json_file_path = os.path.join(data_dir, "test_news_1000.json")
                    news_data = self._load_news_from_json(json_file_path, evaluation_results.get('data_info', {}).get('limit', 1000))
                    
                    if news_data:
                        # 클러스터링 과정을 로그로 캡처하기 위해 임시로 로그 레벨 조정
                        import logging
                        original_level = logging.getLogger().level
                        logging.getLogger().setLevel(logging.INFO)
                        
                        # 실제 분석 수행 (중복 분석 방지를 위해 주석 처리)
                        # analysis_result = self.news_analyzer.analyze_from_db(news_data)
                        
                        # 로그 레벨 복원
                        logging.getLogger().setLevel(original_level)
                        
                        if analysis_result:
                            f.write("🏫 대분류/중분류 분석 결과\n")
                            f.write("-" * 60 + "\n")
                            
                            # 대분류별 통계
                            total_major_categories = len(analysis_result)
                            f.write(f"📊 총 대분류 수: {total_major_categories}개\n\n")
                            
                            for major_idx, major_category in enumerate(analysis_result, 1):
                                major_name = major_category.get('majorKeyword', 'Unknown')
                                middle_keywords = major_category.get('middleKeywords', [])
                                other_news = major_category.get('otherNews', [])
                                
                                # 대분류별 뉴스 수 계산
                                total_news_in_major = len(other_news)
                                for middle_cat in middle_keywords:
                                    total_news_in_major += len(middle_cat.get('relatedNews', []))
                                
                                f.write(f"📁 대분류 {major_idx}: {major_name} (총 {total_news_in_major}개 뉴스)\n")
                                
                                # 중분류 출력
                                if middle_keywords:
                                    f.write(f"   중분류 수: {len(middle_keywords)}개\n")
                                    for middle_idx, middle_cat in enumerate(middle_keywords, 1):
                                        middle_name = middle_cat.get('middleKeyword', 'Unknown')
                                        related_news = middle_cat.get('relatedNews', [])
                                        
                                        f.write(f"   ├─ 중분류 {middle_idx}: {middle_name} ({len(related_news)}개 뉴스)\n")
                                        
                                        # 뉴스 제목 출력 (최대 5개)
                                        for news_idx, news in enumerate(related_news[:5], 1):
                                            news_title = news.get('title', 'Unknown')
                                            # 제목이 너무 길면 자르기
                                            if len(news_title) > 80:
                                                news_title = news_title[:80] + "..."
                                            f.write(f"   │  └─ {news_idx}. {news_title}\n")
                                        
                                        # 더 많은 뉴스가 있으면 표시
                                        if len(related_news) > 5:
                                            f.write(f"   │     ... 외 {len(related_news) - 5}개\n")
                                
                                # 기타 뉴스 출력
                                if other_news:
                                    f.write(f"   └─ 기타 뉴스: {len(other_news)}개\n")
                                    for news_idx, news in enumerate(other_news[:3], 1):
                                        news_title = news.get('title', 'Unknown')
                                        if len(news_title) > 80:
                                            news_title = news_title[:80] + "..."
                                        f.write(f"      └─ {news_idx}. {news_title}\n")
                                    if len(other_news) > 3:
                                        f.write(f"         ... 외 {len(other_news) - 3}개\n")
                                
                                f.write("\n")
                            
                            # 전체 통계 요약
                            f.write("📊 전체 통계 요약\n")
                            f.write("-" * 60 + "\n")
                            f.write(f"총 대분류: {total_major_categories}개\n")
                            
                            total_middle_categories = sum(len(major.get('middleKeywords', [])) for major in analysis_result)
                            f.write(f"총 중분류: {total_middle_categories}개\n")
                            
                            total_news_count = sum(
                                len(major.get('otherNews', [])) + 
                                sum(len(middle.get('relatedNews', [])) for middle in major.get('middleKeywords', []))
                                for major in analysis_result
                            )
                            f.write(f"총 뉴스 수: {total_news_count}개\n")
                            
                        else:
                            f.write("❌ 분석 결과를 가져올 수 없습니다.\n")
                    
                except Exception as e:
                    f.write(f"❌ 분석 결과 저장 중 오류: {e}\n")
                
                f.write("\n" + "=" * 80 + "\n")
                f.write("📝 보고서 끝\n")
                f.write("=" * 80 + "\n")
            
            logger.info(f"📝 상세 평가 결과 보고서 저장 완료: {summary_file}")
            
        except Exception as e:
            logger.error(f"평가 결과 요약 저장 실패: {e}")
    
    def _load_news_from_json(self, json_file_path, limit):
        """
        JSON 파일에서 뉴스 데이터 로드
        
        Args:
            json_file_path (str): JSON 파일 경로
            limit (int): 로드할 뉴스 개수
            
        Returns:
            list: 뉴스 데이터 리스트
        """
        import json
        import os
        
        try:
            # 파일 경로 확인
            if not os.path.exists(json_file_path):
                logger.error(f"JSON 파일을 찾을 수 없습니다: {json_file_path}")
                return []
            
            # JSON 파일 로드
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 뉴스 데이터 추출
            news_data = data.get('news_data', [])
            metadata = data.get('metadata', {})
            
            logger.info(f"📊 JSON 파일 메타데이터:")
            logger.info(f"   - 추출 시간: {metadata.get('extraction_time', 'N/A')}")
            logger.info(f"   - 총 뉴스 수: {metadata.get('total_news_count', len(news_data))}")
            logger.info(f"   - 설명: {metadata.get('description', 'N/A')}")
            
            # limit 적용
            if limit and limit < len(news_data):
                news_data = news_data[:limit]
                logger.info(f"📝 limit 적용: {len(news_data)}개 뉴스")
            
            logger.info(f"✅ JSON 파일에서 {len(news_data)}개 뉴스 로드 완료")
            return news_data
            
        except Exception as e:
            logger.error(f"JSON 파일 로드 실패: {e}")
            return []
    
    def _convert_to_json_serializable(self, obj):
        """
        JSON 직렬화를 위해 float32, numpy 타입 등을 Python 기본 타입으로 변환
        
        Args:
            obj: 변환할 객체
            
        Returns:
            JSON 직렬화 가능한 객체
        """
        import numpy as np
        
        if isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        else:
            return obj
    
    def _calculate_overall_score(self, clustering_results, keyword_results, performance_results, topic_consistency_results=None):
        """
        종합 점수 계산 (ChatGPT 제안 반영)
        
        Args:
            clustering_results: 클러스터링 품질 평가 결과
            keyword_results: 키워드 추출 정확도 평가 결과
            performance_results: 성능 평가 결과
            topic_consistency_results: Topic Consistency 평가 결과 (선택)
        """
        try:
            logger.info("🧮 종합 점수 계산 시작...")
            score_components = {}
            total_score = 0
            max_score = 0
            
            # 클러스터링 품질 점수 (30점 만점) - 내부 평가 지표 개선
            logger.info("📊 클러스터링 품질 점수 계산 중...")
            if "error" not in clustering_results:
                clustering_score = 0
                
                # 실루엣 점수 (10점) - 가중치 조정
                if clustering_results.get("silhouette_score") is not None:
                    silhouette = clustering_results["silhouette_score"]
                    # 실루엣 점수는 -1~1 범위이므로 0~1로 정규화 후 점수화
                    normalized_silhouette = (silhouette + 1) / 2  # -1~1 -> 0~1
                    silhouette_points = normalized_silhouette * 10
                    clustering_score += silhouette_points
                    logger.info(f"   📏 실루엣 점수: {silhouette:.4f} → {silhouette_points:.1f}점")
                else:
                    logger.warning("   ⚠️ 실루엣 점수 없음")
                
                # Calinski-Harabasz Index (10점) - 추가된 지표
                if clustering_results.get("calinski_harabasz_index") is not None:
                    ch_index = clustering_results["calinski_harabasz_index"]
                    # CH Index는 값이 클수록 좋으므로 정규화 필요
                    # 일반적으로 100~10000 범위이므로 로그 스케일 사용
                    if ch_index > 0:
                        normalized_ch = min(1.0, np.log10(ch_index + 1) / 4)  # 대략 0~1 범위로 정규화
                        ch_points = normalized_ch * 10
                        clustering_score += ch_points
                        logger.info(f"   📊 Calinski-Harabasz Index: {ch_index:.2f} → {ch_points:.1f}점")
                else:
                    logger.warning("   ⚠️ Calinski-Harabasz Index 없음")
                
                # Davies-Bouldin Index (10점) - 추가된 지표
                if clustering_results.get("davies_bouldin_index") is not None:
                    db_index = clustering_results["davies_bouldin_index"]
                    # DB Index는 낮을수록 좋으므로 역수 사용
                    # 일반적으로 0~5 범위이므로 정규화
                    normalized_db = max(0, 1 - (db_index / 5))  # 0~1 범위로 정규화
                    db_points = normalized_db * 10
                    clustering_score += db_points
                    logger.info(f"   📏 Davies-Bouldin Index: {db_index:.4f} → {db_points:.1f}점")
                else:
                    logger.warning("   ⚠️ Davies-Bouldin Index 없음")
                
                score_components["clustering"] = clustering_score
                total_score += clustering_score
                logger.info(f"   ✅ 클러스터링 총점: {clustering_score:.1f}/30")
            else:
                logger.warning("   ❌ 클러스터링 평가 실패")
            max_score += 30
            
            # 키워드 추출 정확도 점수 (30점 만점) - 가중치 조정
            logger.info("🔑 키워드 추출 정확도 점수 계산 중...")
            if "error" not in keyword_results:
                keyword_score = 0
                
                # 대학교 키워드 정확도 (15점)
                univ_accuracy = keyword_results.get("university_keyword_accuracy", 0)
                univ_points = univ_accuracy * 15
                keyword_score += univ_points
                logger.info(f"   🏫 대학교 키워드 정확도: {univ_accuracy:.1%} → {univ_points:.1f}점")
                
                # 클러스터 키워드 정확도 (15점)
                cluster_accuracy = keyword_results.get("cluster_keyword_accuracy", 0)
                cluster_points = cluster_accuracy * 15
                keyword_score += cluster_points
                logger.info(f"   🔍 클러스터 키워드 정확도: {cluster_accuracy:.1%} → {cluster_points:.1f}점")
                
                score_components["keyword_extraction"] = keyword_score
                total_score += keyword_score
                logger.info(f"   ✅ 키워드 추출 총점: {keyword_score:.1f}/30")
            else:
                logger.warning("   ❌ 키워드 추출 평가 실패")
            max_score += 30
            
            # Topic Consistency 점수 (20점 만점) - ChatGPT 제안 추가
            logger.info("📊 Topic Consistency 점수 계산 중...")
            if topic_consistency_results and "error" not in topic_consistency_results:
                topic_consistency_score = 0
                consistency = topic_consistency_results.get("topic_consistency_score", 0)
                consistency_points = consistency * 20  # 0~1 범위를 0~20점으로 변환
                topic_consistency_score += consistency_points
                logger.info(f"   📊 Topic Consistency: {consistency:.4f} → {consistency_points:.1f}점")
                
                score_components["topic_consistency"] = topic_consistency_score
                total_score += topic_consistency_score
                logger.info(f"   ✅ Topic Consistency 총점: {topic_consistency_score:.1f}/20")
            else:
                logger.warning("   ⚠️ Topic Consistency 평가 없음 (선택적 지표)")
            max_score += 20
            
            # 성능 점수 (30점 만점)
            logger.info("⚡ 성능 점수 계산 중...")
            if "error" not in performance_results:
                performance_score = 0
                
                # 처리 시간 점수 (15점) - 10초 이내면 만점
                processing_time = performance_results.get("total_processing_time", 100)
                time_score = min(15, max(0, 15 - (processing_time - 10) * 0.5))
                performance_score += time_score
                logger.info(f"   ⏱️ 처리 시간: {processing_time:.2f}초 → {time_score:.1f}점")
                
                # 처리량 점수 (15점) - 뉴스/초
                throughput = performance_results.get("throughput", 0)
                throughput_score = min(15, max(0, throughput * 0.1))
                performance_score += throughput_score
                logger.info(f"   📊 처리량: {throughput:.1f} 뉴스/초 → {throughput_score:.1f}점")
                
                score_components["performance"] = performance_score
                total_score += performance_score
                logger.info(f"   ✅ 성능 총점: {performance_score:.1f}/30")
            else:
                logger.warning("   ❌ 성능 평가 실패")
            max_score += 30
            
            # 최종 점수 계산
            final_score = (total_score / max_score * 100) if max_score > 0 else 0
            grade = self._get_grade(final_score)
            
            logger.info(f"🏆 최종 점수: {total_score:.1f}/{max_score} → {final_score:.1f}/100 ({grade})")
            
            result = {
                "score": final_score,
                "max_possible_score": max_score,
                "components": score_components,
                "grade": grade
            }
            
            # JSON 직렬화를 위해 float32를 float로 변환
            return self._convert_to_json_serializable(result)
            
        except Exception as e:
            logger.error(f"종합 점수 계산 중 오류: {e}")
            return {"error": str(e)}
    
    def _get_grade(self, score):
        """점수에 따른 등급 반환"""
        if score >= 90:
            return "A+"
        elif score >= 80:
            return "A"
        elif score >= 70:
            return "B+"
        elif score >= 60:
            return "B"
        elif score >= 50:
            return "C+"
        elif score >= 40:
            return "C"
        else:
            return "D"


def run_accuracy_evaluation(limit=1000):
    """
    정확도 평가 실행 함수
    
    Args:
        limit (int): 분석할 최대 뉴스 개수
        
    Returns:
        dict: 평가 결과
    """
    evaluator = AccuracyEvaluator()
    return evaluator.comprehensive_evaluation(limit=limit)


if __name__ == "__main__":
    # 테스트 실행
    print("=" * 80)
    print("📊 뉴스 분석 정확도 평가")
    print("=" * 80)
    
    try:
        result = run_accuracy_evaluation(limit=1000)
        
        if "error" in result:
            print(f"❌ 오류 발생: {result['error']}")
        else:
            print(f"✅ 평가 완료!")
            print(f"📅 평가 시간: {result['evaluation_timestamp']}")
            print(f"📰 분석 뉴스: {result['data_info']['total_news']}개")
            print(f"🏆 종합 점수: {result['overall_score']['score']:.1f}/100 ({result['overall_score']['grade']})")
            
            # 상세 결과 출력
            if "clustering_quality" in result and "error" not in result["clustering_quality"]:
                cq = result["clustering_quality"]
                print(f"\n📊 클러스터링 품질:")
                print(f"   • 클러스터 수: {cq.get('n_clusters', 0)}개")
                print(f"   • 노이즈 비율: {cq.get('noise_ratio', 0):.1%}")
                print(f"   • 실루엣 점수: {cq.get('silhouette_score', 'N/A')}")
            
            if "keyword_extraction" in result and "error" not in result["keyword_extraction"]:
                ke = result["keyword_extraction"]
                print(f"\n🔑 키워드 추출:")
                print(f"   • 대학교 분류 정확도: {ke.get('university_keyword_accuracy', 0):.1%}")
                print(f"   • 클러스터 키워드 정확도: {ke.get('cluster_keyword_accuracy', 0):.1%}")
            
            if "performance" in result and "error" not in result["performance"]:
                perf = result["performance"]
                print(f"\n⚡ 성능:")
                print(f"   • 처리 시간: {perf.get('total_processing_time', 0):.2f}초")
                print(f"   • 처리량: {perf.get('throughput', 0):.1f} 뉴스/초")
        
    except Exception as e:
        print(f"❌ 평가 실행 중 오류: {e}")
    
    print("=" * 80)
