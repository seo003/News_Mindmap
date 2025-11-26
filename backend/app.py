import logging
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from database.news_fetcher import fetch_news_from_db
from analysis.tfidf_clusterer import TfidfClusterer
from analysis.simple_clusterer import SimpleClusterer
from analysis.news_analyzer import NewsAnalyzer
from analysis.accuracy_evaluator import AccuracyEvaluator

# 로깅 설정 (콘솔에만 출력)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastTextClusterer는 선택적 의존성 (fasttext 모듈 필요)
try:
    from analysis.fasttext_clusterer import FastTextClusterer
    FASTTEXT_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ FastTextClusterer를 사용할 수 없습니다: {e}")
    logger.warning("⚠️ fasttext 모듈을 설치하려면: pip install fasttext")
    FastTextClusterer = None
    FASTTEXT_AVAILABLE = False

# Flask 앱 생성 및 CORS 설정
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})

# 모든 클러스터러 초기화
try:
    tfidf_clusterer = TfidfClusterer()
    logger.info("✅ TfidfClusterer 초기화 완료")
except Exception as e:
    logger.error(f"❌ TfidfClusterer 초기화 실패: {e}")
    tfidf_clusterer = None

if FASTTEXT_AVAILABLE and FastTextClusterer is not None:
    try:
        fasttext_clusterer = FastTextClusterer()
        logger.info("✅ FastTextClusterer 초기화 완료")
    except Exception as e:
        logger.error(f"❌ FastTextClusterer 초기화 실패: {e}")
        fasttext_clusterer = None
else:
    fasttext_clusterer = None
    logger.warning("⚠️ FastTextClusterer를 사용할 수 없습니다 (fasttext 모듈 없음)")

try:
    simple_clusterer = SimpleClusterer()
    logger.info("✅ SimpleClusterer 초기화 완료")
except Exception as e:
    logger.error(f"❌ SimpleClusterer 초기화 실패: {e}")
    simple_clusterer = None

try:
    news_analyzer = NewsAnalyzer()
    logger.info("✅ NewsAnalyzer 초기화 완료")
except Exception as e:
    logger.error(f"❌ NewsAnalyzer 초기화 실패: {e}")
    news_analyzer = None

try:
    accuracy_evaluator = AccuracyEvaluator()
    logger.info("✅ AccuracyEvaluator 초기화 완료")
except Exception as e:
    logger.error(f"❌ AccuracyEvaluator 초기화 실패: {e}", exc_info=True)
    accuracy_evaluator = None

# 사용 가능한 클러스터링 방법
CLUSTERING_METHODS = {
    'simple': {
        'name': '빈도수',
        'description': '키워드 빈도 기반 클러스터링',
        'clusterer': simple_clusterer
    },
    'tfidf': {
        'name': 'TF-IDF',
        'description': 'TF-IDF + 코사인 유사도 기반 클러스터링',
        'clusterer': tfidf_clusterer
    },
    'news_analyzer': {
        'name': 'HDBSCAN',
        'description': 'Sentence Transformer 임베딩 + HDBSCAN/K-Means 클러스터링',
        'clusterer': news_analyzer
    }
}

# FastText는 선택적 (모듈이 있을 때만 추가)
if fasttext_clusterer is not None:
    CLUSTERING_METHODS['fasttext'] = {
        'name': 'FastText',
        'description': 'FastText 임베딩 + K-Means 클러스터링',
        'clusterer': fasttext_clusterer
    }



def print_analysis_result(result):
    """
    분석 결과를 계층 구조로 콘솔에 출력
    
    Args:
        result (list): 분석 결과 리스트
    """
    print("\n" + "=" * 80)
    print("📊 뉴스 분석 결과")
    print("=" * 80)
    
    if not result:
        print("분석 결과가 없습니다.")
        return
    
    for major_idx, major_cat in enumerate(result, 1):
        major_name = major_cat.get('majorKeyword', 'Unknown')
        middle_keywords = major_cat.get('middleKeywords', [])
        other_news = major_cat.get('otherNews', [])
        
        # 대분류 출력
        total_news = sum(len(mid.get('relatedNews', [])) for mid in middle_keywords) + len(other_news)
        print(f"\n📁 대분류 {major_idx}: {major_name} (총 {total_news}개 뉴스)")
        
        # 중분류 출력
        if middle_keywords:
            for middle_idx, middle_cat in enumerate(middle_keywords, 1):
                middle_name = middle_cat.get('middleKeyword', 'Unknown')
                related_news = middle_cat.get('relatedNews', [])
                
                print(f"  ├─ 중분류 {middle_idx}: {middle_name} ({len(related_news)}개 뉴스)")
                
                # 뉴스 제목 출력 (최대 3개)
                for news_idx, news in enumerate(related_news[:3], 1):
                    news_title = news.get('title', 'Unknown')
                    # 제목이 너무 길면 자르기
                    if len(news_title) > 60:
                        news_title = news_title[:60] + "..."
                    print(f"  │  └─ {news_idx}. {news_title}")
                
                # 더 많은 뉴스가 있으면 표시
                if len(related_news) > 3:
                    print(f"  │     ... 외 {len(related_news) - 3}개")
        
        # 기타 뉴스 출력
        if other_news:
            print(f"  └─ 기타 뉴스: {len(other_news)}개")
    
    print("\n" + "=" * 80 + "\n")


@app.route("/api/news_analysis", methods=["GET"])
def get_news_analysis():
    try:
        # 쿼리 파라미터로 limit와 method 받기
        limit = request.args.get('limit', default=1000, type=int)
        method = request.args.get('method', default='tfidf', type=str).lower()
        
        # 데이터베이스에서 최근 뉴스 데이터 가져오기
        logger.info(f"데이터베이스에서 최근 {limit}개 뉴스 데이터 가져오는 중...")
        news_data = fetch_news_from_db(limit=limit)
        
        if not news_data:
            logger.error("데이터베이스에서 뉴스 데이터를 가져올 수 없습니다")
            return jsonify({"error": "뉴스 데이터를 가져올 수 없습니다."}), 500
        
        logger.info(f"데이터베이스에서 {len(news_data)}개 뉴스 데이터 로드 완료")
        
        # 클러스터링 방법 선택
        if method not in CLUSTERING_METHODS:
            logger.warning(f"알 수 없는 클러스터링 방법: {method}, 기본값(tfidf) 사용")
            method = 'tfidf'
        
        method_info = CLUSTERING_METHODS[method]
        clusterer = method_info['clusterer']
        
        logger.info(f"📊 클러스터링 방법: {method_info['name']} - {method_info['description']}")
        
        # 선택한 클러스터러로 분석 실행
        if method == 'news_analyzer':
            # NewsAnalyzer는 analyze_from_db 메서드 사용
            result = clusterer.analyze_from_db(news_data)
        else:
            # 다른 클러스터러는 analyze_news 메서드 사용
            result = clusterer.analyze_news(news_data)
        
        if result is None:
            logger.warning("분석 가능한 뉴스가 부족합니다")
            return jsonify({"error": "분석할 뉴스가 없습니다."}), 400
        
        # 분석 결과를 계층 구조로 콘솔에 출력
        print_analysis_result(result)
        
        # 결과에 사용된 방법 정보 추가
        response_data = {
            "method": method,
            "method_name": method_info['name'],
            "method_description": method_info['description'],
            "data": result
        }
            
        return jsonify(response_data)
    
    except Exception as e:
        logger.error(f"뉴스 분석 중 오류 발생: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/clustering_methods", methods=["GET"])
def get_clustering_methods():
    """사용 가능한 클러스터링 방법 목록 반환 (순서: 빈도수 - TF-IDF - FastText - HDBSCAN)"""
    # 명시적인 순서 정의
    method_order = ['simple', 'tfidf', 'fasttext', 'news_analyzer']
    methods_list = []
    
    # 순서대로 추가
    for method_id in method_order:
        if method_id in CLUSTERING_METHODS:
            value = CLUSTERING_METHODS[method_id]
            methods_list.append({
                "id": method_id,
                "name": value['name'],
                "description": value['description']
            })
    
    return jsonify({
        "success": True,
        "methods": methods_list
    })


# 클러스터링 평가
@app.route("/api/accuracy", methods=["GET"])
def evaluate_accuracy():
    try:
        # 쿼리 파라미터로 limit와 method 받기
        limit = request.args.get('limit', default=1000, type=int)
        method = request.args.get('method', default='news_analyzer', type=str).lower()
        
        # limit 범위 검증
        if limit < 10:
            return jsonify({
                "success": False,
                "error": "최소 10개 이상의 뉴스가 필요합니다.",
                "message": "limit 파라미터를 10 이상으로 설정해주세요."
            }), 400
        
        if limit > 5000:
            return jsonify({
                "success": False,
                "error": "최대 5000개까지만 분석 가능합니다.",
                "message": "limit 파라미터를 5000 이하로 설정해주세요."
            }), 400
        
        # 클러스터링 방법 검증
        if method not in CLUSTERING_METHODS:
            logger.warning(f"알 수 없는 클러스터링 방법: {method}, 기본값(news_analyzer) 사용")
            method = 'news_analyzer'
        
        logger.info(f"정확도 평가 요청: limit={limit}, method={method}")
        
        # accuracy_evaluator 확인
        if accuracy_evaluator is None:
            return jsonify({
                "success": False,
                "error": "정확도 평가기가 초기화되지 않았습니다.",
                "message": "서버를 재시작해주세요."
            }), 500
        
        # 선택한 클러스터러로 분석 수행 후 정확도 평가
        selected_clusterer = CLUSTERING_METHODS[method]['clusterer']
        if selected_clusterer is None:
            return jsonify({
                "success": False,
                "error": f"클러스터러 '{method}'가 초기화되지 않았습니다.",
                "message": "서버를 재시작해주세요."
            }), 500
        
        result = accuracy_evaluator.comprehensive_evaluation(limit=limit, method=method, clusterer=selected_clusterer)
        
        # 사용된 클러스터링 방법 정보 추가
        if "data" in result or isinstance(result, dict):
            result["clustering_method"] = method
            result["clustering_method_name"] = CLUSTERING_METHODS[method]['name']
        
        if "error" in result:
            logger.error(f"정확도 평가 실패: {result['error']}")
            return jsonify({
                "success": False,
                "error": result["error"],
                "message": "정확도 평가를 다시 시도해주세요."
            }), 500
        
        logger.info(f"정확도 평가 완료: {result['overall_score']['score']:.1f}점")
        
        return jsonify({
            "success": True,
            "data": result,
            "message": "정확도 평가가 완료되었습니다."
        })
    
    except Exception as e:
        logger.error(f"정확도 평가 API 오류: {e}", exc_info=True)
        
        return jsonify({
            "success": False,
            "error": f"정확도 평가 중 오류가 발생했습니다: {str(e)}",
            "message": "정확도 평가를 다시 시도해주세요."
        }), 500


# 정확도 평가 요약 (간단한 버전)
@app.route("/api/accuracy/summary", methods=["GET"])
def evaluate_accuracy_summary():
    try:
        # 쿼리 파라미터로 limit와 method 받기
        limit = request.args.get('limit', default=500, type=int)
        method = request.args.get('method', default='news_analyzer', type=str).lower()
        
        # 클러스터링 방법 검증
        if method not in CLUSTERING_METHODS:
            logger.warning(f"알 수 없는 클러스터링 방법: {method}, 기본값(news_analyzer) 사용")
            method = 'news_analyzer'
        
        logger.info(f"정확도 평가 요약 요청: limit={limit}, method={method}")
        
        # accuracy_evaluator 확인
        if accuracy_evaluator is None:
            return jsonify({
                "success": False,
                "error": "정확도 평가기가 초기화되지 않았습니다."
            }), 500
        
        # 선택한 클러스터러로 분석 수행 후 정확도 평가
        selected_clusterer = CLUSTERING_METHODS[method]['clusterer']
        if selected_clusterer is None:
            return jsonify({
                "success": False,
                "error": f"클러스터러 '{method}'가 초기화되지 않았습니다."
            }), 500
        
        # 정확도 평가 실행 (전역 평가기 사용)
        result = accuracy_evaluator.comprehensive_evaluation(limit=limit, method=method, clusterer=selected_clusterer)
        
        # 사용된 클러스터링 방법 정보 추가
        if "data" in result or isinstance(result, dict):
            result["clustering_method"] = method
            result["clustering_method_name"] = CLUSTERING_METHODS[method]['name']
        
        if "error" in result:
            return jsonify({
                "success": False,
                "error": result["error"]
            }), 500
        
        # 요약 정보만 추출
        summary = {
            "timestamp": result.get("evaluation_timestamp"),
            "news_count": result.get("data_info", {}).get("total_news", 0),
            "overall_score": result.get("overall_score", {}).get("score", 0),
            "grade": result.get("overall_score", {}).get("grade", "N/A"),
            "clustering_score": result.get("overall_score", {}).get("components", {}).get("clustering", 0),
            "keyword_score": result.get("overall_score", {}).get("components", {}).get("keyword_extraction", 0),
            "performance_score": result.get("overall_score", {}).get("components", {}).get("performance", 0)
        }
        
        return jsonify({
            "success": True,
            "data": summary,
            "message": "정확도 평가 요약이 완료되었습니다."
        })
    
    except Exception as e:
        logger.error(f"정확도 평가 요약 API 오류: {e}")
        
        return jsonify({
            "success": False,
            "error": f"정확도 평가 요약 중 오류가 발생했습니다: {str(e)}"
        }), 500


if __name__ == "__main__":
    # 디버그 모드로 실행 (개발 환경용)
    app.run(debug=True, use_reloader=False)
