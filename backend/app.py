from flask import Flask, jsonify
from flask_cors import CORS
from database.news_fetcher import fetch_news_from_db
from analysis.news_analyzer import NewsAnalyzer


# Flask 앱 생성 및 CORS 설정
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})

# 뉴스 분석기 초기화
news_analyzer = NewsAnalyzer()


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
        # 데이터베이스에서 최근 1000개의 뉴스 데이터 가져오기
        news_data = fetch_news_from_db(limit=1000)
        
        if not news_data:
            return jsonify({"error": "뉴스 데이터를 가져올 수 없습니다."}), 500
        
        # 뉴스 분석 실행 (전처리, 클러스터링, 키워드 추출)
        result = news_analyzer.analyze_from_db(news_data)
        
        if result is None:
            return jsonify({"error": "분석할 뉴스가 없습니다."}), 400
        
        # 분석 결과를 계층 구조로 콘솔에 출력
        print_analysis_result(result)
            
        return jsonify(result)
    
    except Exception as e:
        print(f"뉴스 분석 오류: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # 디버그 모드로 실행 (개발 환경용)
    app.run(debug=True, use_reloader=False)
