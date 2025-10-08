from flask import Flask, jsonify
from flask_cors import CORS
from database.news_fetcher import fetch_news_from_db
from analysis.news_analyzer import NewsAnalyzer


# Flask 앱 생성
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})

# 뉴스 분석기 초기화
news_analyzer = NewsAnalyzer()

# 뉴스 분석 API 
@app.route("/api/news_analysis", methods=["GET"])
def get_news_analysis():
    try:
        # 데이터베이스에서 뉴스 데이터 가져오기
        news_data = fetch_news_from_db(limit=1000)
        
        if not news_data:
            return jsonify({"error": "뉴스 데이터를 가져올 수 없습니다."}), 500
        
        # 분석 실행
        result = news_analyzer.analyze_from_db(news_data)
        
        if result is None:
            return jsonify({"error": "분석할 뉴스가 없습니다."}), 400
        
        # API 응답 전 결과 요약 출력
        print(f"\n=== API 응답 데이터 요약 ===")
        print(f"총 대분류 개수: {len(result) if result else 0}")
        for i, major_cat in enumerate(result or [], 1):
            major_name = major_cat.get('majorKeyword', 'Unknown')
            middle_count = len(major_cat.get('middleKeywords', []))
            other_news_count = len(major_cat.get('otherNews', []))
            print(f"  {i}. {major_name}: 중분류 {middle_count}개, 기타뉴스 {other_news_count}개")
        print("=" * 30)
            
        return jsonify(result)
    except Exception as e:
        print(f"뉴스 분석 오류: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
