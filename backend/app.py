from flask import Flask, jsonify, request
from flask_cors import CORS
from database.news_fetcher import fetch_news_from_db
from analysis.keyword_extraction import extract_keywords
from analysis.title_tfidf import analyze_keywords
from analysis.news_analyzer import NewsAnalyzer
from crawlAll import crawl_all


# Flask 앱 생성
app = Flask(__name__)
# CORS(app)  
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})

# 뉴스 분석기 초기화
news_analyzer = NewsAnalyzer()

# 키워드 분석 API
# @app.route("/api/keywords", methods=["GET"])
# def get_keywords():
#     crawl_data = crawl_all()
#     news_data, tokenized_titles = extract_keywords(crawl_data)
#     result = analyze_keywords(news_data, tokenized_titles)
#     return jsonify(result)


# 키워드 분석 API
# @app.route("/api/keywords_from_db", methods=["GET"])
# def get_keywords_from_db():
#     # 데이터베이스에서 최신 뉴스 가져오기
#     news_data = fetch_news_from_db()

#     if not news_data:
#         return jsonify({"error": "뉴스 데이터를 가져올 수 없습니다."}), 500

#     # 키워드 추출 및 분석
#     processed_data = extract_keywords(news_data)
#     result = analyze_keywords(processed_data)

#     return jsonify(result)


# 뉴스 분석 API 
@app.route("/api/news_analysis", methods=["GET"])
def get_news_analysis():
    try:
        # URL 파라미터에서 클러스터링 방법 확인
        use_hdbscan = request.args.get('hdbscan', 'false').lower() == 'true'
        use_hybrid = request.args.get('hybrid', 'false').lower() == 'true'
        
        # 데이터베이스에서 뉴스 데이터 가져오기
        news_data = fetch_news_from_db(limit=1000)
        
        if not news_data:
            return jsonify({"error": "뉴스 데이터를 가져올 수 없습니다."}), 500
        
        # 분석 실행 (클러스터링 옵션 포함)
        result = news_analyzer.analyze_from_db(news_data, use_hdbscan=use_hdbscan, use_hybrid=use_hybrid)
        
        if result is None:
            return jsonify({"error": "분석할 뉴스가 없습니다."}), 400
            
        return jsonify(result)
    except Exception as e:
        print(f"뉴스 분석 오류: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
