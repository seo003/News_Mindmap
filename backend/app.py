from flask import Flask, jsonify
from flask_cors import CORS 
from analysis.keyword_extraction import extract_keywords
from analysis.keyword_analysis import analyze_keywords
from crawlAll import crawl_all
    
# Flask 앱 생성
app = Flask(__name__)
# CORS(app)  
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})

# 키워드 분석 API
@app.route("/api/keywords", methods=["GET"])
def get_keywords():
    crawl_data = crawl_all()
    news_data, tokenized_titles = extract_keywords(crawl_data)
    # news_data, tokenized_titles = extract_keywords()
    result = analyze_keywords(news_data, tokenized_titles)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
