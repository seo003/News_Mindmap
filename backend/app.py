from flask import Flask, jsonify
from flask_cors import CORS 
from data.keyword_extraction import extract_keywords
from data.keyword_analysis import analyze_keywords

# Flask 앱 생성
app = Flask(__name__)
# CORS(app)  
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})

# 키워드 분석 API
@app.route("/api/keywords", methods=["GET"])
def get_keywords():
    news_data, tokenized_titles = extract_keywords()
    result = analyze_keywords(news_data, tokenized_titles)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
