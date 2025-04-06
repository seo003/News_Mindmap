from data.keyword_extraction import extract_keywords
from data.keyword_analysis import analyze_keywords

# 1. 키워드 추출
print("🔍 키워드 추출 중...")
titles, tokenized_titles = extract_keywords()

# 2. 분석
print("🧠 LDA 토픽 모델링 중...")
analyze_keywords(titles, tokenized_titles)
