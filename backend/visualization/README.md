# 논문용 그래프 생성 가이드

## 개요
이 디렉토리에는 논문에 사용할 그래프를 생성하는 스크립트가 포함되어 있습니다.

## 설치

### 필요한 라이브러리
```bash
pip install matplotlib seaborn
```

또는 `requirements.txt`에 이미 추가되어 있으므로:
```bash
pip install -r requirements.txt
```

## 사용 방법

### 1. 정확도 평가 실행 및 결과 저장

먼저 프론트엔드나 API를 통해 각 클러스터링 방법별로 정확도 평가를 실행합니다.

- 프론트엔드에서 "클러스터링 평가" 모달을 열고 각 방법(빈도수, TF-IDF, FastText, HDBSCAN)을 선택하여 평가를 실행합니다.
- 각 평가가 완료되면 자동으로 `backend/data/evaluation_results/` 폴더에 결과가 저장됩니다:
  - 개별 파일: `evaluation_{method}_{timestamp}.json` (예: `evaluation_simple_20241126_143022.json`)
  - 통합 파일: `evaluation_results_combined.json` (모든 방법의 결과가 누적됨)

### 2. 그래프 생성

#### 방법 1: 자동으로 통합 파일 사용 (권장)
```bash
cd backend
python visualization/create_graphs.py
```
이 명령어는 자동으로 `backend/data/evaluation_results/evaluation_results_combined.json` 파일을 찾아서 사용합니다.

#### 방법 2: 특정 JSON 파일 지정
```bash
cd backend
python visualization/create_graphs.py --results data/evaluation_results/evaluation_results_combined.json
```

#### 방법 3: 출력 디렉토리 지정
```bash
python visualization/create_graphs.py --output my_graphs
```

## 생성되는 그래프

1. **종합 점수 비교 막대 그래프** (`1_overall_score_comparison.png`)
   - 4가지 방법의 종합 점수를 비교
   - 논문의 주요 결과 그래프

2. **평가 지표별 레이더 차트** (`2_radar_chart.png`)
   - 7개 평가 지표를 한눈에 비교
   - 각 방법의 강점/약점 파악

3. **처리 시간 vs 정확도 산점도** (`3_time_vs_accuracy.png`)
   - 속도와 정확도의 트레이드오프 시각화
   - 실용성 분석에 유용

4. **클러스터 수 비교 막대 그래프** (`4_cluster_count_comparison.png`)
   - 각 방법이 생성한 클러스터 수 비교

5. **노이즈 비율 비교 막대 그래프** (`5_noise_ratio_comparison.png`)
   - 미분류 뉴스 비율 비교

6. **실루엣 점수 비교 그래프** (`6_silhouette_comparison.png`)
   - 클러스터링 품질 지표 비교

7. **종합 비교 그래프** (`7_comprehensive_comparison.png`)
   - 6개 지표를 한 화면에 표시
   - 논문 부록에 적합

## 평가 결과 데이터 형식

스크립트는 다음 형식의 JSON 데이터를 기대합니다:

```json
{
  "simple": {
    "overall_score": {"score": 65.2},
    "clustering_quality": {
      "silhouette_score": 0.15,
      "calinski_harabasz_index": 45.2,
      "davies_bouldin_index": 2.8,
      "n_clusters": 25,
      "noise_ratio": 0.0
    },
    "keyword_extraction": {
      "cluster_keyword_accuracy": 0.65
    },
    "topic_consistency": {
      "topic_consistency_score": 0.55
    },
    "performance": {
      "total_processing_time": 8.5,
      "throughput": 117.6
    }
  },
  "tfidf": { ... },
  "fasttext": { ... },
  "news_analyzer": { ... }
}
```

## 저장된 결과 파일 구조

평가 결과는 다음 구조로 저장됩니다:

```
backend/
  data/
    evaluation_results/
      evaluation_simple_20241126_143022.json      # 빈도수 방법 결과
      evaluation_tfidf_20241126_143145.json        # TF-IDF 방법 결과
      evaluation_fasttext_20241126_143310.json   # FastText 방법 결과
      evaluation_news_analyzer_20241126_143520.json  # HDBSCAN 방법 결과
      evaluation_results_combined.json           # 통합 결과 (그래프 생성용)
```

### 통합 파일 구조

`evaluation_results_combined.json` 파일은 다음과 같은 구조를 가집니다:

```json
{
  "simple": {
    "overall_score": {"score": 65.2, "grade": "C"},
    "clustering_quality": {
      "silhouette_score": 0.15,
      "calinski_harabasz_index": 45.2,
      "davies_bouldin_index": 2.8,
      "n_clusters": 25,
      "noise_ratio": 0.0
    },
    "keyword_extraction": {"cluster_keyword_accuracy": 0.65},
    "topic_consistency": {"topic_consistency_score": 0.55},
    "performance": {
      "total_processing_time": 8.5,
      "throughput": 117.6
    }
  },
  "tfidf": { ... },
  "fasttext": { ... },
  "news_analyzer": { ... },
  "last_updated": "2024-11-26 14:35:20",
  "last_updated_method": "news_analyzer"
}
```

## 주의사항

- 각 클러스터링 방법별로 평가를 실행하면 통합 파일이 자동으로 업데이트됩니다.
- 모든 방법의 평가를 완료한 후 그래프를 생성하면 더 정확한 비교가 가능합니다.
- 통합 파일이 없거나 일부 방법만 평가된 경우, 평가된 방법만 그래프에 표시됩니다.

## 그래프 커스터마이징

`create_graphs.py` 파일을 수정하여:
- 색상 변경 (`method_colors` 딕셔너리)
- 그래프 크기 조정 (`figsize` 파라미터)
- 폰트 크기 조정
- 스타일 변경 (Seaborn 스타일)

## 출력 형식

각 그래프는 두 가지 형식으로 저장됩니다:
- **PNG**: 고해상도 (300 DPI) - 프레젠테이션용
- **PDF**: 벡터 형식 - 논문 삽입용 (LaTeX 등)

## 논문 삽입 팁

1. **레이더 차트**: 방법론 비교에 유용
2. **산점도**: 속도-정확도 트레이드오프 설명
3. **막대 그래프**: 개별 지표 비교
4. **종합 비교**: 부록이나 보조 자료로 활용

## 문제 해결

### 한글 폰트 깨짐
Windows: `Malgun Gothic` 자동 사용
Linux/Mac: `NanumGothic` 또는 시스템 폰트 사용

### 그래프가 비어있음
평가 결과 데이터 형식 확인 필요

### 색상이 마음에 안 듦
`method_colors` 딕셔너리에서 색상 코드 변경

