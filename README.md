# News Mindmap

실시간 뉴스 크롤링 및 클러스터링을 통한 인터랙티브 마인드맵 시각화 시스템

## 🎯 프로젝트 개요

News Mindmap은 다양한 뉴스 소스에서 수집한 뉴스를 자동으로 클러스터링하고, 계층적 마인드맵 형태로 시각화하는 웹 애플리케이션입니다. 

### 핵심 목표
- **자동 뉴스 분류**: 여러 클러스터링 알고리즘을 통해 뉴스를 주제별로 자동 분류
- **대학교 뉴스 분리**: 대학교 관련 뉴스를 자동으로 식별하여 별도 카테고리로 분류
- **인터랙티브 시각화**: React Force-directed graph 기반 마인드맵으로 뉴스 탐색
- **성능 평가**: 다양한 클러스터링 방법의 정확도와 성능을 비교 평가

## ✨ 주요 기능

### 1. 뉴스 크롤링
- **13개 뉴스 소스** 지원
  - 조선일보, 연합뉴스, 경향신문, 중앙일보 등
  - 대학교 언론사 (인천대, 경인일보, 교수신문 등)
- **멀티스레딩** 기반 병렬 크롤링
- **PostgreSQL** 데이터베이스 연동

### 2. 클러스터링 및 분석
- **4가지 클러스터링 방법** 제공
  1. **빈도 기반** (Simple): 키워드 빈도 기반 빠른 분류
  2. **TF-IDF**: TF-IDF 벡터와 코사인 유사도 기반
  3. **FastText**: FastText 임베딩 + K-Means
  4. **HDBSCAN**: Sentence Transformer + UMAP + HDBSCAN
- **자동 키워드 추출**: KoNLPy, KeyBERT 기반
- **계층적 분류**: 대분류 → 중분류 → 뉴스 구조

### 3. 마인드맵 시각화
- **React Force-directed graph** 기반 인터랙티브 시각화
- **3단계 계층 구조**
  - Level 0: 중앙 노드
  - Level 1: 대분류 
  - Level 2: 중분류 
- **반응형 디자인**: 화면 크기에 따른 자동 조정
- **줌/팬 기능**: 마우스 휠, 드래그, 터치 제스처 지원

### 4. 정확도 평가
- **종합 평가 시스템** 
  - 클러스터링 품질 : Silhouette, CH Index, DB Index
  - 키워드 추출 정확도
  - Topic Consistency : KeyBERT 기반 의미 유사도
  - 성능 : 처리 시간, 처리량

## 🏗️ 시스템 아키텍처

```
┌─────────────────┐
│  뉴스 크롤러     │ → PostgreSQL DB
│  (13개 소스)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Flask API      │ ← 클러스터링 분석
│  (Backend)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  React Frontend │ → Force Graph 시각화
│  (MindMap)      │
└─────────────────┘
```

### 데이터 흐름
1. **크롤링**: 여러 뉴스 소스에서 뉴스 수집 → DB 저장
2. **전처리**: 제목 정제, 형태소 분석, 불용어 제거
3. **대학교 분류**: 정규표현식 기반 대학교 뉴스 자동 분리
4. **클러스터링**: 선택한 방법으로 뉴스 클러스터링
5. **키워드 추출**: 클러스터별 대표 키워드 추출
6. **시각화**: 계층 구조를 마인드맵으로 렌더링

## 🛠️ 기술 스택

### Backend
- **Python 3.x**
- **Flask**: RESTful API 서버
- **PostgreSQL**: 뉴스 데이터 저장
- **클러스터링 라이브러리**:
  - `scikit-learn`: K-Means, 평가 지표
  - `hdbscan`: 밀도 기반 클러스터링
  - `sentence-transformers`: 문장 임베딩
  - `umap-learn`: 차원 축소
- **자연어 처리**:
  - `konlpy`: 한국어 형태소 분석
  - `keybert`: 키워드 추출
  - `fasttext`: FastText 임베딩

### Frontend
- **React 19.x**
- **react-force-graph**: Force-directed graph 시각화
- **D3.js**: 그래프 레이아웃 계산
- **HTML5 Canvas**: 고성능 렌더링


## 📦 설치 및 실행

### 사전 요구사항
- Python 3.8 이상
- Node.js 16 이상
- PostgreSQL 데이터베이스

### 1. 저장소 클론
```bash
git clone <repository-url>
cd News_Mindmap
```

### 2. Backend 설정

#### 의존성 설치
```bash
cd backend
pip install -r requirements.txt
```

#### 환경 변수 설정
`backend/data.env` 파일 생성:
```env
dbhost=localhost
dbport=5432
dbname=news_db
dbuser=your_username
dbpass=your_password
```

#### (선택) FastText 모델 다운로드
```bash
cd models
# FastText 한국어 모델 다운로드
# wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ko.300.bin.gz
# gunzip cc.ko.300.bin.gz
```

#### Backend 실행
```bash
python app.py
```
서버는 `http://localhost:5000`에서 실행됩니다.

### 3. Frontend 설정

#### 의존성 설치
```bash
cd frontend
npm install
```

#### Frontend 실행
```bash
npm start
```
애플리케이션은 `http://localhost:3000`에서 실행됩니다.

### 4. 통합 실행 (개발 환경)
```bash
cd frontend
npm run dev  # concurrently로 프론트엔드와 백엔드 동시 실행
```

---

## 📁 프로젝트 구조

```
News_Mindmap/
├── backend/                    # Flask 백엔드
│   ├── app.py                  # Flask API 서버 메인
│   ├── crawlAll.py             # 통합 크롤러
│   ├── requirements.txt        # Python 의존성
│   ├── data.env                # 환경 변수 (DB 설정)
│   │
│   ├── analysis/               # 클러스터링 분석 모듈
│   │   ├── news_analyzer.py    # HDBSCAN 클러스터링 (최고 정확도)
│   │   ├── tfidf_clusterer.py  # TF-IDF 기반 클러스터링
│   │   ├── simple_clusterer.py # 빈도 기반 클러스터링
│   │   ├── fasttext_clusterer.py # FastText 기반 클러스터링
│   │   └── accuracy_evaluator.py # 정확도 평가 모듈
│   │
│   ├── crawl/                  # 뉴스 크롤러
│   │   ├── chosun.py           # 조선일보
│   │   ├── yna.py              # 연합뉴스
│   │   ├── incheon.py          # 인천대
│   │   └── ...                 # 기타 10개 소스
│   │
│   ├── config/                 # 설정 파일
│   │   ├── config.py           # 전역 설정
│   │   ├── stopwords.txt       # 불용어 목록
│   │   └── non_university_words.txt # 제외 단어
│   │
│   ├── database/               # 데이터베이스 모듈
│   │   └── news_fetcher.py     # DB 조회 모듈
│   │
│   ├── visualization/          # 평가 그래프 생성
│   │   └── create_graphs.py    # matplotlib 기반 그래프
│   │
│   ├── models/                 # ML 모델 파일
│   │   └── cc.ko.300.bin       # FastText 모델 (선택적)
│   │
│   └── data/                   # 테스트 데이터
│       └── test_news_*.json    # 평가용 뉴스 데이터
│
├── frontend/                   # React 프론트엔드
│   ├── src/
│   │   ├── App.js              # 메인 컴포넌트
│   │   ├── components/
│   │   │   ├── MindMap.js      # 마인드맵 시각화
│   │   │   └── AccuracyModal.js # 정확도 평가 모달
│   │   ├── hooks/
│   │   │   ├── useMindmapHandler.js    # 마인드맵 이벤트 처리
│   │   │   └── useMindmapResponsive.js # 반응형 처리
│   │   └── utils/
│   │       ├── mindmapUtil.js          # 유틸리티 함수
│   │       ├── mindmapNodeRenderer.js  # 노드 렌더링
│   │       ├── mindmapNodeCalculations.js # 노드 계산
│   │       └── mindmapConstants.js     # 상수 정의
│   └── package.json
│
└── README.md                  
```

## 🔬 클러스터링 방법

### 1. 빈도 기반 
- **방식**: 키워드 빈도 기반 할당
- **속도**:  매우 빠름
- **정확도**: 낮음
- **용도**: 빠른 프로토타이핑, 대량 데이터 1차 분류

### 2. TF-IDF 
- **방식**: TF-IDF 벡터 + 코사인 유사도
- **속도**:  중간
- **정확도**:  중간
- **용도**: 속도와 정확도의 균형

### 3. FastText 
- **방식**: FastText 임베딩 + K-Means
- **속도**:  중간
- **정확도**:  가장 높음
- **용도**: 의미 기반 클러스터링 (모델 파일 필요)

### 4. HDBSCAN
- **방식**: Sentence Transformer + UMAP + HDBSCAN + K-Means
- **속도**: 느림
- **정확도**: 중간
- **용도**: 밀도 기반 클러스터링
- **특징**:
  - 노이즈 자동 처리
  - 클러스터 병합 및 재할당
  - 다단계 파이프라인

## 📊 평가 시스템

### 평가 지표

#### 1. 클러스터링 품질
- **Silhouette Score**: 클러스터 응집도와 분리도
- **Calinski-Harabasz Index**: 클러스터 간 분산 비율
- **Davies-Bouldin Index**: 클러스터 간/내 거리 비율

#### 2. 키워드 추출 정확도
- 클러스터 키워드가 사전 정의된 카테고리와 일치하는 비율
- 카테고리: 정치, 경제, 사회, 국제, 법무, 문화, 기술, 교육, 의료, 환경

#### 3. Topic Consistency
- KeyBERT 기반 키워드와 뉴스 본문의 의미 유사도
- Sentence Transformer 임베딩 코사인 유사도

#### 4. 성능 
- **처리 시간** : 전체 분석 소요 시간
- **처리량** : 뉴스/초 단위 처리량

## 📖 사용 가이드

### 기본 사용법

1. **백엔드 및 프론트엔드 실행**
   ```bash
   # 터미널 1: 백엔드
   cd backend
   python app.py
   
   # 터미널 2: 프론트엔드
   cd frontend
   npm start
   ```

2. **마인드맵 탐색**
   - 브라우저에서 `http://localhost:3000` 접속
   - 자동으로 뉴스 분석 및 마인드맵 생성
   - 대분류 노드 클릭: 중분류 확장/축소
   - 중분류 노드 클릭: 관련 뉴스 목록 표시

3. **클러스터링 방법 변경**
   - 프론트엔드에서 드롭다운 선택 (추후 구현 예정)

4. **정확도 평가**
   - 우측 상단 "정확도 평가" 버튼 클릭
   - 클러스터링 방법 및 뉴스 수 선택
   - 평가 결과 확인
