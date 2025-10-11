import React, { useState, useEffect } from 'react';
import './App.css';
import MindMap from './components/MindMap';

function App() {
  const [loading, setLoading] = useState(true); 
  const [keywords, setKeywords] = useState(null); 
  const [error, setError] = useState(null); 

  useEffect(() => {
    const fetchKeywords = async () => {
      try {
        setLoading(true); 
        setError(null); 

        // 뉴스 분석 API 사용 (대학교 분류 + 키워드별 중분류)
        const response = await fetch('http://localhost:5000/api/news_analysis');

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json(); 

        setKeywords(data);
        setLoading(false);

      } catch (error) {
        console.error("Error fetching data:", error);
        console.log("API 호출 실패, 예시 데이터를 사용합니다.");
        
        // API 실패 시 예시 데이터 사용
        const sampleData = [
          {
            majorKeyword: "정치",
            middleKeywords: [
              {
                middleKeyword: "선거",
                relatedNews: [
                  {
                    title: "2024년 총선 결과",
                    url: "https://example.com/news1"
                  },
                  {
                    title: "후보자 공약 분석",
                    url: "https://example.com/news2"
                  }
                ]
              },
              {
                middleKeyword: "정당",
                relatedNews: [
                  {
                    title: "새로운 정당 창당",
                    url: "https://example.com/news3"
                  },
                  {
                    title: "정당 정책 발표",
                    url: "https://example.com/news4"
                  }
                ]
              }
            ],
            otherNews: [
              {
                title: "정치 관련 기타 뉴스",
                url: "https://example.com/news5"
              }
            ]
          },
          {
            majorKeyword: "환경",
            middleKeywords: [
              {
                middleKeyword: "기후변화",
                relatedNews: [
                  {
                    title: "지구온난화 대응책",
                    url: "https://example.com/news6"
                  },
                  {
                    title: "탄소중립 정책",
                    url: "https://example.com/news7"
                  }
                ]
              },
              {
                middleKeyword: "재생에너지",
                relatedNews: [
                  {
                    title: "태양광 발전 확대",
                    url: "https://example.com/news8"
                  }
                ]
              }
            ],
            otherNews: [
              {
                title: "환경 보호 캠페인",
                url: "https://example.com/news9"
              }
            ]
          },
          {
            majorKeyword: "경제",
            middleKeywords: [
              {
                middleKeyword: "부동산",
                relatedNews: [
                  {
                    title: "부동산 시장 동향",
                    url: "https://example.com/news10"
                  },
                  {
                    title: "주택 가격 정책",
                    url: "https://example.com/news11"
                  }
                ]
              }
            ],
            otherNews: [
              {
                title: "경제 성장률 발표",
                url: "https://example.com/news12"
              }
            ]
          }
        ];
        
        setKeywords(sampleData);
        setLoading(false);
        setError(null); // 에러를 null로 설정하여 정상 화면 표시
      }
    };

    fetchKeywords(); 

    // cleanup 함수
    return () => {
        // 필요한 경우 클린업 로직 추가
    };

  }, []);

  // MindMap 컴포넌트에 모든 상태 전달
  return (
    <div style={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
      <h1 style={{ 
        textAlign: 'center', 
        margin: '20px 0', 
        fontSize: '28px',
        fontWeight: '600',
        color: '#1976d2',
        textShadow: '0 2px 4px rgba(25, 118, 210, 0.1)'
      }}>
        뉴스 최신 정보 키워드 마인드맵
      </h1>
      <div style={{ flex: 1, minHeight: 0 }}>
        <MindMap keywords={keywords} loading={loading} error={error} /> 
      </div>
    </div>
  );
}
export default App;