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
        setError(error); 
        setLoading(false); 
        setKeywords(null); 
      }
    };

    fetchKeywords(); 

    // cleanup 함수
    return () => {
        // 필요한 경우 클린업 로직 추가
    };

  }, []);

  // 로딩 중 UI
  if (loading) {
    return (
      <section className="loading">
        <h1 className="loading-title">뉴스정보 가져오는 중...</h1>
        <div className="progress-bar" aria-hidden="true">
          <span className="progress-bar-gauge"></span>
        </div>
      </section>
    );
  }

  // 에러 발생 UI
  if (!loading && error) {
     return (
       <section className="error">
         <h1 className="error-title">데이터를 불러오는데 실패했습니다.</h1>
         <p>{error.message}</p>
       </section>
     );
  }


  // 마인드맵
  if (!loading && !error && keywords) {
    return (
      <div>
        <h1>마인드맵</h1>
        <MindMap keywords={keywords} /> 
      </div>
    );
  }
  
  // 예외 상황 또는 데이터 분석 결과가 빈 경우
   return (
       <div>
           <h1>마인드맵</h1>
           <p>표시할 뉴스 정보가 없습니다.</p> 
       </div>
   );
}
export default App;