import React, { useState, useEffect } from 'react';
import './App.css';
import MindMap from './components/MindMap';

function App() {
  const [loading, setLoading] = useState(true);
  const [keywords, setKeywords] = useState(null);

  useEffect(() => {
    const timer = setTimeout(() => {
      setLoading(false);
    }, 1000);

    // Flask API 호출
    fetch("http://localhost:5000/api/keywords")
      .then(response => response.json())
      .then(data => {
        setKeywords(data);
        setLoading(false); 
      })
      .catch(error => {
        console.error("Error fetching data:", error);
        setLoading(false); 
      });

    return () => clearTimeout(timer); // cleanup
  }, []);

  // 로딩 중일 때
  if (loading || !keywords) {
    return (
      <section className="loading">
        <h1 className="loading-title">뉴스정보 가져오는 중...</h1>
        <div className="progress-bar" aria-hidden="true">
          <span className="progress-bar-gauge"></span>
        </div>
      </section>
    );
  }

  // 마인드맵 출력
  return (
    <div>
      <h1>마인드맵</h1>
      <MindMap keywords={keywords} />
    </div>
  );
}

export default App;
