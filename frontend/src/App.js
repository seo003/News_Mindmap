import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setTimeout(() => {
      setLoading(false);
    }, 5000); 
  }, []);

  // 뉴스 크롤링 및 분석 로딩 화면
  if (loading) {
    return (
      <section class="loading">
      <h1 class="loading-title">뉴스정보 가져오는 중...</h1>
      <div class="progress-bar" aria-hidden="true">
        <span class="progress-bar-gauge"></span>
      </div>
      </section>
    );
  }

  return (
    <div>
      <h1>마인드맵</h1>
    </div>
  );
}

export default App;
