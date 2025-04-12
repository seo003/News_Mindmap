import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [loading, setLoading] = useState(true);
  const [keywords, setKeywords] = useState([]);


  useEffect(() => {
    // 로딩화면
    setTimeout(() => {
      setLoading(false);
    }, 80000);

    // Flask API에서 데이터 전달
    fetch("http://localhost:5000/api/keywords")
      .then(response => response.json())
      .then(data => {
        setKeywords(data);
      })
      .catch(error => {
        console.error("Error fetching data:", error);
      });
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
      <div>
        {Object.keys(keywords).map((key) => (
          <div key={key}>
            <h2>{key}</h2> {/* 학교 이름/키워드 */}
            <div>
              <h3>뉴스</h3>
              <ul>
                {keywords[key].뉴스.map((newsItem, index) => (
                  <li key={index}>
                    <a href={newsItem.link} target="_blank" rel="noopener noreferrer">
                      {newsItem.title}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
            <div>
              <h3>소분류</h3>
              <ul>
                {keywords[key].소분류.map((subKeyword, index) => (
                  <li key={index}>{subKeyword}</li>
                ))}
              </ul>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default App;
