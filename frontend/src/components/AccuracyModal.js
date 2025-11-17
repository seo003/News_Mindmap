import React, { useState, useEffect } from 'react';
import './AccuracyModal.css';

const AccuracyModal = ({ isOpen, onClose }) => {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [limit, setLimit] = useState(1000);
  const [clusteringMethod, setClusteringMethod] = useState(null);
  const [availableMethods, setAvailableMethods] = useState([]);

  // 사용 가능한 클러스터링 방법 목록 가져오기
  useEffect(() => {
    if (isOpen) {
      const fetchMethods = async () => {
        try {
          const response = await fetch('http://localhost:5000/api/clustering_methods');
          if (response.ok) {
            const data = await response.json();
            if (data.success && data.methods) {
              setAvailableMethods(data.methods);
            }
          }
        } catch (error) {
          console.error("클러스터링 방법 목록 가져오기 실패:", error);
        }
      };
      fetchMethods();
    } else {
      // 모달이 닫힐 때 상태 초기화
      setClusteringMethod(null);
      setResult(null);
      setError(null);
    }
  }, [isOpen]);

  const runAccuracyEvaluation = async () => {
    if (!clusteringMethod) {
      setError('클러스터링 방법을 선택해주세요.');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch(
        `http://localhost:5000/api/accuracy?limit=${limit}&method=${clusteringMethod}`
      );
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      if (data.success) {
        setResult(data.data);
      } else {
        setError(data.error || '정확도 평가에 실패했습니다.');
      }
    } catch (err) {
      console.error('정확도 평가 오류:', err);
      setError(`정확도 평가 중 오류가 발생했습니다: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const runSummaryEvaluation = async () => {
    if (!clusteringMethod) {
      setError('클러스터링 방법을 선택해주세요.');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch(
        `http://localhost:5000/api/accuracy/summary?limit=${limit}&method=${clusteringMethod}`
      );
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      if (data.success) {
        setResult(data.data);
      } else {
        setError(data.error || '정확도 평가 요약에 실패했습니다.');
      }
    } catch (err) {
      console.error('정확도 평가 요약 오류:', err);
      setError(`정확도 평가 요약 중 오류가 발생했습니다: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const getGradeColor = (grade) => {
    switch (grade) {
      case 'A+': return '#4caf50';
      case 'A': return '#8bc34a';
      case 'B+': return '#cddc39';
      case 'B': return '#ffeb3b';
      case 'C+': return '#ffc107';
      case 'C': return '#ff9800';
      case 'D': return '#ff5722';
      default: return '#9e9e9e';
    }
  };

  const getGradeEmoji = (grade) => {
    switch (grade) {
      case 'A+': return '🏆';
      case 'A': return '🥇';
      case 'B+': return '🥈';
      case 'B': return '🥉';
      case 'C+': return '👍';
      case 'C': return '👌';
      case 'D': return '⚠️';
      default: return '❓';
    }
  };

  if (!isOpen) return null;

  return (
    <div className="accuracy-modal-overlay">
      <div className="accuracy-modal">
        {/* 헤더 */}
        <div className="accuracy-modal-header">
          <div className="accuracy-modal-title">
            <i className="fas fa-chart-line"></i>
            <h3>백엔드 정확도 평가</h3>
          </div>
          <button onClick={onClose} className="accuracy-modal-close">
            <i className="fas fa-times"></i>
          </button>
        </div>

        {/* 설정 섹션 */}
        <div className="accuracy-settings">
          <div className="setting-group">
            <label htmlFor="limit">분석할 뉴스 개수:</label>
            <input
              id="limit"
              type="number"
              min="10"
              max="5000"
              value={limit}
              onChange={(e) => setLimit(parseInt(e.target.value) || 1000)}
              disabled={loading}
            />
          </div>

          {/* 클러스터링 방법 선택 */}
          <div className="setting-group">
            <label>클러스터링 방법:</label>
            <div className="clustering-method-buttons" style={{
              display: 'flex',
              gap: '8px',
              flexWrap: 'wrap',
              marginTop: '8px',
              alignItems: 'stretch'
            }}>
              {(availableMethods.length > 0 ? availableMethods : [
                { id: 'tfidf', name: 'TF-IDF' },
                { id: 'fasttext', name: 'FastText' },
                { id: 'simple', name: '빈도수' },
                { id: 'news_analyzer', name: 'HDBSCAN' }
              ]).map((method) => {
                const isSelected = clusteringMethod === method.id;
                return (
                  <button
                    key={method.id}
                    onClick={() => !loading && setClusteringMethod(method.id)}
                    disabled={loading}
                    style={{
                      padding: '8px 14px',
                      borderRadius: '8px',
                      border: isSelected ? '2px solid #1976d2' : '2px solid #dee2e6',
                      fontSize: '13px',
                      fontWeight: isSelected ? '600' : '500',
                      color: isSelected ? '#fff' : '#495057',
                      background: isSelected 
                        ? 'linear-gradient(135deg, #1976d2 0%, #1565c0 100%)'
                        : '#fff',
                      cursor: loading ? 'not-allowed' : 'pointer',
                      outline: 'none',
                      transition: 'all 0.2s ease',
                      boxShadow: isSelected 
                        ? '0 3px 10px rgba(25, 118, 210, 0.3)'
                        : '0 1px 3px rgba(0, 0, 0, 0.1)',
                      transform: isSelected ? 'translateY(-1px)' : 'none',
                      whiteSpace: 'nowrap',
                      opacity: loading ? 0.6 : 1,
                      flexShrink: 0,
                      height: '36px',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center'
                    }}
                    onMouseEnter={(e) => {
                      if (!loading && !isSelected) {
                        e.target.style.backgroundColor = '#e3f2fd';
                        e.target.style.borderColor = '#90caf9';
                        e.target.style.transform = 'translateY(-1px)';
                      }
                    }}
                    onMouseLeave={(e) => {
                      if (!loading && !isSelected) {
                        e.target.style.backgroundColor = '#fff';
                        e.target.style.borderColor = '#dee2e6';
                        e.target.style.transform = 'none';
                      }
                    }}
                  >
                    {method.name}
                  </button>
                );
              })}
            </div>
          </div>
          
          <div className="button-group">
            <button
              onClick={runSummaryEvaluation}
              disabled={loading || !clusteringMethod}
              className="btn btn-primary"
            >
              <i className="fas fa-chart-bar"></i>
              간단 평가
            </button>
            <button
              onClick={runAccuracyEvaluation}
              disabled={loading || !clusteringMethod}
              className="btn btn-secondary"
            >
              <i className="fas fa-chart-line"></i>
              상세 평가
            </button>
          </div>
        </div>

        {/* 로딩 상태 */}
        {loading && (
          <div className="accuracy-loading">
            <div className="loading-spinner">
              <i className="fas fa-spinner fa-spin"></i>
            </div>
            <p>정확도 평가 중...</p>
          </div>
        )}

        {/* 에러 상태 */}
        {error && (
          <div className="accuracy-error">
            <i className="fas fa-exclamation-triangle"></i>
            <p>{error}</p>
          </div>
        )}

        {/* 결과 표시 */}
        {result && !loading && (
          <div className="accuracy-result">
            {/* 요약 결과 */}
            {result.overall_score && (
              <div className="result-summary">
                <div className="score-display">
                  <div className="score-circle" style={{ 
                    background: `conic-gradient(${getGradeColor(result.grade)} 0deg ${((typeof result.overall_score === 'number' ? result.overall_score : result.overall_score.score || 0) / 100) * 360}deg, #e0e0e0 ${((typeof result.overall_score === 'number' ? result.overall_score : result.overall_score.score || 0) / 100) * 360}deg 360deg)`
                  }}>
                    <div className="score-inner">
                      <span className="score-number">{(typeof result.overall_score === 'number' ? result.overall_score : result.overall_score.score || 0).toFixed(1)}</span>
                      <span className="score-grade">{getGradeEmoji(result.grade)} {result.grade}</span>
                    </div>
                  </div>
                </div>
                
                <div className="score-details">
                  <h4>종합 점수</h4>
                  <p>분석된 뉴스: {result.news_count}개</p>
                  <p>평가 시간: {result.timestamp}</p>
                </div>
              </div>
            )}

            {/* 상세 점수 */}
            {result.clustering_score !== undefined && (
              <div className="detailed-scores">
                <h4>세부 점수</h4>
                <div className="score-bars">
                  <div className="score-bar">
                    <div className="score-label">
                      <i className="fas fa-project-diagram"></i>
                      클러스터링 품질
                    </div>
                    <div className="score-value">{(result.clustering_score || result.overall_score?.components?.clustering || 0).toFixed(1)}/30</div>
                    <div className="progress-bar">
                      <div 
                        className="progress-fill" 
                        style={{ width: `${((result.clustering_score || result.overall_score?.components?.clustering || 0) / 30) * 100}%` }}
                      ></div>
                    </div>
                  </div>

                  <div className="score-bar">
                    <div className="score-label">
                      <i className="fas fa-key"></i>
                      키워드 추출
                    </div>
                    <div className="score-value">{(result.keyword_score || result.overall_score?.components?.keyword_extraction || 0).toFixed(1)}/40</div>
                    <div className="progress-bar">
                      <div 
                        className="progress-fill" 
                        style={{ width: `${((result.keyword_score || result.overall_score?.components?.keyword_extraction || 0) / 40) * 100}%` }}
                      ></div>
                    </div>
                  </div>

                  <div className="score-bar">
                    <div className="score-label">
                      <i className="fas fa-tachometer-alt"></i>
                      성능
                    </div>
                    <div className="score-value">{(result.performance_score || result.overall_score?.components?.performance || 0).toFixed(1)}/30</div>
                    <div className="progress-bar">
                      <div 
                        className="progress-fill" 
                        style={{ width: `${((result.performance_score || result.overall_score?.components?.performance || 0) / 30) * 100}%` }}
                      ></div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* 상세 결과 (상세 평가 시) */}
            {result.clustering_quality && (
              <div className="detailed-results">
                <h4>상세 분석 결과</h4>
                
                {result.clustering_quality.error ? (
                  <p className="error-text">클러스터링 분석 실패: {result.clustering_quality.error}</p>
                ) : (
                  <div className="clustering-details">
                    <h5>클러스터링 품질</h5>
                    <div className="detail-grid">
                      <div className="detail-item">
                        <span className="detail-label">클러스터 수:</span>
                        <span className="detail-value">{result.clustering_quality?.n_clusters || 0}개</span>
                      </div>
                      <div className="detail-item">
                        <span className="detail-label">노이즈 비율:</span>
                        <span className="detail-value">{((result.clustering_quality?.noise_ratio || 0) * 100).toFixed(1)}%</span>
                      </div>
                      <div className="detail-item">
                        <span className="detail-label">평균 클러스터 크기:</span>
                        <span className="detail-value">{(result.clustering_quality?.avg_cluster_size || 0).toFixed(1)}</span>
                      </div>
                      <div className="detail-item">
                        <span className="detail-label">실루엣 점수:</span>
                        <span className="detail-value">{result.clustering_quality?.silhouette_score?.toFixed(3) || 'N/A'}</span>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default AccuracyModal;
