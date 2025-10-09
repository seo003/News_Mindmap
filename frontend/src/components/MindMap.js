import React, { useState, useRef, useEffect, useCallback, useMemo } from "react";
import 'aframe';
import { ForceGraph2D } from 'react-force-graph';
import '../styles/MindMap.css';

import { useMindmapHandler } from "../hooks/useMindmapHandler";
import {
  generateInitialMindMapData,
  parseNodeId,
  calculateNewsCountsByMajor,
} from "../utils/mindmapUtil";
import { renderNode, renderNodePointerArea } from "../utils/mindmapNodeRenderer";
import { useMindmapResponsive, getInitialZoomLevel } from "../hooks/useMindmapResponsive";
import {
  ZOOM_CONFIG,
  FORCE_CONFIG,
  LINK_COLORS,
  LINK_WIDTH,
  ANIMATION_DURATION,
  NODE_LEVELS,
  NEWS_TITLE_MAX_LENGTH,
} from "../utils/mindmapConstants";

/**
 * 줌 컨트롤 컴포넌트
 */
const MindmapZoomControls = ({ onZoomIn, onZoomOut, onReset }) => {
  return (
    <div className="zoom-controls">
      <button
        onClick={onZoomIn}
        className="zoom-button"
        aria-label="확대"
      >
        &#43;
      </button>
      <button
        onClick={onZoomOut}
        className="zoom-button"
        aria-label="축소"
      >
        &#45;
      </button>
      <button
        onClick={onReset}
        className="zoom-button reset"
        aria-label="초기화"
      >
        Reset
      </button>
    </div>
  );
};

/**
 * 뉴스 패널 컴포넌트
 */
const MindmapNewsPanel = ({ newsList, nodeId, onClose }) => {
  if (!newsList || !nodeId) return null;

  const parsed = parseNodeId(nodeId);
  const keyword = parsed?.middleKeyword || parsed?.majorKeyword || '뉴스';

  return (
    <div className="news-panel">
      {/* 헤더 */}
      <div className="news-panel-header">
        <div className="news-panel-title">
          <div className="news-panel-indicator"></div>
          <h4>
            {`'${keyword}' 관련 뉴스`}
            <span className="news-count-badge">
              {newsList?.length || 0}
            </span>
          </h4>
        </div>
        <button
          onClick={onClose}
          className="news-panel-close"
        >
          ×
        </button>
      </div>

      {/* 뉴스 리스트 */}
      <div className="news-list">
        {newsList.length === 0 ? (
          <p className="no-news-message">
            관련 뉴스가 없습니다.
          </p>
        ) : (
          <div>
            {newsList.map((news, index) => (
              news && news.link && news.title ? (
                <div 
                  key={index} 
                  className="news-item"
                  onClick={() => window.open(news.link, '_blank')}
                >
                  <a 
                    href={news.link} 
                    target="_blank" 
                    rel="noopener noreferrer" 
                    className="news-link"
                  >
                    {news.title.length > NEWS_TITLE_MAX_LENGTH 
                      ? `${news.title.substring(0, NEWS_TITLE_MAX_LENGTH)}...` 
                      : news.title}
                  </a>
                </div>
              ) : (
                <div key={index} className="invalid-news-item">
                  유효하지 않은 뉴스 항목
                </div>
              )
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

/**
 * 뉴스 마인드맵 컴포넌트
 * 
 * @param {Object} props - 컴포넌트 속성
 * @param {Array} props.keywords - 마인드맵 데이터
 * @param {boolean} props.loading - 로딩 상태 (선택사항)
 * @param {Object} props.error - 에러 객체 (선택사항)
 */
const MindMap = ({ keywords, loading = false, error = null }) => {
  // 초기 마인드맵 데이터 생성 (훅 호출 전에 early return 불가)
  const initialData = keywords ? generateInitialMindMapData(keywords) : { nodes: [], links: [] };
  
  // 마인드맵 그래프 데이터 상태
  const [graphData, setGraphData] = useState(initialData);
  
  // 노드 확장 상태
  const [expandedNodeIds, setExpandedNodeIds] = useState(new Set());
  
  // 사용자가 수동으로 줌 조정했는지 추적
  const [userZoomed, setUserZoomed] = useState(false);  
  
  // Force-graph 컴포넌트 참조
  const fgRef = useRef();
  
  // 컨테이너 DOM 참조
  const containerRef = useRef();

  // 반응형 화면 크기 훅
  const { dimensions, isMobile, isSmallMobile, isVerySmallMobile } = useMindmapResponsive(containerRef);

  // 화면 크기별 줌 레벨 적용
  const getInitialZoom = useCallback(() => {
    return getInitialZoomLevel(dimensions.width || window.innerWidth);
  }, [dimensions.width]);

  // 초기 줌 레벨 계산
  const initialZoom = getInitialZoom();
  
  // 현재 줌 레벨 상태
  const [currentZoom, setCurrentZoom] = useState(initialZoom);
  
  // 뉴스 개수 데이터 계산
  const newsCountsByMajor = calculateNewsCountsByMajor(keywords);
  
  // 뉴스 개수 순으로 정렬된 대분류 목록 생성
  const sortedMajors = useMemo(() => {
    return Object.entries(newsCountsByMajor)
      .sort(([,a], [,b]) => b - a)
      .map(([major, count]) => ({ major, count }));
  }, [newsCountsByMajor]);
  
  // 각 대분류의 순위 계산
  const majorRankings = useMemo(() => {
    const rankings = {};
    sortedMajors.forEach((item, index) => {
      rankings[item.major] = index;
    });
    return rankings;
  }, [sortedMajors]);

  // 마인드맵 핸들러 훅 사용
  const {
    handleNodeClick,
    selectedNews,
    setSelectedNews,
    selectedNodeIdForNews,
    setSelectedNodeIdForNews,
  } = useMindmapHandler({
    keywords,
    fgRef,
    graphData, 
    setGraphData,
    expandedNodeIds, 
    setExpandedNodeIds,
  });

  /**
   * 줌 컨트롤 함수들
   */
  const applyZoom = useCallback((newZoom) => {
    if (fgRef.current) {
      try {
        fgRef.current.zoom(newZoom);
        setTimeout(() => {
          if (fgRef.current) {
            fgRef.current.zoom(newZoom);
          }
        }, ANIMATION_DURATION.ZOOM_RETRY);
        setCurrentZoom(newZoom);
        setUserZoomed(true);
      } catch (error) {
        console.error('줌 적용 중 오류:', error);
      }
    } else {
      console.warn('fgRef.current가 null입니다');
    }
  }, []);

  const handleZoomIn = useCallback(() => {
    const newZoom = Math.min(currentZoom * ZOOM_CONFIG.MULTIPLIER, ZOOM_CONFIG.MAX);
    applyZoom(newZoom);
  }, [currentZoom, applyZoom]);

  const handleZoomOut = useCallback(() => {
    const newZoom = Math.max(currentZoom / ZOOM_CONFIG.MULTIPLIER, ZOOM_CONFIG.MIN);
    applyZoom(newZoom);
  }, [currentZoom, applyZoom]);

  const handleReset = useCallback(() => {
    if (fgRef.current) {
      fgRef.current.centerAt(0, 0, ANIMATION_DURATION.CENTER_AT);
      fgRef.current.zoom(initialZoom);
      setCurrentZoom(initialZoom);
      setUserZoomed(false); 
    }
  }, [initialZoom]);

  /**
   * 키워드 데이터 변경 시 마인드맵 초기화
   */
  useEffect(() => {
    const newInitialData = keywords ? generateInitialMindMapData(keywords) : { nodes: [], links: [] };
    
    setGraphData(newInitialData);
    setExpandedNodeIds(new Set());
    setSelectedNews(null);
    setSelectedNodeIdForNews(null);
    
    const initialZoomLevel = getInitialZoom();
    setCurrentZoom(initialZoomLevel);
    setUserZoomed(false);
    
    if (fgRef.current) {
      fgRef.current.centerAt(0, 0, ANIMATION_DURATION.CENTER_AT);
      fgRef.current.zoom(initialZoomLevel);
    }
  }, [keywords, setGraphData, setExpandedNodeIds, setSelectedNews, setSelectedNodeIdForNews, getInitialZoom]);

  /**
   * 화면 크기 변경 시 줌 레벨 자동 조정
   */
  useEffect(() => {
    if (fgRef.current && dimensions.width > 0 && dimensions.height > 0 && !userZoomed) {
      const zoomLevel = getInitialZoom();
      
      if (Math.abs(currentZoom - zoomLevel) > 0.1) {
        const timeoutId = setTimeout(() => {
          if (fgRef.current) {
            fgRef.current.centerAt(0, 0, ANIMATION_DURATION.CENTER_AT);
            fgRef.current.zoom(zoomLevel);
            setCurrentZoom(zoomLevel);
          }
        }, ANIMATION_DURATION.ZOOM_DELAY);

        return () => clearTimeout(timeoutId);
      }
    }
  }, [dimensions.width, dimensions.height, getInitialZoom, currentZoom, userZoomed]);

  /**
   * 화면 크기 변경 시 force-graph 설정 업데이트
   */
  useEffect(() => {
    const currentWidth = dimensions.width || window.innerWidth;
    const currentHeight = dimensions.height || window.innerHeight;
    
    if (fgRef.current && currentWidth > 0 && currentHeight > 0) {
      // 노드 간격 조정
      let baseStrength = FORCE_CONFIG.CHARGE_STRENGTH.DEFAULT;
      if (isVerySmallMobile) {
        baseStrength = FORCE_CONFIG.CHARGE_STRENGTH.VERY_SMALL;
      } else if (isSmallMobile) {
        baseStrength = FORCE_CONFIG.CHARGE_STRENGTH.SMALL;
      } else if (isMobile) {
        baseStrength = FORCE_CONFIG.CHARGE_STRENGTH.MEDIUM;
      }
      fgRef.current.d3Force('charge').strength(baseStrength);

      // 링크 거리 조정
      let baseLinkDistance = FORCE_CONFIG.LINK_DISTANCE.DEFAULT;
      if (isVerySmallMobile) {
        baseLinkDistance = FORCE_CONFIG.LINK_DISTANCE.VERY_SMALL;
      } else if (isSmallMobile) {
        baseLinkDistance = FORCE_CONFIG.LINK_DISTANCE.SMALL;
      } else if (isMobile) {
        baseLinkDistance = FORCE_CONFIG.LINK_DISTANCE.MEDIUM;
      }
      fgRef.current.d3Force('link').distance(baseLinkDistance);
      
      // 중력 조정
      const centerStrength = isMobile 
        ? FORCE_CONFIG.CENTER_STRENGTH.MOBILE 
        : FORCE_CONFIG.CENTER_STRENGTH.DESKTOP;
      fgRef.current.d3Force('center').strength(centerStrength);
    }
  }, [dimensions.width, dimensions.height, isMobile, isSmallMobile, isVerySmallMobile]);

  /**
   * 노드 렌더링 함수
   */
  const nodeCanvasObject = useCallback((node, ctx, scale) => {
    const screenWidth = dimensions.width || window.innerWidth;
    renderNode(
      node,
      ctx,
      scale,
      screenWidth,
      newsCountsByMajor,
      majorRankings,
      sortedMajors,
      keywords
    );
  }, [dimensions.width, newsCountsByMajor, majorRankings, sortedMajors, keywords]);

  /**
   * 노드 클릭 영역 렌더링 함수
   */
  const nodePointerAreaCanvasObject = useCallback((node, ctx, scale) => {
    const screenWidth = dimensions.width || window.innerWidth;
    renderNodePointerArea(
      node,
      ctx,
      scale,
      screenWidth,
      newsCountsByMajor,
      keywords
    );
  }, [dimensions.width, newsCountsByMajor, keywords]);

  /**
   * 링크 두께 계산
   */
  const getLinkWidth = useCallback((link) => {
    const sourceId = typeof link.source === 'object' ? link.source?.id : link.source;
    const targetId = typeof link.target === 'object' ? link.target?.id : link.target;

    if (!sourceId || !targetId) {
      return LINK_WIDTH.DEFAULT;
    }

    const targetNode = graphData.nodes.find(n => n.id === targetId);

    if (targetNode) {
      if (targetNode.level === NODE_LEVELS.MAJOR) return LINK_WIDTH.MAJOR;
      if (targetNode.level === NODE_LEVELS.MIDDLE) return LINK_WIDTH.MIDDLE;
    }
    return LINK_WIDTH.DEFAULT;
  }, [graphData.nodes]);

  /**
   * 링크 렌더링 함수
   */
  const linkCanvasObject = useCallback((link, ctx, scale) => {
    const source = typeof link.source === 'object' 
      ? link.source 
      : graphData.nodes.find(node => node.id === link.source); 
    const target = typeof link.target === 'object' 
      ? link.target 
      : graphData.nodes.find(node => node.id === link.target); 

    if (!source || !target || !source.id || !target.id) {
      return;
    }
    
    const start = { x: source.x, y: source.y };
    const end = { x: target.x, y: target.y };

    if (!isFinite(start.x) || !isFinite(start.y) || !isFinite(end.x) || !isFinite(end.y)) {
      return;
    }

    // 링크 그라데이션 효과
    const gradient = ctx.createLinearGradient(start.x, start.y, end.x, end.y);
    gradient.addColorStop(0, LINK_COLORS.GRADIENT_START);
    gradient.addColorStop(1, LINK_COLORS.GRADIENT_END);

    ctx.beginPath();
    ctx.moveTo(start.x, start.y);
    ctx.lineTo(end.x, end.y);
    ctx.strokeStyle = gradient;
    ctx.lineWidth = link.width || LINK_WIDTH.MAJOR;
    ctx.stroke();
  }, [graphData.nodes]);

  /**
   * 마인드맵 컴포넌트 렌더링
   */
  
  // 로딩 중 UI
  if (loading) {
    return (
      <section className="loading">
        <h1 className="loading-title">뉴스 제목 분석 중...</h1>
        <div className="progress-bar" aria-hidden="true">
          <span className="progress-bar-gauge"></span>
        </div>
      </section>
    );
  }

  // 에러 발생 UI
  if (error) {
    return (
      <section className="loading">
        <h1 className="loading-title error-message">
          데이터를 불러오는데 실패했습니다.
        </h1>
        <p className="error-description">
          잠시 후 다시 시도해주세요
        </p>
      </section>
    );
  }

  // 데이터가 없는 경우 (로딩 중이면 빈 화면으로 처리)
  if ((!keywords || keywords.length === 0) && !loading) {
    return (
      <section className="loading">
        <h1 className="loading-title">표시할 뉴스 정보가 없습니다.</h1>
      </section>
    );
  }
  
  return (
    <div 
      ref={containerRef}
      className="mindmap-container"
    >
      <div className="mindmap-graph">
        {dimensions.width > 0 && dimensions.height > 0 ? (
          <ForceGraph2D
          key={`${dimensions.width}x${dimensions.height}`}
          ref={fgRef}
          graphData={graphData}
          width={dimensions.width}
          height={dimensions.height}
          nodeLabel="label"
          nodeCanvasObject={nodeCanvasObject}
          nodePointerAreaCanvasObject={nodePointerAreaCanvasObject}
          onNodeClick={handleNodeClick}
          enableNodeDrag={true}
          enableZoomPanInteraction={true}
          onNodeHover={(node) => {
            if (node) {
              document.body.style.cursor = 'pointer';
            } else {
              document.body.style.cursor = 'default';
            }
          }}
            linkWidth={getLinkWidth}
            linkCanvasObject={linkCanvasObject}
          />
        ) : (
          <div className="loading-placeholder">
            화면 크기 계산 중...
          </div>
        )}
      </div>

      {/* 줌 컨트롤 버튼 */}
      <MindmapZoomControls
        onZoomIn={handleZoomIn}
        onZoomOut={handleZoomOut}
        onReset={handleReset}
      />

      {/* 뉴스 목록 */}
      {selectedNews !== null && selectedNodeIdForNews && (
        <MindmapNewsPanel
          newsList={selectedNews}
          nodeId={selectedNodeIdForNews}
          onClose={() => {
            setSelectedNews(null);
            setSelectedNodeIdForNews(null);
          }}
        />
      )}
    </div>
  );
};

export default MindMap;
