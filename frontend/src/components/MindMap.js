import React, { useState, useRef, useEffect, useCallback } from "react";
import 'aframe';
import { ForceGraph2D } from 'react-force-graph';
import './MindMap.css';

import { useMindmapHandler } from "./useMindmapHandler";
import {
  generateInitialMindMapData,
  parseNodeId,
  calculateNewsCountsByMajor,
  calculateColorIntensity,
  MAJOR_KEY,
  MIDDLE_LIST_KEY,
  MIDDLE_ITEM_KEY,
  RELATED_NEWS_KEY,
} from "../utils/mindmapUtil";

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
  
  // 화면 크기 상태 
  const [dimensions, setDimensions] = useState({
    width: 0, 
    height: 0,
  });


  // 화면 크기별 줌 레벨 적용
  const getInitialZoom = useCallback(() => {
    const viewportWidth = dimensions.width || window.innerWidth;
    let zoomLevel;
    
    if (viewportWidth <= 320) {
      zoomLevel = 1.9; 
    } else if (viewportWidth <= 480) {
      zoomLevel = 2.2; 
    } else if (viewportWidth <= 768) {
      zoomLevel = 2.6; 
    } else if (viewportWidth <= 1024) {
      zoomLevel = 2.9; 
    } else {
      zoomLevel = 3.4; 
    }
    
    return zoomLevel;
  }, [dimensions.width]);

  // 초기 줌 레벨 계산
  const initialZoom = getInitialZoom();

  // 노드 확장 상태
  const [expandedNodeIds, setExpandedNodeIds] = useState(new Set());
  
  // 현재 줌 레벨 상태
  const [currentZoom, setCurrentZoom] = useState(initialZoom);
  
  // 사용자가 수동으로 줌 조정했는지 추적
  const [userZoomed, setUserZoomed] = useState(false);  
  
  /**
   * 줌 컨트롤 함수
   */
  
  // 줌 인 함수: 마인드맵 확대
  const handleZoomIn = useCallback(() => {
    const newZoom = Math.min(currentZoom * 1.2, 10); // 최대 줌 레벨 제한
    
    if (fgRef.current) {
      try {
        fgRef.current.zoom(newZoom);
        setTimeout(() => {
          if (fgRef.current) {
            fgRef.current.zoom(newZoom);
          }
        }, 100);
        setCurrentZoom(newZoom);
        setUserZoomed(true);
      } catch (error) {
        console.error('줌인 적용 중 오류:', error);
      }
    } else {
      console.warn('fgRef.current가 null입니다');
    }
  }, [currentZoom]);

  // 줌 아웃 함수: 마인드맵 축소
  const handleZoomOut = useCallback(() => {
    const newZoom = Math.max(currentZoom / 1.2, 0.1); // 최소 줌 레벨 제한
    
    if (fgRef.current) {
      try {
        fgRef.current.zoom(newZoom);
        setTimeout(() => {
          if (fgRef.current) {
            fgRef.current.zoom(newZoom);
          }
        }, 100);
        setCurrentZoom(newZoom);
        setUserZoomed(true);
      } catch (error) {
        console.error('줌아웃 적용 중 오류:', error);
      }
    } else {
      console.warn('fgRef.current가 null입니다');
    }
  }, [currentZoom]);

  // 리셋 함수: 마인드맵 줌 상태 초기화
  const handleReset = useCallback(() => {
    if (fgRef.current) {
      fgRef.current.centerAt(0, 0, 1000);
      fgRef.current.zoom(initialZoom);
      setCurrentZoom(initialZoom);
      setUserZoomed(false); 
    }
  }, [initialZoom]);
  
  // 뉴스 개수 데이터 계산
  const newsCountsByMajor = calculateNewsCountsByMajor(keywords);
  const maxNewsCount = Math.max(...Object.values(newsCountsByMajor), 0);
  const minNewsCount = Math.min(...Object.values(newsCountsByMajor), 0);
  
  // 뉴스 개수 순으로 정렬된 대분류 목록 생성
  const sortedMajors = Object.entries(newsCountsByMajor)
    .sort(([,a], [,b]) => b - a) // 내림차순 정렬
    .map(([major, count]) => ({ major, count }));
  
  // 각 대분류의 순위 계산
  const majorRankings = {};
  sortedMajors.forEach((item, index) => {
    majorRankings[item.major] = index;
  });

  // Force-graph 컴포넌트 참조
  const fgRef = useRef();
  
  // 컨테이너 DOM 참조
  const containerRef = useRef();

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
    graphData, setGraphData,
    expandedNodeIds, setExpandedNodeIds,
  });


  /**
   * 키워드 데이터 변경 시 마인드맵 초기화
   */
  useEffect(() => {
    const newInitialData = keywords ? generateInitialMindMapData(keywords) : { nodes: [], links: [] };
    
    // 그래프 데이터 및 상태 초기화
    setGraphData(newInitialData);
    setExpandedNodeIds(new Set());
    setSelectedNews(null);
    setSelectedNodeIdForNews(null);
    
    // 초기 줌 레벨 설정 및 사용자 줌 상태 초기화
    const initialZoomLevel = getInitialZoom();
    setCurrentZoom(initialZoomLevel);
    setUserZoomed(false);
    
    // Force-graph 중앙 이동 및 초기 줌 적용
    if (fgRef.current) {
      fgRef.current.centerAt(0, 0, 1000);
      fgRef.current.zoom(initialZoomLevel);
    }
  }, [keywords, setGraphData, setExpandedNodeIds, setSelectedNews, setSelectedNodeIdForNews, getInitialZoom]);

  // 컨테이너 크기 변화 감지 및 force-graph 크기 업데이트
  useEffect(() => {
    const handleResize = () => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        const newDimensions = {
          width: rect.width,
          height: rect.height,
        };
        
        setDimensions(prevDimensions => {
          // 크기가 실제로 변경된 경우에만 업데이트
          if (newDimensions.width !== prevDimensions.width || newDimensions.height !== prevDimensions.height) {
            return newDimensions;
          }
          return prevDimensions;
        });
      }
    };

    // 초기 크기 설정
    handleResize();

    // ResizeObserver 사용
    let resizeObserver;
    if (containerRef.current) {
      resizeObserver = new ResizeObserver((entries) => {
        for (const entry of entries) {
          const { width, height } = entry.contentRect;
          const newDimensions = { width, height };
          
          setDimensions(prevDimensions => {
            // 크기가 실제로 변경된 경우에만 업데이트
            if (newDimensions.width !== prevDimensions.width || newDimensions.height !== prevDimensions.height) {
              return newDimensions;
            }
            return prevDimensions;
          });
        }
      });
      resizeObserver.observe(containerRef.current);
    }

    return () => {
      if (resizeObserver) {
        resizeObserver.disconnect();
      }
    };
  }, []);

  // 화면 크기 변경 시 줌 레벨 자동 조정
  useEffect(() => {
    if (fgRef.current && dimensions.width > 0 && dimensions.height > 0 && !userZoomed) {
      const zoomLevel = getInitialZoom();
      
      // 화면 크기가 변경되면 항상 줌 레벨 업데이트
      if (Math.abs(currentZoom - zoomLevel) > 0.1) {
        // 약간의 지연을 두고 줌 적용
        const timeoutId = setTimeout(() => {
          if (fgRef.current) {
            fgRef.current.centerAt(0, 0, 1000);
            fgRef.current.zoom(zoomLevel);
            setCurrentZoom(zoomLevel);
          }
        }, 200);

        return () => clearTimeout(timeoutId);
      }
    }
  }, [dimensions.width, dimensions.height, getInitialZoom, currentZoom, userZoomed]);

  // 화면 크기 변경 시 force-graph 설정 업데이트
  useEffect(() => {
    const currentWidth = dimensions.width || window.innerWidth;
    const currentHeight = dimensions.height || window.innerHeight;
    
    if (fgRef.current && currentWidth > 0 && currentHeight > 0) {
      // 화면 크기에 따른 force 설정
      const isMobile = currentWidth < 768;
      const isSmallMobile = currentWidth < 480;
      const isVerySmallMobile = currentWidth < 320;
      
      // 노드 간격 조정
      const baseStrength = isVerySmallMobile ? -200 : (isSmallMobile ? -250 : (isMobile ? -300 : -400));
      fgRef.current.d3Force('charge').strength(baseStrength);

      // 링크 거리 조정
      const baseLinkDistance = isVerySmallMobile ? 40 : (isSmallMobile ? 50 : (isMobile ? 60 : 80));
      fgRef.current.d3Force('link').distance(baseLinkDistance);
      
      // 중력 조정
      const centerStrength = isMobile ? 0.2 : 0.1;
      fgRef.current.d3Force('center').strength(centerStrength);
    }
  }, [dimensions.width, dimensions.height]);

  /**
   * 노드 렌더링 함수
   * 각 노드를 타원형으로 그리고 텍스트 표시
   * 노드 레벨별로 다른 크기, 색상, 글자 크기 적용
   */
  const nodeCanvasObject = (node, ctx, scale) => {
    // 유효한지 확인
    if (!node || !node.id) {
      console.warn("Skipping rendering for invalid node:", node);
      return;
    }

    const label = node.label;

    let nodeWidth, nodeHeight;
    let fontSize;
    let nodeColor;
    let borderColor;
    let textColor = 'white';

    // 노드 별 크기 및 색상 조절
    if (node.level === 0) {
      // 중앙 노드 (뉴스)
      if (dimensions.width < 320) {
        fontSize = 9; nodeHeight = 16;
      } else if (dimensions.width < 480) {
        fontSize = 11; nodeHeight = 20; 
      } else if (dimensions.width < 700) {
        fontSize = 13; nodeHeight = 24; 
      } else {
        fontSize = 15; nodeHeight = 28;
      }
      // 화면 크기에 따른 가로 길이 조절
      let widthMultiplier;
      if (dimensions.width < 320) {
        widthMultiplier = 0.6; 
      } else if (dimensions.width < 480) {
        widthMultiplier = 0.7; 
      } else if (dimensions.width < 700) {
        widthMultiplier = 0.8; 
      } else {
        widthMultiplier = 1.0; 
      }
      nodeWidth = Math.max(40 * widthMultiplier, (label.length * fontSize * 0.4 + 12) * widthMultiplier);
      nodeColor = '#1e3a8a'; borderColor = '#1e40af'; // 진한 남색
    } else if (node.level === 1) {
      // 대분류 - 뉴스 개수에 따라 색상 및 크기 조절
      const parsed = parseNodeId(node.id);
      const majorKeyword = parsed.majorKeyword;
      const newsCount = newsCountsByMajor[majorKeyword] || 0;
      const intensity = calculateColorIntensity(newsCount, maxNewsCount, minNewsCount);
      
      // 뉴스 개수에 따른 크기 조절 (기본 0.8배, 최대 2.0배)
      const sizeMultiplier = Math.max(0.8, Math.min(2.0, 0.8 + intensity * 1.2));
      
      // 기본 글자 크기 설정
      const currentWidth = dimensions.width || window.innerWidth;
      let baseFontSize;
      if (currentWidth < 320) {
        baseFontSize = 4; 
      } else if (currentWidth < 480) {
        baseFontSize = 5; 
      } else if (currentWidth < 700) {
        baseFontSize = 6; 
      } else {
        baseFontSize = 7; 
      }
      // 뉴스가 적을수록 글자 크기 줄이기
      fontSize = baseFontSize * Math.max(0.7, sizeMultiplier);
      
      // 화면 크기에 따른 세로 길이 조절
      let baseHeight;
      if (currentWidth < 320) {
        baseHeight = 10; 
      } else if (currentWidth < 480) {
        baseHeight = 12; 
      } else if (currentWidth < 700) {
        baseHeight = 14; 
      } else {
        baseHeight = 16; 
      }
      // 뉴스 개수에 따라 더 크게 증가
      nodeHeight = baseHeight * sizeMultiplier * 1.2;
      
      // 화면 크기에 따른 가로 길이 조절
      let widthMultiplier;
      if (currentWidth < 320) {
        widthMultiplier = 0.6; 
      } else if (currentWidth < 480) {
        widthMultiplier = 0.7; 
      } else if (currentWidth < 700) {
        widthMultiplier = 0.8; 
      } else {
        widthMultiplier = 1.0; 
      }
      nodeWidth = Math.max(25 * widthMultiplier, (label.length * fontSize * 0.3 + 8) * widthMultiplier) * sizeMultiplier;
      
      // 뉴스 개수 순위에 따른 색상 조정  
      const totalMajors = sortedMajors.length;
      const rank = majorRankings[majorKeyword] || 0;
      const rankRatio = rank / Math.max(totalMajors - 1, 1); 
      
      // 순위에 따라 색상 계산
      const darkR = 59;   
      const darkG = 130; 
      const darkB = 246;  
      
      const lightR = 147; 
      const lightG = 197; 
      const lightB = 253; 
      
      // 순위에 따라 색상 보간
      const r = Math.round(darkR + (lightR - darkR) * rankRatio);
      const g = Math.round(darkG + (lightG - darkG) * rankRatio);
      const b = Math.round(darkB + (lightB - darkB) * rankRatio);
      
      nodeColor = `rgb(${r}, ${g}, ${b})`;
      borderColor = `rgb(${Math.round(r * 0.8)}, ${Math.round(g * 0.8)}, ${Math.round(b * 0.8)})`;
      
      // 텍스트 색상도 조정
      if (rankRatio > 0.7) {
        textColor = '#1e40af'; 
      }
    } else if (node.level === 2) {
      // 중분류 - 뉴스 개수에 따른 색상 차이 및 글자 길이에 맞춤
      const parsed = parseNodeId(node.id);
      const majorKeyword = parsed.majorKeyword;
      const middleKeyword = parsed.middleKeyword;
      
      // 뉴스 개수 순위 계산
      const majorData = keywords.find(cat => cat[MAJOR_KEY] === majorKeyword);
      
      if (majorData && majorData[MIDDLE_LIST_KEY]) {
        const middleCategories = majorData[MIDDLE_LIST_KEY];
        const middleNewsCounts = middleCategories.map(middle => ({
          keyword: middle[MIDDLE_ITEM_KEY],
          count: middle[RELATED_NEWS_KEY]?.length || 0
        })).sort((a, b) => b.count - a.count);
        
        const currentMiddleIndex = middleNewsCounts.findIndex(m => m.keyword === middleKeyword);
        const totalMiddles = middleNewsCounts.length;
        
        if (currentMiddleIndex >= 0 && totalMiddles > 0) {
          const rankRatio = currentMiddleIndex / Math.max(totalMiddles - 1, 1);
          
          // 순위에 따라 색상 계산
          const darkR = 147; 
          const darkG = 197; 
          const darkB = 253; 
          
          const lightR = 219; 
          const lightG = 234; 
          const lightB = 254; 
          
          // 순위에 따라 색상 보간
          const r = Math.round(darkR + (lightR - darkR) * rankRatio);
          const g = Math.round(darkG + (lightG - darkG) * rankRatio);
          const b = Math.round(darkB + (lightB - darkB) * rankRatio);
          
          nodeColor = `rgb(${r}, ${g}, ${b})`;
          borderColor = `rgb(${Math.round(r * 0.8)}, ${Math.round(g * 0.8)}, ${Math.round(b * 0.8)})`;
          
          // 뉴스 개수에 따라 텍스트 색상 조절
          if (rankRatio > 0.7) {
            textColor = '#1e40af'; 
          } else {
            textColor = 'white'; 
          }
        } else {
          // 기본 색상
          nodeColor = '#93c5fd'; borderColor = '#60a5fa';
          textColor = 'white'; 
        }
      } else {
        // 기본 색상
        nodeColor = '#93c5fd'; borderColor = '#60a5fa';
        textColor = 'white'; 
      }
      
      // 뉴스 개수 순위 계산
      const majorData2 = keywords.find(cat => cat[MAJOR_KEY] === majorKeyword);
      let sizeMultiplier2 = 1.0; 
      
      if (majorData2 && majorData2[MIDDLE_LIST_KEY]) {
        const middleCategories = majorData2[MIDDLE_LIST_KEY];
        const middleNewsCounts = middleCategories.map(middle => ({
          keyword: middle[MIDDLE_ITEM_KEY],
          count: middle[RELATED_NEWS_KEY]?.length || 0
        })).sort((a, b) => b.count - a.count);
        
        const currentMiddleIndex = middleNewsCounts.findIndex(m => m.keyword === middleKeyword);
        
        if (currentMiddleIndex >= 0 && middleNewsCounts.length > 0) {
          const currentMiddle = middleNewsCounts[currentMiddleIndex];
          const maxCount = middleNewsCounts[0]?.count || 0;
          const minCount = middleNewsCounts[middleNewsCounts.length - 1]?.count || 0;
          
          // 뉴스 개수에 따른 크기 조절
          if (maxCount > minCount) {
            const intensity = (currentMiddle.count - minCount) / (maxCount - minCount);
            sizeMultiplier2 = Math.max(0.7, Math.min(1.4, 0.7 + intensity * 0.7)); 
          }
        }
      }
      
      // 기본 글자 크기 설정
      const currentWidth2 = dimensions.width || window.innerWidth;
      let baseFontSize2;
      if (currentWidth2 < 320) {
        baseFontSize2 = 4; 
      } else if (currentWidth2 < 480) {
        baseFontSize2 = 5; 
      } else if (currentWidth2 < 700) {
        baseFontSize2 = 6; 
      } else {
        baseFontSize2 = 8; 
      }
      // 뉴스가 적을수록 글자 크기 줄이기
      fontSize = baseFontSize2 * Math.max(0.8, sizeMultiplier2); 
      
      let baseHeight2;
      if (currentWidth2 < 320) {
        baseHeight2 = 10;
      } else if (currentWidth2 < 480) {
        baseHeight2 = 12;
      } else if (currentWidth2 < 700) {
        baseHeight2 = 14;
      } else {
        baseHeight2 = 18;
      }
      nodeHeight = baseHeight2 * sizeMultiplier2;
      
      // 화면 크기에 따른 가로 길이 조절
      let widthMultiplier;
      if (currentWidth2 < 320) {
        widthMultiplier = 0.6; 
      } else if (currentWidth2 < 480) {
        widthMultiplier = 0.7; 
      } else if (currentWidth2 < 700) {
        widthMultiplier = 0.8; 
      } else {
        widthMultiplier = 1.0; 
      }
      nodeWidth = Math.max(36 * widthMultiplier, (label.length * fontSize * 0.4 + 6) * widthMultiplier) * sizeMultiplier2;
      } else {
        // 예상치 못한 노드 레벨 - 기본값으로 처리
        console.warn("Rendering node with unexpected level:", node);
        fontSize = 6; nodeHeight = 12; nodeWidth = 50;
        nodeColor = '#dbeafe'; borderColor = '#93c5fd'; textColor = '#1e40af';
      }

    // 타원형 그리기
    let radiusX, radiusY;
    
    if (node.level === 0) {
      // 중앙 노드
      radiusX = nodeWidth / 2;
      radiusY = Math.max(nodeHeight / 2, radiusX * 0.8);
    } else if (node.level === 1) {
      // 대분류 노드
      radiusX = nodeWidth / 2;
      radiusY = Math.max(nodeHeight / 2, radiusX * 0.8);
    } else if (node.level === 2) {
      // 중분류 노드
      radiusX = nodeWidth / 2;
      radiusY = Math.max(nodeHeight / 2, radiusX * 0.8); 
      } else {
        // 예상치 못한 노드 레벨 - 기본 타원형
        radiusX = nodeWidth / 2;
        radiusY = Math.max(nodeHeight / 2, radiusX * 0.8);
      }
    const x = node.x;
    const y = node.y;

    ctx.beginPath();
    ctx.ellipse(x, y, radiusX, radiusY, 0, 0, 2 * Math.PI);
    ctx.fillStyle = nodeColor;
    ctx.fill();
    
    // 테두리 그리기
    ctx.strokeStyle = borderColor;
    ctx.lineWidth = 1;
    ctx.stroke();

    // 폰트 크기
    const adaptiveFontSize = fontSize;
    
    ctx.font = `${adaptiveFontSize}px sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = textColor;
    ctx.fillText(label, node.x, node.y);

  };

  /**
   * 노드 클릭 영역 렌더링 함수
   */
  const nodePointerAreaCanvasObject = (node, ctx, scale) => {
    // 유효한지 확인
    if (!node || !node.id) {
      return;
    }
    
    const label = node.label;
    let nodeWidth, nodeHeight;
    let fontSize;
    
    // 노드 크기를 반응형으로 계산
    if (node.level === 0) {
      if (dimensions.width < 320) {
        fontSize = 9; nodeHeight = 16; 
      } else if (dimensions.width < 480) {
        fontSize = 11; nodeHeight = 20; 
      } else if (dimensions.width < 700) {
        fontSize = 13; nodeHeight = 24; 
      } else {
        fontSize = 15; nodeHeight = 28; 
      }
      // 화면 크기에 따른 가로 길이 조절
      let widthMultiplier;
      if (dimensions.width < 320) {
        widthMultiplier = 0.6; 
      } else if (dimensions.width < 480) {
        widthMultiplier = 0.7; 
      } else if (dimensions.width < 700) {
        widthMultiplier = 0.8; 
      } else {
        widthMultiplier = 1.0; 
      }
      nodeWidth = Math.max(40 * widthMultiplier, (label.length * fontSize * 0.4 + 12) * widthMultiplier);
    } else if (node.level === 1) {
      // 대분류: 뉴스 개수에 따른 크기 조절
      const parsed = parseNodeId(node.id);
      const majorKeyword = parsed.majorKeyword;
      const newsCount = newsCountsByMajor[majorKeyword] || 0;
      const intensity = calculateColorIntensity(newsCount, maxNewsCount, minNewsCount);
      const sizeMultiplier = Math.max(0.8, Math.min(2.0, 0.8 + intensity * 1.2));
      
      // 기본 글자 크기 설정
      let baseFontSize;
      if (dimensions.width < 320) {
        baseFontSize = 4; 
      } else if (dimensions.width < 480) {
        baseFontSize = 5; 
      } else if (dimensions.width < 700) {
        baseFontSize = 6; 
      } else {
        baseFontSize = 7; 
      }
      // 뉴스가 적을수록 글자 크기 줄이기
      fontSize = baseFontSize * Math.max(0.7, sizeMultiplier);
      
      // 화면 크기에 따른 세로 길이 조절
      let baseHeight;
      if (dimensions.width < 320) {
        baseHeight = 10; 
      } else if (dimensions.width < 480) {
        baseHeight = 12; 
      } else if (dimensions.width < 700) {
        baseHeight = 14; 
      } else {
        baseHeight = 16; 
      }
      // 뉴스 개수에 따라 더 크게 증가
      nodeHeight = baseHeight * sizeMultiplier * 1.2;
      
      // 화면 크기에 따른 가로 길이 조절
      let widthMultiplier;
      if (dimensions.width < 320) {
        widthMultiplier = 0.6; 
      } else if (dimensions.width < 480) {
        widthMultiplier = 0.7; 
      } else if (dimensions.width < 700) {
        widthMultiplier = 0.8; 
      } else {
        widthMultiplier = 1.0; 
      }
      nodeWidth = Math.max(25 * widthMultiplier, (label.length * fontSize * 0.3 + 8) * widthMultiplier) * sizeMultiplier;
    } else if (node.level === 2) {
      // 중분류: 뉴스 개수에 따른 크기 조절
      const parsed = parseNodeId(node.id);
      const majorKeyword = parsed.majorKeyword;
      const middleKeyword = parsed.middleKeyword;
      
      const majorData3 = keywords.find(cat => cat[MAJOR_KEY] === majorKeyword);
      let sizeMultiplier3 = 1.0;
      
      if (majorData3 && majorData3[MIDDLE_LIST_KEY]) {
        const middleCategories = majorData3[MIDDLE_LIST_KEY];
        const middleNewsCounts = middleCategories.map(middle => ({
          keyword: middle[MIDDLE_ITEM_KEY],
          count: middle[RELATED_NEWS_KEY]?.length || 0
        })).sort((a, b) => b.count - a.count);
        
        const currentMiddleIndex = middleNewsCounts.findIndex(m => m.keyword === middleKeyword);
        
        if (currentMiddleIndex >= 0 && middleNewsCounts.length > 0) {
          const currentMiddle = middleNewsCounts[currentMiddleIndex];
          const maxCount = middleNewsCounts[0]?.count || 0;
          const minCount = middleNewsCounts[middleNewsCounts.length - 1]?.count || 0;
          
          if (maxCount > minCount) {
            const intensity = (currentMiddle.count - minCount) / (maxCount - minCount);
            sizeMultiplier3 = Math.max(0.7, Math.min(1.4, 0.7 + intensity * 0.7)); 
          }
        }
      }
      
      // 기본 글자 크기 설정
      let baseFontSize2;
      if (dimensions.width < 320) {
        baseFontSize2 = 4; 
      } else if (dimensions.width < 480) {
        baseFontSize2 = 5; 
      } else if (dimensions.width < 700) {
        baseFontSize2 = 6; 
      } else {
        baseFontSize2 = 8; 
      }
      // 뉴스가 적을수록 글자 크기 줄이기
      fontSize = baseFontSize2 * Math.max(0.8, sizeMultiplier3); 
      
      let baseHeight2;
      if (dimensions.width < 320) {
        baseHeight2 = 10;
      } else if (dimensions.width < 480) {
        baseHeight2 = 12;
      } else if (dimensions.width < 700) {
        baseHeight2 = 14;
      } else {
        baseHeight2 = 18;
      }
      nodeHeight = baseHeight2 * sizeMultiplier3;
      
      // 화면 크기에 따른 가로 길이 조절
      let widthMultiplier;
      if (dimensions.width < 320) {
        widthMultiplier = 0.6; 
      } else if (dimensions.width < 480) {
        widthMultiplier = 0.7; 
      } else if (dimensions.width < 700) {
        widthMultiplier = 0.8; 
      } else {
        widthMultiplier = 1.0; 
      }
      nodeWidth = Math.max(36 * widthMultiplier, (label.length * fontSize * 0.4 + 6) * widthMultiplier) * sizeMultiplier3;
      } else {
        // 예상치 못한 노드 레벨 - 기본값으로 처리
        console.warn("Rendering node with unexpected level:", node);
        fontSize = 6; nodeHeight = 12; nodeWidth = 50;
      }

    // 클릭 영역을 타원형으로 설정
    let clickRadiusX, clickRadiusY;
    
    if (node.level === 0) {
      // 중앙 노드
      clickRadiusX = nodeWidth / 2 * 1.2;
      clickRadiusY = Math.max(nodeHeight / 2 * 1.2, clickRadiusX * 0.8);
    } else if (node.level === 1) {
      // 대분류 노드
      clickRadiusX = nodeWidth / 2 * 1.2;
      clickRadiusY = Math.max(nodeHeight / 2 * 1.2, clickRadiusX * 0.8);
    } else if (node.level === 2) {
      // 중분류 노드
      clickRadiusX = nodeWidth / 2 * 1.2;
      clickRadiusY = Math.max(nodeHeight / 2 * 1.2, clickRadiusX * 0.8); 
      } else {
        // 예상치 못한 노드 레벨 - 기본 클릭 영역
        clickRadiusX = nodeWidth / 2 * 1.2;
        clickRadiusY = Math.max(nodeHeight / 2 * 1.2, clickRadiusX * 0.8);
      }
    const x = node.x;
    const y = node.y;

    // 타원형으로 클릭 영역 설정
    ctx.beginPath();
    ctx.ellipse(x, y, clickRadiusX, clickRadiusY, 0, 0, 2 * Math.PI);
    ctx.fillStyle = 'rgba(0, 0, 0, 0)';
    ctx.fill();
  };

  /**
   * 마인드맵 컴포넌트 렌더링
   */
  
  // 로딩 중 UI (제거)
  // if (loading) {
  //   return (
  //     <section className="loading">
  //       <h1 className="loading-title">뉴스 제목 분석 중...</h1>
  //       <div className="progress-bar" aria-hidden="true">
  //         <span className="progress-bar-gauge"></span>
  //       </div>
  //     </section>
  //   );
  // }

  // 에러 발생 UI
  if (error) {
    return (
      <section className="loading">
        <h1 className="loading-title" style={{ color: '#dc2626' }}>
          데이터를 불러오는데 실패했습니다.
        </h1>
        <p style={{ fontSize: '1.2rem', color: '#6b7280' }}>
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
          linkWidth={link => {
            // 유효한지 확인
            const sourceId = typeof link.source === 'object' ? link.source?.id : link.source;
            const targetId = typeof link.target === 'object' ? link.target?.id : link.target;

            if (!sourceId || !targetId) {
              return 1;
            }

            const targetNode = graphData.nodes.find(n => n.id === targetId);

            if (targetNode) {
              if (targetNode.level === 1) return 2;
              if (targetNode.level === 2) return 1.5;
            }
            return 1;
          }}
          linkCanvasObject={(link, ctx, scale) => {
            // 링크 그리기
            const source = typeof link.source === 'object' ? link.source : graphData.nodes.find(node => node.id === link.source); 
            const target = typeof link.target === 'object' ? link.target : graphData.nodes.find(node => node.id === link.target); 

            // 유효하지 않은 링크는 그리지 않음
            if (!source || !target || !source.id || !target.id) {
              return;
            }

            const start = { x: source.x, y: source.y };
            const end = { x: target.x, y: target.y };

            // 좌표가 유효한지 확인
            if (!isFinite(start.x) || !isFinite(start.y) || !isFinite(end.x) || !isFinite(end.y)) {
              return;
            }

            // 링크 그라데이션 효과
            const gradient = ctx.createLinearGradient(start.x, start.y, end.x, end.y);
            gradient.addColorStop(0, 'rgba(59, 130, 246, 0.8)');
            gradient.addColorStop(1, 'rgba(37, 99, 235, 0.6)');

            ctx.beginPath();
            ctx.moveTo(start.x, start.y);
            ctx.lineTo(end.x, end.y);
            ctx.strokeStyle = gradient;
            ctx.lineWidth = link.width || 2;
            ctx.stroke();
          }}
          />
        ) : (
          <div style={{ 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center', 
            height: '600px',
            fontSize: '18px',
            color: '#666'
          }}>
            로딩 중...
          </div>
        )}
      </div>

      {/* 줌 컨트롤 버튼 */}
      <div className="zoom-controls">
        <button
          onClick={handleZoomIn}
          className="zoom-button"
        >
          &#43;
        </button>
        <button
          onClick={handleZoomOut}
          className="zoom-button"
        >
          &#45;
        </button>
        <button
          onClick={handleReset}
          className="zoom-button reset"
        >
          Reset
        </button>
      </div>

      {/* 뉴스 목록 */}
      {selectedNews !== null && selectedNodeIdForNews && (
        <div className="news-panel">
          {/* 헤더 */}
          <div className="news-panel-header">
            <div className="news-panel-title">
              <div className="news-panel-indicator"></div>
              <h4>
                {`'${selectedNodeIdForNews ? (parseNodeId(selectedNodeIdForNews)?.middleKeyword || parseNodeId(selectedNodeIdForNews)?.majorKeyword || '뉴스') : '뉴스'}' 관련 뉴스`}
                <span style={{ 
                  display: 'inline-block',
                  backgroundColor: '#3b82f6',
                  color: 'white',
                  fontSize: '0.75em',
                  fontWeight: 'bold',
                  padding: '2px 8px',
                  borderRadius: '12px',
                  marginLeft: '10px',
                  minWidth: '20px',
                  textAlign: 'center',
                  lineHeight: '1.2',
                  boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
                }}>
                  {selectedNews?.length || 0}
                </span>
              </h4>
            </div>
            <button
              onClick={() => { setSelectedNews(null); setSelectedNodeIdForNews(null); }}
              className="news-panel-close"
            >
              ×
            </button>
          </div>

          {/* 뉴스 리스트*/}
          <div className="news-list">
            {selectedNews.length === 0 ? (
              <p className="no-news-message">
                관련 뉴스가 없습니다.
              </p>
            ) : (
              <div>
                {selectedNews.map((news, index) => (
                  // 뉴스 객체가 유효한지 확인
                  news && news.link && news.title ? (
                    <div key={index} className="news-item"
                    onClick={() => window.open(news.link, '_blank')}
                    >
                      <a 
                        href={news.link} 
                        target="_blank" 
                        rel="noopener noreferrer" 
                        className="news-link"
                      >
                        {news.title.length > 50 ? `${news.title.substring(0, 50)}...` : news.title}
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
      )}
    </div>
  );
};

export default MindMap;