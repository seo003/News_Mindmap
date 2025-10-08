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


const MindMap = ({ keywords }) => {
  const initialData = generateInitialMindMapData(keywords);
  const [graphData, setGraphData] = useState(initialData);
  const [dimensions, setDimensions] = useState({
    width: window.innerWidth, // 초기값을 실제 화면 크기로 설정
    height: window.innerHeight,
  });


  // 화면 크기별 줌 레벨 적용
  const getInitialZoom = useCallback(() => {
    // 디바이스 툴바 대응을 위해 window.innerWidth 직접 사용
    const viewportWidth = window.innerWidth;
    let zoomLevel;
    
    if (viewportWidth <= 320) {
      zoomLevel = 2.0; // 320px 이하에서는 더 작게
    } else if (viewportWidth <= 1024) {
      zoomLevel = 2.0; // 768px 이하에서는 기존 크기
    } else {
      zoomLevel = 3.0; // 768px 이상에서는 기본 크기
    }
    
    console.log(`화면 크기 변경 감지: window.innerWidth=${window.innerWidth}, dimensions.width=${dimensions.width}, viewportWidth=${viewportWidth}, 줌 레벨=${zoomLevel}`);
    return zoomLevel;
  }, [dimensions.width]);

  const initialZoom = getInitialZoom();

  const [expandedNodeIds, setExpandedNodeIds] = useState(new Set());
  const [currentZoom, setCurrentZoom] = useState(initialZoom);
  const [userZoomed, setUserZoomed] = useState(false); // 사용자가 수동으로 줌 조정했는지 추적  
  
  // 줌 컨트롤 함수들
  const handleZoomIn = useCallback(() => {
    console.log('줌인 버튼 클릭됨, 현재 줌:', currentZoom);
    const newZoom = Math.min(currentZoom * 1.2, 10); // 최대 줌 레벨 제한
    console.log('새로운 줌 레벨:', newZoom);
    
    if (fgRef.current) {
      try {
        // 줌 적용을 여러 번 시도
        fgRef.current.zoom(newZoom);
        // 약간의 지연 후 다시 시도
        setTimeout(() => {
          if (fgRef.current) {
            fgRef.current.zoom(newZoom);
          }
        }, 100);
        setCurrentZoom(newZoom);
        setUserZoomed(true);
        console.log('줌인 적용 완료');
      } catch (error) {
        console.error('줌인 적용 중 오류:', error);
      }
    } else {
      console.warn('fgRef.current가 null입니다');
    }
  }, [currentZoom]);

  const handleZoomOut = useCallback(() => {
    console.log('줌아웃 버튼 클릭됨, 현재 줌:', currentZoom);
    const newZoom = Math.max(currentZoom / 1.2, 0.1); // 최소 줌 레벨 제한
    console.log('새로운 줌 레벨:', newZoom);
    
    if (fgRef.current) {
      try {
        // 줌 적용을 여러 번 시도
        fgRef.current.zoom(newZoom);
        // 약간의 지연 후 다시 시도
        setTimeout(() => {
          if (fgRef.current) {
            fgRef.current.zoom(newZoom);
          }
        }, 100);
        setCurrentZoom(newZoom);
        setUserZoomed(true);
        console.log('줌아웃 적용 완료');
      } catch (error) {
        console.error('줌아웃 적용 중 오류:', error);
      }
    } else {
      console.warn('fgRef.current가 null입니다');
    }
  }, [currentZoom]);

  const handleReset = useCallback(() => {
    if (fgRef.current) {
      fgRef.current.centerAt(0, 0, 1000);
      fgRef.current.zoom(initialZoom);
      setCurrentZoom(initialZoom);
      setUserZoomed(false); // 리셋 시 사용자 줌 상태 초기화
    }
  }, [initialZoom]);

  const handleResetZoom = useCallback(() => {
    if (fgRef.current) {
      const resetZoom = getInitialZoom();
      setCurrentZoom(resetZoom);
      fgRef.current.zoom(resetZoom);
      fgRef.current.centerAt(0, 0, 1000);
    }
  }, [getInitialZoom]);
  
  // 뉴스 개수 데이터 계산
  const newsCountsByMajor = calculateNewsCountsByMajor(keywords);
  const maxNewsCount = Math.max(...Object.values(newsCountsByMajor), 0);
  const minNewsCount = Math.min(...Object.values(newsCountsByMajor), 0);
  
  // 뉴스 개수 순으로 정렬된 대분류 목록 생성
  const sortedMajors = Object.entries(newsCountsByMajor)
    .sort(([,a], [,b]) => b - a) // 뉴스 개수 내림차순 정렬
    .map(([major, count]) => ({ major, count }));
  
  // 각 대분류의 순위 계산
  const majorRankings = {};
  sortedMajors.forEach((item, index) => {
    majorRankings[item.major] = index;
  });

  const fgRef = useRef();
  const containerRef = useRef();

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


  useEffect(() => {
    const newInitialData = generateInitialMindMapData(keywords);
    
    
    setGraphData(newInitialData);
    setExpandedNodeIds(new Set());
    setSelectedNews(null);
    setSelectedNodeIdForNews(null);
    
    // 초기 줌 레벨 설정 및 사용자 줌 상태 초기화
    const initialZoomLevel = getInitialZoom();
    setCurrentZoom(initialZoomLevel);
    setUserZoomed(false); // 새로운 데이터 로드 시 사용자 줌 상태 초기화
    
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

    // 초기 크기 설정 - 즉시 실행
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

  // 화면 크기 변경 시 줌 레벨 자동 조정 (사용자가 수동으로 줌을 조정하지 않은 경우에만)
  useEffect(() => {
    if (fgRef.current && dimensions.width > 0 && dimensions.height > 0 && !userZoomed) {
      const zoomLevel = getInitialZoom();
      
      // 화면 크기가 변경되면 항상 줌 레벨 업데이트 (디바이스 툴바 대응)
      if (Math.abs(currentZoom - zoomLevel) > 0.1) {
        // 약간의 지연을 두고 줌 적용 (force-graph가 완전히 로드된 후)
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
    if (fgRef.current && dimensions.width > 0 && dimensions.height > 0) {
      // 화면 크기에 따른 force 설정
      const isMobile = dimensions.width < 768;
      const isSmallMobile = dimensions.width < 480;
      const isVerySmallMobile = dimensions.width < 320;
      
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

  // 줌 상태 변경 시 ForceGraph2D 업데이트는 zoom prop을 통해 자동으로 처리됨


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
      // 중앙 노드 (뉴스) - 진한 남색으로 대분류와 구분
      if (dimensions.width < 320) {
        fontSize = 10; nodeHeight = 20;
      } else if (dimensions.width < 480) {
        fontSize = 12; nodeHeight = 24;
      } else if (dimensions.width < 700) {
        fontSize = 14; nodeHeight = 28;
      } else {
        fontSize = 16; nodeHeight = 32;
      }
      nodeWidth = Math.max(60, label.length * fontSize * 0.6 + 24);
      nodeColor = '#1e3a8a'; borderColor = '#1e40af'; // 진한 남색
    } else if (node.level === 1) {
      // 1차 노드 (대분류) - 뉴스 개수에 따라 색상과 크기 조절
      const parsed = parseNodeId(node.id);
      const majorKeyword = parsed.majorKeyword;
      const newsCount = newsCountsByMajor[majorKeyword] || 0;
      const intensity = calculateColorIntensity(newsCount, maxNewsCount, minNewsCount);
      
      // 뉴스 개수에 따른 크기 조절 (기본 0.8배, 최대 2.0배)
      const sizeMultiplier = Math.max(0.8, Math.min(2.0, 0.8 + intensity * 1.2));
      
      // 기본 글자 크기 설정 (더 세밀한 조정)
      let baseFontSize;
      if (dimensions.width < 320) {
        baseFontSize = 4;
      } else if (dimensions.width < 480) {
        baseFontSize = 5;
      } else if (dimensions.width < 700) {
        baseFontSize = 6;
      } else {
        baseFontSize = 8;
      }
      // 뉴스가 적을수록 글자 크기 줄이기 (최대치에서 시작해서 줄어듦)
      fontSize = baseFontSize * Math.max(0.7, sizeMultiplier);
      
      let baseHeight;
      if (dimensions.width < 320) {
        baseHeight = 12;
      } else if (dimensions.width < 480) {
        baseHeight = 14;
      } else if (dimensions.width < 700) {
        baseHeight = 16;
      } else {
        baseHeight = 20;
      }
      nodeHeight = baseHeight * sizeMultiplier;
      nodeWidth = Math.max(40, label.length * fontSize * 0.6 + 14) * sizeMultiplier;
      
      // 뉴스 개수 순위에 따른 완전히 다른 색상 (순위별로 점점 연해짐)
      const totalMajors = sortedMajors.length;
      const rank = majorRankings[majorKeyword] || 0;
      const rankRatio = rank / Math.max(totalMajors - 1, 1); // 0~1 범위로 정규화
      
      // 순위에 따라 색상 계산 (1위가 가장 진하고, 마지막 순위가 가장 연함)
      const darkR = 59;   // #3b82f6의 R값 (예쁜 파란색)
      const darkG = 130;  // #3b82f6의 G값  
      const darkB = 246;  // #3b82f6의 B값
      
      const lightR = 147; // #93c5fd의 R값 (연한 파란색)
      const lightG = 197; // #93c5fd의 G값
      const lightB = 253; // #93c5fd의 B값
      
      // 순위에 따라 색상 보간 (rankRatio가 0이면 진한 색, 1이면 연한 색)
      const r = Math.round(darkR + (lightR - darkR) * rankRatio);
      const g = Math.round(darkG + (lightG - darkG) * rankRatio);
      const b = Math.round(darkB + (lightB - darkB) * rankRatio);
      
      nodeColor = `rgb(${r}, ${g}, ${b})`;
      borderColor = `rgb(${Math.round(r * 0.8)}, ${Math.round(g * 0.8)}, ${Math.round(b * 0.8)})`;
      
      // 연한 색상일 때는 텍스트 색상도 조정
      if (rankRatio > 0.7) {
        textColor = '#1e40af'; // 진한 파란색 텍스트+
      }
    } else if (node.level === 2) {
      // 2차 노드 (중분류) - 뉴스 개수에 따른 색상 차이, 글자 길이에 맞춤
      const parsed = parseNodeId(node.id);
      const majorKeyword = parsed.majorKeyword;
      const middleKeyword = parsed.middleKeyword;
      
      // 해당 대분류의 중분류들 중에서 뉴스 개수 순위 계산
      const majorData = keywords.find(cat => cat[MAJOR_KEY] === majorKeyword);
      let sizeMultiplier = 1.0; // 기본 크기
      
      if (majorData && majorData[MIDDLE_LIST_KEY]) {
        const middleCategories = majorData[MIDDLE_LIST_KEY];
        const middleNewsCounts = middleCategories.map(middle => ({
          keyword: middle[MIDDLE_ITEM_KEY],
          count: middle[RELATED_NEWS_KEY]?.length || 0
        })).sort((a, b) => b.count - a.count);
        
        const currentMiddleIndex = middleNewsCounts.findIndex(m => m.keyword === middleKeyword);
        const totalMiddles = middleNewsCounts.length;
        
        if (currentMiddleIndex >= 0 && totalMiddles > 0) {
          const currentMiddle = middleNewsCounts[currentMiddleIndex];
          const maxCount = middleNewsCounts[0]?.count || 0;
          const minCount = middleNewsCounts[middleNewsCounts.length - 1]?.count || 0;
          
          // 뉴스 개수에 따른 크기 조절 (기본 1.0배, 최대 1.4배)
          if (maxCount > minCount) {
            const intensity = (currentMiddle.count - minCount) / (maxCount - minCount);
            sizeMultiplier = Math.max(1.0, Math.min(1.4, 1.0 + intensity * 0.4));
          }
          
          const rankRatio = currentMiddleIndex / Math.max(totalMiddles - 1, 1);
          
          // 순위에 따라 색상 계산 (뉴스가 많을수록 진한 색)
          const darkR = 147; // #93c5fd의 R값
          const darkG = 197; // #93c5fd의 G값  
          const darkB = 253; // #93c5fd의 B값
          
          const lightR = 219; // #dbeafe의 R값
          const lightG = 234; // #dbeafe의 G값
          const lightB = 254; // #dbeafe의 B값
          
          // 순위에 따라 색상 보간 (rankRatio가 0이면 진한 색, 1이면 연한 색)
          const r = Math.round(darkR + (lightR - darkR) * rankRatio);
          const g = Math.round(darkG + (lightG - darkG) * rankRatio);
          const b = Math.round(darkB + (lightB - darkB) * rankRatio);
          
          nodeColor = `rgb(${r}, ${g}, ${b})`;
          borderColor = `rgb(${Math.round(r * 0.8)}, ${Math.round(g * 0.8)}, ${Math.round(b * 0.8)})`;
          
          // 뉴스 개수에 따라 텍스트 색상 조절
          if (rankRatio > 0.7) {
            textColor = '#1e40af'; // 연한 배경일 때는 진한 파란색 텍스트
          } else {
            textColor = 'white'; // 진한 배경일 때는 흰색 텍스트
          }
        } else {
          // 기본 색상
          nodeColor = '#93c5fd'; borderColor = '#60a5fa';
          textColor = 'white'; // 흰색 텍스트
        }
      } else {
        // 기본 색상
        nodeColor = '#93c5fd'; borderColor = '#60a5fa';
        textColor = 'white'; // 흰색 텍스트
      }
      
      // 해당 대분류의 중분류들 중에서 뉴스 개수 순위 계산
      const majorData2 = keywords.find(cat => cat[MAJOR_KEY] === majorKeyword);
      let sizeMultiplier2 = 1.0; // 기본 크기
      
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
          
          // 뉴스 개수에 따른 크기 조절 (기본 0.7배, 최대 1.8배)
          if (maxCount > minCount) {
            const intensity = (currentMiddle.count - minCount) / (maxCount - minCount);
            sizeMultiplier2 = Math.max(0.7, Math.min(1.8, 0.7 + intensity * 1.1));
          }
        }
      }
      
      // 기본 글자 크기 설정 (더 세밀한 조정)
      let baseFontSize2;
      if (dimensions.width < 320) {
        baseFontSize2 = 3;
      } else if (dimensions.width < 480) {
        baseFontSize2 = 4;
      } else if (dimensions.width < 700) {
        baseFontSize2 = 5;
      } else {
        baseFontSize2 = 7;
      }
      // 뉴스가 적을수록 글자 크기 줄이기 (최대치에서 시작해서 줄어듦)
      fontSize = baseFontSize2 * Math.max(0.7, sizeMultiplier2);
      
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
      nodeHeight = baseHeight2 * sizeMultiplier2;
      nodeWidth = Math.max(36, label.length * fontSize * 0.6 + 8) * sizeMultiplier2;
    } else {
      // 기타 노드 레벨 - 더 세밀한 조정
      if (dimensions.width < 320) {
        fontSize = 3; nodeHeight = 8;
      } else if (dimensions.width < 480) {
        fontSize = 4; nodeHeight = 10;
      } else if (dimensions.width < 700) {
        fontSize = 5; nodeHeight = 12;
      } else {
        fontSize = 7; nodeHeight = 14;
      }
      nodeWidth = Math.max(25, label.length * fontSize * 0.6 + 6);
      nodeColor = '#dbeafe'; borderColor = '#93c5fd'; textColor = '#1e40af';
      console.warn("Rendering node with unexpected level:", node);
    }

    // 타원형 그리기
    const radiusX = nodeWidth / 2;
    const radiusY = nodeHeight / 2;
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

    // 단순한 폰트 크기 (스케일 팩터 제거)
    const adaptiveFontSize = fontSize;
    
    ctx.font = `${adaptiveFontSize}px sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = textColor;
    ctx.fillText(label, node.x, node.y);

  };

  const nodePointerAreaCanvasObject = (node, ctx, scale) => {
    // 유효한지 확인
    if (!node || !node.id) {
      return;
    }
    
    const label = node.label;
    let nodeWidth, nodeHeight;
    let fontSize;
    
    // 노드 크기를 반응형으로 계산 (뉴스 개수에 따른 크기 조절 포함) - 더 세밀한 조정
    if (node.level === 0) {
      if (dimensions.width < 320) {
        fontSize = 10; nodeHeight = 20;
      } else if (dimensions.width < 480) {
        fontSize = 12; nodeHeight = 24;
      } else if (dimensions.width < 700) {
        fontSize = 14; nodeHeight = 28;
      } else {
        fontSize = 16; nodeHeight = 32;
      }
      nodeWidth = Math.max(60, label.length * fontSize * 0.6 + 24);
    } else if (node.level === 1) {
      // 1차 노드 - 뉴스 개수에 따른 크기 조절
      const parsed = parseNodeId(node.id);
      const majorKeyword = parsed.majorKeyword;
      const newsCount = newsCountsByMajor[majorKeyword] || 0;
      const intensity = calculateColorIntensity(newsCount, maxNewsCount, minNewsCount);
      const sizeMultiplier = Math.max(0.8, Math.min(2.0, 0.8 + intensity * 1.2));
      
      // 기본 글자 크기 설정 (더 세밀한 조정)
      let baseFontSize;
      if (dimensions.width < 320) {
        baseFontSize = 4;
      } else if (dimensions.width < 480) {
        baseFontSize = 5;
      } else if (dimensions.width < 700) {
        baseFontSize = 6;
      } else {
        baseFontSize = 8;
      }
      // 뉴스가 적을수록 글자 크기 줄이기 (최대치에서 시작해서 줄어듦)
      fontSize = baseFontSize * Math.max(0.7, sizeMultiplier);
      
      let baseHeight;
      if (dimensions.width < 320) {
        baseHeight = 12;
      } else if (dimensions.width < 480) {
        baseHeight = 14;
      } else if (dimensions.width < 700) {
        baseHeight = 16;
      } else {
        baseHeight = 20;
      }
      nodeHeight = baseHeight * sizeMultiplier;
      nodeWidth = Math.max(40, label.length * fontSize * 0.6 + 14) * sizeMultiplier;
    } else if (node.level === 2) {
      // 2차 노드 - 뉴스 개수에 따른 크기 조절
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
            sizeMultiplier3 = Math.max(0.7, Math.min(1.8, 0.7 + intensity * 1.1));
          }
        }
      }
      
      // 기본 글자 크기 설정 (더 세밀한 조정)
      let baseFontSize2;
      if (dimensions.width < 320) {
        baseFontSize2 = 3;
      } else if (dimensions.width < 480) {
        baseFontSize2 = 4;
      } else if (dimensions.width < 700) {
        baseFontSize2 = 5;
      } else {
        baseFontSize2 = 7;
      }
      // 뉴스가 적을수록 글자 크기 줄이기 (최대치에서 시작해서 줄어듦)
      fontSize = baseFontSize2 * Math.max(0.7, sizeMultiplier3);
      
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
      nodeWidth = Math.max(36, label.length * fontSize * 0.6 + 8) * sizeMultiplier3;
    } else {
      // 기타 노드 레벨 - 더 세밀한 조정
      if (dimensions.width < 320) {
        fontSize = 3; nodeHeight = 8;
      } else if (dimensions.width < 480) {
        fontSize = 4; nodeHeight = 10;
      } else if (dimensions.width < 700) {
        fontSize = 5; nodeHeight = 12;
      } else {
        fontSize = 7; nodeHeight = 14;
      }
      nodeWidth = Math.max(25, label.length * fontSize * 0.6 + 6);
    }

    // 클릭 영역을 타원형으로 설정 (여유 공간 포함)
    const clickRadiusX = nodeWidth / 2 * 1.2;
    const clickRadiusY = nodeHeight / 2 * 1.2;
    const x = node.x;
    const y = node.y;

    // 타원형으로 클릭 영역 설정
    ctx.beginPath();
    ctx.ellipse(x, y, clickRadiusX, clickRadiusY, 0, 0, 2 * Math.PI);
    ctx.fillStyle = 'rgba(0, 0, 0, 0)';
    ctx.fill();
  };

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

            // 좌표가 유효한지 확인 (무한대, NaN 체크)
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
          +
        </button>
        <button
          onClick={handleZoomOut}
          className="zoom-button"
        >
          −
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

          {/* 뉴스 리스트 - 스크롤 가능한 영역 */}
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