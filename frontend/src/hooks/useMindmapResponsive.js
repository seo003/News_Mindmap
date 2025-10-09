/**
 * 마인드맵 반응형 화면 크기 관리 훅
 */

import { useEffect, useState } from 'react';
import { BREAKPOINTS, ZOOM_CONFIG } from '../utils/mindmapConstants';

/**
 * 화면 크기에 따른 초기 줌 레벨 계산
 * @param {number} width - 화면 너비
 * @returns {number} 줌 레벨
 */
export const getInitialZoomLevel = (width) => {
  if (!width) return ZOOM_CONFIG.INITIAL.DEFAULT;
  
  if (width <= BREAKPOINTS.VERY_SMALL) return ZOOM_CONFIG.INITIAL[BREAKPOINTS.VERY_SMALL];
  if (width <= BREAKPOINTS.SMALL) return ZOOM_CONFIG.INITIAL[BREAKPOINTS.SMALL];
  if (width <= BREAKPOINTS.MEDIUM) return ZOOM_CONFIG.INITIAL[BREAKPOINTS.MEDIUM];
  if (width <= BREAKPOINTS.LARGE) return ZOOM_CONFIG.INITIAL[BREAKPOINTS.LARGE];
  
  return ZOOM_CONFIG.INITIAL.DEFAULT;
};

/**
 * 마인드맵 반응형 화면 크기 훅
 * @param {Object} containerRef - 컨테이너 참조
 * @returns {Object} { dimensions, isMobile, isSmallMobile, isVerySmallMobile }
 */
export const useMindmapResponsive = (containerRef) => {
  const [dimensions, setDimensions] = useState({
    width: window.innerWidth,
    height: window.innerHeight,
  });

  useEffect(() => {
    const handleResize = () => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        const newDimensions = {
          width: rect.width || window.innerWidth,
          height: rect.height || window.innerHeight,
        };
        
        setDimensions(prevDimensions => {
          if (newDimensions.width !== prevDimensions.width || newDimensions.height !== prevDimensions.height) {
            return newDimensions;
          }
          return prevDimensions;
        });
      }
    };

    // 초기 크기 설정
    handleResize();

    let resizeObserver;
    if (containerRef.current) {
      resizeObserver = new ResizeObserver((entries) => {
        for (const entry of entries) {
          const { width, height } = entry.contentRect;
          const newDimensions = { 
            width: width || window.innerWidth, 
            height: height || window.innerHeight 
          };
          
          setDimensions(prevDimensions => {
            if (newDimensions.width !== prevDimensions.width || newDimensions.height !== prevDimensions.height) {
              return newDimensions;
            }
            return prevDimensions;
          });
        }
      });
      resizeObserver.observe(containerRef.current);
    }

    // 윈도우 리사이즈 이벤트도 추가
    window.addEventListener('resize', handleResize);

    return () => {
      if (resizeObserver) {
        resizeObserver.disconnect();
      }
      window.removeEventListener('resize', handleResize);
    };
  }, [containerRef]);

  const currentWidth = dimensions.width;
  
  const isMobile = currentWidth < BREAKPOINTS.MEDIUM;
  const isSmallMobile = currentWidth < BREAKPOINTS.SMALL;
  const isVerySmallMobile = currentWidth < BREAKPOINTS.VERY_SMALL;

  return {
    dimensions,
    isMobile,
    isSmallMobile,
    isVerySmallMobile,
  };
};

