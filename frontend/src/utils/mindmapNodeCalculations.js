/**
 * 마인드맵 노드 크기 및 색상 계산 유틸리티
 */

import {
  BREAKPOINTS,
  NODE_SIZE_CONFIG,
  NODE_COLORS,
  NODE_LEVELS,
  ELLIPSE_RATIO,
  CLICK_AREA_SCALE,
} from './mindmapConstants';
import { parseNodeId, calculateColorIntensity, MAJOR_KEY, MIDDLE_LIST_KEY, MIDDLE_ITEM_KEY, RELATED_NEWS_KEY } from './mindmapUtil';

/**
 * 화면 너비에 따른 반응형 값 가져오기
 * @param {Object} config 
 * @param {number} width 
 * @returns {number} 
 */
export const getResponsiveValue = (config, width) => {
  if (width < BREAKPOINTS.VERY_SMALL) return config.VERY_SMALL;
  if (width < BREAKPOINTS.SMALL) return config.SMALL;
  if (width < BREAKPOINTS.MEDIUM) return config.MEDIUM;
  return config.DEFAULT;
};

/**
 * 대분류 노드의 크기 배율 계산
 * @param {number} newsCount 
 * @param {number} maxNewsCount 
 * @param {number} minNewsCount 
 * @returns {number}
 */
const calculateMajorSizeMultiplier = (newsCount, maxNewsCount, minNewsCount) => {
  const intensity = calculateColorIntensity(newsCount, maxNewsCount, minNewsCount);
  const { MIN, MAX, BASE, RANGE } = NODE_SIZE_CONFIG.MAJOR.SIZE_MULTIPLIER;
  return Math.max(MIN, Math.min(MAX, BASE + intensity * RANGE));
};

/**
 * 중분류 노드의 크기 배율 계산
 * @param {string} majorKeyword 
 * @param {string} middleKeyword 
 * @param {Array} keywords -
 * @returns {number} s
 */
const calculateMiddleSizeMultiplier = (majorKeyword, middleKeyword, keywords) => {
  const majorData = keywords.find(cat => cat[MAJOR_KEY] === majorKeyword);
  
  if (!majorData || !majorData[MIDDLE_LIST_KEY]) {
    return 1.0;
  }
  
  const middleCategories = majorData[MIDDLE_LIST_KEY];
  const middleNewsCounts = middleCategories
    .map(middle => ({
      keyword: middle[MIDDLE_ITEM_KEY],
      count: middle[RELATED_NEWS_KEY]?.length || 0
    }))
    .sort((a, b) => b.count - a.count);
  
  const currentMiddleIndex = middleNewsCounts.findIndex(m => m.keyword === middleKeyword);
  
  if (currentMiddleIndex < 0 || middleNewsCounts.length === 0) {
    return 1.0;
  }
  
  const currentMiddle = middleNewsCounts[currentMiddleIndex];
  const maxCount = middleNewsCounts[0]?.count || 0;
  const minCount = middleNewsCounts[middleNewsCounts.length - 1]?.count || 0;
  
  if (maxCount <= minCount) {
    return 1.0;
  }
  
  const intensity = (currentMiddle.count - minCount) / (maxCount - minCount);
  const { MIN, MAX, BASE, RANGE } = NODE_SIZE_CONFIG.MIDDLE.SIZE_MULTIPLIER;
  return Math.max(MIN, Math.min(MAX, BASE + intensity * RANGE));
};

/**
 * 노드의 크기 계산 (폰트 크기, 너비, 높이)
 * @param {Object} node 
 * @param {number} screenWidth 
 * @param {Object} newsCountsByMajor 
 * @param {Array} keywords 
 * @returns {Object} 
 */
export const calculateNodeDimensions = (node, screenWidth, newsCountsByMajor = {}, keywords = []) => {
  const label = node.label || '';
  
  // 중앙 노드
  if (node.level === NODE_LEVELS.CENTRAL) {
    const fontSize = getResponsiveValue(NODE_SIZE_CONFIG.CENTRAL.FONT_SIZE, screenWidth);
    const nodeHeight = getResponsiveValue(NODE_SIZE_CONFIG.CENTRAL.HEIGHT, screenWidth);
    const widthMultiplier = getResponsiveValue(NODE_SIZE_CONFIG.CENTRAL.WIDTH_MULTIPLIER, screenWidth);
    const nodeWidth = Math.max(
      NODE_SIZE_CONFIG.CENTRAL.MIN_WIDTH * widthMultiplier,
      (label.length * fontSize * 0.4 + 12) * widthMultiplier
    );
    
    return { fontSize, nodeWidth, nodeHeight };
  }
  
  // 대분류 노드
  if (node.level === NODE_LEVELS.MAJOR) {
    const parsed = parseNodeId(node.id);
    const majorKeyword = parsed.majorKeyword;
    const newsCount = newsCountsByMajor[majorKeyword] || 0;
    const maxNewsCount = Math.max(...Object.values(newsCountsByMajor), 0);
    const minNewsCount = Math.min(...Object.values(newsCountsByMajor), 0);
    
    const sizeMultiplier = calculateMajorSizeMultiplier(newsCount, maxNewsCount, minNewsCount);
    
    const baseFontSize = getResponsiveValue(NODE_SIZE_CONFIG.MAJOR.BASE_FONT_SIZE, screenWidth);
    const fontSize = baseFontSize * Math.max(NODE_SIZE_CONFIG.MAJOR.FONT_SIZE_MIN, sizeMultiplier);
    
    const baseHeight = getResponsiveValue(NODE_SIZE_CONFIG.MAJOR.BASE_HEIGHT, screenWidth);
    const nodeHeight = baseHeight * sizeMultiplier * NODE_SIZE_CONFIG.MAJOR.HEIGHT_MULTIPLIER;
    
    const widthMultiplier = getResponsiveValue(NODE_SIZE_CONFIG.MAJOR.WIDTH_MULTIPLIER, screenWidth);
    const nodeWidth = Math.max(
      NODE_SIZE_CONFIG.MAJOR.MIN_WIDTH * widthMultiplier,
      (label.length * fontSize * 0.3 + 8) * widthMultiplier
    ) * sizeMultiplier;
    
    return { fontSize, nodeWidth, nodeHeight };
  }
  
  // 중분류 노드
  if (node.level === NODE_LEVELS.MIDDLE) {
    const parsed = parseNodeId(node.id);
    const majorKeyword = parsed.majorKeyword;
    const middleKeyword = parsed.middleKeyword;
    
    const sizeMultiplier = calculateMiddleSizeMultiplier(majorKeyword, middleKeyword, keywords);
    
    const baseFontSize = getResponsiveValue(NODE_SIZE_CONFIG.MIDDLE.BASE_FONT_SIZE, screenWidth);
    const fontSize = baseFontSize * Math.max(NODE_SIZE_CONFIG.MIDDLE.FONT_SIZE_MIN, sizeMultiplier);
    
    const baseHeight = getResponsiveValue(NODE_SIZE_CONFIG.MIDDLE.BASE_HEIGHT, screenWidth);
    const nodeHeight = baseHeight * sizeMultiplier;
    
    const widthMultiplier = getResponsiveValue(NODE_SIZE_CONFIG.MIDDLE.WIDTH_MULTIPLIER, screenWidth);
    const nodeWidth = Math.max(
      NODE_SIZE_CONFIG.MIDDLE.MIN_WIDTH * widthMultiplier,
      (label.length * fontSize * 0.4 + 6) * widthMultiplier
    ) * sizeMultiplier;
    
    return { fontSize, nodeWidth, nodeHeight };
  }
  
  // 예상치 못한 레벨 - 기본값
  console.warn("Calculating dimensions for node with unexpected level:", node);
  return { fontSize: 6, nodeWidth: 50, nodeHeight: 12 };
};

/**
 * 대분류 노드 색상 계산
 * @param {number} rank 
 * @param {number} totalMajors 
 * @returns {Object}
 */
const calculateMajorNodeColor = (rank, totalMajors) => {
  const rankRatio = rank / Math.max(totalMajors - 1, 1);
  
  const { DARK, LIGHT, RANK_THRESHOLD, TEXT_DARK, TEXT_LIGHT, BORDER_MULTIPLIER } = NODE_COLORS.MAJOR;
  
  const r = Math.round(DARK.R + (LIGHT.R - DARK.R) * rankRatio);
  const g = Math.round(DARK.G + (LIGHT.G - DARK.G) * rankRatio);
  const b = Math.round(DARK.B + (LIGHT.B - DARK.B) * rankRatio);
  
  const nodeColor = `rgb(${r}, ${g}, ${b})`;
  const borderColor = `rgb(${Math.round(r * BORDER_MULTIPLIER)}, ${Math.round(g * BORDER_MULTIPLIER)}, ${Math.round(b * BORDER_MULTIPLIER)})`;
  const textColor = rankRatio > RANK_THRESHOLD ? TEXT_DARK : TEXT_LIGHT;
  
  return { nodeColor, borderColor, textColor };
};

/**
 * 중분류 노드 색상 계산
 * @param {string} majorKeyword 
 * @param {string} middleKeyword 
 * @param {Array} keywords 
 * @returns {Object} 
 */
const calculateMiddleNodeColor = (majorKeyword, middleKeyword, keywords) => {
  const majorData = keywords.find(cat => cat[MAJOR_KEY] === majorKeyword);
  
  if (!majorData || !majorData[MIDDLE_LIST_KEY]) {
    return {
      nodeColor: NODE_COLORS.MIDDLE.DEFAULT_FILL,
      borderColor: NODE_COLORS.MIDDLE.DEFAULT_BORDER,
      textColor: NODE_COLORS.MIDDLE.TEXT_LIGHT,
    };
  }
  
  const middleCategories = majorData[MIDDLE_LIST_KEY];
  const middleNewsCounts = middleCategories
    .map(middle => ({
      keyword: middle[MIDDLE_ITEM_KEY],
      count: middle[RELATED_NEWS_KEY]?.length || 0
    }))
    .sort((a, b) => b.count - a.count);
  
  const currentMiddleIndex = middleNewsCounts.findIndex(m => m.keyword === middleKeyword);
  const totalMiddles = middleNewsCounts.length;
  
  if (currentMiddleIndex < 0 || totalMiddles === 0) {
    return {
      nodeColor: NODE_COLORS.MIDDLE.DEFAULT_FILL,
      borderColor: NODE_COLORS.MIDDLE.DEFAULT_BORDER,
      textColor: NODE_COLORS.MIDDLE.TEXT_LIGHT,
    };
  }
  
  const rankRatio = currentMiddleIndex / Math.max(totalMiddles - 1, 1);
  const { DARK, LIGHT, RANK_THRESHOLD, TEXT_DARK, TEXT_LIGHT, BORDER_MULTIPLIER } = NODE_COLORS.MIDDLE;
  
  const r = Math.round(DARK.R + (LIGHT.R - DARK.R) * rankRatio);
  const g = Math.round(DARK.G + (LIGHT.G - DARK.G) * rankRatio);
  const b = Math.round(DARK.B + (LIGHT.B - DARK.B) * rankRatio);
  
  const nodeColor = `rgb(${r}, ${g}, ${b})`;
  const borderColor = `rgb(${Math.round(r * BORDER_MULTIPLIER)}, ${Math.round(g * BORDER_MULTIPLIER)}, ${Math.round(b * BORDER_MULTIPLIER)})`;
  const textColor = rankRatio > RANK_THRESHOLD ? TEXT_DARK : TEXT_LIGHT;
  
  return { nodeColor, borderColor, textColor };
};

/**
 * 노드 색상 계산
 * @param {Object} node 
 * @param {Object} majorRankings 
 * @param {Array} sortedMajors 
 * @param {Array} keywords 
 * @returns {Object}
 */
export const calculateNodeColors = (node, majorRankings = {}, sortedMajors = [], keywords = []) => {
  // 중앙 노드
  if (node.level === NODE_LEVELS.CENTRAL) {
    return {
      nodeColor: NODE_COLORS.CENTRAL.FILL,
      borderColor: NODE_COLORS.CENTRAL.BORDER,
      textColor: NODE_COLORS.CENTRAL.TEXT,
    };
  }
  
  // 대분류 노드
  if (node.level === NODE_LEVELS.MAJOR) {
    const parsed = parseNodeId(node.id);
    const majorKeyword = parsed.majorKeyword;
    const rank = majorRankings[majorKeyword] || 0;
    const totalMajors = sortedMajors.length;
    
    return calculateMajorNodeColor(rank, totalMajors);
  }
  
  // 중분류 노드
  if (node.level === NODE_LEVELS.MIDDLE) {
    const parsed = parseNodeId(node.id);
    const majorKeyword = parsed.majorKeyword;
    const middleKeyword = parsed.middleKeyword;
    
    return calculateMiddleNodeColor(majorKeyword, middleKeyword, keywords);
  }
  
  // 예상치 못한 레벨 - 기본 색상
  console.warn("Calculating colors for node with unexpected level:", node);
  return {
    nodeColor: NODE_COLORS.DEFAULT.FILL,
    borderColor: NODE_COLORS.DEFAULT.BORDER,
    textColor: NODE_COLORS.DEFAULT.TEXT,
  };
};

/**
 * 타원형 반지름 계산
 * @param {number} nodeWidth 
 * @param {number} nodeHeight
 * @returns {Object} 
 */
export const calculateEllipseRadius = (nodeWidth, nodeHeight) => {
  const radiusX = nodeWidth / 2;
  const radiusY = Math.max(nodeHeight / 2, radiusX * ELLIPSE_RATIO);
  return { radiusX, radiusY };
};

/**
 * 클릭 영역 타원형 반지름 계산
 * @param {number} nodeWidth 
 * @param {number} nodeHeight 
 * @returns {Object} 
 */
export const calculateClickAreaRadius = (nodeWidth, nodeHeight) => {
  const clickRadiusX = (nodeWidth / 2) * CLICK_AREA_SCALE;
  const clickRadiusY = Math.max((nodeHeight / 2) * CLICK_AREA_SCALE, clickRadiusX * ELLIPSE_RATIO);
  return { clickRadiusX, clickRadiusY };
};

