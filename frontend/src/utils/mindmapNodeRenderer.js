/**
 * 마인드맵 노드 렌더링 유틸리티
 */

import {
  calculateNodeDimensions,
  calculateNodeColors,
  calculateEllipseRadius,
  calculateClickAreaRadius,
} from './mindmapNodeCalculations';

/**
 * 노드를 캔버스에 렌더링
 * @param {Object} node - 노드 객체
 * @param {CanvasRenderingContext2D} ctx - 캔버스 컨텍스트
 * @param {number} scale - 스케일
 * @param {number} screenWidth - 화면 너비
 * @param {Object} newsCountsByMajor - 대분류별 뉴스 개수
 * @param {Object} majorRankings - 대분류별 순위
 * @param {Array} sortedMajors - 정렬된 대분류 목록
 * @param {Array} keywords - 전체 키워드 데이터
 */
export const renderNode = (
  node,
  ctx,
  scale,
  screenWidth,
  newsCountsByMajor,
  majorRankings,
  sortedMajors,
  keywords
) => {
  if (!node || !node.id) {
    console.warn("Skipping rendering for invalid node:", node);
    return;
  }

  // 노드 크기 계산
  const { fontSize, nodeWidth, nodeHeight } = calculateNodeDimensions(
    node,
    screenWidth,
    newsCountsByMajor,
    keywords
  );

  // 노드 색상 계산
  const { nodeColor, borderColor, textColor } = calculateNodeColors(
    node,
    majorRankings,
    sortedMajors,
    keywords
  );

  // 타원형 반지름 계산
  const { radiusX, radiusY } = calculateEllipseRadius(nodeWidth, nodeHeight);

  const x = node.x;
  const y = node.y;

  // 타원형 그리기
  ctx.beginPath();
  ctx.ellipse(x, y, radiusX, radiusY, 0, 0, 2 * Math.PI);
  ctx.fillStyle = nodeColor;
  ctx.fill();

  // 테두리 그리기
  ctx.strokeStyle = borderColor;
  ctx.lineWidth = 1;
  ctx.stroke();

  // 텍스트 그리기
  ctx.font = `${fontSize}px sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillStyle = textColor;
  ctx.fillText(node.label, x, y);
};

/**
 * 노드 클릭 영역을 캔버스에 렌더링
 * @param {Object} node - 노드 객체
 * @param {CanvasRenderingContext2D} ctx - 캔버스 컨텍스트
 * @param {number} scale - 스케일
 * @param {number} screenWidth - 화면 너비
 * @param {Object} newsCountsByMajor - 대분류별 뉴스 개수
 * @param {Array} keywords - 전체 키워드 데이터
 */
export const renderNodePointerArea = (
  node,
  ctx,
  scale,
  screenWidth,
  newsCountsByMajor,
  keywords
) => {
  if (!node || !node.id) {
    return;
  }

  // 노드 크기 계산
  const { nodeWidth, nodeHeight } = calculateNodeDimensions(
    node,
    screenWidth,
    newsCountsByMajor,
    keywords
  );

  // 클릭 영역 반지름 계산
  const { clickRadiusX, clickRadiusY } = calculateClickAreaRadius(nodeWidth, nodeHeight);

  const x = node.x;
  const y = node.y;

  // 타원형으로 클릭 영역 설정
  ctx.beginPath();
  ctx.ellipse(x, y, clickRadiusX, clickRadiusY, 0, 0, 2 * Math.PI);
  ctx.fillStyle = 'rgba(0, 0, 0, 0)';
  ctx.fill();
};

