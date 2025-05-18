import React, { useState, useRef, useEffect } from "react";
import 'aframe';
import { ForceGraph2D } from 'react-force-graph';

import { useMindmapHandler } from "./useMindmapHandler";
import {
  generateInitialMindMapData,
  parseNodeId,
} from "../utils/mindmapUtil";


const MindMap = ({ keywords }) => {
  const initialData = generateInitialMindMapData(keywords);
  const [graphData, setGraphData] = useState(initialData);

  const [expandedNodeIds, setExpandedNodeIds] = useState(new Set());

  const fgRef = useRef();

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
    console.log("Keywords prop updated, regenerating initial mind map data.");
    const newInitialData = generateInitialMindMapData(keywords);
    setGraphData(newInitialData);
    setExpandedNodeIds(new Set());
    setSelectedNews(null);
    setSelectedNodeIdForNews(null);
    if (fgRef.current) {
      fgRef.current.centerAt(0, 0, 1000);
    }
  }, [keywords, setGraphData, setExpandedNodeIds, setSelectedNews, setSelectedNodeIdForNews]);


  useEffect(() => {
    if (fgRef.current) {
      fgRef.current.d3Force('charge').strength(-450);
    }
  }, [graphData]);


  const nodeCanvasObject = (node, ctx, scale) => {
    // 유효한지 확인
    if (!node || !node.id) {
      console.warn("Skipping rendering for invalid node:", node);
      return;
    }

    const label = node.label;

    let nodeRadius;
    let fontSize;
    let circleColor;
    let textColor = 'black';

    // 노드 별 크기 조절
    if (node.level === 0) {
      nodeRadius = 25; fontSize = 20; circleColor = '#ff9800'; textColor = 'white';
    } else if (node.level === 1) {
      nodeRadius = 18; fontSize = 20; circleColor = '#2196f3'; textColor = 'white';
    } else if (node.level === 2) {
      nodeRadius = 18; fontSize = 20; circleColor = '#4caf50'; textColor = 'white';
    } else {
      nodeRadius = 8; fontSize = 6; circleColor = '#9e9e9e'; textColor = 'black';
      console.warn("Rendering node with unexpected level:", node);
    }

    ctx.beginPath();
    ctx.arc(node.x, node.y, nodeRadius, 0, 2 * Math.PI, false);
    ctx.fillStyle = circleColor;
    ctx.fill();

    // 줌 관계 없이 글자 크기 일정
    const scaledFontSize = fontSize / scale;
    ctx.font = `${scaledFontSize}px sans-serif`;
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
    let r = 18;
    if (node.level === 0) r = 25;
    else if (node.level === 1) r = 18;

    // 클릭 영역 계산
    const clickRadius = r * 1.5;

    ctx.beginPath();
    ctx.arc(node.x, node.y, clickRadius, 0, 2 * Math.PI, false);
    ctx.fillStyle = 'rgba(0, 0, 0, 0)';
    ctx.fill();
  };


  return (
    <div style={{ position: 'relative', width: '100%', height: '90vh' }}>
      <ForceGraph2D
        ref={fgRef}
        graphData={graphData}
        nodeLabel="label"
        nodeCanvasObject={nodeCanvasObject}
        nodePointerAreaCanvasObject={nodePointerAreaCanvasObject}
        onNodeClick={handleNodeClick}
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

          ctx.beginPath();
          ctx.moveTo(start.x, start.y);
          ctx.lineTo(end.x, end.y);
          ctx.strokeStyle = link.color || '#cccccc';
          ctx.lineWidth = link.width || 1;
          ctx.stroke();
        }}

      />

      {/* 뉴스 목록 */}
      {selectedNews !== null && selectedNodeIdForNews && (
        <div
          style={{
            position: 'absolute', top: '20px', right: '20px',
            backgroundColor: 'white', border: '1px solid #ccc', padding: '15px', borderRadius: '8px', boxShadow: '0 2px 10px rgba(0, 0, 0, 0.1)',
            maxWidth: '400px', maxHeight: '80%', overflowY: 'auto', zIndex: 1000,
          }}
        >
          <button
            onClick={() => { setSelectedNews(null); setSelectedNodeIdForNews(null); }}
            style={{
              position: 'absolute', top: '5px', right: '5px',
              background: 'none', border: 'none', fontSize: '1.2em',
              cursor: 'pointer', color: '#888'
            }}
          >
            &times;
          </button>
          <h4>{`'${selectedNodeIdForNews ? (parseNodeId(selectedNodeIdForNews)?.middleKeyword || parseNodeId(selectedNodeIdForNews)?.majorKeyword || '뉴스') : '뉴스'}' 관련 뉴스`}</h4>
          {selectedNews.length === 0 ? (
            <p>관련 뉴스가 없습니다.</p>
          ) : (
            <ul>
              {selectedNews.map((news, index) => (
                // 뉴스 객체가 유효한지 확인
                news && news.link && news.title ? (
                  // 뉴스 링크 클릭 시 새 탭에 표시
                  <li key={index} style={{ marginBottom: '10px', wordBreak: 'break-word' }}>
                    <a href={news.link} target="_blank" rel="noopener noreferrer" style={{ textDecoration: 'none', color: '#0066cc' }}>
                      <strong>{news.title}</strong>
                    </a>
                  </li>
                ) : (
                  <li key={index} style={{ marginBottom: '10px', color: '#999' }}>유효하지 않은 뉴스 항목</li>
                )
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
};

export default MindMap;