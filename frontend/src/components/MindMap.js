import React, { useState, useRef, useEffect } from "react";
import 'aframe';
import { ForceGraph2D } from 'react-force-graph';
import {
  generateInitialMindMapData,
  getSubNodesAndLinks
} from "../utils/mindmapData";

const MindMap = ({ keywords }) => {
  const initialData = generateInitialMindMapData(keywords);
  const [graphData, setGraphData] = useState(initialData);
  const [expandedCategory, setExpandedCategory] = useState(null);
  const fgRef = useRef();  

  const handleNodeClick = (node) => {
    const isMainCategory = keywords[node.id];

    if (isMainCategory) {
      const newCategory = node.id;

      const prevSubPrefix = expandedCategory ? `${expandedCategory}-` : null;

      const filteredNodes = graphData.nodes.filter(
        (n) => !(prevSubPrefix && n.id.startsWith(prevSubPrefix))
      );

      const filteredLinks = graphData.links.filter((l) => {
        const sourceId = typeof l.source === "object" ? l.source.id : l.source;
        const targetId = typeof l.target === "object" ? l.target.id : l.target;
        return !(
          prevSubPrefix &&
          sourceId === expandedCategory &&
          targetId.startsWith(prevSubPrefix)
        );
      });

      const { nodes: subNodes, links: subLinks } = getSubNodesAndLinks(
        newCategory,
        keywords[newCategory].소분류
      );

      setGraphData({
        nodes: [...filteredNodes, ...subNodes],
        links: [...filteredLinks, ...subLinks],
      });

      setExpandedCategory(newCategory);
    }
  };

  // 최초 렌더 시 중앙으로 줌인 및 퍼지게 설정
  useEffect(() => {
    if (fgRef.current) {
      fgRef.current.centerAt(0, 0, 1000); 
      fgRef.current.zoom(4);  
    }
  }, []);

  // 노드 위에 동그라미와 글자 표시
const nodeCanvasObject = (node, ctx) => {
  let nodeRadius;
  let fontSize;

  if (node.id === "뉴스") { // 메인 
    nodeRadius = 20; 
    fontSize = 12;
  } else if (node.group === 0) { // 대분류 
    nodeRadius = 10; 
    fontSize = 7;
  } else { // 소분류 
    nodeRadius = 10; 
    fontSize = 7;
  }

  // 동그라미 그리기
  ctx.beginPath();
  ctx.arc(node.x, node.y, nodeRadius, 0, Math.PI * 2, false);
  ctx.fillStyle = node.group === 0 ? 'lightblue' : 'lightgreen';
  ctx.fill();

  // 글자 그리기
  ctx.font = `${fontSize}px sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillStyle = node.group === 0 ? 'black' : 'white';
  ctx.fillText(node.id, node.x, node.y);
};


  return (
    <div style={{ width: '100%', height: '90vh' }}>
      <ForceGraph2D
        ref={fgRef} 
        graphData={graphData}
        nodeAutoColorBy="group"
        nodeCanvasObject={nodeCanvasObject}  
        onNodeClick={handleNodeClick}
        linkDirectionalParticles={0}
        linkDirectionalParticleSpeed={0.005}
        d3Force="collide"
        d3ForceCollide={50} 
        d3ForceManyBody={0}
        forceEngine="d3"
        d3ForceCenter={null} 
        linkWidth={2}
        linkColor="#aaa"
        nodeRelSize={10} 
        nodeId="id"
      />
    </div>
  );
};

export default MindMap;
