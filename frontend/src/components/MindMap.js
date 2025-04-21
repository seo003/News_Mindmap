import React from "react";
import 'aframe';
import { ForceGraph2D } from 'react-force-graph';
import { generateMindMapData } from "../utils/mindmapData";

const MindMap = ({ keywords }) => {
  const data = generateMindMapData(keywords);

  return (
    <div >
      <ForceGraph2D
        graphData={data}
        nodeAutoColorBy="group"
        nodeLabel="id"
        linkDirectionalParticles={2}
        linkDirectionalParticleSpeed={0.005}
      />
    </div>
  );
};

export default MindMap;
