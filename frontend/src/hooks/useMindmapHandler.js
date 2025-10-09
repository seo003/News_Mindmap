import { useState, useCallback } from "react";

import {
  generateNodeId,
  parseNodeId,
  findNestedData,
  formatKeywordForDisplay,
  MIDDLE_LIST_KEY,
  MIDDLE_ITEM_KEY,
  RELATED_NEWS_KEY,
} from "../utils/mindmapUtil";
import { MIN_NEWS_COUNT_FOR_DISPLAY, ANIMATION_DURATION, NODE_LEVELS } from "../utils/mindmapConstants";

/**
 * 마인드맵 인터랙션 핸들러 훅
 */
export function useMindmapHandler({
    keywords,
    fgRef,
    graphData,
    setGraphData,
    expandedNodeIds,
    setExpandedNodeIds,
}) {
    // 선택된 뉴스 상태
    const [selectedNews, setSelectedNews] = useState(null);
    
    // 뉴스 선택을 위한 노드 ID 상태
    const [selectedNodeIdForNews, setSelectedNodeIdForNews] = useState(null);
    
    /**
     * 노드 축소 시 하위 노드 검색
     */
    const collectDescendants = useCallback(
        (collapsedNodeId) => {
            const nodesToRemove = new Set();

            const descendants = graphData.nodes.filter((n) => n.parentId === collapsedNodeId);
            descendants.forEach((descendantNode) => {
                if (descendantNode && descendantNode.id && !nodesToRemove.has(descendantNode.id)) {
                    nodesToRemove.add(descendantNode.id);
                } else if (descendantNode && !descendantNode.id) {
                    console.warn("Found descendant node with null or undefined ID:", descendantNode);
                }
            });
            return nodesToRemove;
        },
        [graphData.nodes]
    );

    /**
     * 중앙 노드 클릭 핸들러
     */
    const handleCentralNodeClick = useCallback(() => {
        setSelectedNews(null);
        setSelectedNodeIdForNews(null);
        if (fgRef.current) {
            fgRef.current.centerAt(0, 0, ANIMATION_DURATION.CENTER_AT);
        }
    }, [fgRef, setSelectedNews, setSelectedNodeIdForNews]);

    /**
     * 대분류 노드 클릭 핸들러 - 축소 처리
     */
    const handleMajorNodeCollapse = useCallback((nodeId) => {
        // 중앙 노드('뉴스')로 focus
        if (fgRef.current) {
            fgRef.current.centerAt(0, 0, ANIMATION_DURATION.CENTER_AT);
        }

        const nodesToRemove = collectDescendants(nodeId);

        // 노드 제거
        const newNodes = graphData.nodes.filter((n) => !nodesToRemove.has(n.id));

        const newLinks = graphData.links.filter((link) => {
            const sourceId = typeof link.source === "object" ? link.source?.id : link.source;
            const targetId = typeof link.target === "object" ? link.target?.id : link.target;

            if (!sourceId || !targetId) {
                console.warn("Skipping invalid link:", link);
                return false;
            }

            return (
                newNodes.some((n) => n.id === sourceId) && newNodes.some((n) => n.id === targetId)
            );
        });

        setGraphData({ nodes: newNodes, links: newLinks });

        // 확장된 노드 목록에서 제거
        const newExpandedNodeIds = new Set(expandedNodeIds);
        newExpandedNodeIds.delete(nodeId);
        setExpandedNodeIds(newExpandedNodeIds);
    }, [fgRef, graphData, setGraphData, expandedNodeIds, setExpandedNodeIds, collectDescendants]);

    /**
     * 대분류 노드 클릭 핸들러 - 확장 처리
     */
    const handleMajorNodeExpand = useCallback((node, nodeId, currentNodeData) => {
        // 클릭된 노드로 focus
        if (fgRef.current) {
            fgRef.current.centerAt(node.x, node.y, ANIMATION_DURATION.CENTER_AT);
        }

        const parsed = parseNodeId(nodeId);
        const middleCategoriesData = currentNodeData[MIDDLE_LIST_KEY] || [];

        if (middleCategoriesData.length === 0) {
            return;
        }

        const nodesToAdd = [];
        const linksToAdd = [];

        middleCategoriesData.forEach((middleCatObj) => {
            const middleKeywordValue = middleCatObj[MIDDLE_ITEM_KEY];
            const relatedNews = middleCatObj[RELATED_NEWS_KEY] || [];
            
            // 중분류가 존재하고 뉴스가 충분한 경우에만 처리
            if (middleKeywordValue && Array.isArray(relatedNews) && relatedNews.length >= MIN_NEWS_COUNT_FOR_DISPLAY) {
                const middleNodeId = generateNodeId(parsed.majorKeyword, middleKeywordValue);

                if (middleNodeId && !graphData.nodes.some((n) => n.id === middleNodeId)) {
                    nodesToAdd.push({
                        id: middleNodeId,
                        group: 2,
                        level: NODE_LEVELS.MIDDLE,
                        label: formatKeywordForDisplay(middleKeywordValue),
                        type: "middle",
                        parentId: nodeId,
                    });
                    linksToAdd.push({ source: nodeId, target: middleNodeId });
                } else if (!middleNodeId) {
                    console.warn(`중분류 노드 ID 생성 실패: ${middleKeywordValue}`);
                }
            } else {
                console.warn(`건너뛰기: 뉴스 부족 (뉴스 개수: ${relatedNews.length}, 최소 필요: ${MIN_NEWS_COUNT_FOR_DISPLAY}개)`);
            }
        });

        // 중복되지 않는 노드만 추가
        const uniqueNodesToAdd = nodesToAdd.filter(
            (node) => node && node.id && !graphData.nodes.some((n) => n.id === node.id)
        );

        // 중복되지 않는 링크만 추가
        const uniqueLinksToAdd = linksToAdd.filter((link) => {
            const sourceId = typeof link.source === "object" ? link.source?.id : link.source;
            const targetId = typeof link.target === "object" ? link.target?.id : link.target;

            if (!sourceId || !targetId) {
                console.warn("Skipping link with invalid source/target ID during uniqueness check:", link);
                return false;
            }

            return !graphData.links.some(
                (l) =>
                    (typeof l.source === "object" ? l.source.id : l.source) === sourceId &&
                    (typeof l.target === "object" ? l.target.id : l.target) === targetId
            );
        });

        // 노드 업데이트
        if (uniqueNodesToAdd.length > 0 || uniqueLinksToAdd.length > 0) {
            setGraphData((prevGraphData) => ({
                nodes: [...prevGraphData.nodes, ...uniqueNodesToAdd],
                links: [...prevGraphData.links, ...uniqueLinksToAdd],
            }));

            setExpandedNodeIds((prevIds) => new Set(prevIds).add(nodeId));
        }
    }, [fgRef, graphData, setGraphData, setExpandedNodeIds]);

    /**
     * 대분류 노드 클릭 핸들러
     */
    const handleMajorNodeClick = useCallback((node, nodeId) => {
        // 재클릭 시 축소
        if (expandedNodeIds.has(nodeId)) {
            handleMajorNodeCollapse(nodeId);
            return;
        }

        // 확장
        const currentNodeData = findNestedData(node, keywords);
        if (!currentNodeData) {
            console.warn("Could not find data for node:", nodeId);
            return;
        }

        handleMajorNodeExpand(node, nodeId, currentNodeData);
    }, [expandedNodeIds, keywords, handleMajorNodeCollapse, handleMajorNodeExpand]);

    /**
     * 중분류 노드 클릭 핸들러
     */
    const handleMiddleNodeClick = useCallback((node, nodeId) => {
        // 중분류 재클릭 시 뉴스 패널 닫기 (부모 노드로 focus)
        if (selectedNodeIdForNews === nodeId) {
            const parentNode = graphData.nodes.find(n => n.id === node.parentId);
            if (parentNode && fgRef.current) {
                fgRef.current.centerAt(parentNode.x, parentNode.y, ANIMATION_DURATION.CENTER_AT);
            }
            
            setSelectedNews(null);
            setSelectedNodeIdForNews(null);
            return;
        }
        
        const currentNodeData = findNestedData(node, keywords);
        const newsList = currentNodeData?.[RELATED_NEWS_KEY] || [];
        setSelectedNews(newsList);
        setSelectedNodeIdForNews(nodeId);
    }, [selectedNodeIdForNews, graphData.nodes, fgRef, keywords, setSelectedNews, setSelectedNodeIdForNews]);

    /**
     * 노드 클릭 메인 핸들러
     */
    const handleNodeClick = useCallback(
        (node) => {
            // 클릭된 노드가 유효한지 확인
            if (!node || !node.id) {
                console.warn("Clicked on an invalid node.", node);
                return;
            }

            const nodeId = node.id;
            const nodeLevel = node.level;
            const parsed = parseNodeId(nodeId);

            // 파싱 실패한 노드 처리 X
            if (parsed.level === -1) {
                console.warn(`Clicked on unparseable node ID: ${nodeId}`);
                return;
            }

            // 레벨별 핸들러 호출
            switch (nodeLevel) {
                case NODE_LEVELS.CENTRAL:
                    handleCentralNodeClick();
                    break;
                    
                case NODE_LEVELS.MAJOR:
                    setSelectedNews(null);
                    setSelectedNodeIdForNews(null);
                    handleMajorNodeClick(node, nodeId);
                    break;
                    
                case NODE_LEVELS.MIDDLE:
                    handleMiddleNodeClick(node, nodeId);
                    break;
                    
                default:
                    console.warn(`Clicked on a node with unexpected level. ID: ${nodeId}, Level: ${nodeLevel}`);
                    setSelectedNews(null);
                    setSelectedNodeIdForNews(null);
            }
        },
        [handleCentralNodeClick, handleMajorNodeClick, handleMiddleNodeClick, setSelectedNews, setSelectedNodeIdForNews]
    );


    // 훅에서 반환할 인터랙션 핸들러와 상태
    return {
        handleNodeClick,
        selectedNews,
        setSelectedNews,
        selectedNodeIdForNews,
        setSelectedNodeIdForNews,
    };
}

