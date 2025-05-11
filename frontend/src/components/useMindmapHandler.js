import { useState, useCallback } from "react";

import {
    generateNodeId,
    parseNodeId,
    findNestedData,
    MIDDLE_LIST_KEY,
    MIDDLE_ITEM_KEY,
    RELATED_NEWS_KEY,
} from "../utils/mindmapUtil";

export function useMindmapHandler({
    keywords,
    fgRef,
    graphData,
    setGraphData,
    expandedNodeIds,
    setExpandedNodeIds,
}) {
    const [selectedNews, setSelectedNews] = useState(null);
    const [selectedNodeIdForNews, setSelectedNodeIdForNews] = useState(null);
    
    // 노드 축소 시 하위 노드 검색 함수
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

    // 노드 클릭 시 노드 상태 확인 후 수행행
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

            console.log(`Node clicked: ID=${nodeId}, Level=${nodeLevel}, Type=${parsed.type}`);


            // 메인 노드("뉴스") 클릭
            if (nodeLevel === 0) {
                setSelectedNews(null);
                setSelectedNodeIdForNews(null);
                if (fgRef.current) {
                    fgRef.current.centerAt(0, 0, 1000);
                }
                return;
            }

            if (nodeLevel !== 2) {
                setSelectedNews(null);
                setSelectedNodeIdForNews(null);
            }


            // 대분류 재클릭 시 축소
            if (nodeLevel === 1 && expandedNodeIds.has(nodeId)) {
                console.log(`Collapsing node: ${nodeId} (Level ${nodeLevel})`);

                const nodesToRemove = collectDescendants(nodeId);

                // 노드 제거
                const newNodes = graphData.nodes.filter((n) => !nodesToRemove.has(n.id));

                const newLinks = graphData.links.filter((link) => {
                    // 링크 source/target이 유효한지 확인
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

                return;
            }

            // 하위 노드 생성 및 추가
            const currentNodeData = findNestedData(node, keywords);

            if (!currentNodeData) {
                console.warn("Could not find data for node:", nodeId);
                return;
            }

            const nodesToAdd = [];
            const linksToAdd = [];

            // 대분류 클릭 시 중분류 생성
            if (nodeLevel === 1) {
                console.log(`Expanding Major: ${parsed.majorKeyword}`);

                // 대분류 아래의 중분류 목록 확인
                const middleCategoriesData = currentNodeData[MIDDLE_LIST_KEY] || [];

                if (middleCategoriesData.length > 0) {
                    console.log(`Found ${middleCategoriesData.length} Middle Categories`);
                    middleCategoriesData.forEach((middleCatObj) => {
                        const middleKeywordValue = middleCatObj[MIDDLE_ITEM_KEY];
                        // 중분류가 존재할 경우에만 처리
                        if (middleKeywordValue) {
                            // 중분류 노드 ID 생성
                            const middleNodeId = generateNodeId(
                                parsed.majorKeyword,
                                middleKeywordValue
                            );

                            // 중분류가 유효하고, 그래프에 아직 없는 경우에만 추가
                            if (middleNodeId && !graphData.nodes.some((n) => n.id === middleNodeId)) {
                                nodesToAdd.push({
                                    id: middleNodeId,
                                    group: 2,
                                    level: 2,
                                    label: middleKeywordValue,
                                    type: "middle",
                                    parentId: nodeId,
                                });
                                // 링크 추가
                                linksToAdd.push({ source: nodeId, target: middleNodeId });
                            } else if (!middleNodeId) {
                                console.warn("Skipping middle node due to invalid ID generated for value:", middleKeywordValue);
                            }
                        } else {
                            console.warn("Skipping middle category object with no keyword value:", middleCatObj);
                        }
                    });
                } else {
                    console.log(`No Middle Categories found for Major: ${parsed.majorKeyword}`);
                }
            }

            // 중분류 클릭 시 뉴스 표시
            else if (nodeLevel === 2) {
                console.log(`Clicked on Middle-category: ${parsed.middleKeyword}`);
                const newsList = currentNodeData?.[RELATED_NEWS_KEY] || [];
                setSelectedNews(newsList);
                setSelectedNodeIdForNews(nodeId);
                console.log(`Found ${newsList.length} related news items.`);
                return;
            }

            // 그 외 클릭 시 처리
            else {
                console.warn(`Clicked on a node with unexpected level or type. ID: ${nodeId}, Level: ${nodeLevel}, Type: ${parsed.type}`);
                setSelectedNews(null);
                setSelectedNodeIdForNews(null);
                return;
            }


            // 새로운 노드 및 링크 추가
            const uniqueNodesToAdd = nodesToAdd.filter(
                (node) => node && node.id && !graphData.nodes.some((n) => n.id === node.id)
            );

            // 중복되지 않는 링크만 추가
            const uniqueLinksToAdd = linksToAdd.filter((link) => {
                const sourceId =
                    typeof link.source === "object" ? link.source?.id : link.source;
                const targetId =
                    typeof link.target === "object" ? link.target?.id : link.target;

                // null/undefined/빈 문자열이면 무시
                if (!sourceId || !targetId) {
                    console.warn(
                        "Skipping link with invalid source/target ID during uniqueness check:",
                        link
                    );
                    return false;
                }

                // 동일한 쌍이 있는지 확인
                return !graphData.links.some(
                    (l) =>
                        (typeof l.source === "object" ? l.source.id : l.source) === sourceId &&
                        (typeof l.target === "object" ? l.target.id : l.target) === targetId
                );
            });


            // 노드 업데이트
            if (uniqueNodesToAdd.length > 0 || uniqueLinksToAdd.length > 0) {
                console.log(
                    `Adding ${uniqueNodesToAdd.length} nodes and ${uniqueLinksToAdd.length} links.`
                );
                setGraphData((prevGraphData) => ({
                    nodes: [...prevGraphData.nodes, ...uniqueNodesToAdd],
                    links: [...prevGraphData.links, ...uniqueLinksToAdd],
                }));

                if (nodeLevel === 1) {
                    setExpandedNodeIds((prevIds) => new Set(prevIds).add(nodeId));
                }
            } else {
                console.log(`No new nodes or links to add for node: ${nodeId}`);
            }
        },
        [graphData, setGraphData, expandedNodeIds, setExpandedNodeIds, keywords, fgRef, setSelectedNews, setSelectedNodeIdForNews, collectDescendants]
    );


    return {
        handleNodeClick,
        selectedNews,
        setSelectedNews,
        selectedNodeIdForNews,
        setSelectedNodeIdForNews,
    };
}