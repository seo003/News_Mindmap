/**
 * 마인드맵 유틸리티 함수들
 */

import { FORCE_CONFIG, BREAKPOINTS } from './mindmapConstants';

// 데이터 구조 키 상수
export const MAJOR_KEY = "majorKeyword";
export const MIDDLE_LIST_KEY = "middleKeywords";
export const MIDDLE_ITEM_KEY = "middleKeyword";
export const RELATED_NEWS_KEY = "relatedNews"; 
export const OTHER_NEWS_KEY = "otherNews";

// 키워드 표시용 함수
export const formatKeywordForDisplay = (keyword) => {
    return keyword ? keyword.replace(/-/g, ' ') : '';
};

// 노드 ID 생성 함수
export const generateNodeId = (majorValue, middleValue = null) => {
    const safe = (value) => value ? String(value).replace(/[^a-zA-Z0-9가-힣_.-]/g, '') : '';

    const safeMajor = safe(majorValue);
    const safeMiddle = safe(middleValue);

    // 중앙 노드("뉴스")
    if (!majorValue && !middleValue) return "뉴스"; 

    // 대분류 노드
    if (majorValue && !middleValue) return `MAJOR_${safeMajor}`; 

    // 중분류 노드
    if (majorValue && middleValue) return `MIDDLE_${safeMajor}_${safeMiddle}`; 

    // 유효하지 않은 조합 시 null 반환
    console.error("generateNodeId: Invalid input values for ID.", {majorValue, middleValue});
    return null;
};

// 노드 ID에서 키워드 값 및 타입 파싱
export const parseNodeId = (nodeId) => {
    // null 또는 undefined nodeId 처리
    if (!nodeId) {
         console.warn("parseNodeId: Received null or undefined nodeId.");
         return { level: -1, type: 'invalid', majorKeyword: null, middleKeyword: null };
    }
    
    // 중앙 노드 처리
    if (nodeId === "뉴스") return { level: 0, type: 'central', majorKeyword: null, middleKeyword: null };

    const parts = nodeId.split('_');
    const type = parts[0]; 

    try {
        // 대분류 노드 파싱
        if (type === 'MAJOR' && parts.length === 2) {
            return { level: 1, type: 'major', majorKeyword: parts[1], middleKeyword: null };
        }
        // 중분류 노드 파싱
        if (type === 'MIDDLE' && parts.length === 3) {
            return { level: 2, type: 'middle', majorKeyword: parts[1], middleKeyword: parts[2] };
        }
    } catch (e) {
        console.error(`parseNodeId Error parsing ID "${nodeId}": ${e}`);
        return { level: -1, type: 'error', majorKeyword: null, middleKeyword: null };
    }

    console.warn("parseNodeId: Unknown node ID format:", nodeId);
    return { level: -1, type: 'unknown', majorKeyword: null, middleKeyword: null };
};

// 노드 객체와 원래 키워드를 기반으로 해당 레벨의 데이터 객체를 찾는 헬퍼 함수 
export const findNestedData = (node, keywordsData) => {
    if (!keywordsData || !Array.isArray(keywordsData) || !node) {
        console.error("findNestedData: Invalid inputs.");
        return null;
    }
     // null 또는 undefined nodeId 처리
    if (!node.id) {
         console.warn("findNestedData: Received node with null or undefined id.");
         return null;
    }

    const { id, level } = node;
    const parsed = parseNodeId(id);
    // parseNodeId에서 오류가 발생했거나 알 수 없는 타입이면 여기서 종료
    if (parsed.level === -1) {
         console.warn(`findNestedData: Skipping node with unparseable ID: ${id}`);
         return null;
    }

    const { majorKeyword, middleKeyword, type } = parsed;


    // // 중앙노드는 직접 매핑 X
    if (level === 0) return null;

    // 대분류 데이터 찾기 
    const majorCat = keywordsData.find(cat => cat[MAJOR_KEY] === majorKeyword);
    if (!majorCat) {
        return null;
    }

    // 대분류 클릭 시 대분류 데이터 반환
    if (level === 1 && type === 'major') {
        return majorCat; 
    }

    // 중분류 클릭 시 중분류 데이터 반환
    if (level === 2 && type === 'middle') {
        const middleCat = majorCat[MIDDLE_LIST_KEY]?.find(mid => mid[MIDDLE_ITEM_KEY] === middleKeyword);
        return middleCat; 
    }

    return null;
};


// 대분류별 뉴스 개수 계산 함수
export const calculateNewsCountsByMajor = (keywordsData) => {
    if (!keywordsData || !Array.isArray(keywordsData)) {
        return {};
    }
    
    const newsCounts = {};
    
    keywordsData.forEach((majorCatObj) => {
        const majorKeyword = majorCatObj[MAJOR_KEY];
        if (!majorKeyword) return;
        
        let totalNewsCount = 0;
        
        // 중분류들의 뉴스 개수 합산
        if (majorCatObj[MIDDLE_LIST_KEY] && Array.isArray(majorCatObj[MIDDLE_LIST_KEY])) {
            majorCatObj[MIDDLE_LIST_KEY].forEach(middleCat => {
                if (middleCat[RELATED_NEWS_KEY] && Array.isArray(middleCat[RELATED_NEWS_KEY])) {
                    totalNewsCount += middleCat[RELATED_NEWS_KEY].length;
                }
            });
        }
        
        // 기타 뉴스 개수 추가
        if (majorCatObj[OTHER_NEWS_KEY] && Array.isArray(majorCatObj[OTHER_NEWS_KEY])) {
            totalNewsCount += majorCatObj[OTHER_NEWS_KEY].length;
        }
        
        newsCounts[majorKeyword] = totalNewsCount;
    });
    
    return newsCounts;
};

// 색상 진하기 계산 함수
export const calculateColorIntensity = (newsCount, maxNewsCount, minNewsCount = 0) => {
    if (maxNewsCount === minNewsCount) return 0.5; 
    
    const normalizedCount = (newsCount - minNewsCount) / (maxNewsCount - minNewsCount);
    // 정규화 (너무 연하지 않게)
    return Math.max(0.3, Math.min(1.0, normalizedCount));
};


/**
 * 초기 마인드맵 데이터 생성 함수
 */
export const generateInitialMindMapData = (keywordsData) => {
    // 중앙 노드 ID 생성 
    const centralNodeId = generateNodeId(); 
    if (!centralNodeId) {
         console.error("Failed to generate central node ID.");
         return { nodes: [], links: [] };
    }
    
    // 초기 노드 배열
    const nodes = [{
        id: centralNodeId,
        group: 0,
        level: 0,
        label: "뉴스",
        type: 'central',
        parentId: null,
    }];
    const links = [];

    if (!keywordsData || !Array.isArray(keywordsData)) {
        console.error("generateInitialMindMapData: Invalid keywords data format:", keywordsData);
        return { nodes, links };
    }

    keywordsData.forEach((majorCatObj) => {
        const majorKeywordValue = majorCatObj[MAJOR_KEY];
        
        // 중분류와 기타 뉴스 모두 체크
        const middleCategories = majorCatObj[MIDDLE_LIST_KEY] || [];
        const otherNews = majorCatObj[OTHER_NEWS_KEY] || [];
        
        // 중분류가 및 실제 뉴스가 있는지 확인
        const hasValidMiddleCategories = middleCategories.some(middle => 
            middle[MIDDLE_ITEM_KEY] && 
            middle[RELATED_NEWS_KEY] && 
            Array.isArray(middle[RELATED_NEWS_KEY]) && 
            middle[RELATED_NEWS_KEY].length > 0
        );
        
        // 기타 뉴스가 실제로 있는지 확인
        const hasValidOtherNews = Array.isArray(otherNews) && otherNews.length > 0;
        
        const hasContent = hasValidMiddleCategories || hasValidOtherNews;

        if (majorKeywordValue && hasContent) { // 유효한 경우만 노드 생성
            // 대분류 노드 ID 생성
            const majorNodeId = generateNodeId(majorKeywordValue); 

            if (!majorNodeId) {
                 console.warn("Skipping major category due to invalid ID generated for value:", majorKeywordValue);
                 return; 
            }

            // 이미 존재하는지 확인 
            if (!nodes.some(n => n.id === majorNodeId)) {
                nodes.push({ id: majorNodeId, group: 1, level: 1, label: formatKeywordForDisplay(majorKeywordValue), type: 'major', parentId: centralNodeId });
                // 중앙 노드에서 대분류 노드로 링크 추가
                links.push({ source: centralNodeId, target: majorNodeId });
            }
        } else {
            console.warn("generateInitialMindMapData: Skipping category with no name:", majorCatObj);
        }
    });

    // 화면 크기에 따라 초기 분산 반지름 조정
    const screenWidth = window.innerWidth;
    let initialSpreadRadius = FORCE_CONFIG.INITIAL_SPREAD_RADIUS.DEFAULT;
    if (screenWidth < BREAKPOINTS.SMALL) {
        initialSpreadRadius = FORCE_CONFIG.INITIAL_SPREAD_RADIUS.SMALL;
    } else if (screenWidth < BREAKPOINTS.MEDIUM) {
        initialSpreadRadius = FORCE_CONFIG.INITIAL_SPREAD_RADIUS.MEDIUM;
    }

    nodes.forEach(node => {
      if (node.level === 0) {
        // 중앙 노드 (0,0)
        node.x = 0;
        node.y = 0;
        node.fx = 0;
        node.fy = 0;
      } else {
        // 다른 노드: 랜덤 위치
        const angle = Math.random() * 2 * Math.PI;
        const radius = initialSpreadRadius * Math.sqrt(Math.random()); // 원형으로 고르게
        node.x = radius * Math.cos(angle);
        node.y = radius * Math.sin(angle);
      }
    });

    return { nodes, links };
};