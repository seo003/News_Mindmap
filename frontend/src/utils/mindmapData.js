export const generateMindMapData = (keywords) => {
    const nodes = [{ id: "뉴스", group: 0 }];
    const links = [];
  
    Object.keys(keywords).forEach((category, i) => {
      const groupId = i + 1;
      nodes.push({ id: category, group: groupId });
      links.push({ source: "뉴스", target: category });
  
      keywords[category].소분류.forEach((sub) => {
        const subId = `${category}-${sub}`;
        nodes.push({ id: subId, group: groupId });
        links.push({ source: category, target: subId });
      });
    });
  
    return { nodes, links };
  };
  